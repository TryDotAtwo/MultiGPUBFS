#include "segmented_history.cuh"
#include "generic_route.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

static void require(bool value, const char* message) {
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(1);} } while(0)
using State = std::array<uint8_t,2>;

struct Owned {
    MgbfsSegment segment{};
    uint32_t* count{};
    uint32_t* future{};
    uint32_t* error{};
};

static Owned make_segment(uint32_t capacity, uint32_t banks=1, uint32_t width=1) {
    Owned x;
    x.segment.capacity = capacity;
    x.segment.banks = banks;
    x.segment.slots_count = 32;
    while(x.segment.slots_count < uint64_t(capacity)*2) x.segment.slots_count *= 2;
    CUDA_OK(cudaMalloc((void**)&x.segment.slots, x.segment.slots_count * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc((void**)&x.segment.states, capacity * banks * 2 * width));
    CUDA_OK(cudaMalloc((void**)&x.segment.positions, capacity * banks * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc((void**)&x.count, sizeof(uint32_t)));
    CUDA_OK(cudaMalloc((void**)&x.future, capacity * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc((void**)&x.error, sizeof(uint32_t)));
    CUDA_OK(cudaMemset(x.segment.slots, 0xff, x.segment.slots_count * sizeof(uint64_t)));
    CUDA_OK(cudaMemset(x.count, 0, sizeof(uint32_t)));
    CUDA_OK(cudaMemset(x.error, 0, sizeof(uint32_t)));
    return x;
}
static void destroy(Owned& x) {
    CUDA_OK(cudaFree(x.segment.slots)); CUDA_OK(cudaFree(x.segment.states));
    CUDA_OK(cudaFree(x.segment.positions)); CUDA_OK(cudaFree(x.count));
    CUDA_OK(cudaFree(x.future)); CUDA_OK(cudaFree(x.error));
}
static uint32_t submit(Owned& dst, const std::vector<MgbfsSegment>& sealed, const std::vector<State>& input, uint32_t bank=0, uint64_t hash_value=0x1234567812345678ULL) {
    const uint32_t n = uint32_t(input.size());
    std::vector<uint8_t> soa(n*2);
    for(uint32_t i=0;i<n;++i) { soa[i]=input[i][0]; soa[n+i]=input[i][1]; }
    std::vector<uint64_t> hashes(n, hash_value); // Deliberately force every hash to collide.
    uint8_t* d_input=nullptr; uint64_t* d_hash=nullptr; MgbfsSegment* d_sealed=nullptr;
    CUDA_OK(cudaMalloc((void**)&d_input, soa.size())); CUDA_OK(cudaMalloc((void**)&d_hash, hashes.size()*8));
    CUDA_OK(cudaMemcpy(d_input, soa.data(), soa.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_hash, hashes.data(), hashes.size()*8, cudaMemcpyHostToDevice));
    if(!sealed.empty()) { CUDA_OK(cudaMalloc((void**)&d_sealed, sealed.size()*sizeof(MgbfsSegment)));
        CUDA_OK(cudaMemcpy(d_sealed,sealed.data(),sealed.size()*sizeof(MgbfsSegment),cudaMemcpyHostToDevice)); }
    CUDA_OK(cudaMemset(dst.error,0,sizeof(uint32_t)));
    CUDA_OK((cudaError_t)mgbfs_segmented_accept_u8(d_input,d_hash,n,2,d_sealed,uint32_t(sealed.size()),dst.segment,bank,dst.count,dst.future,dst.error,nullptr));
    CUDA_OK(cudaDeviceSynchronize());
    uint32_t error=0; CUDA_OK(cudaMemcpy(&error,dst.error,4,cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(d_input)); CUDA_OK(cudaFree(d_hash)); if(d_sealed) CUDA_OK(cudaFree(d_sealed));
    return error;
}
static std::set<State> values(const Owned& x, uint32_t expected) {
    uint32_t count=0; CUDA_OK(cudaMemcpy(&count,x.count,4,cudaMemcpyDeviceToHost));
    require(count==expected,"unexpected accepted count");
    std::vector<uint8_t> soa(x.segment.capacity*2);
    CUDA_OK(cudaMemcpy(soa.data(),x.segment.states,soa.size(),cudaMemcpyDeviceToHost));
    std::set<State> result;
    for(uint32_t i=0;i<count;++i) result.insert({soa[i],soa[x.segment.capacity+i]});
    return result;
}
static void multisegment_collision_stress() {
    Owned first=make_segment(256), second=make_segment(256), third=make_segment(256);
    auto batch=[](unsigned unique) {
        std::vector<State> v;
        for(unsigned repeat=0;repeat<3;++repeat)
            for(unsigned i=0;i<unique;++i) v.push_back({uint8_t(i&255),uint8_t(i>>8)});
        std::reverse(v.begin(),v.end());
        return v;
    };
    require(submit(first,{},batch(200))==0,"collision stress first segment");
    require(values(first,200).size()==200,"first segment lost a state");
    require(submit(second,{first.segment},batch(400))==0,"collision stress second segment");
    require(values(second,200).size()==200,"second segment lost a state");
    require(submit(third,{first.segment,second.segment},batch(600))==0,"collision stress third segment");
    auto actual=values(third,200);
    std::set<State> expected;
    for(unsigned i=400;i<600;++i) expected.insert({uint8_t(i&255),uint8_t(i>>8)});
    require(actual==expected,"wrong result across multiple sealed segments");
    for(const Owned* x : {&first,&second,&third}) {
        std::vector<uint64_t> slots(x->segment.slots_count);
        CUDA_OK(cudaMemcpy(slots.data(),x->segment.slots,slots.size()*8,cudaMemcpyDeviceToHost));
        for(uint64_t slot:slots) require(slot==~uint64_t(0) || slot==~uint64_t(1) || !(slot>>63),"pending token survived successful batch");
    }
    destroy(first); destroy(second); destroy(third);
}
static void wide_values() {
    Owned x=make_segment(4,1,sizeof(int64_t));
    std::vector<int64_t> data={-7,1LL<<40,-7,1LL<<40,INT64_MAX,-1,INT64_MAX,-1};
    std::vector<uint64_t> hashes(4,5); int64_t* in=nullptr; uint64_t* dh=nullptr;
    CUDA_OK(cudaMalloc((void**)&in,data.size()*8)); CUDA_OK(cudaMalloc((void**)&dh,hashes.size()*8));
    CUDA_OK(cudaMemcpy(in,data.data(),data.size()*8,cudaMemcpyHostToDevice)); CUDA_OK(cudaMemcpy(dh,hashes.data(),hashes.size()*8,cudaMemcpyHostToDevice));
    CUDA_OK((cudaError_t)mgbfs_segmented_accept_i64(in,dh,4,2,nullptr,0,x.segment,0,x.count,x.future,x.error,nullptr));
    CUDA_OK(cudaDeviceSynchronize());
    uint32_t count=0,error=0; CUDA_OK(cudaMemcpy(&count,x.count,4,cudaMemcpyDeviceToHost)); CUDA_OK(cudaMemcpy(&error,x.error,4,cudaMemcpyDeviceToHost));
    require(error==0 && count==2,"wide state count");
    std::vector<int64_t> states(8); CUDA_OK(cudaMemcpy(states.data(),x.segment.states,states.size()*8,cudaMemcpyDeviceToHost));
    std::set<std::array<int64_t,2>> actual;
    for(uint32_t i=0;i<count;++i) actual.insert({states[i],states[4+i]});
    require(actual==std::set<std::array<int64_t,2>>{{-7,INT64_MAX},{1LL<<40,-1}},"wide values truncated or changed");
    CUDA_OK(cudaFree(in)); CUDA_OK(cudaFree(dh)); destroy(x);
}
static void parallel_shards() {
    Owned a=make_segment(4), b=make_segment(4); uint8_t* in=nullptr; uint64_t* hashes=nullptr;
    std::vector<uint8_t> data={1,2,3,1,2,3,11,12,13,11,12,13}; std::vector<uint64_t> hv(6,7);
    CUDA_OK(cudaMalloc((void**)&in,data.size())); CUDA_OK(cudaMalloc((void**)&hashes,hv.size()*8));
    CUDA_OK(cudaMemcpy(in,data.data(),data.size(),cudaMemcpyHostToDevice)); CUDA_OK(cudaMemcpy(hashes,hv.data(),hv.size()*8,cudaMemcpyHostToDevice));
    cudaStream_t sa=nullptr,sb=nullptr; CUDA_OK(cudaStreamCreate(&sa)); CUDA_OK(cudaStreamCreate(&sb));
    CUDA_OK((cudaError_t)mgbfs_segmented_accept_u8(in,hashes,6,2,nullptr,0,a.segment,0,a.count,a.future,a.error,sa));
    CUDA_OK((cudaError_t)mgbfs_segmented_accept_u8(in,hashes,6,2,nullptr,0,b.segment,0,b.count,b.future,b.error,sb));
    CUDA_OK(cudaStreamSynchronize(sa)); CUDA_OK(cudaStreamSynchronize(sb));
    require(values(a,3)==std::set<State>{{1,11},{2,12},{3,13}},"parallel shard A");
    require(values(b,3)==std::set<State>{{1,11},{2,12},{3,13}},"parallel shard B");
    CUDA_OK(cudaStreamDestroy(sa)); CUDA_OK(cudaStreamDestroy(sb)); CUDA_OK(cudaFree(in)); CUDA_OK(cudaFree(hashes)); destroy(a); destroy(b);
}

static uint64_t host_mix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
static uint64_t host_hash(State s,uint64_t seed){uint64_t h=seed;for(uint32_t i=0;i<s.size();i++)h=host_mix(h^uint64_t(s[i])^i);return host_mix(h);}
// Rehash must reproduce the exact same slot-start rule used during insert.
// The second mix prevents a logical shard from fixing bits of the table bucket.
static void rehash_nontrivial_seed_and_bucket() {
    Owned x=make_segment(4,3);const State a{7,17};const uint64_t seed=0x19f856a2ULL;
    const uint64_t h=host_hash(a,seed);const uint64_t expected=host_mix(h^0x9e3779b97f4a7c15ULL)&(x.segment.slots_count-1);
    require(submit(x,{}, {a},0,h)==0,"hashed insert");
    uint64_t position=~uint64_t(0);CUDA_OK(cudaMemcpy(&position,x.segment.positions,8,cudaMemcpyDeviceToHost));
    require(position==expected,"shard-independent bucket rule");
    CUDA_OK((cudaError_t)mgbfs_segmented_rehash_u8(2,x.segment,1,0,0,seed,64,x.error,nullptr));
    CUDA_OK(cudaDeviceSynchronize());uint32_t error=0;CUDA_OK(cudaMemcpy(&error,x.error,4,cudaMemcpyDeviceToHost));require(error==0,"nontrivial rehash error");
    CUDA_OK(cudaMemset(x.count,0,4));require(submit(x,{}, {a},1,h)==0,"same hash after rehash");
    uint32_t count=0;CUDA_OK(cudaMemcpy(&count,x.count,4,cudaMemcpyDeviceToHost));require(count==0,"rehash lost previously inserted state");
    destroy(x);
}

static void rolling_retirement() {
    Owned x=make_segment(4,3);
    require(submit(x,{},{{1,11},{2,12}},0,0)==0,"rolling bank zero insert");
    CUDA_OK(cudaMemset(x.count,0,sizeof(uint32_t)));
    require(submit(x,{},{{3,13}},1,0)==0,"rolling bank one insert");
    CUDA_OK(cudaMemset(x.error,0,sizeof(uint32_t)));
    CUDA_OK((cudaError_t)mgbfs_segmented_retire(x.segment,0,2,x.error,nullptr));
    CUDA_OK(cudaDeviceSynchronize());
    uint32_t error=0; CUDA_OK(cudaMemcpy(&error,x.error,4,cudaMemcpyDeviceToHost));
    require(error==0,"rolling retirement failed");
    CUDA_OK(cudaMemset(x.count,0,sizeof(uint32_t)));
    require(submit(x,{},{{1,11},{2,12},{3,13},{4,14}},2,0)==0,"post-retirement insert");
    uint32_t count=0; CUDA_OK(cudaMemcpy(&count,x.count,4,cudaMemcpyDeviceToHost));
    require(count==3,"retired rows or surviving bank were mishandled");
    std::vector<uint8_t> data(4*3*2);
    CUDA_OK(cudaMemcpy(data.data(),x.segment.states,data.size(),cudaMemcpyDeviceToHost));
    std::set<State> actual;
    for(uint32_t i=0;i<count;++i) actual.insert({data[8+i],data[12+8+i]});
    require(actual==std::set<State>{{1,11},{2,12},{4,14}},"incorrect rolling states");
    // Rebuild a table with tombstones, preserving both other live banks.
    CUDA_OK(cudaMemset(x.error,0,4));
    CUDA_OK((cudaError_t)mgbfs_segmented_rehash_u8(2,x.segment,0,1,3,0,0,x.error,nullptr));
    CUDA_OK(cudaDeviceSynchronize()); CUDA_OK(cudaMemcpy(&error,x.error,4,cudaMemcpyDeviceToHost));require(!error,"rehash error");
    CUDA_OK(cudaMemset(x.count,0,4));
    require(submit(x,{},{{3,13},{4,14},{5,15},{5,15}},0,0)==0,"post rehash insert");
    CUDA_OK(cudaMemcpy(&count,x.count,4,cudaMemcpyDeviceToHost));require(count==1,"rehash lost live bank or accepted duplicate");
    destroy(x);
}

// Force one collision-heavy batch to span many 7-row segments.  Every item
// must either be committed or kept in a bounded spill list for the next one.
static void spill_across_segments() {
    const uint32_t unique=110, n=unique*4;
    std::vector<uint8_t> data(n*2); std::vector<uint64_t> hv(n,0x777777777ULL);
    for(uint32_t i=0;i<n;i++) { const uint32_t k=(i*73)%unique; data[i]=uint8_t(k); data[n+i]=uint8_t(k+3); }
    uint8_t* input=nullptr;uint64_t* hashes=nullptr;uint32_t *ids[2]{},*out_count=nullptr;
    CUDA_OK(cudaMalloc((void**)&input,data.size()));CUDA_OK(cudaMalloc((void**)&hashes,n*8));
    CUDA_OK(cudaMalloc((void**)&ids[0],n*4));CUDA_OK(cudaMalloc((void**)&ids[1],n*4));CUDA_OK(cudaMalloc((void**)&out_count,4));
    CUDA_OK(cudaMemcpy(input,data.data(),data.size(),cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(hashes,hv.data(),n*8,cudaMemcpyHostToDevice));
    std::vector<Owned> storage;std::vector<MgbfsSegment> sealed;const uint32_t* selected=nullptr;uint32_t remaining=n,iter=0;std::set<State> all;
    while(remaining){
        require(iter<30,"spill failed to converge");storage.push_back(make_segment(7));Owned& dst=storage.back();
        MgbfsSegment* ds=nullptr;if(!sealed.empty()){CUDA_OK(cudaMalloc((void**)&ds,sealed.size()*sizeof(MgbfsSegment)));CUDA_OK(cudaMemcpy(ds,sealed.data(),sealed.size()*sizeof(MgbfsSegment),cudaMemcpyHostToDevice));}
        CUDA_OK(cudaMemset(out_count,0,4));
        CUDA_OK((cudaError_t)mgbfs_segmented_accept_spill_u8(input,hashes,n,selected,remaining,2,ds,uint32_t(sealed.size()),dst.segment,0,dst.count,dst.future,ids[iter%2],out_count,dst.error,nullptr));
        CUDA_OK(cudaDeviceSynchronize());uint32_t count=0,error=0,next=0;
        CUDA_OK(cudaMemcpy(&count,dst.count,4,cudaMemcpyDeviceToHost));CUDA_OK(cudaMemcpy(&error,dst.error,4,cudaMemcpyDeviceToHost));CUDA_OK(cudaMemcpy(&next,out_count,4,cudaMemcpyDeviceToHost));
        require(!error && count<=7 && next<=remaining,"spill bounds or error");
        auto got=values(dst,count);require(got.size()==count,"duplicate inside segment");for(auto x:got)require(all.insert(x).second,"duplicate across segments");
        require(!next||count,"spill made no progress");sealed.push_back(dst.segment);selected=ids[iter%2];remaining=next;++iter;if(ds)CUDA_OK(cudaFree(ds));
    }
    require(all.size()==unique && iter>1,"spill lost states or failed to segment");
    for(uint32_t i=0;i<unique;i++) require(all.count({uint8_t(i),uint8_t(i+3)})==1,"spill missing exact state");
    for(auto& x:storage)destroy(x);CUDA_OK(cudaFree(input));CUDA_OK(cudaFree(hashes));CUDA_OK(cudaFree(ids[0]));CUDA_OK(cudaFree(ids[1]));CUDA_OK(cudaFree(out_count));
}


static void routed_packed_inboxes() {
    constexpr uint32_t world=2, shards=2, q=4, bound=world*q, chosen=1;
    std::vector<GenericRouteRecord> local(world*shards*q),remote(local.size());
    std::vector<uint32_t> lc(world*shards),rc(world*shards);
    lc[chosen]=4;rc[shards+chosen]=4;
    const uint8_t xs[8]={1,2,3,4,2,3,5,6};
    for(uint32_t i=0;i<8;++i){GenericRouteRecord v{};v.hash=0x999;v.parent=uint64_t(xs[i])|(uint64_t(xs[i]+20)<<8);
        if(i<4)local[chosen*q+i]=v;else remote[(shards+chosen)*q+i-4]=v;}
    GenericRouteRecord *dl=nullptr,*dr=nullptr;uint32_t *dcl=nullptr,*dcr=nullptr,*spill=nullptr,*spill_count=nullptr;
    CUDA_OK(cudaMalloc((void**)&dl,local.size()*sizeof(GenericRouteRecord)));CUDA_OK(cudaMalloc((void**)&dr,remote.size()*sizeof(GenericRouteRecord)));
    CUDA_OK(cudaMalloc((void**)&dcl,lc.size()*4));CUDA_OK(cudaMalloc((void**)&dcr,rc.size()*4));CUDA_OK(cudaMalloc((void**)&spill,bound*4));CUDA_OK(cudaMalloc((void**)&spill_count,4));
    CUDA_OK(cudaMemcpy(dl,local.data(),local.size()*sizeof(GenericRouteRecord),cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(dr,remote.data(),remote.size()*sizeof(GenericRouteRecord),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dcl,lc.data(),lc.size()*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(dcr,rc.data(),rc.size()*4,cudaMemcpyHostToDevice));
    Owned first=make_segment(3), second=make_segment(3);CUDA_OK(cudaMemset(spill_count,0,4));
    CUDA_OK((cudaError_t)mgbfs_segmented_accept_routes_u8(2,nullptr,nullptr,dl,dr,dcl,dcr,0,world,chosen,shards,q,1,nullptr,bound,nullptr,0,first.segment,0,first.count,first.future,spill,spill_count,first.error,nullptr));
    CUDA_OK(cudaDeviceSynchronize());uint32_t n=0,err=0;CUDA_OK(cudaMemcpy(&n,spill_count,4,cudaMemcpyDeviceToHost));CUDA_OK(cudaMemcpy(&err,first.error,4,cudaMemcpyDeviceToHost));require(!err&&n>0,"routed packed spill");
    MgbfsSegment* sealed=nullptr;CUDA_OK(cudaMalloc((void**)&sealed,sizeof(MgbfsSegment)));CUDA_OK(cudaMemcpy(sealed,&first.segment,sizeof(MgbfsSegment),cudaMemcpyHostToDevice));
    uint32_t* spill2=nullptr;uint32_t* count2=nullptr;CUDA_OK(cudaMalloc((void**)&spill2,bound*4));CUDA_OK(cudaMalloc((void**)&count2,4));CUDA_OK(cudaMemset(count2,0,4));
    CUDA_OK((cudaError_t)mgbfs_segmented_accept_routes_u8(2,nullptr,nullptr,dl,dr,dcl,dcr,0,world,chosen,shards,q,1,spill,n,sealed,1,second.segment,0,second.count,second.future,spill2,count2,second.error,nullptr));
    CUDA_OK(cudaDeviceSynchronize());uint32_t n2=0,err2=0,c1=0,c2=0;CUDA_OK(cudaMemcpy(&n2,count2,4,cudaMemcpyDeviceToHost));CUDA_OK(cudaMemcpy(&err2,second.error,4,cudaMemcpyDeviceToHost));CUDA_OK(cudaMemcpy(&c1,first.count,4,cudaMemcpyDeviceToHost));CUDA_OK(cudaMemcpy(&c2,second.count,4,cudaMemcpyDeviceToHost));
    require(!err2&&!n2&&c1+c2==6,"routed inbox failed exact dedup");auto a=values(first,c1),b=values(second,c2);a.insert(b.begin(),b.end());require(a.size()==6,"routed inbox lost or duplicated states");
    for(uint8_t i=1;i<=6;i++)require(a.count({i,uint8_t(i+20)})==1,"packed state mismatch");
    destroy(first);destroy(second);for(void* ptr:{(void*)dl,(void*)dr,(void*)dcl,(void*)dcr,(void*)spill,(void*)spill_count,(void*)sealed,(void*)spill2,(void*)count2})CUDA_OK(cudaFree(ptr));
}


static void directed_omits_retirement_metadata() {
    Owned x=make_segment(4);MgbfsSegment lightweight=x.segment;lightweight.positions=nullptr;
    uint8_t data[4]={1,2,11,12};uint64_t hashes[2]={7,7};uint8_t* in=nullptr;uint64_t* hs=nullptr;
    CUDA_OK(cudaMalloc((void**)&in,sizeof(data)));CUDA_OK(cudaMalloc((void**)&hs,sizeof(hashes)));
    CUDA_OK(cudaMemcpy(in,data,sizeof(data),cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(hs,hashes,sizeof(hashes),cudaMemcpyHostToDevice));
    CUDA_OK((cudaError_t)mgbfs_segmented_accept_u8(in,hs,2,2,nullptr,0,lightweight,0,x.count,nullptr,x.error,nullptr));
    CUDA_OK(cudaDeviceSynchronize());require(values(x,2)==std::set<State>{{1,11},{2,12}},"optional metadata changed values");
    CUDA_OK(cudaFree(in));CUDA_OK(cudaFree(hs));destroy(x);
}


// Exercise the actual last encodable local row without materializing millions
// of earlier values; the one inserted row and a full-segment spill are real.
static void boundary_24bit_row() {
    constexpr uint32_t cap=1u<<24;
    MgbfsSegment seg{};seg.capacity=cap;seg.banks=1;seg.slots_count=32;
    uint32_t *count=nullptr,*error=nullptr,*spill=nullptr,*spill_count=nullptr;
    uint8_t* input=nullptr;uint64_t* hashes=nullptr;
    CUDA_OK(cudaMalloc((void**)&seg.slots,seg.slots_count*8));
    CUDA_OK(cudaMalloc((void**)&seg.states,cap));
    CUDA_OK(cudaMalloc((void**)&count,4));CUDA_OK(cudaMalloc((void**)&error,4));
    CUDA_OK(cudaMalloc((void**)&spill,4));CUDA_OK(cudaMalloc((void**)&spill_count,4));
    CUDA_OK(cudaMalloc((void**)&input,2));CUDA_OK(cudaMalloc((void**)&hashes,16));
    CUDA_OK(cudaMemset(seg.slots,0xff,seg.slots_count*8));CUDA_OK(cudaMemset(error,0,4));
    uint32_t initial=cap-1;uint8_t rows[2]={42,73};uint64_t hv[2]={0x1234567800000000ULL,0x2345678900000000ULL};
    CUDA_OK(cudaMemcpy(count,&initial,4,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(input,rows,2,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(hashes,hv,16,cudaMemcpyHostToDevice));
    CUDA_OK((cudaError_t)mgbfs_segmented_accept_spill_u8(input,hashes,1,nullptr,1,1,nullptr,0,seg,0,count,nullptr,spill,spill_count,error,nullptr));
    CUDA_OK(cudaDeviceSynchronize());uint32_t got=0,err=0;uint8_t stored=0;
    CUDA_OK(cudaMemcpy(&got,count,4,cudaMemcpyDeviceToHost));CUDA_OK(cudaMemcpy(&err,error,4,cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(&stored,static_cast<uint8_t*>(seg.states)+cap-1,1,cudaMemcpyDeviceToHost));
    require(got==cap&&!err&&stored==42,"24-bit maximum row not stored");
    std::vector<uint64_t> cells(seg.slots_count);CUDA_OK(cudaMemcpy(cells.data(),seg.slots,cells.size()*8,cudaMemcpyDeviceToHost));
    bool found=false;for(uint64_t cell:cells)if(cell!=~uint64_t(0)&&uint32_t(cell)==cap-1)found=true;
    require(found,"24-bit maximum row not encoded in slot");
    CUDA_OK(cudaMemset(spill_count,0,4));
    CUDA_OK((cudaError_t)mgbfs_segmented_accept_spill_u8(input+1,hashes+1,1,nullptr,1,1,nullptr,0,seg,0,count,nullptr,spill,spill_count,error,nullptr));
    CUDA_OK(cudaDeviceSynchronize());uint32_t spilled=0,index=99;
    CUDA_OK(cudaMemcpy(&spilled,spill_count,4,cudaMemcpyDeviceToHost));CUDA_OK(cudaMemcpy(&index,spill,4,cudaMemcpyDeviceToHost));CUDA_OK(cudaMemcpy(&err,error,4,cudaMemcpyDeviceToHost));
    require(spilled==1&&index==0&&!err,"full 24-bit segment failed to spill");
    for(void* p:{(void*)seg.slots,seg.states,(void*)count,(void*)error,(void*)spill,(void*)spill_count,(void*)input,(void*)hashes})CUDA_OK(cudaFree(p));
}


// Owner descriptor cache may contain the active table, which must be skipped
// during the sealed-table pass and handled with pending-aware active probing.
static void cached_descriptors_include_active() {
    Owned x=make_segment(256);std::vector<State> data;
    for(unsigned rep=0;rep<4;++rep)for(unsigned i=0;i<200;++i)data.push_back({uint8_t(i),uint8_t(i+1)});
    require(submit(x,{x.segment},data)==0,"active descriptor cache");
    require(values(x,200).size()==200,"active descriptor cache lost values");destroy(x);
    // Force a pending marker with a different hash prefix in the first probe.
    // The sealed path rejects pending; the active path can step over it.
    Owned y=make_segment(8); const uint64_t hash=0x1234567812345678ULL;
    const uint64_t pos=host_mix(hash^0x9e3779b97f4a7c15ULL)&(y.segment.slots_count-1);
    const uint64_t pending=(uint64_t(1)<<63)|0x0100000000000000ULL;
    CUDA_OK(cudaMemcpy(y.segment.slots+pos,&pending,8,cudaMemcpyHostToDevice));
    require(submit(y,{y.segment},{{3,7}},0,hash)==0,"active descriptor pending skip");
    require(values(y,1)==std::set<State>{{3,7}},"active descriptor pending result");destroy(y);
}

static void exercise(int device) {
    CUDA_OK(cudaSetDevice(device));
    Owned first=make_segment(4), second=make_segment(4), overflow=make_segment(2);
    require(submit(first,{},{{1,11},{2,12},{3,13},{4,14},{1,11},{2,12},{3,13},{4,14}})==0,"first insert");
    require(values(first,4)==std::set<State>{{1,11},{2,12},{3,13},{4,14}},"first exact states");
    require(submit(second,{first.segment},{{1,11},{2,12},{3,13},{4,14},{5,15},{6,16},{5,15},{6,16}})==0,"sealed lookup");
    require(values(second,2)==std::set<State>{{5,15},{6,16}},"duplicate accepted across segments");
    require((submit(overflow,{},{{7,17},{8,18},{9,19}})&MGBFS_SEGMENTED_OVERFLOW)!=0,"overflow was silent");
    require(mgbfs_segmented_accept_u8(nullptr,nullptr,0,2,nullptr,0,first.segment,3,first.count,first.future,first.error,nullptr)==int(cudaErrorInvalidValue),"invalid bank accepted");
    rehash_nontrivial_seed_and_bucket();
    rolling_retirement();
    spill_across_segments();
    routed_packed_inboxes();
    directed_omits_retirement_metadata();
    multisegment_collision_stress();
    wide_values();
    parallel_shards();
    boundary_24bit_row();
    cached_descriptors_include_active();
    destroy(first); destroy(second); destroy(overflow);
}
int main() {
    int devices=0; CUDA_OK(cudaGetDeviceCount(&devices)); require(devices>=1,"at least one CUDA device required");
    for(int device=0;device<devices;++device) exercise(device);
    std::printf("PASS segmented collision, sealed-history, spill, duplicate and overflow on %d GPUs\n",devices);
}
