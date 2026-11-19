// Independent tiny-graph oracle for real two-rank routing, NCCL and segmented owner.
// Host reads states solely to compare against the CPU oracle; production must not.
#include "segmented_history.cuh"
#include "generic_route.h"
#include <cuda_runtime.h>
#include <nccl.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <set>
#include <vector>
#include <array>
extern "C" int mgbfs_generic_route_packed_u8(uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,const uint8_t*,uint32_t,uint32_t,const uint32_t*,const int64_t*,const uint32_t*,uint64_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint64_t,const uint32_t*,const uint64_t*,GenericRouteRecord*,uint32_t*,uint32_t*,void*);
#define CHECK(x) do {cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define NCCL(x) do {ncclResult_t e=(x);if(e!=ncclSuccess){fprintf(stderr,"NCCL %s:%d %s\n",__FILE__,__LINE__,ncclGetErrorString(e));exit(1);}}while(0)
static void need(bool p,const char* s){if(!p){fprintf(stderr,"FAIL %s\n",s);exit(1);}}
using State=std::vector<uint8_t>;
static uint64_t mix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
static uint64_t state_hash(const State& s,uint64_t seed){uint64_t h=seed;for(size_t i=0;i<s.size();i++)h=mix(h^uint64_t(s[i])^uint64_t(i));return mix(h);}
static std::vector<std::set<State>> oracle(State start,const std::vector<std::vector<uint32_t>>& gs){
 std::set<State> seen{start},front{start};std::vector<std::set<State>> layers;
 while(!front.empty()){layers.push_back(front);std::set<State> next;
  for(const auto& state:front)for(const auto& g:gs){State child;for(auto x:g)child.push_back(state[x]);if(!seen.count(child))next.insert(child);}
  seen.insert(next.begin(),next.end());front=next;need(seen.size()<100000,"oracle bound");}return layers;
}
constexpr uint32_t WORLD=2,SHARDS=2,Q=64,CAP=3;constexpr uint64_t SEED=0x19f856a2;
struct Segment {
 MgbfsSegment seg{};uint32_t *count{},*per_bank[3]{},*future{},*error{};uint32_t known=0,bank_known[3]{},mark=0,begin=0,end=0;
 Segment(uint32_t width,bool rolling){seg.capacity=CAP;seg.banks=rolling?3:1;seg.slots_count=32;CHECK(cudaMalloc((void**)&seg.slots,32*8));CHECK(cudaMemset(seg.slots,255,32*8));CHECK(cudaMalloc(&seg.states,CAP*width*seg.banks));CHECK(cudaMalloc((void**)&seg.positions,CAP*8*seg.banks));for(uint32_t i=0;i<seg.banks;i++){CHECK(cudaMalloc((void**)&per_bank[i],4));CHECK(cudaMemset(per_bank[i],0,4));}count=per_bank[0];CHECK(cudaMalloc((void**)&future,CAP*4));CHECK(cudaMalloc((void**)&error,4));CHECK(cudaMemset(error,0,4));}
 void activate(uint32_t bank){count=per_bank[bank];known=bank_known[bank];}
 ~Segment(){cudaFree(seg.slots);cudaFree(seg.states);cudaFree(seg.positions);for(uint32_t i=0;i<seg.banks;i++)cudaFree(per_bank[i]);cudaFree(future);cudaFree(error);}
};
struct Rank {
 int gpu;bool rolling;uint32_t next_bank=0;std::array<uint32_t,SHARDS> peak_per_shard{};cudaStream_t stream{};ncclComm_t comm{};uint32_t *perms{},*counts{},*received{},*route_error{},*map{},*spill[2]{},*spill_count[2]{};
 GenericRouteRecord *routes{},*inbox{};std::array<std::vector<std::unique_ptr<Segment>>,SHARDS> segments;
 Rank(int device,const std::vector<std::vector<uint32_t>>& gs,const std::array<uint32_t,2>& owner_map,bool ring):gpu(device),rolling(ring){
  CHECK(cudaSetDevice(gpu));CHECK(cudaStreamCreate(&stream));std::vector<uint32_t> flat;for(auto g:gs)flat.insert(flat.end(),g.begin(),g.end());
  CHECK(cudaMalloc((void**)&perms,flat.size()*4));CHECK(cudaMemcpy(perms,flat.data(),flat.size()*4,cudaMemcpyHostToDevice));
  CHECK(cudaMalloc((void**)&counts,WORLD*SHARDS*4));CHECK(cudaMalloc((void**)&received,WORLD*SHARDS*4));CHECK(cudaMalloc((void**)&route_error,4));CHECK(cudaMalloc((void**)&map,WORLD*4));CHECK(cudaMemcpy(map,owner_map.data(),WORLD*4,cudaMemcpyHostToDevice));
  CHECK(cudaMalloc((void**)&routes,WORLD*SHARDS*Q*sizeof(GenericRouteRecord)));CHECK(cudaMalloc((void**)&inbox,WORLD*SHARDS*Q*sizeof(GenericRouteRecord)));
  for(int i=0;i<2;i++){CHECK(cudaMalloc((void**)&spill[i],WORLD*Q*4));CHECK(cudaMalloc((void**)&spill_count[i],4));}
 }
 ~Rank(){cudaSetDevice(gpu);segments={};cudaFree(perms);cudaFree(counts);cudaFree(received);cudaFree(route_error);cudaFree(map);cudaFree(routes);cudaFree(inbox);for(int i=0;i<2;i++){cudaFree(spill[i]);cudaFree(spill_count[i]);}cudaStreamDestroy(stream);}
 Segment& make(uint32_t shard,uint32_t width){segments[shard].push_back(std::make_unique<Segment>(width,rolling));segments[shard].back()->activate(next_bank);return *segments[shard].back();}
};
static uint32_t read32(uint32_t* p){uint32_t x=0;CHECK(cudaMemcpy(&x,p,4,cudaMemcpyDeviceToHost));return x;}
static void accept_queue(Rank& r,uint32_t shard,uint32_t width){
 CHECK(cudaSetDevice(r.gpu));auto& segs=r.segments[shard];if(segs.empty())r.make(shard,width);
 const uint32_t* selection=nullptr;uint32_t selected=WORLD*Q,pass=0;size_t active_index=0;
 do{
  if(active_index==segs.size())r.make(shard,width);
  Segment& active=*segs[active_index];std::vector<MgbfsSegment> sealed;for(size_t i=0;i<segs.size();i++)if(i!=active_index)sealed.push_back(segs[i]->seg);MgbfsSegment* ds=nullptr;
  if(!sealed.empty()){CHECK(cudaMalloc((void**)&ds,sealed.size()*sizeof(MgbfsSegment)));CHECK(cudaMemcpy(ds,sealed.data(),sealed.size()*sizeof(MgbfsSegment),cudaMemcpyHostToDevice));}
  const uint32_t out=pass%2;CHECK(cudaMemsetAsync(r.spill_count[out],0,4,r.stream));CHECK(cudaMemsetAsync(active.error,0,4,r.stream));
  CHECK((cudaError_t)mgbfs_segmented_accept_routes_u8(width,nullptr,nullptr,r.routes,r.inbox,r.counts,r.received,r.gpu,WORLD,shard,SHARDS,Q,1,selection,selected,ds,sealed.size(),active.seg,r.next_bank,active.count,active.future,r.spill[out],r.spill_count[out],active.error,r.stream));
  CHECK(cudaStreamSynchronize(r.stream));uint32_t err=read32(active.error),next=read32(r.spill_count[out]);active.known=read32(active.count);active.bank_known[r.next_bank]=active.known;need(!err&&active.known<=CAP&&next<=selected,"owner error/bounds");
  if(ds)CHECK(cudaFree(ds));selection=r.spill[out];selected=next;pass++;need(pass<200,"spill progress");if(next)active_index++;
 }while(selected);
}
static void run_graph(State start,const std::vector<std::vector<uint32_t>>& gs,bool rolling=false){
 const uint32_t width=start.size(),generators=gs.size();auto expected=oracle(start,gs);const uint64_t hash=state_hash(start,SEED);
 uint32_t owner=(uint64_t(uint32_t(hash>>32))*WORLD)>>32;uint32_t shard=(uint64_t(uint32_t(hash))*SHARDS)>>32;std::array<uint32_t,2> map{0,1};std::swap(map[0],map[owner]);
 Rank a(0,gs,map,rolling),b(1,gs,map,rolling);std::array<Rank*,2> ranks{&a,&b};ncclComm_t comms[2];int devices[2]={0,1};NCCL(ncclCommInitAll(comms,2,devices));a.comm=comms[0];b.comm=comms[1];
 CHECK(cudaSetDevice(0));Segment& root=a.make(shard,width);uint8_t* ds=nullptr;uint64_t* dh=nullptr;CHECK(cudaMalloc((void**)&ds,width));CHECK(cudaMalloc((void**)&dh,8));CHECK(cudaMemcpy(ds,start.data(),width,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dh,&hash,8,cudaMemcpyHostToDevice));
 CHECK((cudaError_t)mgbfs_segmented_accept_u8(ds,dh,1,width,nullptr,0,root.seg,0,root.count,root.future,root.error,nullptr));CHECK(cudaDeviceSynchronize());need(read32(root.error)==0&&read32(root.count)==1,"root insertion");root.known=1;root.bank_known[0]=1;root.end=1;CHECK(cudaFree(ds));CHECK(cudaFree(dh));
 for(size_t depth=0;depth<expected.size();++depth){
  std::set<State> actual;std::array<std::vector<Segment*>,2> work;
  const uint32_t current=rolling?depth%3:0, future=rolling?(depth+1)%3:0;
  for(auto* r:ranks){CHECK(cudaSetDevice(r->gpu));r->next_bank=future;for(auto& bucket:r->segments)for(auto& item:bucket){Segment& s=*item;if(s.end>s.begin){work[r->gpu].push_back(&s);std::vector<uint8_t> raw(CAP*width*s.seg.banks);CHECK(cudaMemcpy(raw.data(),s.seg.states,raw.size(),cudaMemcpyDeviceToHost));for(uint32_t row=s.begin;row<s.end;row++){State x;for(uint32_t e=0;e<width;e++)x.push_back(raw[e*CAP*s.seg.banks+current*CAP+row]);need(actual.insert(x).second,"duplicate frontier");}}
    if(rolling){uint32_t expired=s.bank_known[future];CHECK(cudaMemset(s.error,0,4));if(expired)CHECK((cudaError_t)mgbfs_segmented_retire(s.seg,future,expired,s.error,nullptr));CHECK(cudaMemset(s.per_bank[future],0,4));s.bank_known[future]=0;
     CHECK((cudaError_t)mgbfs_segmented_rehash_u8(width,s.seg,s.bank_known[0],s.bank_known[1],s.bank_known[2],SEED,64,s.error,nullptr));CHECK(cudaDeviceSynchronize());need(!read32(s.error),"rolling maintenance");s.activate(future);
    }else s.mark=s.known;}}
  need(actual==expected[depth],"oracle frontier mismatch");size_t rounds=std::max(work[0].size(),work[1].size());
  for(size_t round=0;round<rounds;round++){
   for(auto* r:ranks){CHECK(cudaSetDevice(r->gpu));CHECK(cudaMemsetAsync(r->counts,0,WORLD*SHARDS*4,r->stream));CHECK(cudaMemsetAsync(r->received,0,WORLD*SHARDS*4,r->stream));CHECK(cudaMemsetAsync(r->route_error,0,4,r->stream));
    if(round<work[r->gpu].size()){Segment* s=work[r->gpu][round];auto* parent=static_cast<uint8_t*>(s->seg.states)+current*CAP+s->begin;uint32_t count=s->end-s->begin;
     CHECK((cudaError_t)mgbfs_generic_route_packed_u8(0,width,width,1,generators,parent,count,CAP*s->seg.banks,r->perms,nullptr,nullptr,SEED,64,WORLD,r->gpu,SHARDS,Q,0,r->map,nullptr,r->routes,r->counts,r->route_error,r->stream));}
    CHECK(cudaStreamSynchronize(r->stream));need(read32(r->route_error)==0,"route overflow");}
   NCCL(ncclGroupStart());for(int src=0;src<2;src++){int dst=1-src;Rank* s=ranks[src];Rank* d=ranks[dst];CHECK(cudaSetDevice(src));NCCL(ncclSend(s->counts+dst*SHARDS,SHARDS,ncclUint32,dst,s->comm,s->stream));NCCL(ncclSend(s->routes+dst*SHARDS*Q,SHARDS*Q*sizeof(GenericRouteRecord),ncclUint8,dst,s->comm,s->stream));CHECK(cudaSetDevice(dst));NCCL(ncclRecv(d->received+src*SHARDS,SHARDS,ncclUint32,src,d->comm,d->stream));NCCL(ncclRecv(d->inbox+src*SHARDS*Q,SHARDS*Q*sizeof(GenericRouteRecord),ncclUint8,src,d->comm,d->stream));}NCCL(ncclGroupEnd());
   for(auto* r:ranks){CHECK(cudaSetDevice(r->gpu));CHECK(cudaStreamSynchronize(r->stream));for(uint32_t sh=0;sh<SHARDS;sh++)accept_queue(*r,sh,width);}
  }
  for(auto* r:ranks){CHECK(cudaSetDevice(r->gpu));for(uint32_t sh=0;sh<SHARDS;sh++){auto& bucket=r->segments[sh];uint32_t bank_total=0;for(auto& item:bucket){item->begin=rolling?0:item->mark;item->end=item->known;bank_total+=item->bank_known[future];}if(rolling){r->peak_per_shard[sh]=std::max(r->peak_per_shard[sh],bank_total);uint32_t necessary=std::max(1u,(r->peak_per_shard[sh]+CAP-1)/CAP);need(bucket.size()<=necessary,"rolling allocator failed to reuse segments");}}}
 }
 for(auto* r:ranks){CHECK(cudaSetDevice(r->gpu));for(auto& bucket:r->segments)for(auto& item:bucket)need(item->begin==item->end,"unexpected extra layer");NCCL(ncclCommDestroy(r->comm));}
 printf("PASS routed NCCL segmented BFS: width=%u rolling=%d layers=%zu states=%zu\n",width,int(rolling),expected.size(),[&](){size_t n=0;for(auto&l:expected)n+=l.size();return n;}());
}
int main(){int n=0;CHECK(cudaGetDeviceCount(&n));if(n<2){printf("SKIP two CUDA devices required\n");return 77;}run_graph({0,1,2,3},{{1,0,2,3},{0,2,1,3},{0,1,3,2}});run_graph({0,0,1,2,3},{{1,0,2,3,4},{0,2,1,3,4},{0,1,3,2,4},{0,1,2,4,3}});run_graph({0,1,2,3},{{1,2,3,0}});run_graph({0,1,2,3},{{1,0,2,3},{0,2,1,3},{0,1,3,2}},true);run_graph({0,0,1,2,3},{{1,0,2,3,4},{0,2,1,3,4},{0,1,3,2,4},{0,1,2,4,3}},true);run_graph({0,1,2,3,4,5},{{1,0,2,3,4,5},{0,2,1,3,4,5},{0,1,3,2,4,5},{0,1,2,4,3,5},{0,1,2,3,5,4}},true);}
