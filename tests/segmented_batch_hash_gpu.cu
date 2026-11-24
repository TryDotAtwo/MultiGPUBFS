// Exact collision, cross-tile lookup, spill, and publication for batch hash.
#include "generic_route.h"
#include "segmented_history.cuh"
#include "segmented_hash_batch.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>
constexpr bool check_refs() {
    constexpr MgbfsHashCompactRefLayout regular{18}, full_row{32}, full_descriptor{0}, bad{33};
    MgbfsHashCompactRefLayout admitted{0};
    if (!mgbfs_hash_compact_ref_layout(16384, 262143, admitted) || admitted.row_bits != 18) return false;
    if (mgbfs_hash_compact_ref_layout(16385, 262143, admitted)) return false;
    if (mgbfs_hash_compact_ref_layout(0, 1, admitted) || mgbfs_hash_compact_ref_layout(1, 0, admitted)) return false;
    if (!mgbfs_hash_compact_ref_layout(1, uint64_t{1} << 32, admitted) || admitted.row_bits != 32) return false;
    if (mgbfs_hash_compact_ref_layout(1, (uint64_t{1} << 32) + 1, admitted)) return false;
    uint32_t ref = 0;
    if (!regular.try_encode(16383, 262143, ref) || ref != UINT32_MAX) return false;
    if (regular.descriptor(ref) != 16383 || regular.row(ref) != 262143) return false;
    if (regular.try_encode(16384, 0, ref) || regular.try_encode(0, 262144, ref)) return false;
    if (!full_row.try_encode(0, UINT32_MAX, ref) || full_row.row(ref) != UINT32_MAX) return false;
    if (full_row.descriptor(ref) != 0 || full_row.try_encode(1, 0, ref)) return false;
    if (!full_descriptor.try_encode(UINT32_MAX, 0, ref) || full_descriptor.descriptor(ref) != UINT32_MAX) return false;
    if (full_descriptor.row(ref) != 0 || full_descriptor.try_encode(0, 1, ref) || bad.try_encode(0, 0, ref)) return false;
    constexpr auto wide = mgbfs_hash_wide_ref(UINT32_MAX, UINT32_MAX - 1);
    return mgbfs_hash_ref_descriptor(wide) == UINT32_MAX && mgbfs_hash_ref_row(wide) == UINT32_MAX - 1;
}
constexpr bool check_series_prefixes(){
 const uint64_t keys[]{0xAABBCC0000000000ULL,0xAABBCC0100000000ULL,0xAABBCCFF00000000ULL};
 const auto shift=mgbfs_hash_series_shift(keys[0],keys[2]);
 return shift==32&&mgbfs_hash_series_prefix_begin(keys,3,0,shift)==0
  &&mgbfs_hash_series_prefix_begin(keys,3,1,shift)==1
  &&mgbfs_hash_series_prefix_begin(keys,3,2,shift)==2
  &&mgbfs_hash_series_prefix_begin(keys,3,255,shift)==2
  &&mgbfs_hash_series_prefix_begin(keys,3,256,shift)==3;
}
static_assert(check_series_prefixes());
static_assert(check_refs());

constexpr bool check_ref_remap() {
    constexpr uint32_t old_to_new[]{1, UINT32_MAX, 0, 16384};
    constexpr MgbfsHashCompactRefLayout layout{18};
    uint32_t ref = 0, out = 0;
    if (!layout.try_encode(2, 262143, ref) || !mgbfs_hash_remap_ref(layout, ref, old_to_new, 4, out)) return false;
    if (layout.descriptor(out) != 0 || layout.row(out) != 262143) return false;
    const uint32_t previous = out;
    for (uint32_t invalid : {1u, 3u, 4u}) {
        if (!layout.try_encode(invalid, 5, ref)) return false;
        if (mgbfs_hash_remap_ref(layout, ref, old_to_new, 4, out) || out != previous) return false;
    }
    auto wide = mgbfs_hash_wide_ref(2, UINT32_MAX);
    if (!mgbfs_hash_remap_ref(wide, old_to_new, 4, wide)) return false;
    if (mgbfs_hash_ref_descriptor(wide) != 0 || mgbfs_hash_ref_row(wide) != UINT32_MAX) return false;
    return !mgbfs_hash_remap_ref(mgbfs_hash_wide_ref(1, 0), old_to_new, 4, wide);
}
static_assert(check_ref_remap());


extern "C" int mgbfs_hash_batch_scratch_bytes(uint32_t,uint32_t,uint32_t,size_t*);
extern "C" int mgbfs_hash_batch_accept_v2(uint32_t,uint32_t,uint32_t,const void*,const void*,
 const GenericRouteRecord*,const GenericRouteRecord*,const uint32_t*,const uint32_t*,
 uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,const uint32_t*,uint32_t,uint32_t,
 const MgbfsBatchHashSegment*,uint32_t,uint32_t,uint32_t,MgbfsBatchHashRun*,uint32_t*,uint32_t,
 uint64_t*,uint32_t*,uint32_t*,uint32_t*,uint64_t*,uint64_t,void*,size_t,void*);
extern "C" int mgbfs_hash_batch_accept_profiled_v3(uint32_t,uint32_t,uint32_t,const void*,const void*,
 const GenericRouteRecord*,const GenericRouteRecord*,const uint32_t*,const uint32_t*,
 uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,const uint32_t*,uint32_t,uint32_t,
 const MgbfsBatchHashSegment*,uint32_t,uint32_t,uint32_t,MgbfsBatchHashRun*,uint32_t*,uint32_t,
 uint64_t*,uint32_t*,uint32_t*,uint32_t*,uint64_t*,uint64_t,void*,size_t,void*,void*,unsigned long long*,void*);
extern "C" int mgbfs_hash_batch_accept_profiled_v4(uint32_t,uint32_t,uint32_t,const void*,const void*,
 const GenericRouteRecord*,const GenericRouteRecord*,const uint32_t*,const uint32_t*,
 uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,const uint32_t*,uint32_t,uint32_t,
 const MgbfsBatchHashSegment*,uint32_t,uint32_t,uint32_t,MgbfsBatchHashRun*,uint32_t*,uint32_t,
 uint64_t*,uint32_t*,uint32_t*,uint32_t*,uint64_t*,uint64_t,void*,size_t,void*,void*,unsigned long long*,void*,const MgbfsHashSeriesView*);
extern "C" int mgbfs_hash_extent_gather_v1(const MgbfsBatchHashSegment*,uint32_t,
 const MgbfsBatchHashRun*,uint32_t,uint32_t,uint32_t*,uint64_t*,void*,uint32_t,uint32_t,uint32_t*,void*);
extern "C" int mgbfs_hash_extent_remap_v1(void*,uint32_t,uint32_t,uint32_t,
 const uint32_t*,uint32_t,uint32_t*,void*);
extern "C" int mgbfs_hash_extent_sort_scratch_v1(uint32_t,uint32_t,uint32_t,size_t*);
extern "C" int mgbfs_hash_extent_sort_v1(uint32_t,uint32_t,const MgbfsBatchHashSegment*,uint32_t,
 const uint64_t*,const void*,uint32_t,uint32_t,uint64_t*,void*,void*,size_t,void*);
extern "C" int mgbfs_hash_extent_prefixes_v1(const uint64_t*,uint32_t,uint32_t*,void*);
extern "C" int mgbfs_hash_batch_seed_v2(uint32_t,uint32_t,const void*,MgbfsBatchHashSegment,uint64_t,MgbfsBatchHashRun*,uint32_t*,uint64_t*,void*);
extern "C" int mgbfs_hash_batch_retire_bank_v2(MgbfsBatchHashRun*,uint32_t*,uint32_t,const MgbfsBatchHashSegment*,uint32_t,uint32_t,uint32_t*,void*);
extern "C" int mgbfs_hash_batch_compact_scratch_bytes(uint32_t,size_t*);
extern "C" int mgbfs_hash_batch_compact_tail_v2(uint32_t,uint32_t,const MgbfsBatchHashSegment*,uint32_t,
 MgbfsBatchHashRun*,uint32_t*,uint32_t,void*,size_t,uint32_t*,uint32_t*,void*);
#define C(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);exit(1);}}while(0)
template<class T> T* alloc(size_t n=1){T*p=nullptr;C(cudaMalloc((void**)&p,n*sizeof(T)));return p;}
template<class T> T get(T*p){T v{};C(cudaMemcpy(&v,p,sizeof(T),cudaMemcpyDeviceToHost));return v;}
template<class T> void put(T*p,T v){C(cudaMemcpy(p,&v,sizeof(T),cudaMemcpyHostToDevice));}
using State=std::pair<uint8_t,uint8_t>;
static GenericRouteRecord record(State v){GenericRouteRecord r{};r.hash=0;r.parent=uint64_t(v.first)|(uint64_t(v.second)<<8);return r;}
static void run(){
 constexpr uint32_t Q=8;
 std::vector<State> first={{5,2},{0,1},{5,2},{3,4},{7,8},{0,1},{9,1},{3,4}};
 std::vector<GenericRouteRecord> host(Q);for(unsigned i=0;i<Q;i++)host[i]=record(first[i]);
 auto* meta=alloc<GenericRouteRecord>(Q);C(cudaMemcpy(meta,host.data(),Q*sizeof(host[0]),cudaMemcpyHostToDevice));
 auto* counts=alloc<uint32_t>();put(counts,Q);
 MgbfsBatchHashSegment hs[3]{};
 for(unsigned i=0;i<3;i++){hs[i].capacity=i==0?3:(i==1?2:1);hs[i].banks=1;hs[i].states=alloc<uint8_t>(hs[i].capacity*2);hs[i].keys=alloc<uint64_t>(hs[i].capacity);hs[i].rows=alloc<uint32_t>(hs[i].capacity);}
 auto* segments=alloc<MgbfsBatchHashSegment>(3);C(cudaMemcpy(segments,hs,sizeof(hs),cudaMemcpyHostToDevice));
 auto* runs=alloc<MgbfsBatchHashRun>(8);auto* nruns=alloc<uint32_t>();put(nruns,uint32_t(0));
 auto* accepted=alloc<uint64_t>(3);C(cudaMemset(accepted,0,3*sizeof(uint64_t)));
 auto* global=alloc<uint64_t>();put(global,uint64_t(0));
 auto* selected=alloc<uint32_t>(Q);auto* spill=alloc<uint32_t>(Q);auto* ns=alloc<uint32_t>();auto* err=alloc<uint32_t>();
 size_t bytes=0;C(cudaError_t(mgbfs_hash_batch_scratch_bytes(Q,1,1,&bytes)));void* scratch=nullptr;C(cudaMalloc(&scratch,bytes));
 MgbfsHashSeriesView snapshot{};
 auto call=[&](uint32_t active,const uint32_t* selection,uint32_t n,uint32_t capacity){
  put(ns,uint32_t(0));put(err,uint32_t(0));
  C(cudaError_t(mgbfs_hash_batch_accept_profiled_v4(1,1,2,nullptr,nullptr,meta,meta,counts,counts,
   0,1,0,1,Q,Q,selection,n,0,segments,3,active,0,runs,nruns,capacity,accepted+active,
   spill,ns,err,global,100,scratch,bytes,nullptr,nullptr,nullptr,nullptr,snapshot.length?&snapshot:nullptr)));
  C(cudaDeviceSynchronize());
 };
 call(0,nullptr,Q,8);CHECK(get(err)==0);CHECK(get(accepted)==3);CHECK(get(nruns)==1);CHECK(get(ns)==2);CHECK(get(global)==3);
 C(cudaMemcpy(selected,spill,2*sizeof(uint32_t),cudaMemcpyDeviceToDevice));
 call(1,selected,2,8);CHECK(get(err)==0);CHECK(get(accepted+1)==2);CHECK(get(nruns)==2);CHECK(get(ns)==0);CHECK(get(global)==5);
 constexpr uint32_t snapshot_rows=5,row_bits=2;
 auto* run_offsets=alloc<uint32_t>(2);
 auto* raw_keys=alloc<uint64_t>(snapshot_rows);auto* raw_refs=alloc<uint32_t>(snapshot_rows);
 auto* sorted_keys=alloc<uint64_t>(snapshot_rows);auto* sorted_refs=alloc<uint32_t>(snapshot_rows);
 auto* prefix=alloc<uint32_t>(MGBFS_HASH_SERIES_DIRECTORY_WORDS);
 CHECK(!mgbfs_hash_extent_gather_v1(segments,3,runs,2,snapshot_rows,run_offsets,raw_keys,raw_refs,4,row_bits,err,nullptr));
 CHECK(get(err)==0);
 size_t sort_bytes=0;
 CHECK(!mgbfs_hash_extent_sort_scratch_v1(1,4,snapshot_rows,&sort_bytes));
 auto* sort_scratch=alloc<uint8_t>(sort_bytes);
 CHECK(!mgbfs_hash_extent_sort_v1(1,2,segments,snapshot_rows,raw_keys,raw_refs,4,row_bits,sorted_keys,
  sorted_refs,sort_scratch,sort_bytes,nullptr));
 CHECK(!mgbfs_hash_extent_prefixes_v1(sorted_keys,snapshot_rows,prefix,nullptr));
 C(cudaDeviceSynchronize());
 snapshot={sorted_keys,sorted_refs,prefix,snapshot_rows,2,4,{row_bits}};
 host[0]=record({0,1});host[1]=record({10,2});host[2]=record({9,1});for(unsigned i=3;i<Q;i++)host[i]=record({0,1});
 C(cudaMemcpy(meta,host.data(),Q*sizeof(host[0]),cudaMemcpyHostToDevice));put(counts,uint32_t(3));
 call(2,nullptr,Q,8);CHECK(get(err)==0);CHECK(get(accepted+2)==1);CHECK(get(nruns)==3);CHECK(get(ns)==0);CHECK(get(global)==6);
 std::set<State> got;
 for(unsigned seg=0;seg<3;seg++){
  std::vector<uint8_t> states(hs[seg].capacity*2);C(cudaMemcpy(states.data(),hs[seg].states,states.size(),cudaMemcpyDeviceToHost));
  for(unsigned row=0;row<hs[seg].capacity;row++)CHECK(got.insert({states[row],states[hs[seg].capacity+row]}).second);
 }
 CHECK(got==std::set<State>({{0,1},{3,4},{5,2},{7,8},{9,1},{10,2}}));
 call(2,nullptr,Q,8);CHECK(get(err)==0);CHECK(get(accepted+2)==1);CHECK(get(nruns)==3);CHECK(get(ns)==0);CHECK(get(global)==6);
 MgbfsBatchHashRun observed[3]{};C(cudaMemcpy(observed,runs,sizeof(observed),cudaMemcpyDeviceToHost));
 for(unsigned i=0;i<3;i++){CHECK(observed[i].segment==i&&observed[i].bank==0&&observed[i].begin==0&&observed[i].length==hs[i].capacity);CHECK(observed[i].min_hash==0&&observed[i].max_hash==0);}
 C(cudaFree(run_offsets));C(cudaFree(raw_keys));C(cudaFree(raw_refs));
 C(cudaFree(sorted_keys));C(cudaFree(sorted_refs));C(cudaFree(prefix));C(cudaFree(sort_scratch));
 for(auto& seg:hs){C(cudaFree(seg.states));C(cudaFree(seg.keys));C(cudaFree(seg.rows));}
 C(cudaFree(meta));C(cudaFree(counts));C(cudaFree(segments));C(cudaFree(runs));C(cudaFree(nruns));C(cudaFree(accepted));C(cudaFree(global));C(cudaFree(selected));C(cudaFree(spill));C(cudaFree(ns));C(cudaFree(err));C(cudaFree(scratch));
 puts("PASS forced equal hashes, exact compare, cross-tile history, segment spill, duplicate-only full bank");
}

static void compact_runs(){
 static constexpr uint32_t Q=4;
 auto* meta=alloc<GenericRouteRecord>(Q);auto* counts=alloc<uint32_t>();put(counts,uint32_t(2));
 MgbfsBatchHashSegment h{};h.capacity=8;h.banks=1;h.states=alloc<uint8_t>(16);h.keys=alloc<uint64_t>(8);h.rows=alloc<uint32_t>(8);
 auto* segments=alloc<MgbfsBatchHashSegment>();C(cudaMemcpy(segments,&h,sizeof(h),cudaMemcpyHostToDevice));
 auto* runs=alloc<MgbfsBatchHashRun>(8);auto* nruns=alloc<uint32_t>();put(nruns,uint32_t(0));
 auto* accepted=alloc<uint64_t>();put(accepted,uint64_t(0));auto* global=alloc<uint64_t>();put(global,uint64_t(0));
 auto* spill=alloc<uint32_t>(Q);auto* ns=alloc<uint32_t>();auto* err=alloc<uint32_t>();auto* result=alloc<uint32_t>();
 size_t scratch_bytes=0;C(cudaError_t(mgbfs_hash_batch_scratch_bytes(Q,1,1,&scratch_bytes)));void* scratch=nullptr;C(cudaMalloc(&scratch,scratch_bytes));
 auto accept=[&](State x,State y){
  GenericRouteRecord rows[Q]={record(x),record(y),record(x),record(y)};
  C(cudaMemcpy(meta,rows,sizeof(rows),cudaMemcpyHostToDevice));put(ns,uint32_t(0));put(err,uint32_t(0));
  C(cudaError_t(mgbfs_hash_batch_accept_v2(1,1,2,nullptr,nullptr,meta,meta,counts,counts,0,1,0,1,Q,Q,nullptr,Q,0,
   segments,1,0,0,runs,nruns,8,accepted,spill,ns,err,global,100,scratch,scratch_bytes,nullptr)));
  C(cudaDeviceSynchronize());CHECK(get(err)==0&&get(ns)==0);
 };
 accept({5,2},{1,1});accept({2,0},{4,4});CHECK(get(accepted)==4&&get(nruns)==2);
 std::vector<uint8_t> before(16),after(16);C(cudaMemcpy(before.data(),h.states,16,cudaMemcpyDeviceToHost));
 size_t mb=0;C(cudaError_t(mgbfs_hash_batch_compact_scratch_bytes(2,&mb)));void* ms=nullptr;C(cudaMalloc(&ms,mb));
 put(result,uint32_t(99));put(err,uint32_t(0));
 C(cudaError_t(mgbfs_hash_batch_compact_tail_v2(1,2,segments,1,runs,nruns,2,ms,mb,result,err,nullptr)));
 C(cudaDeviceSynchronize());CHECK(get(result)==2&&get(err)==0&&get(nruns)==2);
 accept({1,1},{2,0});CHECK(get(nruns)==2&&get(accepted)==4);
 C(cudaFree(ms));C(cudaError_t(mgbfs_hash_batch_compact_scratch_bytes(4,&mb)));C(cudaMalloc(&ms,mb));
 MgbfsBatchHashRun before_failed_merge[2]{},after_failed_merge[2]{};
 C(cudaMemcpy(before_failed_merge,runs,sizeof(before_failed_merge),cudaMemcpyDeviceToHost));
 put(result,uint32_t(99));put(err,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));
 C(cudaError_t(mgbfs_hash_batch_compact_tail_v2(1,2,segments,1,runs,nruns,4,ms,mb,result,err,nullptr)));
 C(cudaDeviceSynchronize());
 CHECK(get(result)==0&&get(err)==MGBFS_SEGMENTED_BAD_REFERENCE&&get(nruns)==2);
 C(cudaMemcpy(after_failed_merge,runs,sizeof(after_failed_merge),cudaMemcpyDeviceToHost));
 for(unsigned i=0;i<2;i++){
  CHECK(before_failed_merge[i].begin==after_failed_merge[i].begin);
  CHECK(before_failed_merge[i].length==after_failed_merge[i].length);
  CHECK(before_failed_merge[i].min_hash==after_failed_merge[i].min_hash);
  CHECK(before_failed_merge[i].max_hash==after_failed_merge[i].max_hash);
 }
 put(result,uint32_t(99));put(err,uint32_t(0));
 C(cudaError_t(mgbfs_hash_batch_compact_tail_v2(1,2,segments,1,runs,nruns,4,ms,mb,result,err,nullptr)));
 C(cudaDeviceSynchronize());CHECK(get(result)==1&&get(err)==0&&get(nruns)==1);
 MgbfsBatchHashRun run{};C(cudaMemcpy(&run,runs,sizeof(run),cudaMemcpyDeviceToHost));CHECK(run.begin==0&&run.length==4&&run.min_hash==0&&run.max_hash==0);
 uint32_t rows[4];C(cudaMemcpy(rows,h.rows,sizeof(rows),cudaMemcpyDeviceToHost));CHECK(rows[0]==0&&rows[1]==2&&rows[2]==3&&rows[3]==1);
 C(cudaMemcpy(after.data(),h.states,16,cudaMemcpyDeviceToHost));CHECK(before==after);
 accept({1,1},{2,0});CHECK(get(nruns)==1&&get(accepted)==4);
 C(cudaFree(ms));C(cudaFree(scratch));C(cudaFree(result));C(cudaFree(err));C(cudaFree(ns));C(cudaFree(spill));C(cudaFree(global));C(cudaFree(accepted));C(cudaFree(nruns));C(cudaFree(runs));C(cudaFree(segments));C(cudaFree(h.states));C(cudaFree(h.keys));C(cudaFree(h.rows));C(cudaFree(counts));C(cudaFree(meta));
 puts("PASS deferred/merged runs remain searchable; no state movement");
}

// Both variants must find duplicate hashes across fence boundaries,
// including after two runs are consolidated into a larger fenced run.
static void set_prefix_probe(bool enabled){
#ifdef _WIN32
 CHECK(_putenv_s("MGBFS_SEGMENT_HASH_PREFIX_PROBE",enabled?"1":"0")==0);
#else
 CHECK(setenv("MGBFS_SEGMENT_HASH_PREFIX_PROBE",enabled?"1":"0",1)==0);
#endif
}

template<uint32_t Q>
static void filtered_runs(){
 MgbfsBatchHashSegment h{};
 h.capacity=2*Q;h.banks=1;h.states=alloc<uint8_t>(2*h.capacity);
 h.keys=alloc<uint64_t>(h.capacity);h.rows=alloc<uint32_t>(h.capacity);
 auto* segments=alloc<MgbfsBatchHashSegment>();put(segments,h);
 auto* meta=alloc<GenericRouteRecord>(Q);auto* counts=alloc<uint32_t>();put(counts,Q);
 auto* runs=alloc<MgbfsBatchHashRun>(4);auto* nruns=alloc<uint32_t>();put(nruns,uint32_t(0));
 auto* accepted=alloc<uint64_t>();put(accepted,uint64_t(0));
 auto* spill=alloc<uint32_t>(Q);auto* ns=alloc<uint32_t>();auto* err=alloc<uint32_t>();
 size_t bytes=0;C(cudaError_t(mgbfs_hash_batch_scratch_bytes(Q,1,1,&bytes)));void* scratch=nullptr;C(cudaMalloc(&scratch,bytes));
 auto* misses=alloc<unsigned long long>(33);
 C(cudaMemset(misses,0,33*sizeof(*misses)));
 auto accept=[&](uint8_t group){
  GenericRouteRecord rows[Q];
  for(uint32_t i=0;i<Q;i++){rows[i]=record({uint8_t(i),group});rows[i].hash=Q>=256 ? i/40 : (i%2)*128+group*64;}
  C(cudaMemcpy(meta,rows,sizeof(rows),cudaMemcpyHostToDevice));put(err,uint32_t(0));put(ns,uint32_t(0));
  C(cudaError_t(mgbfs_hash_batch_accept_profiled_v3(1,1,2,nullptr,nullptr,meta,meta,counts,counts,0,1,0,1,Q,Q,nullptr,Q,0,
    segments,1,0,0,runs,nruns,4,accepted,spill,ns,err,nullptr,0,scratch,bytes,nullptr,nullptr,misses,nullptr)));
  C(cudaDeviceSynchronize());CHECK(get(err)==0&&get(ns)==0);
 };
 accept(0);CHECK(get(accepted)==Q);CHECK(get(nruns)==1);
 auto run=get(runs);
 if constexpr(Q>=256){
  for(uint32_t i=1;i<8;i++)CHECK(run.hash_fences[i-1]==((i*Q/8)/40));
 }else{
  CHECK(run.hash_fences[0]==1&&run.hash_fences[1]==0);
  CHECK(run.hash_fences[2]==1&&run.hash_fences[3]==0);
 }
 accept(0);CHECK(get(accepted)==Q);
 accept(1);CHECK(get(accepted)==2*Q);CHECK(get(nruns)==2);
 unsigned long long buckets[33]{};
 C(cudaMemcpy(buckets,misses,sizeof(buckets),cudaMemcpyDeviceToHost));
 if constexpr(Q>=256){
  CHECK(buckets[0]==(Q+127)/128&&buckets[1]==(Q+127)/128);
 }else{
  CHECK(buckets[0]==2*((Q+127)/128)&&buckets[1]==0);
 }
 for(unsigned i=2;i<33;i++)CHECK(buckets[i]==0);
 size_t merge_bytes=0;C(cudaError_t(mgbfs_hash_batch_compact_scratch_bytes(2*Q,&merge_bytes)));
 void* merge=nullptr;C(cudaMalloc(&merge,merge_bytes));auto* result=alloc<uint32_t>();
 put(err,uint32_t(0));
 C(cudaError_t(mgbfs_hash_batch_compact_tail_v2(1,2,segments,1,runs,nruns,2*Q,merge,merge_bytes,result,err,nullptr)));
 C(cudaDeviceSynchronize());CHECK(get(result)==1&&get(err)==0&&get(nruns)==1);
 run=get(runs);
 if constexpr(Q>=256){
  for(uint32_t i=1;i<8;i++)CHECK(run.hash_fences[i-1]==((i*Q/8)/40));
 }else{
  for(uint32_t word=0;word<4;word++)CHECK(run.hash_fences[word]==1);
 }
 accept(0);accept(1);CHECK(get(accepted)==2*Q);CHECK(get(nruns)==1);
 for(void* p:{merge,scratch,static_cast<void*>(misses),static_cast<void*>(result),static_cast<void*>(err),static_cast<void*>(ns),
   static_cast<void*>(spill),static_cast<void*>(accepted),static_cast<void*>(nruns),static_cast<void*>(runs),
   static_cast<void*>(counts),static_cast<void*>(meta),static_cast<void*>(segments),
   h.states,static_cast<void*>(h.keys),static_cast<void*>(h.rows)})C(cudaFree(p));
}

static void bounded_raw_tiles(){
 constexpr uint32_t Q=6;uint8_t ids[Q]={1,1,1,2,1,3};
 GenericRouteRecord host[Q]{};for(unsigned i=0;i<Q;i++)host[i]=record({ids[i],0});
 auto* meta=alloc<GenericRouteRecord>(Q);C(cudaMemcpy(meta,host,sizeof(host),cudaMemcpyHostToDevice));
 auto* counts=alloc<uint32_t>();put(counts,Q);
 MgbfsBatchHashSegment hs[2]{};
 for(unsigned i=0;i<2;i++){hs[i].capacity=i?1:2;hs[i].banks=1;hs[i].states=alloc<uint8_t>(hs[i].capacity*2);hs[i].keys=alloc<uint64_t>(hs[i].capacity);hs[i].rows=alloc<uint32_t>(hs[i].capacity);}
 auto* segments=alloc<MgbfsBatchHashSegment>(2);C(cudaMemcpy(segments,hs,sizeof(hs),cudaMemcpyHostToDevice));
 auto* runs=alloc<MgbfsBatchHashRun>(8);auto* nruns=alloc<uint32_t>();put(nruns,uint32_t(0));
 auto* accepted=alloc<uint64_t>(2);C(cudaMemset(accepted,0,16));auto* global=alloc<uint64_t>();put(global,uint64_t(0));
 auto* spill=alloc<uint32_t>(2);auto* selection=alloc<uint32_t>(2);auto* ns=alloc<uint32_t>();auto* err=alloc<uint32_t>();
 size_t bytes=0;C(cudaError_t(mgbfs_hash_batch_scratch_bytes(2,1,1,&bytes)));void* scratch=nullptr;C(cudaMalloc(&scratch,bytes));
 auto call=[&](const uint32_t* sel,uint32_t begin,uint32_t active,uint32_t n){
  put(ns,uint32_t(0));put(err,uint32_t(0));
  C(cudaError_t(mgbfs_hash_batch_accept_v2(1,1,2,nullptr,nullptr,meta,meta,counts,counts,0,1,0,1,Q,Q,sel,n,begin,segments,2,active,0,runs,nruns,8,accepted+active,spill,ns,err,global,3,scratch,bytes,nullptr)));
  C(cudaDeviceSynchronize());CHECK(get(err)==0);
 };
 int invalid=mgbfs_hash_batch_accept_v2(1,1,2,nullptr,nullptr,meta,meta,counts,counts,0,1,0,1,Q,Q,nullptr,2,5,segments,2,0,0,runs,nruns,8,accepted,spill,ns,err,global,3,scratch,bytes,nullptr);
 CHECK(invalid==int(cudaErrorInvalidValue));
 call(nullptr,0,0,2);CHECK(get(accepted)==1&&get(ns)==0&&get(global)==1);
 call(nullptr,2,0,2);CHECK(get(accepted)==2&&get(ns)==0&&get(global)==2);
 call(nullptr,4,0,2);CHECK(get(accepted)==2&&get(ns)==1&&get(global)==2);
 CHECK(get(spill)==5);C(cudaMemcpy(selection,spill,4,cudaMemcpyDeviceToDevice));call(selection,0,1,1);
 CHECK(get(accepted+1)==1&&get(ns)==0&&get(global)==3&&get(nruns)==3);
 uint8_t rows_a[4]{},rows_b[2]{};C(cudaMemcpy(rows_a,hs[0].states,4,cudaMemcpyDeviceToHost));C(cudaMemcpy(rows_b,hs[1].states,2,cudaMemcpyDeviceToHost));
 CHECK(rows_a[0]==1&&rows_a[1]==2&&rows_b[0]==3);
 for(auto& seg:hs){C(cudaFree(seg.states));C(cudaFree(seg.keys));C(cudaFree(seg.rows));}
 C(cudaFree(meta));C(cudaFree(counts));C(cudaFree(segments));C(cudaFree(runs));C(cudaFree(nruns));C(cudaFree(accepted));C(cudaFree(global));C(cudaFree(spill));C(cudaFree(selection));C(cudaFree(ns));C(cudaFree(err));C(cudaFree(scratch));
 puts("PASS bounded raw tiles; duplicate+new use last bank row; live queue and spill replay");
}


static void seed_and_retire(){
 MgbfsBatchHashSegment hs[2]{};
 for(auto& seg:hs){seg.capacity=2;seg.banks=3;seg.states=alloc<uint8_t>(12);seg.keys=alloc<uint64_t>(6);seg.rows=alloc<uint32_t>(6);}
 auto* segments=alloc<MgbfsBatchHashSegment>(2);C(cudaMemcpy(segments,hs,sizeof(hs),cudaMemcpyHostToDevice));
 auto* input=alloc<uint8_t>(2);uint8_t start[2]={4,7};C(cudaMemcpy(input,start,2,cudaMemcpyHostToDevice));
 auto* runs=alloc<MgbfsBatchHashRun>(8);auto* count=alloc<uint32_t>();auto* accepted=alloc<uint64_t>();
 C(cudaError_t(mgbfs_hash_batch_seed_v2(1,2,input,hs[0],123,runs,count,accepted,nullptr)));C(cudaDeviceSynchronize());
 CHECK(get(count)==1&&get(accepted)==1&&get(hs[0].keys)==123&&get(hs[0].rows)==0);
 uint8_t states[12];C(cudaMemcpy(states,hs[0].states,12,cudaMemcpyDeviceToHost));CHECK(states[0]==4&&states[6]==7);
 MgbfsBatchHashRun seeded{};C(cudaMemcpy(&seeded,runs,sizeof(seeded),cudaMemcpyDeviceToHost));
 CHECK(seeded.segment==0&&seeded.bank==0&&seeded.length==1&&seeded.min_hash==123&&seeded.max_hash==123);
 MgbfsBatchHashRun host[4]={{0,0,0,1,1,1},{0,1,0,1,2,2},{1,0,0,1,3,3},{1,2,0,1,4,4}};
 C(cudaMemcpy(runs,host,sizeof(host),cudaMemcpyHostToDevice));put(count,uint32_t(4));auto* error=alloc<uint32_t>();put(error,uint32_t(0));
 C(cudaError_t(mgbfs_hash_batch_retire_bank_v2(runs,count,8,segments,2,0,error,nullptr)));C(cudaDeviceSynchronize());
 CHECK(get(count)==2&&get(error)==0);MgbfsBatchHashRun retained[2];C(cudaMemcpy(retained,runs,sizeof(retained),cudaMemcpyDeviceToHost));
 CHECK(retained[0].segment==0&&retained[0].bank==1&&retained[1].segment==1&&retained[1].bank==2);
 for(auto& seg:hs){C(cudaFree(seg.states));C(cudaFree(seg.keys));C(cudaFree(seg.rows));}
 C(cudaFree(segments));C(cudaFree(input));C(cudaFree(runs));C(cudaFree(count));C(cudaFree(accepted));C(cudaFree(error));
 puts("PASS exact seed index and bank-local run retirement");
}

static void directory_growth(){
 static constexpr uint32_t Q=2;auto* meta=alloc<GenericRouteRecord>(Q);auto* counts=alloc<uint32_t>();put(counts,Q);
 MgbfsBatchHashSegment h{};h.capacity=4;h.banks=1;h.states=alloc<uint8_t>(8);h.keys=alloc<uint64_t>(4);h.rows=alloc<uint32_t>(4);
 auto* segments=alloc<MgbfsBatchHashSegment>();C(cudaMemcpy(segments,&h,sizeof(h),cudaMemcpyHostToDevice));
 auto* runs=alloc<MgbfsBatchHashRun>(2);auto* nruns=alloc<uint32_t>();put(nruns,uint32_t(0));
 auto* accepted=alloc<uint64_t>();put(accepted,uint64_t(0));auto* global=alloc<uint64_t>();put(global,uint64_t(0));
 auto* spill=alloc<uint32_t>(Q);auto* selection=alloc<uint32_t>(Q);auto* ns=alloc<uint32_t>();auto* err=alloc<uint32_t>();
 size_t bytes=0;C(cudaError_t(mgbfs_hash_batch_scratch_bytes(Q,1,1,&bytes)));void* scratch=nullptr;C(cudaMalloc(&scratch,bytes));
 auto call=[&](State a,State b,const uint32_t* sel,uint32_t n,uint32_t cap){
  GenericRouteRecord host[Q]={record(a),record(b)};C(cudaMemcpy(meta,host,sizeof(host),cudaMemcpyHostToDevice));
  put(ns,uint32_t(0));put(err,uint32_t(0));
  C(cudaError_t(mgbfs_hash_batch_accept_v2(1,1,2,nullptr,nullptr,meta,meta,counts,counts,
   0,1,0,1,Q,Q,sel,n,0,segments,1,0,0,runs,nruns,cap,accepted,spill,ns,err,global,4,scratch,bytes,nullptr)));
  C(cudaDeviceSynchronize());
 };
 call({1,0},{1,0},nullptr,Q,1);
 CHECK(get(err)==0&&get(ns)==0&&get(nruns)==1&&get(accepted)==1&&get(global)==1);
 call({1,0},{1,0},nullptr,Q,1);
 CHECK(get(err)==0&&get(ns)==0&&get(nruns)==1&&get(accepted)==1&&get(global)==1);
 call({1,0},{2,0},nullptr,Q,1);
 CHECK(get(err)==MGBFS_SEGMENTED_RUN_GROW&&get(ns)==1&&get(spill)==1&&get(nruns)==1&&get(accepted)==1&&get(global)==1);
 C(cudaMemcpy(selection,spill,4,cudaMemcpyDeviceToDevice));
 call({1,0},{2,0},selection,1,2);
 CHECK(get(err)==0&&get(ns)==0&&get(nruns)==2&&get(accepted)==2&&get(global)==2);
 uint8_t states[8];C(cudaMemcpy(states,h.states,8,cudaMemcpyDeviceToHost));CHECK(states[0]==1&&states[1]==2);
 call({2,0},{3,0},nullptr,Q,2);
 CHECK(get(err)==MGBFS_SEGMENTED_RUN_GROW&&get(ns)==1&&get(nruns)==2&&get(global)==2);
 C(cudaMemcpy(selection,spill,sizeof(uint32_t),cudaMemcpyDeviceToDevice));
 size_t merge_bytes=0;
 C(cudaError_t(mgbfs_hash_batch_compact_scratch_bytes(2,&merge_bytes)));
 void* merge_scratch=nullptr;C(cudaMalloc(&merge_scratch,merge_bytes));
 auto* merge_result=alloc<uint32_t>();
 C(cudaError_t(mgbfs_hash_batch_compact_tail_v2(1,2,segments,1,runs,nruns,2,
   merge_scratch,merge_bytes,merge_result,err,nullptr)));
 C(cudaDeviceSynchronize());
 CHECK(get(merge_result)==1&&get(nruns)==1&&get(global)==2);
 call({2,0},{3,0},selection,1,2);
 CHECK(get(err)==0&&get(ns)==0&&get(nruns)==2&&get(accepted)==3&&get(global)==3);
 C(cudaFree(merge_result));C(cudaFree(merge_scratch));
 put(counts,uint32_t(3));call({1,0},{2,0},nullptr,Q,2);
 CHECK((get(err)&32u)!=0&&get(accepted)==3&&get(nruns)==2&&get(global)==3);
 C(cudaFree(scratch));C(cudaFree(err));C(cudaFree(ns));C(cudaFree(selection));C(cudaFree(spill));C(cudaFree(global));C(cudaFree(accepted));
 C(cudaFree(nruns));C(cudaFree(runs));C(cudaFree(segments));C(cudaFree(h.states));C(cudaFree(h.keys));C(cudaFree(h.rows));C(cudaFree(counts));C(cudaFree(meta));
 puts("PASS duplicate-only full directory; deferred growth and exact replay; invalid queue remains fatal");
}
static void compact_long_skewed_tail(){
 MgbfsBatchHashSegment seg{};
 seg.capacity=10;seg.banks=1;
 seg.states=alloc<uint8_t>(10);seg.keys=alloc<uint64_t>(10);seg.rows=alloc<uint32_t>(10);
 constexpr uint64_t prefix_step=(uint64_t(1)<<58)+1;
 uint64_t keys[]={0,1,2,3,4,5,10,11,12,6};
 for(auto& key:keys)key*=prefix_step;
 const uint32_t rows[]={0,1,2,3,4,5,6,7,8,9};
 C(cudaMemcpy(seg.keys,keys,sizeof(keys),cudaMemcpyHostToDevice));
 C(cudaMemcpy(seg.rows,rows,sizeof(rows),cudaMemcpyHostToDevice));
 auto* segments=alloc<MgbfsBatchHashSegment>();put(segments,seg);
 MgbfsBatchHashRun initial[8]{};
 for(uint32_t i=0;i<6;++i)initial[i]={0,0,i,1,keys[i],keys[i]};
 initial[6]={0,0,6,3,keys[6],keys[8],{(1u<<10)|(1u<<11)|(1u<<12)}};
 initial[7]={0,0,9,1,keys[9],keys[9],{1u<<6}};
 auto* runs=alloc<MgbfsBatchHashRun>(8);
 C(cudaMemcpy(runs,initial,sizeof(initial),cudaMemcpyHostToDevice));
 auto* count=alloc<uint32_t>();put(count,uint32_t(8));
 auto* result=alloc<uint32_t>();auto* error=alloc<uint32_t>();put(error,uint32_t(0));
 size_t bytes=0;C(cudaError_t(mgbfs_hash_batch_compact_scratch_bytes(4,&bytes)));
 void* scratch=nullptr;C(cudaMalloc(&scratch,bytes));
 // A short skewed tail waits in ordinary operation. A full directory
 // can merge the same pair to make room before replaying its spill.
 put(count,uint32_t(2));
 C(cudaError_t(mgbfs_hash_batch_compact_tail_v2(
  1,1,segments,1,runs+6,count,4,scratch,bytes,result,error,nullptr)));
 C(cudaDeviceSynchronize());
 CHECK(get(error)==0&&get(result)==0&&get(count)==2);
 put(error,uint32_t(MGBFS_SEGMENTED_RUN_GROW));
 C(cudaError_t(mgbfs_hash_batch_compact_tail_v2(
  1,1,segments,1,runs+6,count,4,scratch,bytes,result,error,nullptr)));
 C(cudaDeviceSynchronize());
 CHECK(get(error)==MGBFS_SEGMENTED_RUN_GROW&&get(result)==1&&get(count)==1);
 CHECK(get(seg.rows+6)==9&&get(seg.rows+7)==6);
 C(cudaMemcpy(seg.keys,keys,sizeof(keys),cudaMemcpyHostToDevice));
 C(cudaMemcpy(seg.rows,rows,sizeof(rows),cudaMemcpyHostToDevice));
 C(cudaMemcpy(runs,initial,sizeof(initial),cudaMemcpyHostToDevice));
 put(count,uint32_t(8));put(error,uint32_t(0));
 C(cudaError_t(mgbfs_hash_batch_compact_tail_v2(
  1,1,segments,1,runs,count,4,scratch,bytes,result,error,nullptr)));
 C(cudaDeviceSynchronize());
 CHECK(get(error)==0&&get(result)==1&&get(count)==7);
 const auto tail=get(runs+6);
 CHECK(tail.begin==6&&tail.length==4&&tail.min_hash==6*prefix_step&&tail.max_hash==12*prefix_step);
 for(uint32_t word=0;word<4;word++){
  uint64_t expected=0;
  for(uint64_t key:{6*prefix_step,10*prefix_step,11*prefix_step,12*prefix_step})
   if(((key>>6)&3)==word)expected|=uint64_t(1)<<(key&63);
  CHECK(tail.hash_fences[word]==expected);
 }
 CHECK(tail.hash_fences[4]==1);
 CHECK(get(seg.rows+6)==9&&get(seg.rows+7)==6);
 C(cudaFree(scratch));C(cudaFree(error));C(cudaFree(result));C(cudaFree(count));
 C(cudaFree(runs));C(cudaFree(segments));C(cudaFree(seg.rows));C(cudaFree(seg.keys));C(cudaFree(seg.states));
}

extern "C" size_t mgbfs_hash_extent_merge_control_bytes();
extern "C" int mgbfs_hash_extent_merge_v1(uint32_t,uint32_t,const MgbfsBatchHashSegment*,
 uint32_t,const MgbfsBatchHashRun*,uint32_t,uint64_t*,MgbfsHashRowRef*,void*,size_t,uint32_t*,void*);

extern "C" int mgbfs_hash_extent_merge_compact_v1(uint32_t,uint32_t,const MgbfsBatchHashSegment*,
 uint32_t,const MgbfsBatchHashRun*,uint32_t,uint64_t*,uint32_t*,uint32_t,void*,size_t,uint32_t*,void*);


__global__ void probe_merged_series(const uint64_t* keys,const uint32_t* refs,
 const uint32_t* offsets,const MgbfsBatchHashSegment* segments,uint32_t* results){
 const uint8_t query=threadIdx.x;
 const auto collision=mgbfs_hash_series_lookup(keys,refs,offsets,4,segments,2,{1},42,&query,1);
 const auto wrong_prefix=mgbfs_hash_series_lookup(keys,refs,offsets,4,segments,2,{1},UINT64_MAX,&query,1);
 results[query]=uint32_t(collision);
 results[6+query]=uint32_t(wrong_prefix);
}

static void merge_across_extents(){
 MgbfsBatchHashSegment physical[2]{};
 uint8_t states[2][2]{{1,4},{2,3}};
 const uint64_t hashes[]{42,42};
 const uint32_t row_ids[]{0,1};
 for(unsigned i=0;i<2;i++){
  auto& extent=physical[i];
  extent.capacity=2;extent.banks=1;
  extent.states=alloc<uint8_t>(2);extent.keys=alloc<uint64_t>(2);extent.rows=alloc<uint32_t>(2);
  C(cudaMemcpy(extent.states,states[i],sizeof(states[i]),cudaMemcpyHostToDevice));
  C(cudaMemcpy(extent.keys,hashes,sizeof(hashes),cudaMemcpyHostToDevice));
  C(cudaMemcpy(extent.rows,row_ids,sizeof(row_ids),cudaMemcpyHostToDevice));
 }
 auto* segments=alloc<MgbfsBatchHashSegment>(2);
 C(cudaMemcpy(segments,physical,sizeof(physical),cudaMemcpyHostToDevice));
 const MgbfsBatchHashRun source[2]{{0,0,0,2,42,42,{}},{1,0,0,2,42,42,{}}};
 auto* inputs=alloc<MgbfsBatchHashRun>(2);
 C(cudaMemcpy(inputs,source,sizeof(source),cudaMemcpyHostToDevice));
 const size_t scratch_bytes=mgbfs_hash_extent_merge_control_bytes();
 auto* scratch=alloc<uint8_t>(scratch_bytes);
 auto* error=alloc<uint32_t>();put(error,uint32_t(0));
 auto* keys=alloc<uint64_t>(4);
 auto* refs=alloc<MgbfsHashRowRef>(4);
 CHECK(!mgbfs_hash_extent_merge_v1(1,1,segments,2,inputs,4,keys,refs,scratch,scratch_bytes,error,nullptr));
 C(cudaDeviceSynchronize());
 uint32_t status=0;
 C(cudaMemcpy(&status,error,sizeof(status),cudaMemcpyDeviceToHost));
 CHECK(status==0);
 uint64_t read_keys[4]{};
 MgbfsHashRowRef read_refs[4]{};
 C(cudaMemcpy(read_keys,keys,sizeof(read_keys),cudaMemcpyDeviceToHost));
 C(cudaMemcpy(read_refs,refs,sizeof(read_refs),cudaMemcpyDeviceToHost));
 const uint32_t descriptors[]{0,3,3,0},rows[]{0,0,1,1};
 for(unsigned i=0;i<4;i++){
  CHECK(read_keys[i]==42);
  CHECK(mgbfs_hash_ref_descriptor(read_refs[i])==descriptors[i]);
  CHECK(mgbfs_hash_ref_row(read_refs[i])==rows[i]);
 }
 constexpr MgbfsHashCompactRefLayout layout{1};
 auto* packed=alloc<uint32_t>(4);
 CHECK(!mgbfs_hash_extent_merge_compact_v1(1,1,segments,2,inputs,4,keys,packed,layout.row_bits,
  scratch,scratch_bytes,error,nullptr));
 C(cudaDeviceSynchronize());
 C(cudaMemcpy(&status,error,sizeof(status),cudaMemcpyDeviceToHost));
 CHECK(status==0);
 uint32_t packed_refs[4]{};
 C(cudaMemcpy(packed_refs,packed,sizeof(packed_refs),cudaMemcpyDeviceToHost));
 for(unsigned i=0;i<4;i++){
  CHECK(layout.descriptor(packed_refs[i])==descriptors[i]);
  CHECK(layout.row(packed_refs[i])==rows[i]);
 }
 MgbfsBatchHashSegment host_segments[2]{physical[0],physical[1]};
 host_segments[0].states=states[0];host_segments[1].states=states[1];
 uint32_t prefix[MGBFS_HASH_SERIES_DIRECTORY_WORDS]{};
 const uint32_t shift=mgbfs_hash_series_shift(read_keys[0],read_keys[3]);
 prefix[MGBFS_HASH_SERIES_SHIFT_INDEX]=shift;
 for(uint32_t i=0;i<=MGBFS_HASH_SERIES_PREFIXES;i++)
  prefix[i]=mgbfs_hash_series_prefix_begin(read_keys,4,i,shift);
 for(uint8_t query=0;query<=5;query++){
  const auto expected=query>=1&&query<=4?MgbfsHashSeriesMatch::present:MgbfsHashSeriesMatch::absent;
  CHECK(mgbfs_hash_series_lookup(read_keys,packed_refs,prefix,4,host_segments,2,layout,42,&query,1)==expected);
  CHECK(mgbfs_hash_series_lookup(read_keys,read_refs,prefix,4,host_segments,2,layout,42,&query,1)==expected);
  CHECK(mgbfs_hash_series_lookup(read_keys,packed_refs,prefix,4,host_segments,2,layout,UINT64_MAX,&query,1)==MgbfsHashSeriesMatch::absent);
 }
 auto* device_offsets=alloc<uint32_t>(MGBFS_HASH_SERIES_DIRECTORY_WORDS);
 auto* device_results=alloc<uint32_t>(12);
 CHECK(!mgbfs_hash_extent_prefixes_v1(keys,4,device_offsets,nullptr));
 probe_merged_series<<<1,6>>>(keys,packed,device_offsets,segments,device_results);
 C(cudaGetLastError());
 uint32_t results[12]{};
 C(cudaMemcpy(results,device_results,sizeof(results),cudaMemcpyDeviceToHost));
 for(unsigned i=0;i<6;i++){
  CHECK(results[i]==uint32_t(i>=1&&i<=4?MgbfsHashSeriesMatch::present:MgbfsHashSeriesMatch::absent));
  CHECK(results[6+i]==uint32_t(MgbfsHashSeriesMatch::absent));
 }
 auto* device_remap=alloc<uint32_t>(2);
 const uint32_t invalid_remap[]{1,UINT32_MAX};
 C(cudaMemcpy(device_remap,invalid_remap,sizeof(invalid_remap),cudaMemcpyHostToDevice));
 CHECK(!mgbfs_hash_extent_remap_v1(packed,4,4,layout.row_bits,device_remap,2,error,nullptr));
 CHECK(get(error)&MGBFS_SEGMENTED_BAD_REFERENCE);
 uint32_t unchanged[4]{};
 C(cudaMemcpy(unchanged,packed,sizeof(unchanged),cudaMemcpyDeviceToHost));
 CHECK(std::equal(unchanged,unchanged+4,packed_refs));
 const uint32_t valid_remap[]{1,0};
 C(cudaMemcpy(device_remap,valid_remap,sizeof(valid_remap),cudaMemcpyHostToDevice));
 put(error,uint32_t(0));
 CHECK(!mgbfs_hash_extent_remap_v1(packed,4,4,layout.row_bits,device_remap,2,error,nullptr));
 CHECK(get(error)==0);
 C(cudaMemcpy(packed_refs,packed,sizeof(packed_refs),cudaMemcpyDeviceToHost));
 for(unsigned i=0;i<4;i++){
  CHECK(layout.descriptor(packed_refs[i])==3-descriptors[i]);
  CHECK(layout.row(packed_refs[i])==rows[i]);
 }
 C(cudaFree(device_remap));
 C(cudaFree(device_offsets));C(cudaFree(device_results));
 C(cudaFree(packed));
 for(auto& extent:physical){C(cudaFree(extent.states));C(cudaFree(extent.keys));C(cudaFree(extent.rows));}
 C(cudaFree(segments));C(cudaFree(inputs));C(cudaFree(scratch));
 C(cudaFree(error));C(cudaFree(keys));C(cudaFree(refs));
}

int main(int argc,char** argv){
 const bool single=argc==2&&std::strcmp(argv[1],"--single-gpu")==0;
 if(argc!=1&&!single){
  std::fprintf(stderr,"usage: %s [--single-gpu]\n",argv[0]);
  return 2;
 }
 const int requested=single?1:2;
 int count=0;
 C(cudaGetDeviceCount(&count));
 CHECK(count>=requested);
 for(int d=0;d<requested;d++){
  C(cudaSetDevice(d));
  compact_runs();
  merge_across_extents();
  compact_long_skewed_tail();
  for(bool prefix:{false,true}){
   set_prefix_probe(prefix);
   run();
   // A short run rejects a miss sharing its six low hash bits.
   filtered_runs<16>();
   filtered_runs<64>();
   filtered_runs<256>();
  }
  set_prefix_probe(false);
  bounded_raw_tiles();
  seed_and_retire();
  directory_growth();
 }
 puts(single?"BATCH_HASH_ONE_GPU_PASS":"BATCH_HASH_TWO_GPU_PASS");
}
