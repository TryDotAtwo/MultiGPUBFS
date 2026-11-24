// Experimental bounded exact sorted-run owner, one exclusive shard lane.
// Index = one u64 key + one u32 local row per live state; states stay in SoA.
// No old hash table, no global u32 rank addressing, and no per-state CPU work.
#include "generic_action.cuh"
#include "segmented_incoming.cuh"
#include "segmented_history.cuh"
#include "segmented_hash_batch.cuh"
#include "generic_sorted_origin_exact.cuh"
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace {
constexpr size_t kAlign=256;
struct Grant {uint64_t base;uint32_t total,granted,run_index;};
struct Workspace {size_t hashes,origins,sorted,flags,prefix,grant,temp,temp_bytes,total;};
struct NoPathOutput {static constexpr bool enabled=false;};
struct PathOutput {
 static constexpr bool enabled=true;
 MgbfsHashPathParentV1* parents;
 uint32_t source_world,generators;
};
bool region(size_t& c,size_t n,size_t* off){
 if(c>std::numeric_limits<size_t>::max()-(kAlign-1))return false;
 c=(c+kAlign-1)&~(kAlign-1);*off=c;
 if(n>std::numeric_limits<size_t>::max()-c)return false;c+=n;return true;
}
template<class State,bool Packed> struct SelectedAction {
 uint32_t elements;
 MgbfsBoundedIncomingAction<State,Packed> in;
 const uint32_t* selection;
 uint32_t raw_begin;
 __device__ uint32_t child(uint32_t i)const{return selection?selection[i]:in.child(raw_begin+i);}
 __device__ bool valid(uint32_t c,uint32_t* e)const{return in.valid(c,e);}
 __device__ int64_t value(uint32_t c,uint32_t e)const{return in.value(c,e);}
 __device__ uint64_t hash(uint32_t c,uint64_t seed,uint32_t bits)const{return in.hash(c,seed,bits);}
 __device__ const GenericRouteRecord& route(uint32_t c)const{return in.route(c);}
};
template<class Action> cudaError_t layout(uint32_t n,Workspace* w){
 if(!w||n>=0x7fffffffU)return cudaErrorInvalidValue;*w={};if(!n)return cudaSuccess;
 size_t sort=0,scan=0;Action empty{};GenericSortedOriginLess<Action> cmp{empty,nullptr};
 cudaError_t rc=cub::DeviceMergeSort::SortKeysCopy(nullptr,sort,
  static_cast<const uint32_t*>(nullptr),static_cast<uint32_t*>(nullptr),int(n),cmp);
 if(rc!=cudaSuccess)return rc;
 rc=cub::DeviceScan::ExclusiveSum(nullptr,scan,static_cast<const uint32_t*>(nullptr),
  static_cast<uint32_t*>(nullptr),int(n));if(rc!=cudaSuccess)return rc;
 size_t c=0,temp=sort>scan?sort:scan;
 if(!region(c,size_t(n)*8,&w->hashes)||!region(c,size_t(n)*4,&w->origins)||
    !region(c,size_t(n)*4,&w->sorted)||!region(c,size_t(n)*4,&w->flags)||
    !region(c,size_t(n)*4,&w->prefix)||!region(c,sizeof(Grant),&w->grant)||
    !region(c,temp,&w->temp))return cudaErrorInvalidValue;
 w->temp_bytes=temp;w->total=c;return cudaSuccess;
}
__device__ inline bool fatal(uint32_t x){return (x&~uint32_t(MGBFS_SEGMENTED_CAP_RETRY|MGBFS_SEGMENTED_RUN_GROW))!=0;}
template<class Action,class Lineage>
__global__ void prepare(Action action,uint32_t n,uint64_t* hashes,uint32_t* origins,uint32_t* error,Lineage lineage){
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
  uint32_t child=action.child(i);bool valid=action.valid(child,error);
  if constexpr(Lineage::enabled){
   if(valid){
    const auto& record=action.route(child);
    valid=record.source<lineage.source_world&&record.generator<lineage.generators;
    if(!valid)atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));
   }
  }
  origins[i]=valid?i:UINT32_MAX;hashes[i]=valid?action.hash(child,0,64):0;
 }
}
template<class T,class Action>
__device__ int compare_candidate(Action action,uint32_t origin,uint64_t hash,
 const MgbfsBatchHashSegment& segment,const MgbfsBatchHashRun& run,uint64_t mid,uint32_t* error){
 uint64_t at=uint64_t(run.bank)*segment.capacity+run.begin+mid;
 uint64_t stored_hash=segment.keys[at];
 if(hash<stored_hash)return -1;if(hash>stored_hash)return 1;
 uint32_t row=segment.rows[at];
 if(uint64_t(row)>=segment.capacity){atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return -1;}
 const T* states=static_cast<const T*>(segment.states);
 uint64_t stride=segment.capacity*segment.banks,offset=uint64_t(run.bank)*segment.capacity+row;
 uint32_t child=action.child(origin);
 for(uint32_t e=0;e<action.elements;e++){
  int64_t x=action.value(child,e),y=int64_t(states[uint64_t(e)*stride+offset]);
  if(x<y)return -1;if(x>y)return 1;
 }
 return 0;
}
// Sample the sorted hash column at fixed quantiles. Strict inequalities
// keep every occurrence of a hash that straddles one or more fences.
constexpr uint64_t kFencePartitions=8;
constexpr uint64_t kMinFenceRows=256;
constexpr uint64_t kLegacyPrefixMaskRows=32;
__device__ uint64_t prefix_bit(uint64_t hash){
 // Routing already partitions by the high bits of each hash word.
 // Use low bits so a rank or shard does not collapse the mask.
 return uint64_t(1)<<(hash&63);
}
__device__ bool may_contain_hash(const MgbfsBatchHashRun& run,uint64_t hash){
 if(hash<run.min_hash||hash>run.max_hash)return false;
 if(run.length>=kMinFenceRows)return true;
 if(run.length<=kLegacyPrefixMaskRows&&!run.hash_fences[4]){
  const uint64_t prefixes=run.hash_fences[0];
  return !prefixes||(prefixes&prefix_bit(hash));
 }
 const uint32_t word=(hash>>6)&3;
 return (run.hash_fences[word]&prefix_bit(hash))!=0;
}
__device__ void fill_fences(MgbfsBatchHashRun& run,const uint64_t* keys){
 if(run.length<kMinFenceRows){
  // Four words cover the occupied low-byte buckets without changing the ABI.
  uint64_t words[4]{};
  for(uint64_t i=0;i<run.length;i++){
   const uint64_t hash=keys[i];
   words[(hash>>6)&3]|=prefix_bit(hash);
  }
  for(uint32_t i=0;i<4;i++)run.hash_fences[i]=words[i];
  run.hash_fences[4]=1;
  return;
 }
 for(uint64_t i=1;i<kFencePartitions;i++)
  run.hash_fences[i-1]=keys[run.length*i/kFencePartitions];
}
__device__ void fence_bounds(const MgbfsBatchHashRun& run,uint64_t first,uint64_t last,uint64_t& lo,uint64_t& hi){
 lo=0;hi=run.length;
 if(run.length<kMinFenceRows)return;
 for(uint64_t i=1;i<kFencePartitions;i++){
  const uint64_t fence=run.hash_fences[i-1];
  const uint64_t offset=run.length*i/kFencePartitions;
  if(fence<first)lo=offset+1;
  if(fence>last){hi=offset;break;}
 }
}

template<class Action> struct SeriesCandidate {
 Action action;
 uint32_t child;

 __device__ int64_t value(uint32_t element) const { return action.value(child,element); }
};

template<class T,bool ReturnHit=false,class Action>
__device__ bool probe_snapshot(const MgbfsHashSeriesView& snapshot,Action action,uint32_t origin,
 uint64_t hash,const MgbfsBatchHashSegment* segments,uint32_t segment_count,uint32_t* error,
 MgbfsHashPathHitV1* hit=nullptr){
 if(!snapshot.length)return false;
 const SeriesCandidate<Action> query{action,action.child(origin)};
 const auto result=snapshot.reference_bytes==4
  ?mgbfs_hash_series_probe<T,ReturnHit>(snapshot.keys,static_cast<const uint32_t*>(snapshot.references),
    snapshot.prefix_offsets,snapshot.length,segments,segment_count,snapshot.layout,hash,query,action.elements,hit)
  :mgbfs_hash_series_probe<T,ReturnHit>(snapshot.keys,static_cast<const MgbfsHashRowRef*>(snapshot.references),
    snapshot.prefix_offsets,snapshot.length,segments,segment_count,snapshot.layout,hash,query,action.elements,hit);
 if(result==MgbfsHashSeriesMatch::bad_reference)atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));
 return result==MgbfsHashSeriesMatch::present;
}

template<class T,bool ReturnHit=false,class Action>
__device__ bool probe_snapshots(const MgbfsHashSeriesDirectory& directory,Action action,uint32_t origin,
 uint64_t hash,const MgbfsBatchHashSegment* segments,uint32_t segment_count,uint32_t* error,uint32_t& searched,
 MgbfsHashPathHitV1* hit=nullptr){
 for(uint32_t i=0;i<directory.count;i++){
  ++searched;
  if(probe_snapshot<T,ReturnHit>(directory.series[i],action,origin,hash,segments,segment_count,error,hit))return true;
 }
 return false;
}

template<class T,bool ReturnHit=false,class Action>
__device__ bool contains(Action action,uint32_t origin,uint64_t hash,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,const MgbfsBatchHashRun* runs,
 uint32_t run_count,uint32_t* error,uint32_t& searched,uint32_t first_run=0,MgbfsHashPathHitV1* hit=nullptr){
 for(uint32_t j=first_run;j<run_count;j++){
  MgbfsBatchHashRun run=runs[j];
  if(run.segment>=segment_count){atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return false;}
  if(!run.length||run.min_hash>run.max_hash){
   atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return false;
  }
  if(!may_contain_hash(run,hash))continue;
  ++searched;
  MgbfsBatchHashSegment s=segments[run.segment];
  if(!s.states||!s.keys||!s.rows||!s.capacity||s.capacity>(uint64_t(1)<<32)||
   !(s.banks==1||s.banks==3)||run.bank>=s.banks||run.begin>s.capacity||run.length>s.capacity-run.begin){
   atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return false;
  }
  uint64_t lo,hi;fence_bounds(run,hash,hash,lo,hi);
  const uint64_t past=hi;
  while(lo<hi){uint64_t mid=lo+(hi-lo)/2;int cmp=compare_candidate<T>(action,origin,hash,s,run,mid,error);
   if(cmp>0)lo=mid+1;else hi=mid;}
  if(lo<past&&compare_candidate<T>(action,origin,hash,s,run,lo,error)==0){
   if constexpr(ReturnHit)*hit={run.segment,run.bank,s.rows[uint64_t(run.bank)*s.capacity+run.begin+lo],1};
   return true;
  }
 }
 return false;
}

__device__ uint64_t hash_bound(const uint64_t* keys,uint64_t lo,uint64_t hi,uint64_t hash,bool upper){
 while(lo<hi){
  const uint64_t mid=lo+(hi-lo)/2,key=keys[mid];
  if(key<hash||(upper&&key==hash))lo=mid+1;
  else hi=mid;
 }
 return lo;
}

// Origins are already ordered by full hash. Each warp shares the enclosing
// hash range and narrows the probe into each run once for all 32 candidates.
// Equality still uses the original state and can span the entire hash run.
// Exact diagnostics count lookup executions, including repeated probes on
// spill/retry. A warp aggregates equal counts before updating the histogram.
constexpr uint32_t exact_miss_limit=64;
__device__ void record_miss(unsigned long long* histogram,uint64_t index,uint32_t searched,bool exact){
 if(!histogram)return;
 if(!exact){
  if((index&127)==0)atomicAdd(histogram+(searched?32-__clz(searched):0),1ULL);
  return;
 }
 const uint32_t bin=searched<=exact_miss_limit?searched:exact_miss_limit+1;
#if __CUDA_ARCH__ >= 700
 const uint32_t group=__match_any_sync(__activemask(),bin);
 if(uint32_t(__ffs(group)-1)==(threadIdx.x&31))atomicAdd(histogram+bin,uint64_t(__popc(group)));
#else
 atomicAdd(histogram+bin,1ULL);
#endif
}

__device__ uint32_t lookup_catalog_error(uint32_t runs,uint32_t capacity,
 const MgbfsHashSeriesDirectory& directory,uint32_t max_series){
 if(runs>capacity||directory.replaced_runs>runs)return MGBFS_SEGMENTED_BAD_REFERENCE;
 if(max_series&&uint64_t(runs-directory.replaced_runs)+directory.count>max_series)
  return MGBFS_SEGMENTED_LOOKUP_LIMIT;
 return 0;
}

__device__ bool valid_lookup_catalog(uint32_t runs,uint32_t capacity,MgbfsHashSeriesDirectory directory,
 uint32_t max_series,uint32_t* flags,uint32_t n,uint32_t* error){
 const uint32_t code=lookup_catalog_error(runs,capacity,directory,max_series);
 if(!code)return true;
 if(!blockIdx.x&&!threadIdx.x)atomicOr(error,code);
 // The scan still runs before the error-aware reserve kernel.
 for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=uint64_t(blockDim.x)*gridDim.x)
  flags[i]=0;
 return false;
}

struct PathTarget {
 uint32_t elements;
 const int64_t* coordinates;
 __device__ uint32_t child(uint32_t)const{return 0;}
 __device__ int64_t value(uint32_t,uint32_t e)const{return coordinates[e];}
};

template<class T, class Query>
__device__ MgbfsHashPathHitV1 find_path_state(Query query, uint64_t hash,
 const MgbfsBatchHashSegment* segments, uint32_t segment_count, const MgbfsBatchHashRun* runs,
 uint32_t run_count, uint32_t* error, const MgbfsHashSeriesDirectory& directory) {
 MgbfsHashPathHitV1 hit{};
 uint32_t searched = 0;
 if (!probe_snapshots<T, true>(directory, query, 0, hash, segments, segment_count, error, searched, &hit))
  contains<T, true>(query, 0, hash, segments, segment_count, runs, run_count, error,
   searched, directory.replaced_runs, &hit);
 return hit;
}

template<class T>
__global__ void find_path_targets(uint32_t elements,const int64_t* targets,const uint64_t* target_hashes,
 uint32_t target_count,const MgbfsBatchHashSegment* segments,uint32_t segment_count,
 const MgbfsBatchHashRun* runs,const uint32_t* run_count,uint32_t run_capacity,
 MgbfsHashPathHitV1* hits,uint32_t* error,MgbfsHashSeriesDirectory directory,uint32_t max_series){
 const uint32_t nr=*run_count;
 const uint32_t invalid=lookup_catalog_error(nr,run_capacity,directory,max_series);
 if(invalid&&!blockIdx.x&&!threadIdx.x)atomicOr(error,invalid);
 for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<target_count;i+=uint64_t(blockDim.x)*gridDim.x){
  MgbfsHashPathHitV1 hit{};
  if(!invalid){
   const PathTarget query{elements,targets+i*elements};
   hit = find_path_state<T>(query, target_hashes[i], segments, segment_count, runs, nr, error, directory);
  }
  hits[i]=hit;
 }
}

template<class T> struct PathFrontierRow {
 uint32_t elements;
 const T* row;
 uint64_t stride;

 __device__ uint32_t child(uint32_t) const { return 0; }
 __device__ int64_t value(uint32_t, uint32_t e) const { return row[uint64_t(e) * stride]; }
 __device__ uint64_t hash(uint64_t seed, uint32_t bits) const {
  uint64_t hash = seed;
  for (uint32_t e = 0; e < elements; ++e) hash = mix64(hash ^ uint64_t(value(0, e)) ^ uint64_t(e));
  return finish_hash(hash, bits);
 }
};

template<class T>
__device__ MgbfsHashPathHitV1 find_frontier_row(uint32_t elements, const T* row,
 uint64_t stride, uint64_t seed, uint32_t hash_bits, const MgbfsBatchHashSegment* segments,
 uint32_t segment_count, const MgbfsBatchHashRun* runs, uint32_t run_count, uint32_t* error,
 const MgbfsHashSeriesDirectory& directory) {
 const PathFrontierRow<T> query{elements, row, stride};
 return find_path_state<T>(query, query.hash(seed, hash_bits), segments, segment_count,
  runs, run_count, error, directory);
}

template<class T>
__global__ void meet_path_frontier(uint32_t elements, const T* source_states, uint64_t source_stride,
 uint64_t begin, uint64_t end, uint32_t source_range, uint64_t seed, uint32_t hash_bits,
 const MgbfsBatchHashSegment* segments, uint32_t segment_count, const MgbfsBatchHashRun* runs,
 const uint32_t* run_count, uint32_t run_capacity, MgbfsHashPathMeetingV1* hit, uint32_t* error,
 MgbfsHashSeriesDirectory directory, uint32_t max_series) {
 const uint32_t nr = *run_count;
 if (const uint32_t invalid = lookup_catalog_error(nr, run_capacity, directory, max_series)) {
  if (!blockIdx.x && !threadIdx.x) atomicOr(error, invalid);
  return;
 }
 cuda::atomic_ref<uint32_t, cuda::thread_scope_device> found(hit->found);
 const uint64_t step = uint64_t(blockDim.x) * gridDim.x;
 for (uint64_t i = begin + uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < end; i += step) {
  if (found.load(cuda::memory_order_relaxed)) return;
  const auto opposite = find_frontier_row(elements, source_states + i, source_stride, seed,
   hash_bits, segments, segment_count, runs, nr, error, directory);
  if (opposite.found) {
   uint32_t unclaimed = 0;
   if (found.compare_exchange_strong(unclaimed, 1, cuda::memory_order_relaxed)) {
    // The flag arbitrates the writer. The caller reads the record after stream completion.
    hit->source_range = source_range;
    hit->source_row = uint32_t(i);
    hit->opposite_segment = opposite.segment;
    hit->opposite_bank = opposite.bank;
    hit->opposite_row = opposite.row;
   }
   return;
  }
 }
}

template<class T>
__global__ void join_window_frontier(uint32_t elements, const T* left_states, uint64_t left_stride,
 uint64_t begin, uint64_t end, uint64_t seed, uint32_t hash_bits, uint32_t expected_bank,
 const MgbfsBatchHashSegment* segments, uint32_t segment_count, const MgbfsBatchHashRun* runs,
 const uint32_t* run_count, uint32_t run_capacity, MgbfsHashPathHitV1* hits, uint32_t* error,
 MgbfsHashSeriesDirectory directory, uint32_t max_series) {
 const uint32_t nr = *run_count;
 const uint32_t invalid = lookup_catalog_error(nr, run_capacity, directory, max_series);
 if (invalid && !blockIdx.x && !threadIdx.x) atomicOr(error, invalid);
 const uint64_t step = uint64_t(blockDim.x) * gridDim.x;
 for (uint64_t i = begin + uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < end; i += step) {
  MgbfsHashPathHitV1 hit{};
  if (!invalid) {
   hit = find_frontier_row(elements, left_states + i, left_stride, seed, hash_bits,
    segments, segment_count, runs, nr, error, directory);
   if (hit.found && hit.bank != expected_bank) {
    atomicOr(error, uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));
    hit = {};
   }
  }
  hits[i - begin] = hit;
 }
}

template<class T>
__global__ void resolve_window_edges(uint32_t elements, const T* states,
 const GenericRouteRecord* records, uint32_t rows, uint32_t begin, uint32_t count,
 uint32_t source, uint32_t shard, uint32_t generators, uint32_t expected_bank,
 const MgbfsBatchHashSegment* segments, uint32_t segment_count, const MgbfsBatchHashRun* runs,
 const uint32_t* run_count, uint32_t run_capacity, MgbfsHashWindowEdgeV1* edges, uint32_t* error,
 MgbfsHashSeriesDirectory directory, uint32_t max_series) {
 const uint32_t runs_live = *run_count;
 const uint32_t invalid = lookup_catalog_error(runs_live, run_capacity, directory, max_series);
 if (invalid && !blockIdx.x && !threadIdx.x) atomicOr(error, invalid);
 const uint64_t step = uint64_t(blockDim.x) * gridDim.x;
 for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += step) {
  edges[i] = {};
  if (invalid) continue;
  const uint64_t row = uint64_t(begin) + i;
  const auto record = records[row];
  if (record.source != source || record.shard != shard || record.generator >= generators) {
   atomicOr(error, uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));
   continue;
  }
  const PathFrontierRow<T> query{elements, states + row, rows};
  const auto child = find_path_state<T>(query, record.hash, segments, segment_count, runs,
   runs_live, error, directory);
  if (!child.found || child.bank != expected_bank) {
   atomicOr(error, uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));
   continue;
  }
  edges[i] = {record.parent, child.segment, child.bank, child.row, record.source, record.generator, 1};
 }
}

template<class T,class Action>
__global__ void unique_and_probe_prefix(GenericSortedOriginLess<Action> order,const uint32_t* sorted,uint32_t n,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,const MgbfsBatchHashRun* runs,
 const uint32_t* run_count,uint32_t run_capacity,uint32_t* flags,uint32_t* error,unsigned long long* misses,MgbfsHashSeriesDirectory directory,uint32_t max_series,bool exact_misses){
 constexpr uint32_t warp_width=32, mask=UINT32_MAX;
 const uint32_t nr=*run_count;
 if(!valid_lookup_catalog(nr,run_capacity,directory,max_series,flags,n,error))return;
 const uint32_t lane=threadIdx.x%warp_width;
 for(uint64_t begin=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x-lane;begin<n;begin+=uint64_t(blockDim.x)*gridDim.x){
  const uint64_t i=begin+lane;
  const uint32_t origin=i<n?sorted[i]:UINT32_MAX;
  const uint64_t hash=origin!=UINT32_MAX?order.hashes[origin]:0;
  const uint32_t valid=__ballot_sync(mask,origin!=UINT32_MAX);
  bool fresh=origin!=UINT32_MAX&&(!i||order.exact_compare(sorted[i-1],origin)!=0);
  uint32_t searched=0;
  if(fresh)
   fresh=!probe_snapshots<T>(directory,order.action,origin,hash,segments,segment_count,error,searched);
  if(valid){
   for(uint32_t j=directory.replaced_runs;j<nr&&__any_sync(mask,fresh);j++){
    const auto run=runs[j];
    if(run.segment>=segment_count||!run.length||run.min_hash>run.max_hash){
     if(!lane)atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));
     break;
    }
    // Use only unresolved lanes whose hash can occur in this run.
    // Otherwise a hit in an earlier run, or a gap between lane hashes,
    // can still trigger two unnecessary searches here.
    const bool eligible=fresh&&may_contain_hash(run,hash);
    searched+=eligible;
    const uint32_t probes=__ballot_sync(mask,eligible);
    if(!probes)continue;
    const auto s=segments[run.segment];
    if(!s.states||!s.keys||!s.rows||!s.capacity||s.capacity>(uint64_t(1)<<32)||
     !(s.banks==1||s.banks==3)||run.bank>=s.banks||run.begin>s.capacity||run.length>s.capacity-run.begin){
     if(!lane)atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));
     break;
    }
    const uint64_t first=__shfl_sync(mask,static_cast<unsigned long long>(hash),__ffs(probes)-1);
    const uint64_t last=__shfl_sync(mask,static_cast<unsigned long long>(hash),31-__clz(probes));
    uint64_t lo=0,hi=0;
    const uint64_t* keys=s.keys+uint64_t(run.bank)*s.capacity+run.begin;
    if(!lane){
     fence_bounds(run,first,last,lo,hi);
     // A single eligible lane can do one exact search within the fences.
     // Two extra hash searches only pay off when lanes share the bounds.
     if(__popc(probes)>1&&run.length>kLegacyPrefixMaskRows){
      lo=hash_bound(keys,lo,hi,first,false);
      hi=hash_bound(keys,lo,hi,last,true);
     }
    }
    lo=__shfl_sync(mask,static_cast<unsigned long long>(lo),0);
    hi=__shfl_sync(mask,static_cast<unsigned long long>(hi),0);
    if(!eligible||lo==hi)continue;
    const uint64_t past=hi;
    while(lo<hi){
     const uint64_t mid=lo+(hi-lo)/2;
     const int cmp=compare_candidate<T>(order.action,origin,hash,s,run,mid,error);
     if(cmp>0)lo=mid+1;else hi=mid;
    }
    if(lo<past&&compare_candidate<T>(order.action,origin,hash,s,run,lo,error)==0)fresh=false;
   }
  }
  if(fresh)record_miss(misses,i,searched,exact_misses);
  if(i<n)flags[i]=fresh;
 }
}
template<class T,class Action>
__global__ void unique_and_probe(GenericSortedOriginLess<Action> order,const uint32_t* sorted,uint32_t n,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,const MgbfsBatchHashRun* runs,
 const uint32_t* run_count,uint32_t run_capacity,uint32_t* flags,uint32_t* error,unsigned long long* misses,MgbfsHashSeriesDirectory directory,uint32_t max_series,bool exact_misses){
 uint32_t nr=*run_count;
 if(!valid_lookup_catalog(nr,run_capacity,directory,max_series,flags,n,error))return;
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
  uint32_t origin=sorted[i];uint32_t fresh=0;
  if(origin!=UINT32_MAX&&(!i||order.exact_compare(sorted[i-1],origin)!=0))
  {
   uint32_t searched=0;
   const uint64_t hash=order.hashes[origin];
   fresh=!probe_snapshots<T>(directory,order.action,origin,hash,segments,segment_count,error,searched)
    &&!contains<T>(order.action,origin,hash,segments,segment_count,runs,nr,error,searched,directory.replaced_runs);
   if(fresh)record_miss(misses,i,searched,exact_misses);
  }
  flags[i]=fresh;
 }
}
__global__ void reserve(const uint32_t* flags,const uint32_t* prefix,uint32_t n,
 const MgbfsBatchHashSegment* segments,uint32_t active,uint32_t bank,uint32_t segment_count,
 uint64_t* accepted,const uint32_t* run_count,uint32_t run_capacity,uint64_t* global_count,uint64_t limit,
 Grant* grant,uint32_t* spill_count,uint32_t* error){
 if(blockIdx.x||threadIdx.x||fatal(*error))return;
 if(active>=segment_count){atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;}
 MgbfsBatchHashSegment seg=segments[active];
 uint64_t base=*accepted;
 if(!seg.states||!seg.keys||!seg.rows||!seg.capacity||seg.capacity>(uint64_t(1)<<32)||
  !(seg.banks==1||seg.banks==3)||bank>=seg.banks||base>seg.capacity){
  atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;
 }
 uint32_t total=prefix[n-1]+flags[n-1];
 uint64_t free=seg.capacity-base;uint32_t want=free<total?uint32_t(free):total,got=want;
 uint32_t index=*run_count;
 if(want&&index>=run_capacity){*grant={base,total,0,index};*spill_count=total;atomicOr(error,uint32_t(MGBFS_SEGMENTED_RUN_GROW));return;}
 if(global_count&&want){
  cuda::atomic_ref<uint64_t,cuda::thread_scope_device> count(*global_count);uint64_t old=count.load(cuda::memory_order_relaxed);
  for(;;){if(old>=limit){got=0;break;}uint64_t remaining=limit-old;got=remaining<want?uint32_t(remaining):want;
   if(count.compare_exchange_weak(old,old+got,cuda::memory_order_acq_rel,cuda::memory_order_relaxed))break;}
  if(got<want)atomicOr(error,uint32_t(MGBFS_SEGMENTED_CAP_RETRY));
 }
 *grant={base,total,got,index};*spill_count=total-got;
}
template<class T,class Action,class Lineage>
__global__ void materialize(Action action,const uint64_t* hashes,const uint32_t* sorted,const uint32_t* flags,
 const uint32_t* prefix,uint32_t n,const Grant* grant,const MgbfsBatchHashSegment* segments,uint32_t active,
 uint32_t bank,uint32_t* spill,const uint32_t* error,Lineage lineage){
 if(fatal(*error))return;
 MgbfsBatchHashSegment seg=segments[active];
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)if(flags[i]){
  uint32_t origin=sorted[i],ordinal=prefix[i],child=action.child(origin);
  if(ordinal>=grant->granted){spill[ordinal-grant->granted]=child;continue;}
  uint64_t row=grant->base+ordinal,at=uint64_t(bank)*seg.capacity+row,stride=seg.capacity*seg.banks;
  T* states=static_cast<T*>(seg.states);
  for(uint32_t e=0;e<action.elements;e++)states[uint64_t(e)*stride+at]=T(action.value(child,e));
  seg.keys[at]=hashes[origin];seg.rows[at]=uint32_t(row);
  if constexpr(Lineage::enabled){
   // A spill stores the raw child index, so replay selects the same route
   // record even after the active extent or the tile order has changed.
   const auto& record=action.route(child);
   lineage.parents[at]={record.parent,record.source,record.generator};
  }
 }
}
__global__ void publish(const MgbfsBatchHashSegment* segments,uint32_t active,uint32_t bank,
 MgbfsBatchHashRun* runs,uint32_t* run_count,uint64_t* accepted,const Grant* g,const uint32_t* error){
 if(blockIdx.x||threadIdx.x||fatal(*error))return;
 if(g->granted){
  MgbfsBatchHashSegment seg=segments[active];uint64_t at=uint64_t(bank)*seg.capacity+g->base;
  MgbfsBatchHashRun run{active,bank,g->base,g->granted,seg.keys[at],seg.keys[at+g->granted-1],{}};
  fill_fences(run,seg.keys+at);
  runs[g->run_index]=run;
  *run_count=g->run_index+1;
 }
 *accepted=g->base+g->granted;
}
template<class T,bool Packed,class Lineage>
int launch(uint32_t elements,const T* local,const T* remote,const GenericRouteRecord* lm,const GenericRouteRecord* rm,
 const uint32_t* lc,const uint32_t* rc,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint32_t visible,
 const uint32_t* selection,uint32_t selected,uint32_t raw_begin,const MgbfsBatchHashSegment* segments,uint32_t segment_count,
 uint32_t active,uint32_t bank,MgbfsBatchHashRun* runs,uint32_t* run_count,uint32_t run_capacity,uint64_t* accepted,
 uint32_t* spill,uint32_t* spill_count,uint32_t* error,uint64_t* global_count,uint64_t limit,void* scratch,const Workspace& w,cudaEvent_t lookup_start,cudaEvent_t lookup_end,unsigned long long* misses,cudaStream_t st,MgbfsHashSeriesDirectory directory,uint32_t max_series,bool exact_misses,Lineage lineage){
 if(!selected)return int(cudaSuccess);
 char* base=static_cast<char*>(scratch);auto* hashes=reinterpret_cast<uint64_t*>(base+w.hashes);
 auto* origins=reinterpret_cast<uint32_t*>(base+w.origins);auto* sorted=reinterpret_cast<uint32_t*>(base+w.sorted);
 auto* flags=reinterpret_cast<uint32_t*>(base+w.flags);auto* prefix=reinterpret_cast<uint32_t*>(base+w.prefix);
 auto* grant=reinterpret_cast<Grant*>(base+w.grant);
 MgbfsBoundedIncomingAction<T,Packed> in{{elements,world*q,q,1,rank,world,shard,shards,local,remote,lm,rm,lc,rc,nullptr},visible};
 SelectedAction<T,Packed> action{elements,in,selection,raw_begin};
 uint32_t blocks=(selected+255)/256;if(blocks>65535)blocks=65535;
 prepare<<<blocks,256,0,st>>>(action,selected,hashes,origins,error,lineage);
 cudaError_t status=cudaGetLastError();if(status!=cudaSuccess)return int(status);
 GenericSortedOriginLess<SelectedAction<T,Packed>> cmp{action,hashes};size_t tmp=w.temp_bytes;
 status=cub::DeviceMergeSort::SortKeysCopy(base+w.temp,tmp,origins,sorted,int(selected),cmp,st);
 if(status!=cudaSuccess)return int(status);
 if(lookup_start){status=cudaEventRecord(lookup_start,st);if(status!=cudaSuccess)return int(status);}
 const char* probe_mode=std::getenv("MGBFS_SEGMENT_HASH_PREFIX_PROBE");
 if(probe_mode&&probe_mode[0]=='1'&&!probe_mode[1])
  unique_and_probe_prefix<T><<<blocks,256,0,st>>>(cmp,sorted,selected,segments,segment_count,runs,run_count,run_capacity,flags,error,misses,directory,max_series,exact_misses);
 else
  unique_and_probe<T><<<blocks,256,0,st>>>(cmp,sorted,selected,segments,segment_count,runs,run_count,run_capacity,flags,error,misses,directory,max_series,exact_misses);
 if((status=cudaGetLastError())!=cudaSuccess)return int(status);
 if(lookup_end){status=cudaEventRecord(lookup_end,st);if(status!=cudaSuccess)return int(status);}
 tmp=w.temp_bytes;status=cub::DeviceScan::ExclusiveSum(base+w.temp,tmp,flags,prefix,int(selected),st);
 if(status!=cudaSuccess)return int(status);
 reserve<<<1,1,0,st>>>(flags,prefix,selected,segments,active,bank,segment_count,accepted,run_count,run_capacity,global_count,limit,grant,spill_count,error);
 if((status=cudaGetLastError())!=cudaSuccess)return int(status);
 materialize<T><<<blocks,256,0,st>>>(action,hashes,sorted,flags,prefix,selected,grant,segments,active,bank,spill,error,lineage);
 if((status=cudaGetLastError())!=cudaSuccess)return int(status);
 publish<<<1,1,0,st>>>(segments,active,bank,runs,run_count,accepted,grant,error);
 return int(cudaGetLastError());
}
}
extern "C" int mgbfs_hash_batch_scratch_bytes(uint32_t n,uint32_t state_bytes,uint32_t packed,size_t* bytes){
 if(!bytes||(state_bytes!=1&&state_bytes!=8)||packed>1||(packed&&state_bytes!=1))return int(cudaErrorInvalidValue);
 Workspace w{};cudaError_t status=state_bytes==1?(packed?layout<SelectedAction<uint8_t,true>>(n,&w):layout<SelectedAction<uint8_t,false>>(n,&w)):layout<SelectedAction<int64_t,false>>(n,&w);
 if(status==cudaSuccess)*bytes=w.total;return int(status);
}

static bool valid_directory(const MgbfsHashSeriesDirectory& directory){
 if(directory.count>3)return false;
 uint64_t covered=0;
 for(uint32_t i=0;i<directory.count;i++){
  const auto& snapshot=directory.series[i];
  if(!snapshot.length||!snapshot.keys||!snapshot.references||!snapshot.prefix_offsets||
     !snapshot.replaced_runs||(snapshot.reference_bytes!=4&&snapshot.reference_bytes!=8)||
     snapshot.layout.row_bits>32)return false;
  covered+=snapshot.replaced_runs;
 }
 return covered==directory.replaced_runs;
}

extern "C" int mgbfs_hash_batch_find_targets_v1(uint32_t state_bytes,uint32_t elements,
 const int64_t* targets,const uint64_t* target_hashes,uint32_t target_count,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,const MgbfsBatchHashRun* runs,
 const uint32_t* run_count,uint32_t run_capacity,MgbfsHashPathHitV1* hits,uint32_t* error,void* stream,
 const MgbfsHashSeriesDirectory* view,uint32_t max_series){
 if((state_bytes!=1&&state_bytes!=8)||!elements||!max_series||
    target_count>std::numeric_limits<size_t>::max()/sizeof(int64_t)/elements)
  return int(cudaErrorInvalidValue);
 if(!target_count)return int(cudaSuccess);
 const MgbfsHashSeriesDirectory directory=view?*view:MgbfsHashSeriesDirectory{};
 if(!targets||!target_hashes||!segments||!segment_count||!runs||!run_count||!run_capacity||
    !hits||!error||!valid_directory(directory))return int(cudaErrorInvalidValue);
 const cudaStream_t st=static_cast<cudaStream_t>(stream);
 const uint32_t requested=(uint64_t(target_count)+255)/256;
 const uint32_t blocks=requested<65535?requested:65535;
 if(state_bytes==1)
  find_path_targets<uint8_t><<<blocks,256,0,st>>>(elements,targets,target_hashes,target_count,
   segments,segment_count,runs,run_count,run_capacity,hits,error,directory,max_series);
 else
  find_path_targets<int64_t><<<blocks,256,0,st>>>(elements,targets,target_hashes,target_count,
   segments,segment_count,runs,run_count,run_capacity,hits,error,directory,max_series);
 return int(cudaGetLastError());
}

static bool valid_path_frontier_span(uint32_t state_bytes, uint32_t elements, uint32_t hash_bits,
 uint64_t stride, uint64_t begin, uint64_t end) {
 constexpr uint64_t physical_rows = uint64_t{1} << 32;
 if ((state_bytes != 1 && state_bytes != 8) || !elements || hash_bits > 64 ||
     begin > end || end > stride || end > physical_rows) return false;
 if (begin == end) return true;
 const uint64_t max_items = std::numeric_limits<size_t>::max() / state_bytes;
 return end <= max_items && uint64_t(elements - 1) <= (max_items - end) / stride;
}

// One cleared hit may be shared by ordered intervals, each with a caller's range tag.
extern "C" int mgbfs_hash_batch_meet_frontier_v1(uint32_t state_bytes, uint32_t elements,
 const void* source_states, uint64_t source_stride, uint64_t begin, uint64_t end, uint32_t source_range,
 uint64_t seed, uint32_t hash_bits, const MgbfsBatchHashSegment* segments, uint32_t segment_count,
 const MgbfsBatchHashRun* runs, const uint32_t* run_count, uint32_t run_capacity,
 MgbfsHashPathMeetingV1* hit, uint32_t* error, void* stream, const MgbfsHashSeriesDirectory* view,
 uint32_t max_series) {
 if (!max_series || !valid_path_frontier_span(state_bytes, elements, hash_bits, source_stride,
     begin, end)) return int(cudaErrorInvalidValue);
 if (begin == end) return int(cudaSuccess);
 const MgbfsHashSeriesDirectory directory = view ? *view : MgbfsHashSeriesDirectory{};
 if (!source_states || !segments || !segment_count || !runs || !run_count || !run_capacity ||
     !hit || !error || !valid_directory(directory)) return int(cudaErrorInvalidValue);
 const auto st = static_cast<cudaStream_t>(stream);
 const uint32_t requested = (end - begin + 255) / 256;
 const uint32_t blocks = requested < 65535 ? requested : 65535;
 if (state_bytes == 1)
  meet_path_frontier<<<blocks, 256, 0, st>>>(elements, static_cast<const uint8_t*>(source_states),
   source_stride, begin, end, source_range, seed, hash_bits, segments, segment_count, runs, run_count,
   run_capacity, hit, error, directory, max_series);
 else
  meet_path_frontier<<<blocks, 256, 0, st>>>(elements, static_cast<const int64_t*>(source_states),
   source_stride, begin, end, source_range, seed, hash_bits, segments, segment_count, runs, run_count,
   run_capacity, hit, error, directory, max_series);
 return int(cudaGetLastError());
}

extern "C" int mgbfs_hash_batch_join_window_v1(uint32_t state_bytes, uint32_t elements,
 const void* left_states, uint64_t left_stride, uint64_t begin, uint64_t end, uint64_t seed,
 uint32_t hash_bits, uint32_t expected_bank, const MgbfsBatchHashSegment* segments,
 uint32_t segment_count, const MgbfsBatchHashRun* runs, const uint32_t* run_count,
 uint32_t run_capacity, MgbfsHashPathHitV1* hits, uint32_t hit_capacity, uint32_t* error,
 void* stream, const MgbfsHashSeriesDirectory* view, uint32_t max_series) {
 if (!max_series || expected_bank >= 3 ||
     !valid_path_frontier_span(state_bytes, elements, hash_bits, left_stride, begin, end) ||
     end - begin > hit_capacity) return int(cudaErrorInvalidValue);
 if (begin == end) return int(cudaSuccess);
 const MgbfsHashSeriesDirectory directory = view ? *view : MgbfsHashSeriesDirectory{};
 if (!left_states || !segments || !segment_count || !runs || !run_count || !run_capacity ||
     !hits || !error || !valid_directory(directory)) return int(cudaErrorInvalidValue);
 const auto st = static_cast<cudaStream_t>(stream);
 const uint32_t requested = (end - begin + 255) / 256;
 const uint32_t blocks = requested < 65535 ? requested : 65535;
 if (state_bytes == 1)
  join_window_frontier<<<blocks, 256, 0, st>>>(elements, static_cast<const uint8_t*>(left_states),
   left_stride, begin, end, seed, hash_bits, expected_bank, segments, segment_count, runs,
   run_count, run_capacity, hits, error, directory, max_series);
 else
  join_window_frontier<<<blocks, 256, 0, st>>>(elements, static_cast<const int64_t*>(left_states),
   left_stride, begin, end, seed, hash_bits, expected_bank, segments, segment_count, runs,
   run_count, run_capacity, hits, error, directory, max_series);
 return int(cudaGetLastError());
}

extern "C" int mgbfs_hash_batch_resolve_window_v1(uint32_t state_bytes, uint32_t elements,
 const void* states, const GenericRouteRecord* records, uint32_t rows, uint32_t begin,
 uint32_t count, uint32_t source, uint32_t shard, uint32_t generators, uint32_t expected_bank,
 const MgbfsBatchHashSegment* segments, uint32_t segment_count, const MgbfsBatchHashRun* runs,
 const uint32_t* run_count, uint32_t run_capacity, MgbfsHashWindowEdgeV1* edges,
 uint32_t output_capacity, uint32_t* error, void* stream, const MgbfsHashSeriesDirectory* view,
 uint32_t max_series) {
 if ((state_bytes != 1 && state_bytes != 8) || !elements || !generators || source >= 128 ||
     shard >= 4096 || expected_bank >= 3 || !count || count > output_capacity ||
     begin > rows || count > rows - begin ||
     uint64_t(rows) > std::numeric_limits<size_t>::max() / state_bytes / elements ||
     !states || !records || !segments || !segment_count || !runs || !run_count || !run_capacity ||
     !edges || !error || !max_series) return int(cudaErrorInvalidValue);
 const MgbfsHashSeriesDirectory directory = view ? *view : MgbfsHashSeriesDirectory{};
 if (!valid_directory(directory)) return int(cudaErrorInvalidValue);
 const auto st = static_cast<cudaStream_t>(stream);
 const uint32_t requested = (uint64_t(count) + 255) / 256;
 const uint32_t blocks = requested < 65535 ? requested : 65535;
 if (state_bytes == 1)
  resolve_window_edges<<<blocks, 256, 0, st>>>(elements, static_cast<const uint8_t*>(states),
   records, rows, begin, count, source, shard, generators, expected_bank, segments, segment_count,
   runs, run_count, run_capacity, edges, error, directory, max_series);
 else
  resolve_window_edges<<<blocks, 256, 0, st>>>(elements, static_cast<const int64_t*>(states),
   records, rows, begin, count, source, shard, generators, expected_bank, segments, segment_count,
   runs, run_count, run_capacity, edges, error, directory, max_series);
 return int(cudaGetLastError());
}

template<class Lineage=NoPathOutput>
static int accept_profiled(uint32_t state_bytes,uint32_t packed,uint32_t elements,const void* local,const void* remote,
 const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,
 uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint32_t visible,const uint32_t* selection,uint32_t selected,uint32_t raw_begin,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,uint32_t active,uint32_t bank,MgbfsBatchHashRun* runs,
 uint32_t* run_count,uint32_t run_capacity,uint64_t* accepted,uint32_t* spill,uint32_t* spill_count,uint32_t* error,
 uint64_t* global_count,uint64_t limit,void* scratch,size_t scratch_bytes,void* lookup_start,void* lookup_end,unsigned long long* misses,void* stream,const MgbfsHashSeriesDirectory* view,uint32_t max_series,bool exact_misses,Lineage lineage={}){
 if(exact_misses&&(!misses||!max_series||max_series>exact_miss_limit))return int(cudaErrorInvalidValue);
 const MgbfsHashSeriesDirectory directory=view?*view:MgbfsHashSeriesDirectory{};
 if(!valid_directory(directory))return int(cudaErrorInvalidValue);
 uint64_t bound=uint64_t(world)*q;
 if((state_bytes!=1&&state_bytes!=8)||packed>1||(packed&&(state_bytes!=1||elements>24))||!elements||
  !world||world>128||rank>=world||!shards||shard>=shards||!q||bound>=0x7fffffffULL||visible>q||selected>bound||
  (!selection&&uint64_t(raw_begin)+selected>uint64_t(world)*visible)||(selection&&raw_begin!=0)||!lm||!rm||!lc||!rc||(!packed&&(!local||!remote))||
  !segments||!segment_count||active>=segment_count||!runs||!run_count||!run_capacity||!accepted||!spill||!spill_count||!error||
  bool(lookup_start)!=bool(lookup_end))
  return int(cudaErrorInvalidValue);
 Workspace w{};cudaError_t status=state_bytes==1?(packed?layout<SelectedAction<uint8_t,true>>(selected,&w):layout<SelectedAction<uint8_t,false>>(selected,&w)):layout<SelectedAction<int64_t,false>>(selected,&w);
 if(status!=cudaSuccess)return int(status);
 if(selected&&(!scratch||uintptr_t(scratch)%kAlign||scratch_bytes<w.total))return int(cudaErrorInvalidValue);
 cudaStream_t st=static_cast<cudaStream_t>(stream);
 if constexpr(!Lineage::enabled){
  if(state_bytes==1&&packed)return launch<uint8_t,true>(elements,static_cast<const uint8_t*>(local),static_cast<const uint8_t*>(remote),lm,rm,lc,rc,rank,world,shard,shards,q,visible,selection,selected,raw_begin,segments,segment_count,active,bank,runs,run_count,run_capacity,accepted,spill,spill_count,error,global_count,limit,scratch,w,static_cast<cudaEvent_t>(lookup_start),static_cast<cudaEvent_t>(lookup_end),misses,st,directory,max_series,exact_misses,lineage);
 }
 if(state_bytes==1)return launch<uint8_t,false>(elements,static_cast<const uint8_t*>(local),static_cast<const uint8_t*>(remote),lm,rm,lc,rc,rank,world,shard,shards,q,visible,selection,selected,raw_begin,segments,segment_count,active,bank,runs,run_count,run_capacity,accepted,spill,spill_count,error,global_count,limit,scratch,w,static_cast<cudaEvent_t>(lookup_start),static_cast<cudaEvent_t>(lookup_end),misses,st,directory,max_series,exact_misses,lineage);
 return launch<int64_t,false>(elements,static_cast<const int64_t*>(local),static_cast<const int64_t*>(remote),lm,rm,lc,rc,rank,world,shard,shards,q,visible,selection,selected,raw_begin,segments,segment_count,active,bank,runs,run_count,run_capacity,accepted,spill,spill_count,error,global_count,limit,scratch,w,static_cast<cudaEvent_t>(lookup_start),static_cast<cudaEvent_t>(lookup_end),misses,st,directory,max_series,exact_misses,lineage);
}

extern "C" int mgbfs_hash_batch_accept_profiled_v6(uint32_t state_bytes,uint32_t packed,uint32_t elements,const void* local,const void* remote,
 const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,
 uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint32_t visible,const uint32_t* selection,uint32_t selected,uint32_t raw_begin,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,uint32_t active,uint32_t bank,MgbfsBatchHashRun* runs,
 uint32_t* run_count,uint32_t run_capacity,uint64_t* accepted,uint32_t* spill,uint32_t* spill_count,uint32_t* error,
 uint64_t* global_count,uint64_t limit,void* scratch,size_t scratch_bytes,void* lookup_start,void* lookup_end,unsigned long long* misses,void* stream,const MgbfsHashSeriesDirectory* view,uint32_t max_series){
 return accept_profiled(state_bytes,packed,elements,local,remote,lm,rm,lc,rc,rank,world,shard,shards,q,visible,
  selection,selected,raw_begin,segments,segment_count,active,bank,runs,run_count,run_capacity,accepted,spill,spill_count,error,
  global_count,limit,scratch,scratch_bytes,lookup_start,lookup_end,misses,stream,view,max_series,false);
}

// Path V1 receives one immutable RAW SoA shard run. The flat queue's
// synthetic rank is zero; the route record contains the actual source.
extern "C" int mgbfs_hash_batch_accept_path_v1(uint32_t state_bytes,uint32_t packed,uint32_t elements,const void* local,const void* remote,
 const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,
 uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint32_t visible,const uint32_t* selection,uint32_t selected,uint32_t raw_begin,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,uint32_t active,uint32_t bank,MgbfsBatchHashRun* runs,
 uint32_t* run_count,uint32_t run_capacity,uint64_t* accepted,uint32_t* spill,uint32_t* spill_count,uint32_t* error,
 uint64_t* global_count,uint64_t limit,void* scratch,size_t scratch_bytes,void* lookup_start,void* lookup_end,unsigned long long* misses,void* stream,const MgbfsHashSeriesDirectory* view,uint32_t max_series,
 MgbfsHashPathParentV1* parents,uint32_t source_world,uint32_t generators){
 if(packed||rank||world!=1||shard||shards!=1||!max_series||!parents||!source_world||source_world>128||!generators)
  return int(cudaErrorInvalidValue);
 return accept_profiled(state_bytes,packed,elements,local,remote,lm,rm,lc,rc,rank,world,shard,shards,q,visible,
  selection,selected,raw_begin,segments,segment_count,active,bank,runs,run_count,run_capacity,accepted,spill,spill_count,error,
  global_count,limit,scratch,scratch_bytes,lookup_start,lookup_end,misses,stream,view,max_series,false,
  PathOutput{parents,source_world,generators});
}

// bins 0..64 are exact; bin 65 catches any violation of the supported cap.
extern "C" int mgbfs_hash_batch_accept_exact_miss_v1(uint32_t state_bytes,uint32_t packed,uint32_t elements,const void* local,const void* remote,
 const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,
 uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint32_t visible,const uint32_t* selection,uint32_t selected,uint32_t raw_begin,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,uint32_t active,uint32_t bank,MgbfsBatchHashRun* runs,
 uint32_t* run_count,uint32_t run_capacity,uint64_t* accepted,uint32_t* spill,uint32_t* spill_count,uint32_t* error,
 uint64_t* global_count,uint64_t limit,void* scratch,size_t scratch_bytes,void* lookup_start,void* lookup_end,unsigned long long* misses,void* stream,const MgbfsHashSeriesDirectory* view,uint32_t max_series){
 return accept_profiled(state_bytes,packed,elements,local,remote,lm,rm,lc,rc,rank,world,shard,shards,q,visible,
  selection,selected,raw_begin,segments,segment_count,active,bank,runs,run_count,run_capacity,accepted,spill,spill_count,error,
  global_count,limit,scratch,scratch_bytes,lookup_start,lookup_end,misses,stream,view,max_series,true);
}
extern "C" int mgbfs_hash_batch_accept_profiled_v5(uint32_t state_bytes,uint32_t packed,uint32_t elements,const void* local,const void* remote,
 const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,
 uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint32_t visible,const uint32_t* selection,uint32_t selected,uint32_t raw_begin,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,uint32_t active,uint32_t bank,MgbfsBatchHashRun* runs,
 uint32_t* run_count,uint32_t run_capacity,uint64_t* accepted,uint32_t* spill,uint32_t* spill_count,uint32_t* error,
 uint64_t* global_count,uint64_t limit,void* scratch,size_t scratch_bytes,void* lookup_start,void* lookup_end,unsigned long long* misses,void* stream,const MgbfsHashSeriesDirectory* view){
 return mgbfs_hash_batch_accept_profiled_v6(state_bytes,packed,elements,local,remote,lm,rm,lc,rc,rank,world,shard,shards,q,visible,
  selection,selected,raw_begin,segments,segment_count,active,bank,runs,run_count,run_capacity,accepted,spill,spill_count,error,
  global_count,limit,scratch,scratch_bytes,lookup_start,lookup_end,misses,stream,view,0);
}
extern "C" int mgbfs_hash_batch_accept_profiled_v4(uint32_t state_bytes,uint32_t packed,uint32_t elements,const void* local,const void* remote,
 const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,
 uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint32_t visible,const uint32_t* selection,uint32_t selected,uint32_t raw_begin,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,uint32_t active,uint32_t bank,MgbfsBatchHashRun* runs,
 uint32_t* run_count,uint32_t run_capacity,uint64_t* accepted,uint32_t* spill,uint32_t* spill_count,uint32_t* error,
 uint64_t* global_count,uint64_t limit,void* scratch,size_t scratch_bytes,void* lookup_start,void* lookup_end,unsigned long long* misses,void* stream,const MgbfsHashSeriesView* view){
 MgbfsHashSeriesDirectory directory{};
 if(view&&view->length){
  directory.series[0]=*view;directory.count=1;directory.replaced_runs=view->replaced_runs;
 }else if(view&&view->replaced_runs)return int(cudaErrorInvalidValue);
 return mgbfs_hash_batch_accept_profiled_v5(state_bytes,packed,elements,local,remote,lm,rm,lc,rc,rank,world,shard,shards,q,visible,
  selection,selected,raw_begin,segments,segment_count,active,bank,runs,run_count,run_capacity,accepted,spill,spill_count,error,
  global_count,limit,scratch,scratch_bytes,lookup_start,lookup_end,misses,stream,&directory);
}
extern "C" int mgbfs_hash_batch_accept_profiled_v3(uint32_t state_bytes,uint32_t packed,uint32_t elements,const void* local,const void* remote,
 const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,
 uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint32_t visible,const uint32_t* selection,uint32_t selected,uint32_t raw_begin,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,uint32_t active,uint32_t bank,MgbfsBatchHashRun* runs,
 uint32_t* run_count,uint32_t run_capacity,uint64_t* accepted,uint32_t* spill,uint32_t* spill_count,uint32_t* error,
 uint64_t* global_count,uint64_t limit,void* scratch,size_t scratch_bytes,void* lookup_start,void* lookup_end,unsigned long long* misses,void* stream){
 return mgbfs_hash_batch_accept_profiled_v4(state_bytes,packed,elements,local,remote,lm,rm,lc,rc,rank,world,shard,shards,q,visible,
  selection,selected,raw_begin,segments,segment_count,active,bank,runs,run_count,run_capacity,accepted,spill,spill_count,error,
  global_count,limit,scratch,scratch_bytes,lookup_start,lookup_end,misses,stream,nullptr);
}
extern "C" int mgbfs_hash_batch_accept_timed_v2(uint32_t state_bytes,uint32_t packed,uint32_t elements,const void* local,const void* remote,
 const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,
 uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint32_t visible,const uint32_t* selection,uint32_t selected,uint32_t raw_begin,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,uint32_t active,uint32_t bank,MgbfsBatchHashRun* runs,
 uint32_t* run_count,uint32_t run_capacity,uint64_t* accepted,uint32_t* spill,uint32_t* spill_count,uint32_t* error,
 uint64_t* global_count,uint64_t limit,void* scratch,size_t scratch_bytes,void* lookup_start,void* lookup_end,void* stream){
 return mgbfs_hash_batch_accept_profiled_v3(state_bytes,packed,elements,local,remote,lm,rm,lc,rc,rank,world,shard,shards,q,visible,
  selection,selected,raw_begin,segments,segment_count,active,bank,runs,run_count,run_capacity,accepted,spill,spill_count,error,
  global_count,limit,scratch,scratch_bytes,lookup_start,lookup_end,nullptr,stream);
}



// Seed the unique canonical start using the same exact run/index contract as
// accepted tiles. The caller has already admitted both segment and catalog.
namespace {
template<class T>
__global__ void seed_exact(const T* input,uint32_t elements,MgbfsBatchHashSegment seg,uint64_t hash,
 MgbfsBatchHashRun* runs,uint32_t* run_count,uint64_t* accepted){
 if(blockIdx.x||threadIdx.x)return;
 T* states=static_cast<T*>(seg.states);uint64_t stride=seg.capacity*seg.banks;
 for(uint32_t e=0;e<elements;e++)states[uint64_t(e)*stride]=input[e];
 seg.keys[0]=hash;seg.rows[0]=0;runs[0]={0,0,0,1,hash,hash,{prefix_bit(hash)}};*run_count=1;*accepted=1;
}
__global__ void retire_bank_runs(MgbfsBatchHashRun* runs,uint32_t* count,uint32_t capacity,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,uint32_t bank,uint32_t* error){
 if(blockIdx.x||threadIdx.x)return;
 uint32_t n=*count;if(n>capacity){atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;}
 for(uint32_t i=0;i<n;i++){
  auto r=runs[i];if(r.segment>=segment_count||r.bank>=segments[r.segment].banks){
   atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;
  }
 }
 uint32_t out=0;for(uint32_t i=0;i<n;i++)if(runs[i].bank!=bank)runs[out++]=runs[i];
 *count=out;
}

// Repack the descriptor IDs after a drained bank releases empty segments.
// Validate every reference before mutating any run.
__global__ void remap_run_segments(MgbfsBatchHashRun* runs,uint32_t count,
 const uint32_t* old_to_new,uint32_t segments,uint32_t* error){
 if(blockIdx.x||threadIdx.x)return;
 for(uint32_t i=0;i<count;i++){
  const uint32_t old=runs[i].segment;
  if(old>=segments||old_to_new[old]==UINT32_MAX){
   atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;
  }
 }
 for(uint32_t i=0;i<count;i++)runs[i].segment=old_to_new[runs[i].segment];
}
}

extern "C" int mgbfs_hash_batch_remap_segments_v2(MgbfsBatchHashRun* runs,uint32_t count,
 const uint32_t* old_to_new,uint32_t segments,uint32_t* error,void* stream){
 if(!runs||!old_to_new||!segments||!error)return int(cudaErrorInvalidValue);
 if(count){
  remap_run_segments<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(runs,count,old_to_new,segments,error);
  return int(cudaGetLastError());
 }
 return int(cudaSuccess);
}
extern "C" int mgbfs_hash_batch_accept_v2(uint32_t state_bytes,uint32_t packed,uint32_t elements,const void* local,const void* remote,
 const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,
 uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint32_t visible,const uint32_t* selection,uint32_t selected,uint32_t raw_begin,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,uint32_t active,uint32_t bank,MgbfsBatchHashRun* runs,
 uint32_t* run_count,uint32_t run_capacity,uint64_t* accepted,uint32_t* spill,uint32_t* spill_count,uint32_t* error,
 uint64_t* global_count,uint64_t limit,void* scratch,size_t scratch_bytes,void* stream){
 return mgbfs_hash_batch_accept_timed_v2(state_bytes,packed,elements,local,remote,lm,rm,lc,rc,rank,world,shard,shards,q,visible,
  selection,selected,raw_begin,segments,segment_count,active,bank,runs,run_count,run_capacity,accepted,spill,spill_count,error,
  global_count,limit,scratch,scratch_bytes,nullptr,nullptr,stream);
}

extern "C" int mgbfs_hash_batch_seed_v2(uint32_t state_bytes,uint32_t elements,const void* input,
 MgbfsBatchHashSegment seg,uint64_t hash,MgbfsBatchHashRun* runs,uint32_t* run_count,
 uint64_t* accepted,void* stream){
 if((state_bytes!=1&&state_bytes!=8)||!elements||!input||!seg.states||!seg.keys||!seg.rows||
  !seg.capacity||seg.capacity>(uint64_t(1)<<32)||!(seg.banks==1||seg.banks==3)||!runs||!run_count||!accepted)
  return int(cudaErrorInvalidValue);
 auto st=static_cast<cudaStream_t>(stream);
 if(state_bytes==1)seed_exact<uint8_t><<<1,1,0,st>>>(static_cast<const uint8_t*>(input),elements,seg,hash,runs,run_count,accepted);
 else seed_exact<int64_t><<<1,1,0,st>>>(static_cast<const int64_t*>(input),elements,seg,hash,runs,run_count,accepted);
 return int(cudaGetLastError());
}
extern "C" int mgbfs_hash_batch_retire_bank_v2(MgbfsBatchHashRun* runs,uint32_t* count,uint32_t capacity,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,uint32_t bank,uint32_t* error,void* stream){
 if(!runs||!count||!capacity||!segments||!segment_count||bank>=3||!error)return int(cudaErrorInvalidValue);
 retire_bank_runs<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(runs,count,capacity,segments,segment_count,bank,error);
 return int(cudaGetLastError());
}

// Adjacent-run compaction is optional. Insufficient merge workspace is a
// successful deferred result: both original published runs stay searchable.
namespace {
struct MergeControl {MgbfsBatchHashRun left,right;uint32_t run_index,count,status;};
struct MergeLayout {size_t keys,rows,control,total;};
bool merge_layout(uint32_t entries,MergeLayout* w){
 if(!w||entries>=0x7fffffffU)return false;*w={};size_t c=0;
 if(!region(c,size_t(entries)*8,&w->keys)||!region(c,size_t(entries)*4,&w->rows)||
    !region(c,sizeof(MergeControl),&w->control))return false;
 w->total=c;return true;
}
__global__ void merge_prepare(const MgbfsBatchHashSegment* segments,uint32_t segment_count,
 MgbfsBatchHashRun* runs,const uint32_t* count,uint32_t max_entries,MergeControl* control,uint32_t* result,uint32_t* error){
 if(blockIdx.x||threadIdx.x)return;
 *result=0;control->status=0;
 if(fatal(*error))return;
 uint32_t n=*count;if(n<2)return;
 MgbfsBatchHashRun a=runs[n-2],b=runs[n-1];
 if(a.segment!=b.segment||a.bank!=b.bank)return;
 if(a.segment>=segment_count){atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;}
 MgbfsBatchHashSegment seg=segments[a.segment];
 if(!seg.states||!seg.keys||!seg.rows||!seg.capacity||seg.capacity>(uint64_t(1)<<32)||
  !(seg.banks==1||seg.banks==3)||a.bank>=seg.banks||!a.length||!b.length||
  a.begin>seg.capacity||a.length>seg.capacity-a.begin||b.begin>seg.capacity||b.length>seg.capacity-b.begin||
  a.min_hash>a.max_hash||b.min_hash>b.max_hash){
  atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;
 }
 if(a.begin+a.length!=b.begin)return;
 // Directory pressure is an explicit reason to merge a skewed pair:
 // both runs stay in the same extent and the caller can replay its spill.
 if(a.length>2*b.length&&(*error&MGBFS_SEGMENTED_RUN_GROW)==0){
  // Usually merge runs of comparable size. Once a single extent has
  // accumulated a long tail, prefer one extra merge to repeated probes.
  constexpr uint32_t tail_runs=8;
  uint32_t same_extent=2;
  while(same_extent<tail_runs&&same_extent<n){
   const auto& older=runs[n-same_extent-1];
   if(older.segment!=a.segment||older.bank!=a.bank)break;
   ++same_extent;
  }
  if(same_extent<tail_runs)return;
 }
 uint64_t total=a.length+b.length;
 if(total>max_entries){*result=2;return;}
 *control={a,b,n-2,uint32_t(total),1};
}
// A series reader resolves its own row references. The merge comparator can
// therefore compare states from distinct extents without copying state columns.
struct MergeSeries {
 MgbfsBatchHashSegment segment;
 uint32_t bank;
 uint64_t begin;

 __device__ uint64_t bank_base() const { return uint64_t(bank)*segment.capacity; }
 __device__ uint64_t key(uint64_t index) const {
  return segment.keys[bank_base()+begin+index];
 }
 __device__ uint32_t row(uint64_t index) const {
  return segment.rows[bank_base()+begin+index];
 }
 template<class T>
 __device__ int64_t value(uint32_t element,uint32_t row_index) const {
  const auto* states=static_cast<const T*>(segment.states);
  const uint64_t stride=segment.capacity*segment.banks;
  return int64_t(states[uint64_t(element)*stride+bank_base()+row_index]);
 }
};

template<class T>
__device__ int compare_rows(const MergeSeries& left,uint64_t a,const MergeSeries& right,uint64_t b,
 uint32_t elements,uint32_t* error){
 const uint64_t ka=left.key(a),kb=right.key(b);
 if(ka<kb)return -1;if(ka>kb)return 1;
 const uint32_t ra=left.row(a),rb=right.row(b);
 if(uint64_t(ra)>=left.segment.capacity||uint64_t(rb)>=right.segment.capacity){
  atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));
  return 0;
 }
 for(uint32_t e=0;e<elements;e++){
  const int64_t x=left.value<T>(e,ra),y=right.value<T>(e,rb);
  if(x<y)return -1;if(x>y)return 1;
 }
 return 0;
}
template<class T,bool Logical=false,class Ref=uint32_t>
__global__ void merge_path(const MgbfsBatchHashSegment* segments,uint32_t elements,
 const MergeControl* control,uint64_t* out_keys,Ref* out_rows,uint32_t* error,MgbfsHashCompactRefLayout layout){
 if(fatal(*error)||control->status!=1)return;
 const auto a=control->left,b=control->right;
 const MergeSeries left{segments[a.segment],a.bank,a.begin};
 const MergeSeries right{segments[b.segment],b.bank,b.begin};
 for(uint64_t k=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;k<control->count;k+=blockDim.x*gridDim.x){
  uint64_t low=k>b.length?uint64_t(k)-b.length:0,high=uint64_t(k)<a.length?k:a.length;
  while(low<high){
   const uint64_t amid=low+(high-low)/2,bmid=uint64_t(k)-amid;
   if(amid<a.length&&bmid>0&&compare_rows<T>(right,bmid-1,left,amid,elements,error)>=0)low=amid+1;
   else high=amid;
  }
  const uint64_t ai=low,bi=uint64_t(k)-ai;
  const bool from_a=ai<a.length&&(bi>=b.length||compare_rows<T>(left,ai,right,bi,elements,error)<=0);
  const auto& source=from_a?left:right;
  const uint64_t index=from_a?ai:bi;
  out_keys[k]=source.key(index);
  if constexpr (!Logical){
   out_rows[k]=source.row(index);
  }else{
   const auto& run=from_a?a:b;
   const uint32_t descriptor=run.segment*3+run.bank;
   const uint32_t row=source.row(index);
   if(uint64_t(row)>=source.segment.capacity){
    atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;
   }
   if constexpr (sizeof(Ref)==sizeof(uint32_t)){
    uint32_t reference=0;
    if(!layout.try_encode(descriptor,row,reference)){
     atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;
    }
    out_rows[k]=reference;
   }else{
    out_rows[k]=mgbfs_hash_wide_ref(descriptor,row);
   }
  }
 }
}
// Publish preparation only after both source descriptors and the destination
// capacity have been validated. Source columns remain unchanged.
__global__ void prepare_extent_merge(const MgbfsBatchHashSegment* segments,uint32_t segment_count,
 const MgbfsBatchHashRun* inputs,uint32_t output_capacity,MergeControl* control,uint32_t* error){
 control->status=0;
 if(fatal(*error))return;
 const auto a=inputs[0],b=inputs[1];
 const MgbfsBatchHashRun sources[]{a,b};
 for(const auto& run:sources){
  if(run.segment>=segment_count||uint64_t(run.segment)*3+run.bank>UINT32_MAX){
   atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;
  }
  const auto seg=segments[run.segment];
  if(!seg.states||!seg.keys||!seg.rows||!seg.capacity||seg.capacity>(uint64_t(1)<<32)||
   !(seg.banks==1||seg.banks==3)||run.bank>=seg.banks||
   !run.length||run.begin>seg.capacity||run.length>seg.capacity-run.begin||
   run.min_hash>run.max_hash){
   atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;
  }
 }
 const uint64_t total=a.length+b.length;
 if(total>output_capacity){
  atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;
 }
 *control={a,b,0,uint32_t(total),1};
}
__global__ void merge_copy(const MgbfsBatchHashSegment* segments,const MergeControl* control,
 const uint64_t* keys,const uint32_t* rows,const uint32_t* error){
 if(fatal(*error)||control->status!=1)return;
 MgbfsBatchHashRun a=control->left;MgbfsBatchHashSegment seg=segments[a.segment];
 for(uint32_t k=blockIdx.x*blockDim.x+threadIdx.x;k<control->count;k+=blockDim.x*gridDim.x){
  uint64_t at=uint64_t(a.bank)*seg.capacity+a.begin+k;seg.keys[at]=keys[k];seg.rows[at]=rows[k];
 }
}
__global__ void merge_publish(const MgbfsBatchHashSegment* segments,MgbfsBatchHashRun* runs,uint32_t* count,
 const MergeControl* control,uint32_t* result,const uint32_t* error){
 if(blockIdx.x||threadIdx.x||fatal(*error)||control->status!=1)return;
 MgbfsBatchHashRun a=control->left;MgbfsBatchHashSegment seg=segments[a.segment];
 uint64_t at=uint64_t(a.bank)*seg.capacity+a.begin;
 a.length=control->count;a.min_hash=seg.keys[at];a.max_hash=seg.keys[at+a.length-1];
 if(a.length<kMinFenceRows&&a.hash_fences[4]&&control->right.hash_fences[4]){
  for(uint32_t word=0;word<4;word++)a.hash_fences[word]|=control->right.hash_fences[word];
 }else{
  fill_fences(a,seg.keys+at);
 }
 runs[control->run_index]=a;*count=control->run_index+1;*result=1;
}
}
namespace {
__global__ void build_extent_series_prefixes(const uint64_t* keys,uint32_t count,uint32_t* offsets){
 __shared__ uint32_t shift;
 if(threadIdx.x==0){
  shift=count?mgbfs_hash_series_shift(keys[0],keys[count-1]):0;
  offsets[MGBFS_HASH_SERIES_PREFIXES]=count;
  offsets[MGBFS_HASH_SERIES_SHIFT_INDEX]=shift;
 }
 __syncthreads();
 const uint32_t prefix=threadIdx.x;
 offsets[prefix]=mgbfs_hash_series_prefix_begin(keys,count,prefix,shift);
}
}

extern "C" int mgbfs_hash_extent_prefixes_v1(const uint64_t* keys,uint32_t count,uint32_t* offsets,void* stream){
 if((count&&!keys)||!offsets)return int(cudaErrorInvalidValue);
 build_extent_series_prefixes<<<1,MGBFS_HASH_SERIES_PREFIXES,0,
  static_cast<cudaStream_t>(stream)>>>(keys,count,offsets);
 return int(cudaGetLastError());
}

namespace {
// Build one offset per input run before gathering. This validates coverage
// against the committed row count and makes the gather independent of tile size.
__global__ void plan_series_gather(const MgbfsBatchHashSegment* segments,uint32_t segment_count,
 const MgbfsBatchHashRun* runs,uint32_t run_count,uint32_t rows,uint32_t* offsets,uint32_t* error){
 if(fatal(*error))return;
 uint64_t total=0;
 for(uint32_t i=0;i<run_count;i++){
  const auto& run=runs[i];
  if(run.segment>=segment_count||uint64_t(run.segment)*3+run.bank>UINT32_MAX){
   atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;
  }
  const auto& segment=segments[run.segment];
  if(!segment.states||!segment.keys||!segment.rows||!segment.capacity||
     segment.capacity>(uint64_t{1}<<32)||!(segment.banks==1||segment.banks==3)||
     run.bank>=segment.banks||!run.length||run.begin>segment.capacity||
     run.length>segment.capacity-run.begin||run.min_hash>run.max_hash||run.length>rows-total){
   atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;
  }
  offsets[i]=uint32_t(total);
  total+=run.length;
 }
 if(total!=rows)atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));
}

template<class Ref>
__global__ void gather_series(const MgbfsBatchHashSegment* segments,const MgbfsBatchHashRun* runs,
 uint32_t run_count,const uint32_t* offsets,uint64_t* keys,Ref* refs,
 MgbfsHashCompactRefLayout layout,uint32_t* error){
 if(fatal(*error))return;
 for(uint64_t i=blockIdx.x;i<run_count;i+=gridDim.x){
  const auto& run=runs[i];
  const auto& segment=segments[run.segment];
  const uint64_t begin=segment.capacity*run.bank+run.begin;
  const uint32_t descriptor=run.segment*3+run.bank;
  // Distribute long extents within a run across independent CTAs.
  const uint64_t first=uint64_t(blockIdx.y)*blockDim.x+threadIdx.x;
  const uint64_t stride=uint64_t(gridDim.y)*blockDim.x;
  for(uint64_t j=first;j<run.length;j+=stride){
   const uint32_t row=segment.rows[begin+j];
   if(row>=segment.capacity){
    atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;
   }
   const uint64_t out=uint64_t(offsets[i])+j;
   keys[out]=segment.keys[begin+j];
   if constexpr(sizeof(Ref)==sizeof(uint32_t)){
    uint32_t ref=0;
    if(!layout.try_encode(descriptor,row,ref)){
     atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;
    }
    refs[out]=ref;
   }else{
    refs[out]=mgbfs_hash_wide_ref(descriptor,row);
   }
  }
 }
}
}

namespace {
template<class Ref,bool Write>
__global__ void remap_series_segments(Ref* refs,uint32_t rows,MgbfsHashCompactRefLayout layout,
 const uint32_t* remap,uint32_t segments,uint32_t* error){
 if(fatal(*error))return;
 for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows;i+=uint64_t(gridDim.x)*blockDim.x){
  uint32_t descriptor;
  if constexpr(sizeof(Ref)==4)descriptor=layout.descriptor(refs[i]);
  else descriptor=mgbfs_hash_ref_descriptor(refs[i]);
  const uint32_t segment=descriptor/3,bank=descriptor%3;
  if(segment>=segments||remap[segment]==UINT32_MAX||uint64_t(remap[segment])*3+bank>UINT32_MAX){
   atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;
  }
  if constexpr(sizeof(Ref)==4){
   uint32_t packed=0;
   if(!layout.try_encode(remap[segment]*3+bank,layout.row(refs[i]),packed)){
    atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;
   }
   if constexpr(Write)refs[i]=packed;
  }else if constexpr(Write){
   refs[i]=mgbfs_hash_wide_ref(remap[segment]*3+bank,mgbfs_hash_ref_row(refs[i]));
  }
 }
}

template<class Ref>
int remap_series(Ref* refs,uint32_t rows,MgbfsHashCompactRefLayout layout,
 const uint32_t* remap,uint32_t segments,uint32_t* error,cudaStream_t stream){
 constexpr uint32_t threads=256,max_blocks=65535;
 const uint32_t needed=(rows-1)/threads+1,blocks=needed<max_blocks?needed:max_blocks;
 remap_series_segments<Ref,false><<<blocks,threads,0,stream>>>(refs,rows,layout,remap,segments,error);
 const auto status=cudaGetLastError();
 if(status!=cudaSuccess)return int(status);
 remap_series_segments<Ref,true><<<blocks,threads,0,stream>>>(refs,rows,layout,remap,segments,error);
 return int(cudaGetLastError());
}
}

extern "C" int mgbfs_hash_extent_remap_v1(void* refs,uint32_t rows,uint32_t ref_bytes,uint32_t row_bits,
 const uint32_t* remap,uint32_t segments,uint32_t* error,void* stream){
 if(!refs||!rows||!remap||!segments||!error||row_bits>32)return int(cudaErrorInvalidValue);
 const auto st=static_cast<cudaStream_t>(stream);
 if(ref_bytes==4)return remap_series(static_cast<uint32_t*>(refs),rows,{row_bits},remap,segments,error,st);
 if(ref_bytes==8)return remap_series(static_cast<MgbfsHashRowRef*>(refs),rows,{row_bits},remap,segments,error,st);
 return int(cudaErrorInvalidValue);
}

namespace {
__global__ void summarize_series_bank(const MgbfsBatchHashRun* runs,uint32_t count,uint64_t* result){
 const uint32_t bank=runs[0].bank;
 uint32_t covered=0;
 uint64_t rows=0;
 while(covered<count&&runs[covered].bank==bank){
  if(runs[covered].length>UINT64_MAX-rows){rows=UINT64_MAX;break;}
  rows+=runs[covered++].length;
 }
 result[0]=bank;result[1]=covered;result[2]=rows;
}
}

extern "C" int mgbfs_hash_extent_bank_summary_v1(const MgbfsBatchHashRun* runs,uint32_t count,
 uint64_t* result,void* stream){
 if(!runs||!count||!result)return int(cudaErrorInvalidValue);
 summarize_series_bank<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(runs,count,result);
 return int(cudaGetLastError());
}

extern "C" int mgbfs_hash_extent_gather_v1(const MgbfsBatchHashSegment* segments,uint32_t segment_count,
 const MgbfsBatchHashRun* runs,uint32_t run_count,uint32_t rows,uint32_t* run_offsets,
 uint64_t* keys,void* refs,uint32_t ref_bytes,uint32_t row_bits,uint32_t* error,void* stream){
 if(!segments||!segment_count||!runs||!run_count||!rows||!run_offsets||!keys||!refs||!error||
    (ref_bytes!=4&&ref_bytes!=8)||row_bits>32)return int(cudaErrorInvalidValue);
 auto st=static_cast<cudaStream_t>(stream);
 plan_series_gather<<<1,1,0,st>>>(segments,segment_count,runs,run_count,rows,run_offsets,error);
 auto status=cudaGetLastError();
 if(status!=cudaSuccess)return int(status);
 const uint32_t blocks=run_count<65535?run_count:65535;
 // Aim for one CTA per 4K rows of the mean run. The bounded y
 // dimension keeps small and irregular run directories inexpensive.
 constexpr uint64_t rows_per_cta=4096,max_ctas_per_run=32;
 const uint64_t target=uint64_t(run_count)*rows_per_cta;
 const uint64_t tiles=(uint64_t(rows)+target-1)/target;
 const dim3 grid(blocks,uint32_t(tiles<max_ctas_per_run?tiles:max_ctas_per_run));
 if(ref_bytes==4)gather_series<<<grid,256,0,st>>>(segments,runs,run_count,run_offsets,keys,
  static_cast<uint32_t*>(refs),{row_bits},error);
 else gather_series<<<grid,256,0,st>>>(segments,runs,run_count,run_offsets,keys,
  static_cast<MgbfsHashRowRef*>(refs),{row_bits},error);
 return int(cudaGetLastError());
}

namespace {
template<class Ref>
__device__ MgbfsHashRowRef decode_series_ref(Ref ref,MgbfsHashCompactRefLayout layout){
 if constexpr(sizeof(Ref)==sizeof(uint32_t))
  return mgbfs_hash_wide_ref(layout.descriptor(ref),layout.row(ref));
 else return ref;
}

template<class State,class Ref>
struct SeriesOriginLess {
 const uint64_t* keys;
 const Ref* refs;
 const MgbfsBatchHashSegment* segments;
 MgbfsHashCompactRefLayout layout;
 uint32_t elements;

 __device__ bool operator()(uint32_t a,uint32_t b)const{
  if(keys[a]!=keys[b])return keys[a]<keys[b];
  const auto ra=decode_series_ref(refs[a],layout),rb=decode_series_ref(refs[b],layout);
  const uint32_t da=mgbfs_hash_ref_descriptor(ra),db=mgbfs_hash_ref_descriptor(rb);
  const MergeSeries left{segments[da/3],da%3,0},right{segments[db/3],db%3,0};
  for(uint32_t e=0;e<elements;e++){
   const auto x=left.value<State>(e,mgbfs_hash_ref_row(ra));
   const auto y=right.value<State>(e,mgbfs_hash_ref_row(rb));
   if(x!=y)return x<y;
  }
  return a<b;
 }
};

__global__ void series_sort_origins(uint32_t* origins,uint32_t rows){
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<rows;i+=gridDim.x*blockDim.x)origins[i]=i;
}

template<class Ref>
__global__ void copy_sorted_series(const uint64_t* keys,const Ref* refs,const uint32_t* origins,
 uint32_t rows,uint64_t* out_keys,Ref* out_refs){
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<rows;i+=gridDim.x*blockDim.x){
  const uint32_t origin=origins[i];
  out_keys[i]=keys[origin];out_refs[i]=refs[origin];
 }
}

struct SeriesSortSpace {size_t indices,sorted,temp,temp_bytes,total;};
template<class State,class Ref>
cudaError_t series_sort_space(uint32_t rows,SeriesSortSpace& space){
 size_t temp=0;
 const auto status=cub::DeviceMergeSort::SortKeysCopy(nullptr,temp,static_cast<const uint32_t*>(nullptr),
  static_cast<uint32_t*>(nullptr),int(rows),SeriesOriginLess<State,Ref>{});
 if(status!=cudaSuccess)return status;
 size_t total=0;
 if(!region(total,size_t(rows)*4,&space.indices)||!region(total,size_t(rows)*4,&space.sorted)||
    !region(total,temp,&space.temp))return cudaErrorInvalidValue;
 space.temp_bytes=temp;space.total=total;
 return cudaSuccess;
}

template<class State,class Ref>
int sort_gathered_series(uint32_t elements,const MgbfsBatchHashSegment* segments,
 MgbfsHashCompactRefLayout layout,uint32_t rows,const uint64_t* keys,const void* refs,
 uint64_t* out_keys,void* out_refs,void* scratch,size_t scratch_bytes,cudaStream_t stream){
 SeriesSortSpace space{};
 auto status=series_sort_space<State,Ref>(rows,space);
 if(status!=cudaSuccess)return int(status);
 if(scratch_bytes<space.total)return int(cudaErrorInvalidValue);
 auto* base=static_cast<char*>(scratch);
 auto* indices=reinterpret_cast<uint32_t*>(base+space.indices);
 auto* sorted=reinterpret_cast<uint32_t*>(base+space.sorted);
 constexpr uint32_t threads=256,max_blocks=65535;
 const uint32_t needed=(rows-1)/threads+1,blocks=needed<max_blocks?needed:max_blocks;
 series_sort_origins<<<blocks,threads,0,stream>>>(indices,rows);
 if((status=cudaGetLastError())!=cudaSuccess)return int(status);
 const auto* input_refs=static_cast<const Ref*>(refs);
 const SeriesOriginLess<State,Ref> less{keys,input_refs,segments,layout,elements};
 status=cub::DeviceMergeSort::SortKeysCopy(base+space.temp,space.temp_bytes,indices,sorted,int(rows),less,stream);
 if(status!=cudaSuccess)return int(status);
 copy_sorted_series<<<blocks,threads,0,stream>>>(keys,input_refs,sorted,rows,out_keys,static_cast<Ref*>(out_refs));
 return int(cudaGetLastError());
}
}

extern "C" int mgbfs_hash_extent_sort_scratch_v1(uint32_t state_bytes,uint32_t ref_bytes,uint32_t rows,size_t* bytes){
 if(!bytes||!rows||rows>=0x7fffffffU||(state_bytes!=1&&state_bytes!=8)||
    (ref_bytes!=4&&ref_bytes!=8))return int(cudaErrorInvalidValue);
 SeriesSortSpace s{};
 const auto rc=state_bytes==1
  ?(ref_bytes==4?series_sort_space<uint8_t,uint32_t>(rows,s):series_sort_space<uint8_t,MgbfsHashRowRef>(rows,s))
  :(ref_bytes==4?series_sort_space<int64_t,uint32_t>(rows,s):series_sort_space<int64_t,MgbfsHashRowRef>(rows,s));
 if(rc==cudaSuccess)*bytes=s.total;
 return int(rc);
}

// Inputs must come from a successful gather. Check its error word before
// starting the sort, since the comparator assumes that all row IDs are valid.
extern "C" int mgbfs_hash_extent_sort_v1(uint32_t state_bytes,uint32_t elements,
 const MgbfsBatchHashSegment* segments,uint32_t rows,const uint64_t* keys,const void* refs,
 uint32_t ref_bytes,uint32_t row_bits,uint64_t* out_keys,void* out_refs,void* scratch,
 size_t scratch_bytes,void* stream){
 if(!elements||!segments||!rows||rows>=0x7fffffffU||!keys||!refs||!out_keys||!out_refs||
    keys==out_keys||refs==out_refs||!scratch||uintptr_t(scratch)%kAlign||
    (ref_bytes!=4&&ref_bytes!=8)||row_bits>32)return int(cudaErrorInvalidValue);
 const auto st=static_cast<cudaStream_t>(stream);
 const MgbfsHashCompactRefLayout layout{row_bits};
 if(state_bytes==1){
  if(ref_bytes==4)return sort_gathered_series<uint8_t,uint32_t>(elements,segments,layout,rows,keys,refs,out_keys,out_refs,scratch,scratch_bytes,st);
  return sort_gathered_series<uint8_t,MgbfsHashRowRef>(elements,segments,layout,rows,keys,refs,out_keys,out_refs,scratch,scratch_bytes,st);
 }
 if(state_bytes!=8)return int(cudaErrorInvalidValue);
 if(ref_bytes==4)return sort_gathered_series<int64_t,uint32_t>(elements,segments,layout,rows,keys,refs,out_keys,out_refs,scratch,scratch_bytes,st);
 return sort_gathered_series<int64_t,MgbfsHashRowRef>(elements,segments,layout,rows,keys,refs,out_keys,out_refs,scratch,scratch_bytes,st);
}

extern "C" size_t mgbfs_hash_extent_merge_control_bytes(){
 return sizeof(MergeControl);
}

// Inputs describe two local-row series. Output is a pair of separate
// columns and must not alias either input. The caller synchronizes the
// stream and checks error before making the merged column searchable.
template<class Ref>
static int merge_extents(uint32_t state_bytes,uint32_t elements,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,const MgbfsBatchHashRun* inputs,
 uint32_t output_capacity,uint64_t* keys,Ref* references,MgbfsHashCompactRefLayout layout,
 void* scratch,size_t scratch_bytes,uint32_t* error,void* stream){
 if((state_bytes!=1&&state_bytes!=8)||!elements||!segments||!segment_count||!inputs||
    !output_capacity||!keys||!references||!scratch||uintptr_t(scratch)%alignof(MergeControl)||
    scratch_bytes<sizeof(MergeControl)||!error)return int(cudaErrorInvalidValue);
 auto* control=static_cast<MergeControl*>(scratch);
 auto st=static_cast<cudaStream_t>(stream);
 prepare_extent_merge<<<1,1,0,st>>>(segments,segment_count,inputs,output_capacity,control,error);
 auto status=cudaGetLastError();
 if(status!=cudaSuccess)return int(status);
 constexpr uint32_t threads=256,max_blocks=1024;
 const uint32_t needed=(output_capacity-1)/threads+1;
 const uint32_t blocks=needed<max_blocks?needed:max_blocks;
 if(state_bytes==1)merge_path<uint8_t,true><<<blocks,threads,0,st>>>(segments,elements,control,keys,references,error,layout);
 else merge_path<int64_t,true><<<blocks,threads,0,st>>>(segments,elements,control,keys,references,error,layout);
 return int(cudaGetLastError());
}

extern "C" int mgbfs_hash_extent_merge_v1(uint32_t state_bytes,uint32_t elements,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,const MgbfsBatchHashRun* inputs,
 uint32_t output_capacity,uint64_t* keys,MgbfsHashRowRef* references,
 void* scratch,size_t scratch_bytes,uint32_t* error,void* stream){
 return merge_extents(state_bytes,elements,segments,segment_count,inputs,output_capacity,keys,
  references,{0},scratch,scratch_bytes,error,stream);
}

extern "C" int mgbfs_hash_extent_merge_compact_v1(uint32_t state_bytes,uint32_t elements,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,const MgbfsBatchHashRun* inputs,
 uint32_t output_capacity,uint64_t* keys,uint32_t* references,uint32_t row_bits,
 void* scratch,size_t scratch_bytes,uint32_t* error,void* stream){
 if(row_bits>32)return int(cudaErrorInvalidValue);
 return merge_extents(state_bytes,elements,segments,segment_count,inputs,output_capacity,keys,
  references,{row_bits},scratch,scratch_bytes,error,stream);
}

extern "C" int mgbfs_hash_batch_compact_scratch_bytes(uint32_t entries,size_t* bytes){
 MergeLayout w{};if(!bytes||!merge_layout(entries,&w))return int(cudaErrorInvalidValue);
 *bytes=w.total;return int(cudaSuccess);
}
extern "C" int mgbfs_hash_batch_compact_tail_v2(uint32_t state_bytes,uint32_t elements,const MgbfsBatchHashSegment* segments,
 uint32_t segment_count,MgbfsBatchHashRun* runs,uint32_t* count,uint32_t max_entries,void* scratch,
 size_t scratch_bytes,uint32_t* result,uint32_t* error,void* stream){
 MergeLayout w{};
 if((state_bytes!=1&&state_bytes!=8)||!elements||!segments||!segment_count||!runs||!count||!scratch||
  uintptr_t(scratch)%kAlign||!result||!error||!merge_layout(max_entries,&w)||scratch_bytes<w.total)
  return int(cudaErrorInvalidValue);
 char* base=static_cast<char*>(scratch);auto* keys=reinterpret_cast<uint64_t*>(base+w.keys);
 auto* rows=reinterpret_cast<uint32_t*>(base+w.rows);auto* control=reinterpret_cast<MergeControl*>(base+w.control);
 cudaStream_t st=static_cast<cudaStream_t>(stream);
 merge_prepare<<<1,1,0,st>>>(segments,segment_count,runs,count,max_entries,control,result,error);
 cudaError_t rc=cudaGetLastError();if(rc!=cudaSuccess)return int(rc);
 // The workspace is a capacity bound, while most tail merges are smaller.
 // Both kernels use grid-stride loops over the actual merge count.
 constexpr uint32_t threads=256,max_merge_blocks=1024;
 const uint32_t needed=max_entries?(max_entries+threads-1)/threads:1;
 const uint32_t blocks=needed<max_merge_blocks?needed:max_merge_blocks;
 if(state_bytes==1)merge_path<uint8_t><<<blocks,threads,0,st>>>(segments,elements,control,keys,rows,error,{0});
 else merge_path<int64_t><<<blocks,threads,0,st>>>(segments,elements,control,keys,rows,error,{0});
 if((rc=cudaGetLastError())!=cudaSuccess)return int(rc);
 merge_copy<<<blocks,threads,0,st>>>(segments,control,keys,rows,error);
 if((rc=cudaGetLastError())!=cudaSuccess)return int(rc);
 merge_publish<<<1,1,0,st>>>(segments,runs,count,control,result,error);
 return int(cudaGetLastError());
}

// Marker for the complete batch route, bitmap and exact sorted-hash ABI.
extern "C" unsigned int mgbfs_segmented_batch_strategy_abi_version(){return 1;}
