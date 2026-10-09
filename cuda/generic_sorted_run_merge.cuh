#pragma once
#include "generic_sorted_history.cuh"
#include <cub/device/device_scan.cuh>

struct GenericSortedRunMergeShape {
 uint64_t merged_hash_bytes,merged_row_bytes,flag_bytes,prefix_bytes;
 size_t scan_temporary_bytes;
 uint64_t control_bytes,aligned_workspace_bytes;
};
inline uint64_t generic_sorted_merge_align256(uint64_t bytes){return (bytes+255ull)&~255ull;}
// Query before allocation. Destination run credits and canonical states belong
// to separate admitted arenas, and are NOT included in this scratch query.
inline cudaError_t generic_sorted_run_merge_shape(uint32_t capacity,
 GenericSortedRunMergeShape* shape,cudaStream_t stream=0){
 *shape={};shape->control_bytes=2*sizeof(GenericSortedHistoryRun)+3*sizeof(uint32_t);
 shape->merged_hash_bytes=uint64_t(capacity)*sizeof(uint64_t);
 shape->merged_row_bytes=uint64_t(capacity)*sizeof(uint32_t);
 shape->flag_bytes=shape->prefix_bytes=uint64_t(capacity)*sizeof(uint32_t);
 if(capacity){
  cudaError_t status=cub::DeviceScan::ExclusiveSum(nullptr,shape->scan_temporary_bytes,
    static_cast<const uint32_t*>(nullptr),static_cast<uint32_t*>(nullptr),capacity,stream);
  if(status!=cudaSuccess)return status;
 }
 shape->aligned_workspace_bytes=generic_sorted_merge_align256(shape->merged_hash_bytes)+
  generic_sorted_merge_align256(shape->merged_row_bytes)+generic_sorted_merge_align256(shape->flag_bytes)+
  generic_sorted_merge_align256(shape->prefix_bytes)+generic_sorted_merge_align256(shape->scan_temporary_bytes)+
  generic_sorted_merge_align256(shape->control_bytes);
 return cudaSuccess;
}
// Hash orders buckets; full canonical coordinates resolve every hash tie.
// Equal states compare equal, independently of physical row and input run.
template<class State>
__device__ int generic_sorted_run_compare(uint64_t ah,uint32_t ar,uint64_t bh,uint32_t br,
 const State* arena,uint32_t stride,uint32_t elements,uint32_t* error) {
 if(ar>=stride||br>=stride){atomicOr(error,8u);return 0;}
 if(ah<bh)return -1;if(ah>bh)return 1;
 for(uint32_t c=0;c<elements;++c){
  State av=arena[uint64_t(c)*stride+ar],bv=arena[uint64_t(c)*stride+br];
  if(av<bv)return -1;if(av>bv)return 1;
 }
 return 0;
}
// Stable left-before-right merge-path partition at an output diagonal.
template<class State>
__device__ uint32_t generic_sorted_run_partition(uint32_t diagonal,
 GenericSortedHistoryRun a,GenericSortedHistoryRun b,const State* arena,
 uint32_t stride,uint32_t elements,uint32_t* error) {
 uint32_t lo=diagonal>b.count?diagonal-b.count:0;
 uint32_t hi=diagonal<a.count?diagonal:a.count;
 while(lo<=hi){
  uint32_t ai=lo+(hi-lo)/2,bi=diagonal-ai;
  if(ai&&bi<b.count&&generic_sorted_run_compare(a.hashes[ai-1],a.rows[ai-1],
    b.hashes[bi],b.rows[bi],arena,stride,elements,error)>0){hi=ai-1;continue;}
  if(bi&&ai<a.count&&generic_sorted_run_compare(b.hashes[bi-1],b.rows[bi-1],
    a.hashes[ai],a.rows[ai],arena,stride,elements,error)>=0){lo=ai+1;continue;}
  return ai;
 }
 atomicOr(error,256u);return lo;
}
template<class State>
__global__ void generic_sorted_run_validate_exact(const GenericSortedHistoryRun* view,
 uint32_t admitted_capacity,const State* arena,uint32_t stride,uint32_t elements,uint32_t* error){
 auto run=*view;
 if(run.count>admitted_capacity){if(!blockIdx.x&&!threadIdx.x)atomicOr(error,32u);return;}
 for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<run.count;i+=uint64_t(blockDim.x)*gridDim.x){
  if(run.rows[i]>=stride){atomicOr(error,8u);continue;}
  if(i&&generic_sorted_run_compare(run.hashes[i-1],run.rows[i-1],run.hashes[i],run.rows[i],
    arena,stride,elements,error)>0)atomicOr(error,256u);
 }
}
// Device views/counts: launch geometry depends only on admitted destination capacity.
// Input and destination planes must have distinct, held pool leases.
template<class State>
__global__ void generic_sorted_run_merge(const GenericSortedHistoryRun* av,
 const GenericSortedHistoryRun* bv,const State* arena,uint32_t stride,uint32_t elements,
 uint64_t* hashes,uint32_t* rows,uint32_t a_capacity,uint32_t b_capacity,
 uint32_t capacity,uint32_t* count,uint32_t* error){
 auto a=*av,b=*bv;uint64_t total64=uint64_t(a.count)+b.count;
 if(a.count>a_capacity||b.count>b_capacity||total64>capacity){if(!blockIdx.x&&!threadIdx.x){*count=0;atomicOr(error,32u);}return;}
 uint32_t total=uint32_t(total64);if(!blockIdx.x&&!threadIdx.x)*count=total;
 uint64_t start64=uint64_t(blockIdx.x)*blockDim.x;
 if(start64>=total)return;
 uint32_t start=uint32_t(start64),end=total-start<blockDim.x?total:start+blockDim.x;
 __shared__ uint32_t a0,b0,a1,b1;
 if(!threadIdx.x){
  a0=generic_sorted_run_partition(start,a,b,arena,stride,elements,error);b0=start-a0;
  a1=generic_sorted_run_partition(end,a,b,arena,stride,elements,error);b1=end-a1;
 }
 __syncthreads();uint32_t lane=threadIdx.x;
 if(lane>=end-start)return;
 GenericSortedHistoryRun at{a.hashes?a.hashes+a0:nullptr,a.rows?a.rows+a0:nullptr,a1-a0};
 GenericSortedHistoryRun bt{b.hashes?b.hashes+b0:nullptr,b.rows?b.rows+b0:nullptr,b1-b0};
 uint32_t ai=generic_sorted_run_partition(lane,at,bt,arena,stride,elements,error),bi=lane-ai;
 bool left=ai<at.count&&(bi>=bt.count||generic_sorted_run_compare(at.hashes[ai],at.rows[ai],
   bt.hashes[bi],bt.rows[bi],arena,stride,elements,error)<=0);
 hashes[start+lane]=left?at.hashes[ai]:bt.hashes[bi];
 rows[start+lane]=left?at.rows[ai]:bt.rows[bi];
}
template<class State>
__global__ void generic_sorted_run_unique_flags(const uint64_t* hashes,const uint32_t* rows,
 const uint32_t* count,uint32_t capacity,const State* arena,uint32_t stride,uint32_t elements,
 uint32_t* flags,uint32_t* error){
 uint32_t n=*count;
 if(n>capacity){if(!blockIdx.x&&!threadIdx.x)atomicOr(error,32u);n=0;}
 for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<capacity;i+=uint64_t(blockDim.x)*gridDim.x){
  if(i>=n){flags[i]=0;continue;}
  if(rows[i]>=stride){atomicOr(error,8u);flags[i]=0;continue;}
  flags[i]=!i||generic_sorted_run_compare(hashes[i-1],rows[i-1],hashes[i],rows[i],
    arena,stride,elements,error)!=0;
 }
}
// Prefix is a caller-owned CUB exclusive scan over the fixed admitted capacity.
__global__ void generic_sorted_run_unique_scatter(const uint64_t* hashes,const uint32_t* rows,
 const uint32_t* flags,const uint32_t* prefix,uint32_t capacity,uint64_t* out_hashes,
 uint32_t* out_rows,uint32_t* out_count){
 if(!blockIdx.x&&!threadIdx.x)*out_count=capacity?prefix[capacity-1]+flags[capacity-1]:0;
 for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<capacity;i+=uint64_t(blockDim.x)*gridDim.x)
  if(flags[i]){out_hashes[prefix[i]]=hashes[i];out_rows[prefix[i]]=rows[i];}
}
