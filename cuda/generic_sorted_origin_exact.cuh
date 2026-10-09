#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_scan.cuh>

// An origin names one immutable leased incoming action. Full states are never
// copied into the sorting workspace; only hashes and child indices move.
template<class Action> struct GenericSortedOriginLess {
 Action action;const uint64_t* hashes;
 __device__ int exact_compare(uint32_t a,uint32_t b)const{
  if(a==0xffffffffu)return b==0xffffffffu?0:1;
  if(b==0xffffffffu)return -1;
  uint64_t ah=hashes[a],bh=hashes[b];if(ah<bh)return -1;if(ah>bh)return 1;
  for(uint32_t c=0;c<action.elements;++c){
   int64_t av=action.value(action.child(a),c),bv=action.value(action.child(b),c);
   if(av<bv)return -1;if(av>bv)return 1;
  }
  return 0;
 }
 __device__ bool operator()(uint32_t a,uint32_t b)const{
  int order=exact_compare(a,b);
  // Deterministic representative; the origin tie-break is NOT state identity.
  return order<0||(!order&&a<b);
 }
};
template<class Action>
__global__ void generic_sorted_origin_prepare(Action action,uint32_t capacity,uint64_t seed,
 uint32_t bits,uint64_t* hashes,uint32_t* origins,uint32_t* error){
 for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<capacity;i+=uint64_t(blockDim.x)*gridDim.x){
  uint32_t child=action.child(uint32_t(i));
  bool valid=action.valid(child,error);
  origins[i]=valid?uint32_t(i):0xffffffffu;
  // Padding is invalid origin, not a reserved hash. Every hash remains legal.
  hashes[i]=valid?action.hash(child,seed,bits):0;
 }
}
template<class Action>
__global__ void generic_sorted_origin_unique_flags(GenericSortedOriginLess<Action> order,
 const uint32_t* sorted,uint32_t capacity,uint32_t* flags){
 for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<capacity;i+=uint64_t(blockDim.x)*gridDim.x){
  uint32_t origin=sorted[i];
  flags[i]=origin!=0xffffffffu&&(!i||order.exact_compare(sorted[i-1],origin)!=0);
 }
}
template<class Action>
__global__ void generic_sorted_origin_scatter(Action action,const uint64_t* hashes,const uint32_t* sorted,
 const uint32_t* flags,const uint32_t* prefix,uint32_t capacity,uint64_t* unique_hashes,
 uint32_t* unique_origins,uint32_t* unique_count){
 if(!blockIdx.x&&!threadIdx.x)*unique_count=capacity?prefix[capacity-1]+flags[capacity-1]:0;
 for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<capacity;i+=uint64_t(blockDim.x)*gridDim.x)
  if(flags[i]){uint32_t origin=sorted[i];unique_hashes[prefix[i]]=hashes[origin];unique_origins[prefix[i]]=action.child(origin);}
}
struct GenericSortedOriginShape {
 size_t sort_temporary_bytes,scan_temporary_bytes,shared_temporary_bytes;
 uint64_t hash_bytes,origin_bytes,flag_bytes,aligned_workspace_bytes;
};
inline uint64_t generic_sorted_origin_align(uint64_t value){return (value+255ull)&~255ull;}
template<class Action>
inline cudaError_t generic_sorted_origin_shape(Action action,uint32_t capacity,
 GenericSortedOriginShape* shape,cudaStream_t stream=0){
 *shape={};if(capacity>=0x7fffffffu)return cudaErrorInvalidValue;
 shape->hash_bytes=uint64_t(capacity)*8;shape->origin_bytes=uint64_t(capacity)*4;
 shape->flag_bytes=uint64_t(capacity)*4;
 if(capacity){
  GenericSortedOriginLess<Action> compare{action,nullptr};
  cudaError_t status=cub::DeviceMergeSort::SortKeysCopy(nullptr,shape->sort_temporary_bytes,
   static_cast<const uint32_t*>(nullptr),static_cast<uint32_t*>(nullptr),int(capacity),compare,stream);
  if(status!=cudaSuccess)return status;
  status=cub::DeviceScan::ExclusiveSum(nullptr,shape->scan_temporary_bytes,
   static_cast<const uint32_t*>(nullptr),static_cast<uint32_t*>(nullptr),capacity,stream);
  if(status!=cudaSuccess)return status;
 }
 shape->shared_temporary_bytes=shape->sort_temporary_bytes>shape->scan_temporary_bytes?
  shape->sort_temporary_bytes:shape->scan_temporary_bytes;
 // Cached and compact hashes; input, sorted and compact origins; flag and prefix.
 shape->aligned_workspace_bytes=2*generic_sorted_origin_align(shape->hash_bytes)+
  5*generic_sorted_origin_align(shape->origin_bytes)+generic_sorted_origin_align(shape->shared_temporary_bytes)+256;
 return cudaSuccess;
}
// Launch over a fixed admitted capacity, never a CPU readback of valid count.
// Scratch is caller-owned, reused only after the owner event retires. Action
// parents/materialized input planes remain immutable through scatter and accept.
template<class Action>
inline cudaError_t generic_sorted_origin_exact(Action action,uint32_t capacity,uint64_t seed,
 uint32_t bits,uint64_t* hashes,uint32_t* origins,uint32_t* sorted,uint32_t* flags,
 uint32_t* prefix,uint64_t* unique_hashes,uint32_t* unique_origins,uint32_t* unique_count,
 void* temporary,size_t temporary_bytes,uint32_t* error,cudaStream_t stream=0){
 if(!unique_count||!error||!action.elements||bits>64||capacity>=0x7fffffffu)return cudaErrorInvalidValue;
 if(!capacity){generic_sorted_origin_scatter<<<1,1,0,stream>>>(action,nullptr,nullptr,nullptr,nullptr,0,nullptr,nullptr,unique_count);return cudaGetLastError();}
 if(!hashes||!origins||!sorted||!flags||!prefix||!unique_hashes||!unique_origins||!temporary)return cudaErrorInvalidValue;
 uint32_t blocks=capacity/256+(capacity%256!=0);if(blocks>65535)blocks=65535;
 generic_sorted_origin_prepare<<<blocks,256,0,stream>>>(action,capacity,seed,bits,hashes,origins,error);
 cudaError_t status=cudaGetLastError();if(status!=cudaSuccess)return status;
 GenericSortedOriginLess<Action> compare{action,hashes};size_t bytes=temporary_bytes;
 status=cub::DeviceMergeSort::SortKeysCopy(temporary,bytes,origins,sorted,int(capacity),compare,stream);
 if(status!=cudaSuccess)return status;
 generic_sorted_origin_unique_flags<<<blocks,256,0,stream>>>(compare,sorted,capacity,flags);
 status=cudaGetLastError();if(status!=cudaSuccess)return status;
 bytes=temporary_bytes;status=cub::DeviceScan::ExclusiveSum(temporary,bytes,flags,prefix,capacity,stream);
 if(status!=cudaSuccess)return status;
 generic_sorted_origin_scatter<<<blocks,256,0,stream>>>(action,hashes,sorted,flags,prefix,capacity,unique_hashes,unique_origins,unique_count);
 return cudaGetLastError();
}
