#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// Immutable admitted run. Hash is ordering only; row refers to canonical SoA.
struct GenericSortedHistoryRun {
 const uint64_t* hashes;
 const uint32_t* rows;
 uint32_t count;
};

template<class Candidate,class State>
__device__ bool generic_sorted_history_contains(const Candidate& action,uint32_t child,
 uint64_t hash,GenericSortedHistoryRun run,const State* arena,uint32_t stride,uint32_t* error){
 uint32_t lo=0,hi=run.count;
 while(lo<hi){uint32_t middle=lo+(hi-lo)/2;if(run.hashes[middle]<hash)lo=middle+1;else hi=middle;}
 // A full hash collision cannot establish state identity.
 for(uint32_t at=lo;at<run.count&&run.hashes[at]==hash;++at){
  uint32_t row=run.rows[at];if(row>=stride){atomicOr(error,8u);return false;}
  bool equal=true;for(uint32_t coordinate=0;coordinate<action.elements;++coordinate){
   if(int64_t(arena[uint64_t(coordinate)*stride+row])!=action.value(child,coordinate)){equal=false;break;}
  }
  if(equal)return true;
 }
 return false;
}

// Run validation is a publication gate, never an O(history) per-candidate scan.
__global__ void generic_sorted_history_validate(GenericSortedHistoryRun run,uint32_t capacity,
 uint32_t stride,uint32_t* error){
 if(run.count>capacity){if(!blockIdx.x&&!threadIdx.x)atomicOr(error,32u);return;}
 for(uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;row<run.count;row+=blockDim.x*gridDim.x){
  if(run.rows[row]>=stride)atomicOr(error,8u);
  if(row&&run.hashes[row-1]>run.hashes[row])atomicOr(error,256u);
 }
}
