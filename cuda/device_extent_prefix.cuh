#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace mgbfs_resident {
// Prefix includes one terminal offset. Empty extents create duplicate offsets;
// upper_bound over the terminal offsets skips them without losing a parent.
// Every total and local row is 64-bit; a local row of UINT32_MAX is valid.
__host__ __device__ __forceinline__ bool locate_parent_extent(
    const uint64_t* prefix,uint64_t count,uint64_t ordinal,uint64_t* extent,uint64_t* row) {
  if(!prefix||!extent||!row||!count||ordinal>=prefix[count])return false;
  uint64_t low=0,high=count;
  while(low<high) {
    const uint64_t mid=low+(high-low)/2;
    if(prefix[mid+1]<=ordinal)low=mid+1;else high=mid;
  }
  if(low>=count||prefix[low]>ordinal)return false;
  *extent=low;*row=ordinal-prefix[low];return true;
}
}
