#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define MGBFS_PARTITION_HD __host__ __device__
#else
#define MGBFS_PARTITION_HD
#endif
// Reads high words of sorted AoS Hash128 keys; no pointer dereference for n=0.
// Caller validates world in {1,2,4,8} and boundary <= world.
MGBFS_PARTITION_HD inline uint32_t mgbfs_owner_boundary(
    const uint32_t* words, uint32_t n, uint32_t boundary, uint32_t world) {
  if (!boundary || !n) return 0;
  if (boundary == world) return n;
  uint32_t bits=0;
  for(uint32_t w=world;w>1;w>>=1) ++bits;
  uint32_t lo=0,hi=n;
  while(lo<hi) {
    const uint32_t mid=lo+(hi-lo)/2;
    const uint32_t owner=words[uint64_t(mid)*4+3]>>(32-bits);
    if(owner<boundary) lo=mid+1; else hi=mid;
  }
  return lo;
}
#undef MGBFS_PARTITION_HD
