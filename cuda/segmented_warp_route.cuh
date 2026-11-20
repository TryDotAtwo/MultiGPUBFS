#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// All lanes in the active mask call match_any at the same instruction.
// Each destination subgroup reserves one contiguous queue range.
// Overflow remains a per-lane decision by the caller.
__device__ __forceinline__ uint32_t mgbfs_warp_route_reserve(uint32_t* counts,uint32_t queue) {
 const unsigned active=__activemask();
 const unsigned group=__match_any_sync(active,queue);
 const unsigned lane=threadIdx.x & 31u;
 const int leader=__ffs(int(group))-1;
 uint32_t base=0;
 if(int(lane)==leader)base=atomicAdd(counts+queue,uint32_t(__popc(group)));
 base=__shfl_sync(group,base,leader);
 const unsigned before=(uint32_t(1)<<lane)-1u;
 return base+uint32_t(__popc(group&before));
}
