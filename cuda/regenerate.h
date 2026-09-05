#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct MgbfsRegenerateOrigin {
  uint32_t source;
  uint16_t move, reserved;
  uint64_t parent;
} MgbfsRegenerateOrigin;
/* Selected-parent matrix regeneration baseline (not Tensor Core backend).
 * Device inputs: canonical u8 parents [parent_count,stride], generators
 * [moves,n,n], requests [capacity], count and sticky fatal. stride %16==0.
 * Parent identifiers are absolute in [parent_begin,parent_begin+parent_count).
 * Caller owns the live contiguous parent extent until stream completion.
 * Disjoint output [capacity,stride], request order, zero padding. No allocation,
 * no host sync. Prevalidation rejects whole batch before any output write.
 * fatal 1=count capacity, 2=origin range/source/reserved. Input fatal is never
 * cleared. Host status 0 means enqueue only. Caller validates canonical inputs.
 */
int mgbfs_regenerate_selected(uint32_t n,uint32_t moves,uint32_t modulus,
 uint32_t stride,uint32_t capacity,uint32_t source_rank,uint64_t parent_begin,
 uint32_t parent_count,const uint8_t* parents,const uint8_t* generators,
 const MgbfsRegenerateOrigin* requests,const uint32_t* count,uint8_t* output,
 uint32_t* fatal,void* stream);
#ifdef __cplusplus
}
#endif
