#pragma once
#include "regenerate.h"
#ifdef __cplusplus
extern "C" {
#endif
/* CUDA reference backend, not Tensor Core. Device inputs: canonical matrix
 * parents[parent_capacity,stride], generators[moves,n,n], coefficients[n*n,4]
 * in F_p, offsets[4] in F_p and parent_count. Caller validates canonical data.
 * Outputs hashes[candidate_capacity,4], origins[candidate_capacity] are dense
 * parent-major, move-minor. No child-state allocation or output exists.
 * All pointers device resident except stream. No allocation or host sync.
 * fatal sticky: 1=capacity, 2=absolute parent reference overflow. On fatal,
 * candidate_count=0, hashes/origins untouched. Caller retains input lifetimes.
 */
int mgbfs_generate_hash_only(
 uint32_t n,uint32_t moves,uint32_t modulus,uint32_t stride,uint32_t parent_capacity,
 uint32_t candidate_capacity,uint32_t source,uint64_t parent_begin,
 const uint8_t* parents,const uint8_t* generators,const uint32_t* coefficients,
 const uint32_t* offsets,const uint32_t* parent_count,uint32_t* hashes,
 MgbfsRegenerateOrigin* origins,uint32_t* candidate_count,uint32_t* fatal,void* stream);
/* Experimental SM75 integer-MMA generation with register-only modular/hash
 * reduction. Same wire/lifetime contract; no materialized child-state buffer.
 * This does not claim Tensor Core hash projection or a performance win. */
int mgbfs_generate_hash_only_tc(
 uint32_t n,uint32_t moves,uint32_t modulus,uint32_t stride,uint32_t parent_capacity,
 uint32_t candidate_capacity,uint32_t source,uint64_t parent_begin,
 const uint8_t* parents,const uint8_t* generators,const uint32_t* coefficients,
 const uint32_t* offsets,const uint32_t* parent_count,uint32_t* hashes,
 MgbfsRegenerateOrigin* origins,uint32_t* candidate_count,uint32_t* fatal,void* stream);
#ifdef __cplusplus
}
#endif
