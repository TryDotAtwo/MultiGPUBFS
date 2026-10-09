#pragma once
#include <cstdint>
extern "C" int mgbfs_generic_seed_i64(uint32_t elements,const int64_t* states,uint32_t state_stride,uint32_t state_count,
 uint64_t* slots,uint32_t slot_capacity,uint64_t seed,uint32_t hash_bits,uint32_t* error,void* stream);
extern "C" int mgbfs_generic_expand_i64(uint32_t kind,uint32_t elements,uint32_t rows,uint32_t cols,uint32_t generators,
 const int64_t* parents,uint32_t parent_count,uint32_t parent_stride,const uint32_t* permutation_tables,
 const int64_t* matrix_tables,const uint32_t* moduli,uint64_t* slots,uint32_t slot_capacity,
 int64_t* visited,uint32_t visited_capacity,uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,
 uint32_t* future_count,uint64_t seed,uint32_t hash_bits,uint32_t* error,void* stream);
extern "C" int mgbfs_generic_gather_i64(uint32_t elements,const int64_t* source,uint32_t source_stride,
 const uint32_t* indices,uint32_t count,int64_t* output,uint32_t output_stride,void* stream);

/* Exact destination dedup of regenerated state planes. Metadata hashes must
 * be from the agreed graph/seed and the same ordered regeneration requests.
 * Each shard table has one exclusive consumer lease. A/B buffers overlap
 * source work with owner work; never run concurrent consumers on one table.
 * Different shard tables are independent and may execute concurrently.
 * count is device-resident, bound is the preallocated queue capacity.
 * Error=32 rejects count overflow; other bits match generic_expand.
 * Future is invalid on any error; immutable incoming banks stay leased until
 * stream completion so pending-origin comparisons cannot observe reuse. */
#ifdef __cplusplus
extern "C" {
#endif
struct GenericRouteRecord;
int mgbfs_generic_accept_i64(uint32_t elements,const int64_t* incoming,uint32_t stride,
 const struct GenericRouteRecord* metadata,const uint32_t* received,uint32_t bound,uint64_t* slots,
 uint32_t slot_capacity,int64_t* visited,uint32_t visited_capacity,uint32_t* visited_count,
 uint32_t* future,uint32_t future_capacity,uint32_t* future_count,uint32_t* error,void* stream);
#ifdef __cplusplus
}
#endif

// Compact permutation state payloads; matrix tables remain int64.
#ifdef __cplusplus
extern "C" {
#endif
int mgbfs_generic_seed_u8(uint32_t elements,const uint8_t* states,uint32_t stride,uint32_t count,uint64_t* slots,
 uint32_t capacity,uint64_t seed,uint32_t bits,uint32_t* error,void* stream);
int mgbfs_generic_expand_u8(uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const uint8_t* parents,uint32_t count,uint32_t stride,const uint32_t* permutations,const int64_t* matrices,const uint32_t* moduli,
 uint64_t* slots,uint32_t slot_capacity,uint8_t* visited,uint32_t visited_capacity,uint32_t* visited_count,
 uint32_t* future,uint32_t future_capacity,uint32_t* future_count,uint64_t seed,uint32_t bits,uint32_t* error,void* stream);
int mgbfs_generic_gather_u8(uint32_t elements,const uint8_t* source,uint32_t stride,const uint32_t* indices,
 uint32_t count,uint8_t* output,uint32_t output_stride,void* stream);
int mgbfs_generic_accept_u8(uint32_t elements,const uint8_t* incoming,uint32_t stride,
 const GenericRouteRecord* metadata,const uint32_t* received,uint32_t bound,uint64_t* slots,
 uint32_t slot_capacity,uint8_t* visited,uint32_t visited_capacity,uint32_t* visited_count,
 uint32_t* future,uint32_t future_capacity,uint32_t* future_count,uint32_t* error,void* stream);
#ifdef __cplusplus
}
#endif
