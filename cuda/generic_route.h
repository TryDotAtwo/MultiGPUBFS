#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Flat 32-byte route records. Immutable parent banks are leased until all
 * routed origins have been compared/materialized on destination owners. */
struct GenericRouteRecord {uint64_t hash,parent;uint32_t source,generator,shard,reserved;};
/* Optional owner_cuts[world+1]: increasing cumulative high-word boundaries,
 * first=0,last=2^32. Optional owner_to_rank[world]: a validated permutation.
 * Queues/counts are preallocated [world*local_shards][queue_capacity].
 * No full children, allocation, CPU dedup or readback in this operation.
 * error bit 1=queue capacity,2=rank map,4=owner boundaries. Counts/queues on
 * error are invalid; do not resume or call a layer complete. */
int mgbfs_generic_route_i64(uint32_t kind,uint32_t elements,uint32_t rows,uint32_t cols,uint32_t generators,
 const int64_t* parents,uint32_t count,uint32_t stride,const uint32_t* permutations,
 const int64_t* matrices,const uint32_t* moduli,uint64_t seed,uint32_t hash_bits,
 uint32_t world,uint32_t source,uint32_t local_shards,uint32_t queue_capacity,uint64_t parent_begin,
 const uint32_t* owner_to_rank,const uint64_t* owner_cuts,struct GenericRouteRecord* queues,
 uint32_t* counts,uint32_t* error,void* stream);
/* Regenerate only requested origins on their source GPU. The request lease
 * identifies the immutable parent range; foreign/stale origins set error=2.
 * No host state generation or dense child materialization. */
int mgbfs_generic_regenerate_routes_i64(uint32_t kind,uint32_t elements,uint32_t rows,uint32_t cols,uint32_t generators,
 const int64_t* parents,uint32_t parent_count,uint32_t parent_stride,const uint32_t* permutations,
 const int64_t* matrices,const uint32_t* moduli,uint32_t source,uint64_t parent_begin,
 const struct GenericRouteRecord* requests,uint32_t request_count,int64_t* output,uint32_t output_stride,
 uint32_t* error,void* stream);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
/* Fixed-capacity request launch with device count: no count readback between
 * routing, source regeneration and destination acceptance. */
int mgbfs_generic_regenerate_routes_count_i64(uint32_t kind,uint32_t elements,uint32_t rows,uint32_t cols,uint32_t generators,
 const int64_t* parents,uint32_t parent_count,uint32_t parent_stride,const uint32_t* permutations,
 const int64_t* matrices,const uint32_t* moduli,uint32_t source,uint64_t parent_begin,
 const struct GenericRouteRecord* requests,uint32_t request_capacity,const uint32_t* device_count,
 int64_t* output,uint32_t output_stride,uint32_t* error,void* stream);
#ifdef __cplusplus
}
#endif

// Compact permutation state payloads; matrix tables remain int64.
#ifdef __cplusplus
extern "C" {
#endif
int mgbfs_generic_route_u8(uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const uint8_t* parents,uint32_t count,uint32_t stride,const uint32_t* permutations,const int64_t* matrices,
 const uint32_t* moduli,uint64_t seed,uint32_t bits,uint32_t world,uint32_t source,uint32_t shards,
 uint32_t capacity,uint64_t begin,const uint32_t* map,const uint64_t* cuts,GenericRouteRecord* queues,
 uint32_t* counts,uint32_t* error,void* stream);
int mgbfs_generic_regenerate_routes_u8(uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const uint8_t* parents,uint32_t count,uint32_t stride,const uint32_t* permutations,const int64_t* matrices,
 const uint32_t* moduli,uint32_t source,uint64_t begin,const GenericRouteRecord* requests,uint32_t request_count,
 uint8_t* output,uint32_t output_stride,uint32_t* error,void* stream);
int mgbfs_generic_regenerate_routes_count_u8(uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const uint8_t* parents,uint32_t count,uint32_t stride,const uint32_t* permutations,const int64_t* matrices,
 const uint32_t* moduli,uint32_t source,uint64_t begin,const GenericRouteRecord* requests,uint32_t request_count,const uint32_t* device_count,
 uint8_t* output,uint32_t output_stride,uint32_t* error,void* stream);
#ifdef __cplusplus
}
#endif

extern "C" int mgbfs_generic_route_retry_vote(const uint32_t*,const uint32_t*,uint32_t*,void*);
