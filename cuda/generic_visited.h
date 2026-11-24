#pragma once
#include <cstdint>
extern "C" int mgbfs_generic_seed_i64_wide(uint32_t elements,const int64_t* states,uint32_t state_stride,uint32_t state_count,
 uint64_t* slots,uint64_t slot_capacity,uint64_t seed,uint32_t hash_bits,uint32_t* error,void* stream);
extern "C" int mgbfs_generic_expand_i64_wide(uint32_t kind,uint32_t elements,uint32_t rows,uint32_t cols,uint32_t generators,
 const int64_t* parents,uint32_t parent_count,uint32_t parent_stride,const uint32_t* permutation_tables,
 const int64_t* matrix_tables,const uint32_t* moduli,uint64_t* slots,uint64_t slot_capacity,
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
int mgbfs_generic_accept_i64_wide(uint32_t elements,const int64_t* incoming,uint32_t stride,
 const struct GenericRouteRecord* metadata,const uint32_t* received,uint32_t bound,uint64_t* slots,
 uint64_t slot_capacity,int64_t* visited,uint32_t visited_capacity,uint32_t* visited_count,
 uint32_t* future,uint32_t future_capacity,uint32_t* future_count,uint32_t* error,void* stream);
#ifdef __cplusplus
}
#endif

// Compact permutation state payloads; matrix tables remain int64.
#ifdef __cplusplus
extern "C" {
#endif
int mgbfs_generic_seed_u8_wide(uint32_t elements,const uint8_t* states,uint32_t stride,uint32_t count,uint64_t* slots,
 uint64_t capacity,uint64_t seed,uint32_t bits,uint32_t* error,void* stream);
int mgbfs_generic_expand_u8_wide(uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const uint8_t* parents,uint32_t count,uint32_t stride,const uint32_t* permutations,const int64_t* matrices,const uint32_t* moduli,
 uint64_t* slots,uint64_t slot_capacity,uint8_t* visited,uint32_t visited_capacity,uint32_t* visited_count,
 uint32_t* future,uint32_t future_capacity,uint32_t* future_count,uint64_t seed,uint32_t bits,uint32_t* error,void* stream);
int mgbfs_generic_gather_u8(uint32_t elements,const uint8_t* source,uint32_t stride,const uint32_t* indices,
 uint32_t count,uint8_t* output,uint32_t output_stride,void* stream);
int mgbfs_generic_accept_u8_wide(uint32_t elements,const uint8_t* incoming,uint32_t stride,
 const GenericRouteRecord* metadata,const uint32_t* received,uint32_t bound,uint64_t* slots,
 uint64_t slot_capacity,uint8_t* visited,uint32_t visited_capacity,uint32_t* visited_count,
 uint32_t* future,uint32_t future_capacity,uint32_t* future_count,uint32_t* error,void* stream);
#ifdef __cplusplus
}
#endif

// One exclusive owner launch scans all source inboxes for a shard.
extern "C" {
int mgbfs_generic_accept_all_i64_wide(uint32_t elements,const int64_t* local,const int64_t* remote,const GenericRouteRecord* local_meta,const GenericRouteRecord* remote_meta,
 const uint32_t* local_counts,const uint32_t* remote_counts,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t stride,
 uint64_t* slots,uint64_t slot_capacity,int64_t* visited,uint32_t visited_capacity,uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,
 uint32_t* future_count,uint32_t* error,void* stream);
int mgbfs_generic_accept_all_u8_wide(uint32_t elements,const uint8_t* local,const uint8_t* remote,const GenericRouteRecord* local_meta,const GenericRouteRecord* remote_meta,
 const uint32_t* local_counts,const uint32_t* remote_counts,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t stride,
 uint64_t* slots,uint64_t slot_capacity,uint8_t* visited,uint32_t visited_capacity,uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,
 uint32_t* future_count,uint32_t* error,void* stream);
}

struct GenericRouteRecord;
extern "C" int mgbfs_generic_resolve_incoming_edges_v1(
    uint32_t state_bytes, uint32_t elements, const void* local, const void* remote,
    const GenericRouteRecord* local_meta, const GenericRouteRecord* remote_meta,
    const uint32_t* local_counts, const uint32_t* remote_counts, uint32_t rank,
    uint32_t world, uint32_t shard, uint32_t shards, uint32_t queue_capacity,
    const uint64_t* slots, uint64_t slot_count, const void* arena, uint32_t stride,
    uint32_t first_child, uint32_t child_limit, uint32_t* children, uint32_t output_capacity,
    uint32_t* error, void* raw_stream);
extern "C" int mgbfs_generic_resolve_incoming_edges_v2(
    uint32_t state_bytes, uint32_t elements, const void* local, const void* remote,
    const GenericRouteRecord* local_meta, const GenericRouteRecord* remote_meta,
    const uint32_t* local_counts, const uint32_t* remote_counts, uint32_t rank,
    uint32_t world, uint32_t shard, uint32_t shards, uint32_t queue_capacity,
    const uint64_t* slots, uint64_t slot_count, const void* arena, uint32_t stride,
    uint32_t first_child, uint32_t child_limit, uint32_t* children, uint32_t output_capacity,
    uint32_t begin, uint32_t length, uint32_t* error, void* raw_stream);

extern "C" int mgbfs_generic_seed_shared_u8_wide(
    uint32_t elements, const uint8_t* states, uint32_t stride, uint32_t count,
    uint64_t* slots, uint64_t capacity, uint64_t seed, uint32_t bits, uint32_t* error, void* stream);


// First-admission lineage for unpacked distributed candidates. Indexed by
// visited row. Roots are initialized by the caller.
struct GenericPathParent {
 uint32_t parent,source_label;
};
static_assert(sizeof(GenericPathParent)==8,"path parent ABI v2");
extern "C" int mgbfs_generic_accept_path_parents_v2(
 uint32_t state_bytes,uint32_t elements,uint32_t generators,const void* local,const void* remote,
 const GenericRouteRecord* local_meta,const GenericRouteRecord* remote_meta,
 const uint32_t* local_counts,const uint32_t* remote_counts,
 uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t stride,
 uint64_t* slots,uint64_t slot_capacity,void* visited,uint32_t visited_capacity,
 uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,uint32_t* future_count,
 GenericPathParent* lineage,uint32_t* error,void* stream);

// One raw contiguous run with one state plane per coordinate, each of length
// rows. Every record retains its source rank, parent row and generator label.
struct MgbfsCompactPathInputV1 {
    uint32_t state_bytes, elements, generators, rows, source, shard;
    const void* states;
    const GenericRouteRecord* records;
};

struct MgbfsCompactPathOwnerV1 {
    uint32_t capacity;
    uint64_t slot_count;
    uint64_t* slots;
    void* states;
    uint32_t* visited_count;
    uint32_t* future;
    uint32_t* future_count;
    GenericPathParent* lineage;
    uint32_t* error;
};

struct MgbfsCompactWindowEdgeV1 {
    uint64_t parent;
    uint32_t child, generator;
};

// All runs for one table use one ordered stream. Keep the run live until
// its owner operations drain. Nonnull lineage records the first parent.
extern "C" int mgbfs_generic_accept_compact_path_v1(
    const MgbfsCompactPathInputV1*, const MgbfsCompactPathOwnerV1*, void* stream);

// Resolve every candidate in the requested subrange, including duplicates,
// to a committed row relative to first_child. Advance the run's lease only
// after both the output and owner error are available. Any owner error
// invalidates this output.
extern "C" int mgbfs_generic_resolve_compact_window_edges_v1(
    const MgbfsCompactPathInputV1*, const MgbfsCompactPathOwnerV1*,
    uint32_t first_child, uint32_t begin, uint32_t length,
    MgbfsCompactWindowEdgeV1* children, uint32_t output_capacity, void* stream);
