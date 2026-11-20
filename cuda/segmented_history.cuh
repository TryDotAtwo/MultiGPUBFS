#pragma once
#include <stdint.h>

// A segment owns an independent exact hash table and SoA state storage. The
// caller keeps every input and sealed segment alive until the stream finishes.
struct MgbfsSegment {
    uint64_t* slots;
    void* states;
    uint64_t* positions; // required for rolling banks; directed history may omit
    uint64_t slots_count;
    uint64_t capacity; // rows per bank, at most 2^32
    uint32_t banks;    // 1 (directed history) or 3 (rolling history)
    uint32_t position_item_bytes; // 4 or 8; 0 is legacy 8-byte encoding
};
enum : uint32_t {
    MGBFS_SEGMENTED_TABLE_FULL = 1,
    MGBFS_SEGMENTED_OVERFLOW = 2,
    MGBFS_SEGMENTED_BAD_REFERENCE = 4,
    MGBFS_SEGMENTED_CAPACITY_LIMIT = 8
};

// Exactly one active input batch may write a logical shard at a time. Sealed
// segments are immutable and contain no pending slots. Every input hash must
// equal the generic route hash of that full state with the same seed and bit
// width later supplied to rehash. Equal states alone are not sufficient. This
// invariant holds across all batches, active and sealed segments of the job.
// On any error the batch is terminal, not replayable.
extern "C" int mgbfs_segmented_accept_u8(const uint8_t* input, const uint64_t* hashes,
    uint32_t input_count, uint32_t elements, const MgbfsSegment* sealed, uint32_t sealed_count,
    MgbfsSegment active, uint32_t bank, uint64_t* accepted, uint64_t* future,
    uint32_t* error, void* stream);
extern "C" int mgbfs_segmented_accept_i64(const int64_t* input, const uint64_t* hashes,
    uint32_t input_count, uint32_t elements, const MgbfsSegment* sealed, uint32_t sealed_count,
    MgbfsSegment active, uint32_t bank, uint64_t* accepted, uint64_t* future,
    uint32_t* error, void* stream);

// Bounded spill keeps the original input alive. The output has room for at
// least selected_count indices; its count is zeroed by the caller. A selection
// entry names an original input row, or nullptr means identity. No state is
// silently dropped when this segment fills. Sealed segments must remain fixed.
extern "C" int mgbfs_segmented_accept_spill_u8(const uint8_t* input, const uint64_t* hashes,
    uint32_t input_count, const uint32_t* selection, uint32_t selected_count, uint32_t elements,
    const MgbfsSegment* sealed, uint32_t sealed_count, MgbfsSegment active, uint32_t bank,
    uint64_t* accepted, uint64_t* future, uint32_t* spill, uint32_t* spill_count,
    uint32_t* error, void* stream);
extern "C" int mgbfs_segmented_accept_spill_i64(const int64_t* input, const uint64_t* hashes,
    uint32_t input_count, const uint32_t* selection, uint32_t selected_count, uint32_t elements,
    const MgbfsSegment* sealed, uint32_t sealed_count, MgbfsSegment active, uint32_t bank,
    uint64_t* accepted, uint64_t* future, uint32_t* spill, uint32_t* spill_count,
    uint32_t* error, void* stream);

// Consume the existing bounded local/remote GPU route inbox directly, without
// copying states to the CPU. The inbox and selection are immutable until all
// segment passes complete. packed=1 means each metadata row embeds <=24 bytes.
// future may be null: accepted states occupy rows [0, accepted) in their bank.
struct GenericRouteRecord;
extern "C" int mgbfs_segmented_accept_routes_u8(uint32_t elements,const uint8_t* local,const uint8_t* remote,
    const GenericRouteRecord* local_meta,const GenericRouteRecord* remote_meta,const uint32_t* local_counts,const uint32_t* remote_counts,
    uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t queue_capacity,uint32_t packed,
    const uint32_t* selection,uint32_t selected_count,const MgbfsSegment* sealed,uint32_t sealed_count,
    MgbfsSegment active,uint32_t bank,uint64_t* accepted,uint64_t* future,uint32_t* spill,uint32_t* spill_count,uint32_t* error,void* stream);
extern "C" int mgbfs_segmented_accept_routes_i64(uint32_t elements,const int64_t* local,const int64_t* remote,
    const GenericRouteRecord* local_meta,const GenericRouteRecord* remote_meta,const uint32_t* local_counts,const uint32_t* remote_counts,
    uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t queue_capacity,
    const uint32_t* selection,uint32_t selected_count,const MgbfsSegment* sealed,uint32_t sealed_count,
    MgbfsSegment active,uint32_t bank,uint64_t* accepted,uint64_t* future,uint32_t* spill,uint32_t* spill_count,uint32_t* error,void* stream);

// Rebuild a drained segment after retirement accumulates tombstones. Every
// live row is rehashed; on any error the segment is terminal and not reusable.
extern "C" int mgbfs_segmented_rehash_u8(uint32_t elements,MgbfsSegment segment,uint64_t count0,
    uint64_t count1,uint64_t count2,uint64_t seed,uint32_t hash_bits,uint32_t* error,void* stream);
extern "C" int mgbfs_segmented_rehash_i64(uint32_t elements,MgbfsSegment segment,uint64_t count0,
    uint64_t count1,uint64_t count2,uint64_t seed,uint32_t hash_bits,uint32_t* error,void* stream);

// Exclusive maintenance: drain ALL readers/writers of this segment before
// retire or rehash, and prohibit any new lookup/accept until it completes.
// Caller must enforce the barrier across CUDA streams; readers alone are not
// enough because reuse of a tombstone concurrent with accept can duplicate rows.
extern "C" int mgbfs_segmented_retire(MgbfsSegment segment, uint32_t bank,
    uint64_t count, uint32_t* error, void* stream);
