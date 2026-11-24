#pragma once

#include <cstdint>

// Targets are signed-int64 AoS, frontier rows use the native SoA arena.
// The output is the first relative pair or UINT64_MAX when there is no match.
struct MgbfsPathScanV1 {
    uint32_t state_bytes, kind, elements, n, m, generators;
    uint32_t count, target_count, transitions;
    uint64_t begin, stride;
    const void* states;
    const uint32_t* permutations;
    const int64_t* matrices;
    const uint32_t* moduli;
    const int64_t* targets;
};

extern "C" int mgbfs_path_scan_v1(const MgbfsPathScanV1*, uint64_t* best, void* stream);
// State matches are ordered by target then row. Transition matches keep row then generator.
extern "C" int mgbfs_path_scan_v2(const MgbfsPathScanV1*, uint64_t* best, void* stream);

// Lookup AoS targets in the committed SoA table, restricted to [begin, begin + count).
// best packs the first target index above its absolute row. A low UINT32_MAX
// requests a fallback scan when that target exhausted the probe budget.
// UINT64_MAX means all targets missed.
struct MgbfsPathTargetsV1 {
    uint32_t state_bytes, elements, hash_bits, shared_bucket;
    uint32_t begin, count, target_count;
    uint64_t seed, stride, slot_count;
    const void* states;
    const int64_t* targets;
    const uint64_t* slots;
};

extern "C" int mgbfs_path_find_targets_v1(const MgbfsPathTargetsV1*, uint64_t* best, void* stream);

// Both arenas are local to the same owner and have completed all GPU writes.
// The left range is one committed layer. The right table is append-only.
// best packs the earliest right row above the absolute left row.
struct MgbfsPathMeetV1 {
    uint32_t state_bytes, elements, hash_bits, shared_bucket;
    uint32_t left_begin, left_count, right_count;
    uint64_t seed, left_stride, right_stride, slot_count;
    const void* left;
    const void* right;
    const uint64_t* slots;
};

extern "C" int mgbfs_path_meet_v1(const MgbfsPathMeetV1*, uint64_t* best, void* stream);

// Resolve one bounded tile of candidate rows against the committed destination
// layer. Preserve input order and duplicates. UINT32_MAX denotes a miss.
extern "C" int mgbfs_path_resolve_children_v1(
    const MgbfsPathMeetV1*, uint32_t* children, uint32_t output_capacity, void* stream);
