#pragma once
#include "generic_route.h"
#include "segmented_history.cuh"
#include <cstddef>
#include <cstdint>

// The input is one rebased queue. Bitmap identity still uses the true domain
// distribution, including the logical world and the owner's storage shard.
struct MgbfsBitmapPacketOwnerArgs {
    uint32_t state_bytes, packed, elements;
    const void* states;
    const GenericRouteRecord* records;
    const uint32_t* counts;
    uint32_t world, owner, shard, shards;
    uint64_t total, owner_len;
    uint64_t* bits;
    uint32_t capacity, visible;
    const uint32_t* selection;
    uint32_t selected, raw_begin;
    MgbfsSegment active;
    uint32_t bank;
    uint64_t* accepted;
    uint32_t* spill;
    uint32_t* spill_count;
    uint32_t* error;
    uint64_t* global_count;
    uint64_t limit;
    void* scratch;
    size_t scratch_bytes;
};
