#pragma once
#include "device_pack4_accept.cuh"
#include "segmented_history.cuh"
#include <cstdint>

// Adapt the real segmented bitmap state allocation to the resident owner.
// Called by host admission, before any scheduler starts. The caller owns the
// allocations and must verify lengths, non-overlapping writable row ranges and
// counters, bank lifetime, budget and rank-wide agreement. A valid shape alone does not establish those obligations.
namespace mgbfs_device_pack4 {
inline bool extent_from_segment(const MgbfsSegment& segment,uint64_t* bank_counts,Extent* out) {
    if(!out||!segment.states||!bank_counts||
       (reinterpret_cast<uintptr_t>(segment.states)%alignof(uint64_t))||
       (reinterpret_cast<uintptr_t>(bank_counts)%alignof(uint64_t))||
       !segment.capacity||segment.capacity>(uint64_t(1)<<32)||
       !segment.banks||segment.banks>3)return false;
    *out={static_cast<uint64_t*>(segment.states),bank_counts,segment.capacity,segment.banks,0};
    return true;
}
}
