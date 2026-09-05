#pragma once
#include "owner_job.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Sorted-input owner leaf. All run pointers are device pointers. Inputs are
 * sorted per bucket; ties already have deterministic provenance order.
 * One plan per lane, exclusive shard lease until commit completes. No run
 * allocations or host synchronization. Compare never writes persistent data.
 * Caller reserves StateRing/materialization/archive credits BEFORE commit;
 * granted_rows is the device-side reservation result (not a capacity guess).
 * Failure poisons the job: caller must fail the rank group, not retry/fallback.
 */
typedef struct MgbfsOwnerCounts {
  uint32_t duplicates, prev, curr, accepted, survivors, new_count;
  uint64_t output_offset;
} MgbfsOwnerCounts;
typedef struct MgbfsOwnerControl {
  uint32_t error, stage, survivors, reserved;
  uint64_t padding[6];
} MgbfsOwnerControl;
typedef struct MgbfsBoundedOwnerBytes {
  uint64_t flags, indices, merged, refinement_errors;
} MgbfsBoundedOwnerBytes;
/* Allocation-free shape query. Byte counts are cudaMalloc payload requests,
 * not driver residency. Shared-memory BMMA tiles are not VRAM allocations.
 * Does not validate hardware support; failed queries zero their output. */
int mgbfs_bounded_owner_query(uint32_t i, uint32_t j, uint32_t k,
    uint32_t backend, uint32_t refinement_capacity, uint32_t tile_limit,
    MgbfsBoundedOwnerBytes* out);
int mgbfs_bounded_owner_create(uint32_t i, uint32_t j, uint32_t k, void** plan);
/* Explicit backend: 0=CUB, 1=SM75 BMMA. Refinement descriptors and tile bound
 * are fixed before allocation. Unsupported backend/device is an error. */
int mgbfs_bounded_owner_create_backend(uint32_t i, uint32_t j, uint32_t k,
    uint32_t backend, uint32_t refinement_capacity, uint32_t tile_limit, void** plan);
void mgbfs_bounded_owner_destroy(void* plan);
int mgbfs_bounded_owner_compare(void* plan, const MgbfsBucketJob* jobs,
    uint32_t job_count, uint32_t rows, const void* incoming,
    const void* prev, uint64_t prev_count, const void* curr, uint64_t curr_count,
    const void* accepted, const uint32_t* accepted_counts, uint32_t buckets,
    uint32_t buckets_per_shard, uint32_t lane, uint32_t generation,
    MgbfsOwnerCounts* counts, MgbfsOwnerControl* control, void* stream);
int mgbfs_bounded_owner_commit(void* plan, const MgbfsBucketJob* jobs,
    uint32_t job_count, const void* incoming, void* accepted,
    uint32_t* accepted_counts, const MgbfsOwnerCounts* counts,
    MgbfsOwnerControl* control, const uint32_t* granted_rows,
    uint32_t* survivor_indices, void* stream);
#ifdef __cplusplus
}
#endif
