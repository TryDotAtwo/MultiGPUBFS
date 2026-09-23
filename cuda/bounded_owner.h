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
/* Compact future arena variant. offsets has buckets+1 elements and capacities
 * has buckets elements. Both are immutable device arrays. Compare validates
 * touched extents against the prefix directory and accepted_records before
 * any commit; preflight validates the whole directory and uploads it once.
 * Both calls must use the same plan, ordered stream and layout arrays. */
int mgbfs_bounded_owner_compare_layout(void* plan, const MgbfsBucketJob* jobs,
    uint32_t job_count, uint32_t rows, const void* incoming,
    const void* prev, uint64_t prev_count, const void* curr, uint64_t curr_count,
    const void* accepted, const uint32_t* accepted_counts,
    const uint64_t* accepted_offsets, const uint32_t* accepted_capacities,
    uint64_t accepted_records, uint32_t buckets, uint32_t buckets_per_shard,
    uint32_t lane, uint32_t generation, MgbfsOwnerCounts* counts,
    MgbfsOwnerControl* control, void* stream);
int mgbfs_bounded_owner_commit_layout(void* plan, const MgbfsBucketJob* jobs,
    uint32_t job_count, const void* incoming, void* accepted,
    uint32_t* accepted_counts, const uint64_t* accepted_offsets,
    const uint32_t* accepted_capacities, const MgbfsOwnerCounts* counts,
    MgbfsOwnerControl* control, const uint32_t* granted_rows,
    uint32_t* survivor_indices, void* stream);
/* Macro settlement compare. history_ranges is [history_slots][buckets] in
 * device memory; each range indexes one concatenated, immutable history hash
 * arena and contains at most K sorted keys. Matching any history slot marks
 * category `prev`. Commit uses mgbfs_bounded_owner_commit_layout unchanged.
 * A fixed history_slots launch sequence has no host count readback. */
int mgbfs_bounded_owner_compare_history_layout(void* plan,
    const MgbfsBucketJob* jobs, uint32_t job_count, uint32_t rows,
    const void* incoming, const void* history,
    const MgbfsOwnerRange* history_ranges, uint32_t history_slots,
    uint64_t history_records, const void* accepted,
    const uint32_t* accepted_counts, const uint64_t* accepted_offsets,
    const uint32_t* accepted_capacities, uint64_t accepted_records,
    uint32_t buckets, uint32_t buckets_per_shard, uint32_t lane,
    uint32_t generation, MgbfsOwnerCounts* counts,
    MgbfsOwnerControl* control, void* stream);
#ifdef __cplusplus
}
#endif
