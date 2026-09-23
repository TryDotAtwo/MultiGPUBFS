#ifndef MGBFS_LIBRARY_OWNER_ABI_H
#define MGBFS_LIBRARY_OWNER_ABI_H
#include <stdint.h>
#include "../../cuda/state_commit.h"

#ifdef __cplusplus
extern "C" {
#endif

/* V1, Linux x86_64. Every data pointer is device-resident on the owner device.
 * Counts/structs themselves are host control data. SoA conversion from the
 * native AoS Hash128 format is explicit and must be counted in time/memory.
 * Row counts and capacities must not exceed INT32_MAX (libcudf size_type).
 */
typedef struct MgbfsLibraryKeysV1 {
  const uint32_t* words[4];
  uint32_t rows;
  uint32_t reserved; /* must be zero */
} MgbfsLibraryKeysV1;

typedef struct MgbfsLibraryCandidatesV1 {
  MgbfsLibraryKeysV1 keys;
  const uint32_t* source_indices;
} MgbfsLibraryCandidatesV1;

typedef struct MgbfsLibrarySurvivorsV1 {
  const uint32_t* source_indices;
  uint64_t epoch;
  uint32_t rows;
  uint32_t reserved;
} MgbfsLibrarySurvivorsV1;

/* Device-count, multi-shard cuCO owner. Creation borrows host arrays only for
 * setup, then owns its tables; previous/current GPU key planes remain borrowed
 * until destruction. Every pointer in the result is device-resident and is
 * borrowed through the final ordered consumer of this epoch. Call complete
 * only after all such consumers have completed on the creating stream/event.
 * No hot-path call reads a survivor count or synchronizes the host.
 */
typedef struct MgbfsLibraryRankDeviceBatchV1 {
  const uint32_t* high_words;
  const uint32_t* valid_rows;
  const uint32_t* selected;
  const uint32_t* selected_count;
  const uint32_t* source_indices;
  const uint32_t* accepted_counts;
  const uint32_t* accepted_capacities;
  uint32_t* shard_counts;
  uint32_t* shard_offsets;
} MgbfsLibraryRankDeviceBatchV1;
int mgbfs_library_rank_create_cuco_v1(const MgbfsLibraryKeysV1* previous,
    const MgbfsLibraryKeysV1* current, const uint32_t* accepted_capacities,
    uint32_t shards, uint32_t incoming_capacity, uint32_t logical_owner,
    uint32_t world, void* cuda_stream, void** rank_owner);
int mgbfs_library_rank_compare_v1(void* rank_owner, uint64_t epoch,
    MgbfsLibraryCandidatesV1 input, const uint32_t* valid_rows,
    MgbfsOwnerControl* owner, MgbfsStateRingControl* ring,
    MgbfsLibraryRankDeviceBatchV1* result);
int mgbfs_library_rank_commit_v1(void* rank_owner, uint64_t epoch,
    MgbfsOwnerControl* owner, MgbfsStateRingControl* ring,
    const MgbfsStateExtent* extent);
int mgbfs_library_rank_complete_v1(void* rank_owner, uint64_t epoch);
/* FinalizeDepth only: caller drains all GPU readers first. Drop borrowed
 * previous/current membership tables but retain committed key planes for
 * export. Reject a pending epoch; no host/device synchronization here.
 */
int mgbfs_library_rank_seal_v1(void* rank_owner);
int mgbfs_library_rank_export_shard_v1(void* rank_owner, uint32_t shard,
    uint32_t rows, MgbfsLibraryKeysV1* result);
int mgbfs_library_rank_destroy_v1(void* rank_owner);

/* One fixed RMM pool per device, installed before any owner/input allocation.
 * bytes is nonzero and 256-byte aligned; reserve_bytes is at least 1 GiB.
 * Initial and maximum pool size are identical. No growth or fallback.
 * Calls are serialized by the caller on the creating device. All users of
 * the current RMM resource must be drained before destruction. Destroy rejects
 * live suballocations and leaves the handle valid on failure. On success it
 * restores the previous resource; the handle must never be reused.
 */
int mgbfs_library_pool_create_v1(uint64_t bytes, uint64_t reserve_bytes, void** pool);
int mgbfs_library_pool_destroy_v1(void* pool);
typedef struct MgbfsLibraryPoolUsageV1 {
  uint64_t reserved_bytes;
  uint64_t live_bytes;
  uint64_t peak_bytes;
} MgbfsLibraryPoolUsageV1;
/* Requested suballocations since pool creation, not full VRAM and not a
 * fragmentation-aware minimum reservation. Read outside timed GPU stages;
 * no device drain/reset. Invalid handle/device/resource clears output.
 */
int mgbfs_library_pool_usage_v1(void* pool, MgbfsLibraryPoolUsageV1* usage);

/* Allocation-free layout bridge. AoS is four consecutive u32 words per key.
 * scratch is 256-aligned with 5 * align_up(capacity*4,256) bytes: four
 * hash planes followed by span-local u32 source ordinals. rows <= capacity,
 * 0 < capacity <= INT32_MAX. Input/output ranges must not alias.
 * Result metadata is host-visible; data is ready in stream order. Failure
 * clears result. Tail padding is untouched. These calls never synchronize.
 */
int mgbfs_library_candidates_from_aos_v1(const void* hashes, uint32_t rows,
    uint32_t capacity, void* scratch, uint64_t scratch_bytes, void* cuda_stream,
    MgbfsLibraryCandidatesV1* result);
/* Device-count window of an AoS source. begin/rows are device u32 words;
 * invalid windows poison ring/control before any scratch write. Source
 * indices are absolute offsets into the original source_states plane.
 * Result keys.rows is the fixed window capacity; valid rows remain on GPU.
 */
int mgbfs_library_candidates_from_aos_window_v1(const void* hashes,
    const uint32_t* begin, const uint32_t* rows, uint32_t source_capacity,
    uint32_t window_capacity, void* scratch, uint64_t scratch_bytes,
    MgbfsStateRingControl* ring, MgbfsOwnerControl* control,
    void* cuda_stream, MgbfsLibraryCandidatesV1* result);
int mgbfs_library_keys_to_aos_v1(MgbfsLibraryKeysV1 keys, void* output,
    uint32_t capacity, void* cuda_stream);

/* Caller installs the fixed RMM resource before creation; it and immutable
 * history outlives its owner or successful seal. The resource outlives owners.
 * Exactly one serialized writer/stream per owner.
 * Factory returns a null handle on error. Status 0 means success, never a
 * backend fallback. Exceptions must not cross the C boundary.
 */
int mgbfs_library_owner_create_v1(MgbfsLibraryKeysV1 history, uint32_t capacity,
                                 void* cuda_stream, void** owner);
/* Independent immutable previous/current history views for inverse-closed
 * depth-one BFS. Views are borrowed; no explicit combined history buffer.
 * Both outlive the owner. Remaining V1 operations accept the returned handle.
 */
int mgbfs_library_owner_create_window_v1(MgbfsLibraryKeysV1 previous,
    MgbfsLibraryKeysV1 current, uint32_t capacity, void* cuda_stream, void** owner);
/* Explicit cuCollections implementation of the SAME remaining V1 operations.
 * incoming_capacity is fixed and nonzero before creation. No table growth or
 * fallback; an SDK built without CUCO_ROOT rejects creation and clears owner.
 */
int mgbfs_library_owner_create_cuco_window_v1(MgbfsLibraryKeysV1 previous,
    MgbfsLibraryKeysV1 current, uint32_t capacity, uint32_t incoming_capacity,
    void* cuda_stream, void** owner);
/* One shared workspace per serialized stream/device and captured fixed pool.
 * Factories clear output on failure. All handles are opaque, valid handles
 * only; calls are externally serialized. Destroy rejects live owner references
 * or a live/poisoned lease and retains the workspace handle on failure.
 * Owners must be destroyed before workspace, then workspace before pool.
 */
int mgbfs_library_cuco_workspace_create_v1(uint32_t incoming_capacity,
    void* cuda_stream, void** workspace);
int mgbfs_library_cuco_workspace_destroy_v1(void* workspace);
int mgbfs_library_owner_create_cuco_shared_v1(MgbfsLibraryKeysV1 previous,
    MgbfsLibraryKeysV1 current, uint32_t capacity, void* workspace, void** owner);
/* Caller has drained EVERY result reader and checked native fatal controls.
 * Releases shared scratch lease, not accepted/history storage. Does not wait.
 * Exactly once for the last committed epoch; wrong order poisons the owner.
 */
int mgbfs_library_owner_complete_v1(void* owner, uint64_t epoch);
/* Compare zeros a non-null result on failure and does not publish. Returned indices are borrowed until next compare
 * or destroy. Library count synchronization is included in this call's timing.
 */
int mgbfs_library_owner_compare_v1(void* owner, uint64_t epoch,
                                  MgbfsLibraryCandidatesV1 input,
                                  MgbfsLibrarySurvivorsV1* result);
/* granted_rows must be actual reserved state/archive credits, not a requested
 * capacity. Success only means commit was enqueued. Caller records/waits its
 * stream event before publishing host counts or releasing external leases.
 * Any error poisons the owner; the rank group must fail, not retry.
 */
int mgbfs_library_owner_commit_v1(void* owner, uint64_t epoch, uint32_t granted_rows);
/* Borrow append-order committed SoA keys for depth finalization. Not sorted.
 * Rejects pending/poisoned owners and zeros output on error. Does not wait:
 * consume on the owner stream or after its completion event. Borrow ends at
 * owner destruction; do not mutate keys. No host copy or allocation of key data.
 */
int mgbfs_library_owner_export_v1(void* owner, MgbfsLibraryKeysV1* keys);
/* Finalize this owner after its stream and all external consumers drain.
 * Releases borrowed-history indexes and transient result storage; retains
 * committed keys for export. Only export/destroy are legal afterwards.
 * History buffers may then be reused without an additional depth buffer.
 */
int mgbfs_library_owner_seal_v1(void* owner);
/* Caller drains the owner's stream before destroy. No hidden global barrier. */
void mgbfs_library_owner_destroy_v1(void* owner);

#ifdef __cplusplus
}
static_assert(sizeof(MgbfsLibraryKeysV1) == 40, "keys ABI");
static_assert(sizeof(MgbfsLibraryCandidatesV1) == 48, "candidate ABI");
static_assert(sizeof(MgbfsLibrarySurvivorsV1) == 24, "survivor ABI");
static_assert(sizeof(MgbfsLibraryPoolUsageV1) == 24, "pool usage ABI");
static_assert(sizeof(MgbfsLibraryRankDeviceBatchV1) == 72, "rank batch ABI");
#endif
#endif
