#ifndef MGBFS_CUDA_H
#define MGBFS_CUDA_H
#include <stdint.h>
#include <stddef.h>
#include "regenerate.h"
#ifdef __cplusplus
extern "C" {
#endif
/* ABI V1, Linux x86_64. Status 0 means successful enqueue, not GPU completion.
 * All pointers except coefficients/generators, out-plan and error are device
 * pointers. Stream is cudaStream_t cast to void*. No pointer aliasing is allowed.
 * Create allocates fixed buffers before depth zero. Run never allocates device
 * memory. Plans are non-reentrant: use one per concurrent stream/route slot.
 * Caller must wait for outstanding stream work before destroying a plan.
 * Canonical matrix bytes are row-major; physical stride is align_up(n*n,16).
 * Parents and children are canonical u8 (<modulus); padding is zero. Hash128
 * is four little-endian u32 residues, numerical order word3 through word0.
 */
int mgbfs_hash_create(uint32_t bytes,uint32_t capacity,const uint8_t* limbs,const uint32_t* offsets,void** out,char* error,size_t error_capacity);
/* Pre-allocation queries. Host outputs, no cudaMalloc or CUDA launches.
 * Bytes exclude caller-owned input/output and allocator/runtime overhead.
 * Failed query zeroes output. Workspace is queried from the compiled GEMM.
 */
typedef struct MgbfsGenerateBytes {
  uint64_t generators, packed_parents, products_s32, workspace;
  uint32_t k, stride, rows, columns;
} MgbfsGenerateBytes;
typedef struct MgbfsHashBytes {
  uint64_t weights, offsets, partials_s32, workspace;
  uint32_t stride, reserved;
} MgbfsHashBytes;
typedef struct MgbfsMaterializeBytes {
  uint64_t keys, sorted, indices, order, scratch;
} MgbfsMaterializeBytes;
typedef struct MgbfsFutureMergeBytes {
  uint64_t merged, unique, tags, unique_tags, indices, selected;
  uint64_t selected_count, flags, states, state, scratch;
} MgbfsFutureMergeBytes;
int mgbfs_generate_query(uint32_t n,uint32_t moves,uint32_t modulus,uint32_t capacity,uint32_t variant,MgbfsGenerateBytes* out);
int mgbfs_hash_query(uint32_t bytes,uint32_t capacity,MgbfsHashBytes* out);
int mgbfs_materialize_query(uint32_t stride,uint32_t capacity,uint32_t frontier,MgbfsMaterializeBytes* out);
int mgbfs_future_merge_query(uint32_t stride,uint32_t future,uint32_t incoming,MgbfsFutureMergeBytes* out);
int mgbfs_hash_run(void* plan,const uint8_t* input,uint32_t* output,uint32_t count,void* stream);
void mgbfs_hash_destroy(void* plan);
int mgbfs_generate_create(uint32_t n,uint32_t moves,uint32_t modulus,uint32_t capacity,const uint8_t* generators,void** out,char* error,size_t error_capacity);
/* Explicit fixed generation variants: 0 legacy 64x32x64; 1 transposed
 * 64x32x32; 2 transposed 128x32x32; 3 transposed 64x32x64;
 * 4 transposed 64x32x32 with vector U4 output (requires n=4). No auto selection.
 * Same parent/child wire layout and arithmetic. Run status 7 rejects grid
 * overflow before any enqueue/output write. Existing create selects 0.
 */
int mgbfs_generate_create_variant(uint32_t n,uint32_t moves,uint32_t modulus,uint32_t capacity,const uint8_t* generators,uint32_t variant,void** out,char* error,size_t error_capacity);
/* Macro transitions must be sorted by nondecreasing positive original-edge
 * weight. They are evaluated by one GEMM, but children are materialized in
 * move-major order: [move][parent][canonical padded state]. Thus each weight
 * run is already contiguous and needs no full-state transpose before routing.
 */
int mgbfs_generate_create_macro_variant(uint32_t n,uint32_t moves,uint32_t modulus,uint32_t capacity,
  const uint8_t* generators,const uint32_t* weights,uint32_t variant,void** out,char* error,size_t error_capacity);
int mgbfs_generate_run(void* plan,const uint8_t* parents,uint8_t* children,uint32_t count,void* stream);
/* Measurement only: host array of four already-created CUDA timing events.
 * Records start, packed, GEMM done, children done. No synchronization/allocation.
 * Ordinary run does not record these events. count must be nonzero.
 */
int mgbfs_generate_profile_run(void* plan,const uint8_t* parents,uint8_t* children,uint32_t count,void* stream,void* const* marks);
void mgbfs_generate_destroy(void* plan);
int mgbfs_route_create(uint32_t capacity,void** out,char* error,size_t error_capacity);
/* CUB query uses the CURRENT CUDA device and compiled CUB policy. No kernels
 * or explicit device allocation. Query/create/run must use the same device.
 * sort/select share scratch sequentially: only max is allocated, not their sum.
 * Caller input/output/count banks and CUDA context overhead are excluded.
 */
typedef struct MgbfsRouteBytes {
  uint64_t sorted, refs, indices, selected, flags, scratch, sort_scratch, select_scratch;
} MgbfsRouteBytes;
int mgbfs_route_query(uint32_t capacity,MgbfsRouteBytes* out);
int mgbfs_route_run(void* plan,const void* hashes,const uint64_t* refs,void* sorted_hashes,uint64_t* sorted_refs,uint32_t* output_count,uint32_t count,int pre_dedup,void* stream);
void mgbfs_route_destroy(void* plan);
typedef struct MgbfsOwnerState {
  uint64_t last_epoch;
  uint32_t count, initialized, fatal, reserved;
} MgbfsOwnerState;
/* Owner bucket primitive, not the rank scheduler. prev/curr and candidates are
 * sorted Hash128 runs for ONE bucket. They may contain duplicate hashes.
 * accepted is a caller-owned flat bucket span of bucket_capacity records;
 * state must be zero-initialized before its first epoch. One scratch plan can
 * process different spans serially; concurrent jobs require separate plans.
 * candidate_count lives on device and is bounded by candidate_capacity.
 * All output spans have candidate_capacity entries; accepted has bucket_capacity.
 * state.fatal: 1 capacity, 2 epoch order, 3 unsorted input. Sticky until the
 * enclosing failed run is destroyed. Fatal preserves accepted and publishes
 * zero survivors. The host must treat any nonzero ABI return as group-fatal too.
 * Successful publication follows all dense writes in the supplied stream.
 */
int mgbfs_owner_create(uint32_t candidate_capacity,uint32_t bucket_capacity,void** out,char* error,size_t error_capacity);
int mgbfs_owner_run(void* plan,const void* prev,uint32_t prev_count,const void* curr,uint32_t curr_count,
  void* accepted,MgbfsOwnerState* state,const void* candidates,const uint64_t* refs,const uint32_t* candidate_count,
  void* survivors,uint64_t* survivor_refs,uint32_t* survivor_count,uint64_t epoch,void* stream);
void mgbfs_owner_destroy(void* plan);
typedef struct MgbfsMacroSettleBytes {
  uint64_t indices, selected, flags, count, scratch;
} MgbfsMacroSettleBytes;
typedef struct MgbfsMacroSettleState {
  uint64_t last_epoch;
  uint32_t count, fatal;
} MgbfsMacroSettleState;
/* One target-depth/bucket settlement. Future and each fixed-stride history run
 * are sorted Hash128 arrays. The kernel rejects duplicates inside future and
 * membership in every one of the 2K history runs, then densely compacts hashes
 * and StateRefs. history is [history_layers][history_capacity]. No allocation
 * or host synchronization occurs in run. fatal: 1 capacity, 2 epoch, 3 sort.
 */
int mgbfs_macro_settle_query(uint32_t candidate_capacity,uint32_t history_layers,uint32_t history_capacity,MgbfsMacroSettleBytes* out);
int mgbfs_macro_settle_create(uint32_t candidate_capacity,uint32_t history_layers,uint32_t history_capacity,void** out,char* error,size_t error_capacity);
int mgbfs_macro_settle_run(void* plan,const void* future,const uint64_t* refs,const uint32_t* count,
  const void* history,const uint32_t* history_counts,void* survivors,uint64_t* survivor_refs,
  uint32_t* survivor_count,MgbfsMacroSettleState* state,uint64_t epoch,void* stream);
void mgbfs_macro_settle_destroy(void* plan);
typedef struct MgbfsFrontierState { uint32_t count, fatal; } MgbfsFrontierState;
/* Incrementally commit one sorted candidate run into a sorted provisional
 * target-depth set. Existing rows win equal hashes. State payloads are copied
 * only for unique hashes; output is copied back densely into the fixed future
 * slot. No host count read or allocation occurs in run. */
int mgbfs_future_merge_create(uint32_t stride,uint32_t future_capacity,uint32_t incoming_capacity,void** out,char* error,size_t error_capacity);
int mgbfs_future_merge_run(void* plan,uint8_t* future_states,void* future_hashes,MgbfsFrontierState* future_state,
  const uint8_t* source_states,uint32_t source_count,const void* incoming_hashes,const uint64_t* incoming_refs,
  const uint32_t* incoming_count,void* stream);
/* Same operation with caller-proven upper bounds for the live old and incoming
 * runs. This avoids launching capacity-sized grids when synchronized host
 * routing already knows tighter bounds. The device counts remain authoritative
 * and exceeding either bound poisons the result. */
int mgbfs_future_merge_run_bounded(void* plan,uint8_t* future_states,void* future_hashes,MgbfsFrontierState* future_state,
  uint32_t old_count_bound,const uint8_t* source_states,uint32_t source_count,const void* incoming_hashes,
  const uint64_t* incoming_refs,const uint32_t* incoming_count,uint32_t incoming_count_bound,void* stream);
void mgbfs_future_merge_destroy(void* plan);
int mgbfs_exchange_pack(uint32_t stride,uint32_t capacity,const uint8_t* source_states,uint32_t source_count,
  const void* sorted_hashes,const uint64_t* sorted_refs,uint32_t count,uint8_t* packed_states,uint32_t* owner_counts,void* stream);
int mgbfs_archive_pack_permutation_u8(uint32_t n,uint32_t stride,const uint8_t* states,uint32_t count,
  uint8_t* permutations,void* ring,void* stream);
int mgbfs_nccl_unique_id(void* id128);
int mgbfs_nccl_create(uint32_t rank,uint32_t world,uint32_t device,const void* id128,void** out,char* error,size_t error_capacity);
int mgbfs_nccl_send_recv(void* comm,const void* send,uint64_t send_bytes,uint32_t peer,void* recv,uint64_t recv_bytes,void* stream);
int mgbfs_nccl_all_gather_u32(void* comm,const uint32_t* send,uint32_t* receive,void* stream);
int mgbfs_nccl_all_reduce_max_u32(void* comm,const uint32_t* send,uint32_t* receive,void* stream);
void mgbfs_nccl_destroy(void* comm);
/* Terminal, idempotent abort; wrapper remains owned until destroy. Not thread-safe:
 * serialize with all other communicator calls. Does not provide a watchdog. */
int mgbfs_nccl_abort(void* comm);
/* Health query for the blocking communicator created above, not transfer
 * completion. 0 healthy, 1 invalid/aborted, 2 query failure, 3 async failure. */
int mgbfs_nccl_poll(void* comm);
/* Single-source scatter, same source and matching byte counts on every rank.
 * sizes is a host array [world] on source; send holds dense rank-ordered ranges.
 * Source's own range stays a view (no copy). Receivers provide prevalidated
 * recv_bytes and actual allocated recv_capacity; recv_bytes <= recv_capacity is
 * checked before GroupStart. Source sizes sum <= send_capacity. Cross-rank
 * agreement remains the caller's responsibility; rebuild C and Rust together.
 * Zero-byte peers still participate. Success means enqueue only. */
int mgbfs_nccl_scatter(void* comm,uint32_t source,const void* send,uint64_t send_capacity,const uint64_t* sizes,void* recv,uint64_t recv_bytes,uint64_t recv_capacity,void* stream);
/* Materialize one committed batch from a live source slot. References are local
 * source row indices, not global OriginRefs. Source stride is a multiple of 16.
 * Requests are sorted by source index; output states and hashes share that order.
 * All destination spans are disjoint from inputs. Append jobs serialize on one
 * stream. Device count may be zero. fatal: 1 capacity, 2 invalid source ref.
 * Failure is sticky and leaves destination bytes/count unchanged. The caller
 * retains source and request leases until stream completion and aborts the run
 * on any fatal (including a previous owner failure). No allocations in run.
 */
int mgbfs_materialize_create(uint32_t stride,uint32_t candidate_capacity,uint32_t frontier_capacity,void** out,char* error,size_t error_capacity);
int mgbfs_materialize_run(void* plan,const uint8_t* source,uint32_t source_count,const void* hashes,const uint64_t* refs,const uint32_t* count,uint8_t* states,void* out_hashes,MgbfsFrontierState* state,void* stream);
void mgbfs_materialize_destroy(void* plan);
/* Stable per-source sort by absolute parent StateRef, preserving target pairing.
 * Reuses MaterializePlan buffers/scratch, no allocation or host sync. All data
 * pointers are device pointers. Count may be below capacity, including zero.
 * fatal is sticky: 1=count overflow, 2=mixed source/reserved origin. Invalid
 * input leaves output records untouched. Full uint64 parent range is supported.
 */
int mgbfs_materialize_sort_origins(void* plan,uint32_t source_rank,
    const MgbfsRegenerateOrigin* origins,const uint64_t* targets,const uint32_t* count,
    MgbfsRegenerateOrigin* sorted_origins,uint64_t* sorted_targets,uint32_t* fatal,void* stream);
#ifdef __cplusplus
}
#endif
#endif
