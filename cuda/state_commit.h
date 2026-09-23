#pragma once
#include "bounded_owner.h"
#include "regenerate.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Single-writer GPU reservation sequencer. All pointers are device pointers.
 * Caller serializes reservations and publishes head/descriptor_head only after
 * parent enumeration, origin leases and D2H/archive obligations have drained.
 * Those release obligations are NOT implemented by this leaf.
 */
typedef struct MgbfsStateRingControl {
  uint64_t head, tail, descriptor_head, descriptor_tail;
  uint64_t capacity, descriptor_capacity;
  uint32_t fatal, reserved;
  uint64_t padding;
} MgbfsStateRingControl;
typedef struct MgbfsStateExtent {
  uint64_t sequence, begin, count, descriptor;
  uint32_t granted_rows, ready;
  uint64_t padding[3];
} MgbfsStateExtent;
int mgbfs_state_reserve(MgbfsStateRingControl* ring, MgbfsOwnerControl* owner,
    MgbfsStateExtent* extent, void* stream);
int mgbfs_state_reserve_layer(MgbfsStateRingControl* ring, MgbfsOwnerControl* owner,
    MgbfsStateExtent* extent, uint32_t* layer_count, uint32_t layer_capacity, void* stream);
/* One rank-batch transaction. shard_survivors, accepted_counts and capacities
 * have shard_count device entries. offsets has shard_count+1 entries and is
 * valid only if owner.error==0. One contiguous extent covers the whole batch;
 * no accepted table is modified here. request_capacity is checked only for
 * HASH_FIRST (hash_first != 0). All validation precedes ring/layer mutation.
 * Caller must serialize this single-writer operation on the owner stream.
 */
int mgbfs_state_reserve_rank_batch(MgbfsStateRingControl* ring,
    MgbfsOwnerControl* owner, MgbfsStateExtent* extent,
    const uint32_t* shard_survivors, const uint32_t* accepted_counts,
    const uint32_t* accepted_capacities, uint32_t shard_count,
    uint32_t* offsets, uint32_t* layer_count, uint32_t layer_capacity,
    uint32_t request_capacity, uint32_t hash_first, void* stream);
/* Single-writer DENSE rank-batch publication after materialization on the
 * owner stream. Adjacent physical extents merge; a wrap uses the second slot.
 * count and out[capacity] are preallocated device storage and read at
 * FinalizeDepth only. Sticky fatal leaves count/output unchanged. */
int mgbfs_state_publish_next_extent(MgbfsStateRingControl* ring,
    MgbfsOwnerControl* owner, const MgbfsStateExtent* extent,
    uint32_t* count, MgbfsStateExtent* out, uint32_t capacity, void* stream);
/* Build per-shard counts from CUB-selected indices in sorted Hash128 order.
 * high_words points to candidate SoA word 3; candidate_count and selected_count
 * are device words. Outputs are valid only when owner.error==0. On malformed
 * indices/order/owner, sticky fatal is set before any output write. Fixed
 * launch shape supports graph capture; no host count readback.
 */
int mgbfs_owner_shard_counts(const uint32_t* high_words,
    const uint32_t* candidate_count, const uint32_t* selected,
    const uint32_t* selected_count, uint32_t candidate_capacity,
    uint32_t logical_owner, uint32_t world, uint32_t shards,
    uint32_t* shard_counts, uint32_t* shard_offsets,
    MgbfsStateRingControl* ring, MgbfsOwnerControl* owner, void* stream);
/* DENSE-only FIFO reclamation after both generation and archive DMA have
 * completed for this prefix. The caller owns those event dependencies. */
int mgbfs_state_retire_dense_prefix(MgbfsStateRingControl* ring,
    MgbfsStateExtent* current, uint64_t records, void* stream);
/* Same FIFO validation, but the caller passes its immutable extent snapshot
 * by value. No per-batch H2D extent upload or device extent mutation. The
 * caller must retain generation, transport and archive reader ordering. */
int mgbfs_state_retire_dense_prefix_value(MgbfsStateRingControl* ring,
    MgbfsStateExtent current, uint64_t records, void* stream);
/* Build the NCCL max-reduction input on the device after queued retirement.
 * The result is exactly 0 or 1. Caller orders this kernel after the ring
 * writer and before the collective on the same stream. */
int mgbfs_state_ring_fatal_vote_word(const MgbfsStateRingControl* ring,
    uint32_t* word, void* stream);
/* Dense input is source-order, sorted_refs maps sorted hashes to those rows.
 * All output rows and indices are validated before any state copy. Hash commit
 * must already have completed on this stream. Extent.ready publishes StateReady.
 * Consumer must wait for completion event, not poll ready concurrently.
 */
int mgbfs_state_materialize(const uint8_t* candidates, uint32_t candidate_count,
    const uint64_t* sorted_refs, uint32_t sorted_count, const uint32_t* selected,
    uint32_t selected_capacity, uint32_t stride, uint8_t* states,
    MgbfsStateRingControl* ring, MgbfsOwnerControl* owner,
    MgbfsStateExtent* extent, void* stream);
/* Input already follows sorted hash order (e.g. owner receive frame).
 * selected is span-local; input points to that span's first packed state.
 * Same validation/publication contract, without an identity-reference load.
 * Caller retains the transport consumer through the completion event. */
int mgbfs_state_materialize_packed(const uint8_t* input, uint32_t rows,
    const uint32_t* selected, uint32_t selected_capacity, uint32_t stride,
    uint8_t* states, MgbfsStateRingControl* ring, MgbfsOwnerControl* owner,
    MgbfsStateExtent* extent, void* stream);
/* Rank-batch DENSE path: both source row count and survivor count are device
 * words. source_indices holds compacted source-row ordinals, not hash values.
 * All indices are checked before any state copy; output is one dense extent.
 * No host count readback, allocation, or stream synchronization.
 */
int mgbfs_state_materialize_rank_batch(const uint8_t* input,
    const uint32_t* source_rows, uint32_t source_capacity,
    const uint32_t* source_indices, const uint32_t* selected_count,
    uint32_t selected_capacity, uint32_t stride, uint8_t* states,
    MgbfsStateRingControl* ring, MgbfsOwnerControl* owner,
    MgbfsStateExtent* extent, void* stream);
/* HASH_FIRST after irreversible owner commit (stage 2). Compact selected
 * origins and absolute target StateRefs into dense request order. Does not
 * publish StateReady or release source origins. Caller preserves request/target
 * pairing through routing and waits for all responses before materialization.
 * All indices are validated before any request/target write; request_count=0
 * on device fatal. Output buffers are disjoint, each selected_capacity rows.
 */
int mgbfs_state_build_requests(const MgbfsRegenerateOrigin* origins,uint32_t candidate_count,
    const uint64_t* sorted_refs,uint32_t sorted_count,const uint32_t* selected,
    uint32_t selected_capacity,MgbfsRegenerateOrigin* requests,uint64_t* target_refs,
    uint32_t* request_count,MgbfsStateRingControl* ring,MgbfsOwnerControl* owner,
    MgbfsStateExtent* extent,void* stream);
/* Apply all responses for one committed extent. Reuses MaterializePlan CUB
 * scratch to sort absolute target refs; every target must occur exactly once.
 * Any missing/duplicate/foreign ref or group fatal poisons owner/ring with 18
 * before state writes. Dense writes and StateReady publication use the same
 * checked materializer as DENSE. No allocation/host sync; plans are exclusive.
 */
int mgbfs_state_apply_responses(void* materialize_plan,const uint8_t* responses,
    const uint64_t* targets,const uint32_t* count,const uint32_t* group_fatal,
    uint8_t* states,MgbfsStateRingControl* ring,MgbfsOwnerControl* owner,
    MgbfsStateExtent* extent,void* stream);
/* Same validation for a subrange of the sorted packet, e.g. ring wrap splits.
 * Caller enumerates disjoint extents covering the whole packet exactly once.
 */
int mgbfs_state_apply_response_span(void* materialize_plan,const uint8_t* responses,
    const uint64_t* targets,const uint32_t* count,uint32_t sorted_offset,const uint32_t* group_fatal,
    uint8_t* states,MgbfsStateRingControl* ring,MgbfsOwnerControl* owner,
    MgbfsStateExtent* extent,void* stream);
#ifdef __cplusplus
}
#endif
