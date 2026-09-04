#pragma once
#include "bounded_owner.h"
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
/* DENSE-only FIFO reclamation after both generation and archive DMA have
 * completed for this prefix. The caller owns those event dependencies. */
int mgbfs_state_retire_dense_prefix(MgbfsStateRingControl* ring,
    MgbfsStateExtent* current, uint64_t records, void* stream);
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
#ifdef __cplusplus
}
#endif
