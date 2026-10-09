#pragma once
#include "state_commit.h"
/* V1: one flat 32-byte record per physical allocation descriptor. No state
 * storage, pointer chains, device allocation, or host-visible count. phase:
 * 0 unregistered, 1 current, 2 provisional, 3 released. sequence/count is the
 * remaining live contiguous prefix (released sequence is allocation end).
 * All writers use the owner stream. Register only after StateReady. Release
 * only after generation/origin/transport/archive readers have joined it.
 * Settlement copies survivors into current extents in the SAME StateRing,
 * then discards the original provisional target depth after its final reader.
 */
typedef struct MgbfsWeightedExtentV1 {
  uint64_t sequence, count, descriptor;
  uint32_t target_depth, phase;
} MgbfsWeightedExtentV1;
#ifdef __cplusplus
static_assert(sizeof(MgbfsWeightedExtentV1)==32);
extern "C" {
#endif
int mgbfs_weighted_extent_register(MgbfsStateRingControl*,MgbfsOwnerControl*,
    const MgbfsStateExtent*,MgbfsWeightedExtentV1*,uint32_t target_depth,
    uint32_t provisional,void* stream);
int mgbfs_weighted_extent_retire(MgbfsStateRingControl*,MgbfsWeightedExtentV1*,
    MgbfsStateExtent,uint64_t rows,void* stream);
int mgbfs_weighted_discard_depth(MgbfsStateRingControl*,MgbfsWeightedExtentV1*,
    uint32_t target_depth,void* stream);
/* Reserved current extent and all provisional source descriptors remain leased
 * through this stream's completion. Reuses MaterializePlan request sort; no new
 * allocation or host counts. Validates source generation/target before any copy. */
int mgbfs_state_materialize_weighted_refs(void* plan,const uint64_t* refs,
    const uint32_t* count,uint32_t target_depth,uint8_t* states,
    MgbfsStateRingControl*,MgbfsOwnerControl*,MgbfsStateExtent*,
    const MgbfsWeightedExtentV1*,void* stream);
#ifdef __cplusplus
}
#endif
