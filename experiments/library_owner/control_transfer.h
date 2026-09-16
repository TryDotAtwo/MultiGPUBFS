#pragma once
#include "../../cuda/state_commit.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Host snapshot, Linux x86_64 V1. No commit policy lives in this transport.
 * The caller must still validate every error, grant, ready and count field. */
typedef struct MgbfsControlSnapshotV1 {
  MgbfsOwnerControl control;
  MgbfsStateExtent extent;
  MgbfsStateRingControl ring;
  uint32_t count, reserved;
} MgbfsControlSnapshotV1;
/* Preallocated pinned storage; stream/device must outlive the handle. */
int mgbfs_control_transfer_create_v1(void* stream, void** out);
int mgbfs_control_transfer_destroy_v1(void* handle);
/* Enqueues upload, without a host drain. Another upload before read is fatal. */
int mgbfs_control_transfer_upload_v1(void* handle, const MgbfsOwnerControl* host,
                                   MgbfsOwnerControl* device);
/* All inputs are device pointers. count may be NULL (returned count=0).
 * Enqueues copies on the same stream, drains once, then returns host values.
 * Errors poison the handle and clear output; destroy still drains all DMA. */
int mgbfs_control_transfer_read_v1(void* handle, const MgbfsOwnerControl* control,
    const MgbfsStateExtent* extent, const MgbfsStateRingControl* ring,
    const uint32_t* count, MgbfsControlSnapshotV1* out);
#ifdef __cplusplus
}
static_assert(sizeof(MgbfsControlSnapshotV1) == 200);
#endif
