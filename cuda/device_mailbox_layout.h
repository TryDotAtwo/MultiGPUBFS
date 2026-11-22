#pragma once
#include <stddef.h>
#include <stdint.h>

// Immutable per-job layout for one rank-owned symmetric mailbox window.
// Callers must agree on the entire descriptor before allocation/registration.
// Ring index is destination*lanes+lane; different sources own different
// windows. All offsets and total bytes are 64-bit. The layout reserves no
// communication or BFS state memory beyond these rings and close records.
// The window base must be 64-byte aligned and remain live until every peer
// has drained and retired all references, including cancellation paths.
struct MgbfsDeviceMailboxLayoutV1 {
    uint32_t abi_version,struct_bytes,world,lanes;
    uint32_t slots_per_ring,requested_payload_bytes,payload_stride_bytes,reserved;
    uint64_t ring_count,slots_offset,payload_offset,close_offset,total_bytes;
};
enum MgbfsMailboxLayoutStatus {
    MGBFS_MAILBOX_LAYOUT_OK=0,
    MGBFS_MAILBOX_LAYOUT_INVALID=1,
    MGBFS_MAILBOX_LAYOUT_OVERFLOW=2,
    MGBFS_MAILBOX_LAYOUT_BUDGET=3
};
// budget_bytes is a hard admission bound, not a request to allocate.
// A successful layout may round payload up to 16 bytes; the resulting stride
// is the actual maximum packet size. 0 payload bytes are valid. On error the
// output is not modified. Larger future output structs are allowed.
extern "C" int mgbfs_device_mailbox_layout_v1(uint32_t world,uint32_t lanes,
    uint32_t slots_per_ring,uint32_t payload_bytes,uint64_t budget_bytes,
    size_t output_bytes,MgbfsDeviceMailboxLayoutV1* output);
