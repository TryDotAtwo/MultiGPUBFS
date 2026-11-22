#include "device_mailbox_layout.h"
#include <limits>

namespace {
bool mul(uint64_t a,uint64_t b,uint64_t& out) {
    if(a && b>UINT64_MAX/a)return false;
    out=a*b;return true;
}
bool add(uint64_t a,uint64_t b,uint64_t& out) {
    if(b>UINT64_MAX-a)return false;
    out=a+b;return true;
}
bool align(uint64_t value,uint64_t alignment,uint64_t& out) {
    uint64_t x;
    if(!add(value,alignment-1,x))return false;
    out=x&~(alignment-1);return true;
}
}
extern "C" int mgbfs_device_mailbox_layout_v1(uint32_t world,uint32_t lanes,
    uint32_t slots,uint32_t payload,uint64_t budget,size_t output_bytes,
    MgbfsDeviceMailboxLayoutV1* output) {
    static_assert(sizeof(MgbfsDeviceMailboxLayoutV1)==72,"mailbox layout ABI");
    if(!world||!lanes||!slots||!output||output_bytes<sizeof(*output))
        return MGBFS_MAILBOX_LAYOUT_INVALID;
    MgbfsDeviceMailboxLayoutV1 v{};
    v.abi_version=1;v.struct_bytes=sizeof(v);v.world=world;v.lanes=lanes;
    v.slots_per_ring=slots;v.requested_payload_bytes=payload;
    uint64_t stride=0,slot_count=0,slot_bytes=0,payload_bytes=0,close_bytes=0,end=0;
    if(!align(payload,16,stride)||stride>UINT32_MAX||
       !mul(world,lanes,v.ring_count)||!mul(v.ring_count,slots,slot_count)||
       !mul(slot_count,64,slot_bytes)||!mul(slot_count,stride,payload_bytes)||
       !mul(v.ring_count,64,close_bytes))return MGBFS_MAILBOX_LAYOUT_OVERFLOW;
    v.payload_stride_bytes=uint32_t(stride);
    v.slots_offset=0;v.payload_offset=slot_bytes;
    if(!add(v.payload_offset,payload_bytes,end)||!align(end,64,v.close_offset)||
       !add(v.close_offset,close_bytes,v.total_bytes)||
       v.total_bytes>std::numeric_limits<size_t>::max())return MGBFS_MAILBOX_LAYOUT_OVERFLOW;
    if(v.total_bytes>budget)return MGBFS_MAILBOX_LAYOUT_BUDGET;
    *output=v;return MGBFS_MAILBOX_LAYOUT_OK;
}
