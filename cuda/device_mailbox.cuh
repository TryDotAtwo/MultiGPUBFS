#pragma once
#include "device_mailbox_layout.h"
#include "device_packet_ring.cuh"

// Device address helpers for an already validated, immutable V1 descriptor.
// Caller checks rank<world and lane<lanes; base addresses refer to the same
// registered layout on every rank, with 64-byte alignment and valid leases.
// Only one producer and one consumer may use a given (destination,lane).
namespace mgbfs_device_mailbox {
static_assert(sizeof(mgbfs_packet_ring::PacketSlot)==64,"layout slot bytes");
static_assert(sizeof(mgbfs_packet_ring::CloseSlot)==64,"layout close bytes");
__device__ __forceinline__ uint64_t index(
    const MgbfsDeviceMailboxLayoutV1& v,uint32_t destination,uint32_t lane) {
    return uint64_t(destination)*v.lanes+lane;
}
__device__ __forceinline__ mgbfs_packet_ring::View view(
    unsigned char* base,const MgbfsDeviceMailboxLayoutV1& v,uint32_t destination,uint32_t lane) {
    const uint64_t r=index(v,destination,lane);
    auto* slots=reinterpret_cast<mgbfs_packet_ring::PacketSlot*>(
        base+v.slots_offset+r*uint64_t(v.slots_per_ring)*64);
    auto* payload=v.payload_stride_bytes ?
        base+v.payload_offset+r*uint64_t(v.slots_per_ring)*v.payload_stride_bytes : nullptr;
    return {slots,payload,v.slots_per_ring,v.payload_stride_bytes};
}
__device__ __forceinline__ mgbfs_packet_ring::CloseSlot& close(
    unsigned char* base,const MgbfsDeviceMailboxLayoutV1& v,uint32_t destination,uint32_t lane) {
    return *reinterpret_cast<mgbfs_packet_ring::CloseSlot*>(
        base+v.close_offset+index(v,destination,lane)*64);
}
}
