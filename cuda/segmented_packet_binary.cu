#include <cuda_runtime.h>
#include <stdint.h>
#include "generic_route.h"

struct PacketShardRun {
    uint32_t shard;
    uint32_t rows;
};
static_assert(sizeof(PacketShardRun) == 8);
static_assert(sizeof(GenericRouteRecord) == 32);

struct BinaryPacketRow {
    uint64_t hash;
    uint64_t state;
};
static_assert(sizeof(BinaryPacketRow) == 16);

__device__ uint32_t pack_byte_nibbles(uint64_t bytes) {
    bytes &= 0x0f0f0f0f0f0f0f0full;
    bytes = (bytes | (bytes >> 4)) & 0x00ff00ff00ff00ffull;
    bytes = (bytes | (bytes >> 8)) & 0x0000ffff0000ffffull;
    return uint32_t(bytes | (bytes >> 16));
}

__device__ uint64_t unpack_byte_nibbles(uint32_t packed) {
    uint64_t bytes = packed;
    bytes = (bytes | (bytes << 16)) & 0x0000ffff0000ffffull;
    bytes = (bytes | (bytes << 8)) & 0x00ff00ff00ff00ffull;
    return (bytes | (bytes << 4)) & 0x0f0f0f0f0f0f0f0full;
}

template<bool Decode>
__global__ void pack4_byte_packet(GenericRouteRecord* records, BinaryPacketRow* packet,
                                  uint32_t rows) {
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows; row += gridDim.x * blockDim.x) {
        // Labels stay in [0, 15], and the compact route zeroes unused bytes.
        if constexpr (Decode) {
            const BinaryPacketRow wire = packet[row];
            const uint64_t low = unpack_byte_nibbles(uint32_t(wire.state));
            const uint64_t high = unpack_byte_nibbles(uint32_t(wire.state >> 32));
            records[row] = {wire.hash, low, uint32_t(high), uint32_t(high >> 32), 0, 0};
        } else {
            const GenericRouteRecord record = records[row];
            const uint64_t high = uint64_t(record.source) | (uint64_t(record.generator) << 32);
            const uint64_t word = uint64_t(pack_byte_nibbles(record.parent)) |
                                  (uint64_t(pack_byte_nibbles(high)) << 32);
            packet[row] = {record.hash, word};
        }
    }
}

extern "C" int mgbfs_segmented_packet_pack4_v1(const void*, uint32_t, uint32_t rows,
                                                uint32_t elements, void*, void* records,
                                                void* packet, uint32_t decode, void* stream) {
    if (!rows || !elements || elements > 16 || !records || !packet || decode > 1)
        return int(cudaErrorInvalidValue);
    const uint32_t blocks = (rows - 1) / 128 + 1;
    auto* metadata = static_cast<GenericRouteRecord*>(records);
    auto* wire = static_cast<BinaryPacketRow*>(packet);
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    if (decode)
        pack4_byte_packet<true><<<blocks, 128, 0, cuda_stream>>>(metadata, wire, rows);
    else
        pack4_byte_packet<false><<<blocks, 128, 0, cuda_stream>>>(metadata, wire, rows);
    return int(cudaGetLastError());
}

__device__ bool packet_run(const PacketShardRun* runs, uint32_t total,
                           uint32_t& begin, uint32_t& rows) {
    begin = 0;
    for (uint32_t i = 0; i < blockIdx.y; ++i) {
        if (runs[i].rows > total - begin) return false;
        begin += runs[i].rows;
    }
    rows = runs[blockIdx.y].rows;
    return rows <= total - begin;
}

template<bool Decode, uint32_t Bits>
__global__ void binary_packet_states(const PacketShardRun* runs, uint32_t elements,
                                     uint32_t total, uint8_t* states, GenericRouteRecord* records,
                                     BinaryPacketRow* packed) {
    uint32_t begin, rows;
    if (!packet_run(runs, total, begin, rows)) return;
    // The compact route keeps a separate SoA span for each shard.
    uint8_t* plane = states + uint64_t(begin) * elements;
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows; row += gridDim.x * blockDim.x) {
        const uint32_t index = begin + row;
        uint64_t word = Decode ? packed[index].state : 0;
        for (uint32_t element = 0; element < elements; ++element) {
            uint8_t& value = plane[uint64_t(element) * rows + row];
            if constexpr (Decode) value = uint8_t((word >> (Bits * element)) & ((1u << Bits) - 1));
            else word |= uint64_t(value) << (Bits * element);
        }
        // Unpacked segmented owners compare the state plane and use
        // only the hash field of the route record.
        if constexpr (Decode) records[index] = {packed[index].hash, 0, 0, 0, 0, 0};
        else packed[index] = {records[index].hash, word};
    }
}

template<uint32_t Bits>
int launch_byte_packet(const void* header, uint32_t runs,
                                                 uint32_t rows, uint32_t elements,
                                                 void* states, void* records, void* packed_rows,
                                                 uint32_t decode, void* stream) {
    if (!header || !states || !records || !packed_rows || !runs || runs > 65535 ||
        rows < runs || elements == 0 || elements > 64 / Bits || decode > 1)
        return int(cudaErrorInvalidValue);
    const uint32_t average = (rows - 1) / runs + 1;
    const dim3 grid((average - 1) / 128 + 1, runs);
    const auto* descriptors = static_cast<const PacketShardRun*>(header);
    auto* values = static_cast<uint8_t*>(states);
    auto* metadata = static_cast<GenericRouteRecord*>(records);
    auto* packed = static_cast<BinaryPacketRow*>(packed_rows);
    auto cuda_stream = static_cast<cudaStream_t>(stream);
    if (decode)
        binary_packet_states<true, Bits><<<grid, 128, 0, cuda_stream>>>(descriptors, elements, rows, values, metadata, packed);
    else
        binary_packet_states<false, Bits><<<grid, 128, 0, cuda_stream>>>(descriptors, elements, rows, values, metadata, packed);
    return int(cudaGetLastError());
}

extern "C" int mgbfs_segmented_packet_binary_v2(const void* header, uint32_t runs, uint32_t rows,
                                       uint32_t elements, void* states, void* records,
                                       void* packed, uint32_t decode, void* stream) {
    return launch_byte_packet<1>(header, runs, rows, elements, states, records, packed, decode, stream);
}

extern "C" int mgbfs_segmented_packet_quaternary_v1(const void* header, uint32_t runs, uint32_t rows,
                                       uint32_t elements, void* states, void* records,
                                       void* packed, uint32_t decode, void* stream) {
    return launch_byte_packet<2>(header, runs, rows, elements, states, records, packed, decode, stream);
}

template<bool Decode>
__global__ void range_packet_states(const PacketShardRun* runs, uint32_t elements, uint32_t total,
                                    int64_t base, int64_t* states, GenericRouteRecord* records,
                                    uint64_t* hashes, uint16_t* offsets) {
    uint32_t begin, rows;
    if (!packet_run(runs, total, begin, rows)) return;
    const uint64_t plane = uint64_t(begin) * elements;
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows; row += gridDim.x * blockDim.x) {
        const uint32_t index = begin + row;
        if constexpr (Decode) records[index] = {hashes[index], 0, 0, 0, 0, 0};
        else hashes[index] = records[index].hash;
        for (uint32_t element = 0; element < elements; ++element) {
            const uint64_t offset = plane + uint64_t(element) * rows + row;
            if constexpr (Decode) states[offset] = int64_t(uint64_t(base) + offsets[offset]);
            else offsets[offset] = uint16_t(uint64_t(states[offset]) - uint64_t(base));
        }
    }
}

extern "C" int mgbfs_segmented_packet_range16_v1(const void* header, uint32_t runs,
                                                  uint32_t rows, uint32_t elements, int64_t base,
                                                  void* states, void* records, void* packet,
                                                  uint32_t decode, void* stream) {
    if (!header || !states || !records || !packet || !runs || runs > 65535 ||
        rows < runs || !elements || decode > 1)
        return int(cudaErrorInvalidValue);
    const uint32_t average = (rows - 1) / runs + 1;
    const dim3 grid((average - 1) / 128 + 1, runs);
    const auto* descriptors = static_cast<const PacketShardRun*>(header);
    auto* values = static_cast<int64_t*>(states);
    auto* metadata = static_cast<GenericRouteRecord*>(records);
    auto* hashes = static_cast<uint64_t*>(packet);
    auto* offsets = reinterpret_cast<uint16_t*>(hashes + rows);
    auto cuda_stream = static_cast<cudaStream_t>(stream);
    if (decode)
        range_packet_states<true><<<grid, 128, 0, cuda_stream>>>(descriptors, elements, rows, base, values, metadata, hashes, offsets);
    else
        range_packet_states<false><<<grid, 128, 0, cuda_stream>>>(descriptors, elements, rows, base, values, metadata, hashes, offsets);
    return int(cudaGetLastError());
}
