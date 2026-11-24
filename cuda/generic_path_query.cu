#include "generic_path_query.cuh"
#include "generic_action.cuh"

#include <cstddef>

static_assert(sizeof(MgbfsPathTargetsV1) == 80);
static_assert(offsetof(MgbfsPathTargetsV1, seed) == 32);
static_assert(offsetof(MgbfsPathTargetsV1, states) == 56);

template<class State>
__global__ void scan_paths(MgbfsPathScanV1 request, uint64_t work, uint64_t* best, bool target_first) {
    const auto* states = static_cast<const State*>(request.states) + request.begin;
    ActionT<State> action{
        request.kind, request.elements, request.n, request.m,
        request.generators, request.count, request.stride, states, request.permutations,
        request.matrices, request.moduli,
    };
    for (uint64_t pair = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < work; pair += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t row = pair / request.target_count;
        const uint64_t target_id = pair % request.target_count;
        const auto* target = request.targets + target_id * request.elements;
        bool equal = true;
        for (uint32_t i = 0; i < request.elements; ++i) {
            const int64_t value = request.transitions
                ? action.value(static_cast<uint32_t>(pair), i)
                : int64_t(states[i * request.stride + row]);
            if (value != target[i]) {
                equal = false;
                break;
            }
        }
        if (equal) {
            const uint64_t score = target_first ? target_id * request.count + row : pair;
            atomicMin(reinterpret_cast<unsigned long long*>(best), static_cast<unsigned long long>(score));
        }
    }
}

static int scan_path_query(const MgbfsPathScanV1* query, uint64_t* best, void* raw_stream,
                           bool target_first) {
    if (!query || !best) return cudaErrorInvalidValue;
    const auto q = *query;
    if ((q.state_bytes != 1 && q.state_bytes != 8) || q.kind > 1 || !q.elements
        || !q.target_count || !q.targets || !q.states || q.transitions > 1
        || q.begin > q.stride || q.count > q.stride - q.begin
        || (q.transitions && (q.target_count != 1 || !q.generators
            || (q.kind == 0 && (q.n != q.elements || q.m != 1 || !q.permutations))
            || (q.kind == 1 && (uint64_t(q.n) * q.m != q.elements || !q.matrices || !q.moduli))))) {
        return cudaErrorInvalidValue;
    }
    const uint64_t work = uint64_t(q.count) * (q.transitions ? q.generators : q.target_count);
    if (q.transitions && work > UINT32_MAX) return cudaErrorInvalidValue;
    const auto stream = static_cast<cudaStream_t>(raw_stream);
    const auto clear = cudaMemsetAsync(best, 255, sizeof(uint64_t), stream);
    if (clear != cudaSuccess || !work) return clear;
    const unsigned blocks = unsigned((work + 255) / 256 < 65535 ? (work + 255) / 256 : 65535);
    if (q.state_bytes == 1) {
        scan_paths<uint8_t><<<blocks, 256, 0, stream>>>(q, work, best, target_first && !q.transitions);
    } else {
        scan_paths<int64_t><<<blocks, 256, 0, stream>>>(q, work, best, target_first && !q.transitions);
    }
    return cudaGetLastError();
}

extern "C" int mgbfs_path_scan_v1(const MgbfsPathScanV1* query, uint64_t* best, void* stream) {
    return scan_path_query(query, best, stream, false);
}

extern "C" int mgbfs_path_scan_v2(const MgbfsPathScanV1* query, uint64_t* best, void* stream) {
    return scan_path_query(query, best, stream, true);
}

struct PathLookup {
    uint32_t row;
    bool exhausted;
};

// Lookup only committed rows, comparing the full state even for equal hashes.
template<class State, class Coordinate>
__device__ PathLookup lookup_path_row(
    const State* states, uint64_t stride, uint32_t begin, uint32_t end,
    uint32_t elements, uint64_t seed, uint32_t hash_bits, bool shared_bucket,
    const uint64_t* slots, uint64_t slot_count, uint64_t probe_count, Coordinate value) {
    constexpr uint64_t prefix_mask = 0x7fffffff00000000ULL;
    uint64_t hash = seed;
    for (uint32_t e = 0; e < elements; ++e)
        hash = mix64(hash ^ uint64_t(value(e)) ^ uint64_t(e));
    hash = finish_hash(hash, hash_bits);
    const uint64_t mask = slot_count - 1;
    uint64_t slot = shared_bucket ? mgbfs_shared_table_bucket(hash, slot_count) : hash & mask;
    for (uint64_t probe = 0; probe < probe_count; ++probe, slot = (slot + 1) & mask) {
        const uint64_t token = slots[slot];
        if (token == UINT64_MAX) return {UINT32_MAX, false};
        const uint32_t row = uint32_t(token);
        if ((token >> 63) || row < begin || row >= end || ((token ^ hash) & prefix_mask)) continue;
        bool equal = true;
        for (uint32_t e = 0; e < elements; ++e) {
            if (value(e) != int64_t(states[e * stride + row])) {
                equal = false;
                break;
            }
        }
        if (equal) return {row, false};
    }
    return {UINT32_MAX, probe_count < slot_count};
}

template<class State>
__device__ uint32_t committed_child(const MgbfsPathMeetV1& q, uint64_t row) {
    const auto* left = static_cast<const State*>(q.left) + row;
    const auto value = [left, stride = q.left_stride](uint32_t e) {
        return int64_t(left[e * stride]);
    };
    return lookup_path_row(
        static_cast<const State*>(q.right), q.right_stride, 0, q.right_count, q.elements,
        q.seed, q.hash_bits, q.shared_bucket, q.slots, q.slot_count, q.slot_count, value).row;
}

template<class State>
__global__ void find_targets(MgbfsPathTargetsV1 q, uint64_t* best) {
    constexpr uint64_t max_probes = 1024;
    const uint64_t probes = q.slot_count < max_probes ? q.slot_count : max_probes;
    const auto* states = static_cast<const State*>(q.states);
    for (uint64_t id = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         id < q.target_count; id += uint64_t(gridDim.x) * blockDim.x) {
        const auto* target = q.targets + id * q.elements;
        const auto value = [target](uint32_t e) { return target[e]; };
        const auto hit = lookup_path_row(
            states, q.stride, q.begin, q.begin + q.count, q.elements, q.seed, q.hash_bits,
            q.shared_bucket, q.slots, q.slot_count, probes, value);
        if (hit.row != UINT32_MAX || hit.exhausted) {
            const uint64_t score = (id << 32) | hit.row;
            atomicMin(reinterpret_cast<unsigned long long*>(best), static_cast<unsigned long long>(score));
        }
    }
}

template<class State>
__global__ void meet_paths(MgbfsPathMeetV1 q, uint64_t* best) {
    for (uint64_t offset = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         offset < q.left_count; offset += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t row = q.left_begin + offset;
        const auto child = committed_child<State>(q, row);
        if (child != UINT32_MAX) {
            const uint64_t pair = (uint64_t(child) << 32) | row;
            atomicMin(reinterpret_cast<unsigned long long*>(best), static_cast<unsigned long long>(pair));
        }
    }
}

template<class State>
__global__ void resolve_children(MgbfsPathMeetV1 q, uint32_t* children) {
    for (uint64_t offset = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         offset < q.left_count; offset += uint64_t(gridDim.x) * blockDim.x) {
        children[offset] = q.right_count ? committed_child<State>(q, q.left_begin + offset)
                                         : UINT32_MAX;
    }
}

static bool valid_path_lookup(const MgbfsPathMeetV1& q) {
    return (q.state_bytes == 1 || q.state_bytes == 8) && q.elements && q.hash_bits <= 64
        && q.shared_bucket <= 1 && q.slot_count && !(q.slot_count & (q.slot_count - 1))
        && q.left && q.right && q.slots && q.left_begin <= q.left_stride
        && q.left_count <= q.left_stride - q.left_begin && q.right_count <= q.right_stride
        && uint64_t(q.left_begin) + q.left_count <= UINT32_MAX;
}

static unsigned lookup_grid(uint32_t count) {
    const uint64_t blocks = (uint64_t(count) + 255) / 256;
    return unsigned(blocks < 65535 ? blocks : 65535);
}

extern "C" int mgbfs_path_find_targets_v1(
    const MgbfsPathTargetsV1* query, uint64_t* best, void* raw_stream) {
    if (!query || !best) return cudaErrorInvalidValue;
    const auto q = *query;
    if ((q.state_bytes != 1 && q.state_bytes != 8) || !q.elements || !q.target_count
        || q.hash_bits > 64 || q.shared_bucket > 1 || !q.slot_count
        || (q.slot_count & (q.slot_count - 1)) || q.slot_count > SIZE_MAX / sizeof(uint64_t)
        || !q.states || !q.targets || !q.slots
        || uint64_t(q.begin) + q.count > UINT32_MAX
        || uint64_t(q.begin) + q.count > q.stride
        || q.stride > SIZE_MAX / q.state_bytes / q.elements
        || uint64_t(q.target_count) * q.elements > SIZE_MAX / sizeof(int64_t)) {
        return cudaErrorInvalidValue;
    }
    const auto stream = static_cast<cudaStream_t>(raw_stream);
    const auto clear = cudaMemsetAsync(best, 255, sizeof(uint64_t), stream);
    if (clear != cudaSuccess || !q.count) return clear;
    const unsigned grid = lookup_grid(q.target_count);
    if (q.state_bytes == 1) {
        find_targets<uint8_t><<<grid, 256, 0, stream>>>(q, best);
    } else {
        find_targets<int64_t><<<grid, 256, 0, stream>>>(q, best);
    }
    return cudaGetLastError();
}

extern "C" int mgbfs_path_meet_v1(const MgbfsPathMeetV1* query, uint64_t* best, void* raw_stream) {
    if (!query || !best) return cudaErrorInvalidValue;
    const auto q = *query;
    if (!valid_path_lookup(q)) return cudaErrorInvalidValue;
    const auto stream = static_cast<cudaStream_t>(raw_stream);
    const auto clear = cudaMemsetAsync(best, 255, sizeof(uint64_t), stream);
    if (clear != cudaSuccess || !q.left_count || !q.right_count) return clear;
    const unsigned grid = lookup_grid(q.left_count);
    if (q.state_bytes == 1) {
        meet_paths<uint8_t><<<grid, 256, 0, stream>>>(q, best);
    } else {
        meet_paths<int64_t><<<grid, 256, 0, stream>>>(q, best);
    }
    return cudaGetLastError();
}

extern "C" int mgbfs_path_resolve_children_v1(
    const MgbfsPathMeetV1* query, uint32_t* children, uint32_t output_capacity, void* raw_stream) {
    if (!query || !valid_path_lookup(*query) || query->left_count > output_capacity
        || (query->left_count && !children)) return cudaErrorInvalidValue;
    const auto q = *query;
    if (!q.left_count) return cudaSuccess;
    const auto stream = static_cast<cudaStream_t>(raw_stream);
    const unsigned grid = lookup_grid(q.left_count);
    if (q.state_bytes == 1) {
        resolve_children<uint8_t><<<grid, 256, 0, stream>>>(q, children);
    } else {
        resolve_children<int64_t><<<grid, 256, 0, stream>>>(q, children);
    }
    return cudaGetLastError();
}
