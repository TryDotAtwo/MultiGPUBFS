#include "generic_path_query.cuh"
#include "generic_route.h"
#include "generic_visited.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

void require(cudaError_t result) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(result));
        std::exit(1);
    }
}

template<class T>
T* buffer(const T* host, size_t count) {
    T* device = nullptr;
    require(cudaMalloc(&device, sizeof(T) * count));
    if (host) require(cudaMemcpy(device, host, sizeof(T) * count, cudaMemcpyHostToDevice));
    return device;
}

bool scan(const MgbfsPathScanV1& config, uint64_t* best, uint64_t expected,
          bool target_first = false) {
    const auto scan_kernel = target_first ? mgbfs_path_scan_v2 : mgbfs_path_scan_v1;
    require(static_cast<cudaError_t>(scan_kernel(&config, best, nullptr)));
    uint64_t actual = 0;
    require(cudaMemcpy(&actual, best, sizeof(actual), cudaMemcpyDeviceToHost));
    return actual == expected;
}

bool meet_with_collisions(uint64_t* best) {
    // One hash bucket contains a nonmatch, an uncommitted row, then two hits.
    const int64_t left[] = {90, -7, -4, -3};
    const int64_t right[] = {-3, -4, -7, 77, 88};
    const uint64_t table[] = {4, 2, 1, 0, UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
    auto* source = buffer(left, 4);
    auto* arena = buffer(right, 5);
    auto* slots = buffer(table, 8);
    MgbfsPathMeetV1 query{8, 1, 0, 0, 1, 3, 2, 42, 4, 5, 8, source, arena, slots};
    require(static_cast<cudaError_t>(mgbfs_path_meet_v1(&query, best, nullptr)));
    uint64_t actual = 0;
    require(cudaMemcpy(&actual, best, sizeof(actual), cudaMemcpyDeviceToHost));
    bool correct = actual == 3; // right row zero, absolute left row three
    query.left_count = 1; // only the uncommitted matching row remains
    require(static_cast<cudaError_t>(mgbfs_path_meet_v1(&query, best, nullptr)));
    require(cudaMemcpy(&actual, best, sizeof(actual), cudaMemcpyDeviceToHost));
    correct = correct && actual == UINT64_MAX;
    require(cudaFree(source));
    require(cudaFree(arena));
    require(cudaFree(slots));
    return correct;
}

bool lookup_targets(const MgbfsPathTargetsV1& query, uint64_t* best, uint64_t expected) {
    require(static_cast<cudaError_t>(mgbfs_path_find_targets_v1(&query, best, nullptr)));
    uint64_t actual = 0;
    require(cudaMemcpy(&actual, best, sizeof(actual), cudaMemcpyDeviceToHost));
    return actual == expected;
}

bool targets_with_collisions(uint64_t* best) {
    const int64_t states[] = {-9, -7, -4, -7, 90, 9, 10, 11, 12, 14};
    const int64_t goals[] = {90, 14, -9, 9, -7, 12, -7, 10, -7, 13};
    // The first goal is pending, the second is older than the requested layer,
    // and the third must skip a committed collision with equal first coordinate.
    const uint64_t slots[] = {
        0, (uint64_t(1) << 63) | 4, 1, UINT64_MAX - 1, 2, 3, UINT64_MAX, UINT64_MAX,
    };
    auto* arena = buffer(states, 10);
    auto* targets = buffer(goals, 10);
    auto* table = buffer(slots, 8);
    MgbfsPathTargetsV1 query{8, 2, 0, 0, 1, 4, 5, 42, 5, 8, arena, targets, table};
    const bool first = lookup_targets(query, best, (uint64_t(2) << 32) | 3);
    query.count = 2;
    const bool later = lookup_targets(query, best, (uint64_t(3) << 32) | 1);
    query.target_count = 2;
    const bool absent = lookup_targets(query, best, UINT64_MAX);
    require(cudaFree(arena));
    require(cudaFree(targets));
    require(cudaFree(table));
    return first && later && absent;
}

bool byte_targets_use_shared_bucket(uint64_t* best) {
    const uint8_t states[] = {7, 255, 11, 3, 9, 5};
    const int64_t goals[] = {-1, 9, 511, 9, 255, 9};
    auto* arena = buffer(states, 6);
    auto* targets = buffer(goals, 6);
    auto* slots = buffer<uint64_t>(nullptr, 8);
    const uint32_t zero = 0;
    auto* error = buffer(&zero, 1);
    require(cudaMemset(slots, 255, 8 * sizeof(uint64_t)));
    require(static_cast<cudaError_t>(
        mgbfs_generic_seed_shared_u8_wide(2, arena, 3, 3, slots, 8, 42, 64, error, nullptr)));
    const MgbfsPathTargetsV1 query{1, 2, 64, 1, 1, 2, 3, 42, 3, 8, arena, targets, slots};
    const bool hit = lookup_targets(query, best, (uint64_t(2) << 32) | 1);
    uint32_t status = 0;
    require(cudaMemcpy(&status, error, sizeof(status), cudaMemcpyDeviceToHost));
    require(cudaFree(arena));
    require(cudaFree(targets));
    require(cudaFree(slots));
    require(cudaFree(error));
    return hit && status == 0;
}

bool bounded_target_probes(uint64_t* best) {
    int64_t states[1024];
    uint64_t slots[2048];
    for (uint32_t i = 0; i < 1024; ++i) {
        states[i] = i;
        slots[i] = i;
        slots[i + 1024] = UINT64_MAX;
    }
    const int64_t goals[] = {1024, 1023};
    auto* arena = buffer(states, 1024);
    auto* targets = buffer(goals, 2);
    auto* table = buffer(slots, 2048);
    MgbfsPathTargetsV1 query{8, 1, 0, 0, 0, 1024, 2, 42, 1024, 2048, arena, targets, table};
    const bool needs_fallback = lookup_targets(query, best, UINT32_MAX);
    query.slot_count = 1024;
    const bool full_table_miss = lookup_targets(query, best, (uint64_t(1) << 32) | 1023);
    require(cudaFree(arena));
    require(cudaFree(targets));
    require(cudaFree(table));
    return needs_fallback && full_table_miss;
}

bool resolve_keeps_all_incoming_edges() {
    const uint8_t candidates[] = {99, 42, 13, 42, 17};
    const uint8_t right[] = {13, 42, 17};
    const uint64_t table[] = {2, 1, 0, UINT64_MAX};
    auto* rows = buffer(candidates, 5);
    auto* arena = buffer(right, 3);
    auto* slots = buffer(table, 4);
    auto* children = buffer<uint32_t>(nullptr, 4);
    const MgbfsPathMeetV1 query{1, 1, 0, 0, 1, 4, 2, 0, 5, 3, 4, rows, arena, slots};
    bool ok = mgbfs_path_resolve_children_v1(&query, children, 3, nullptr) == cudaErrorInvalidValue;
    require(static_cast<cudaError_t>(mgbfs_path_resolve_children_v1(&query, children, 4, nullptr)));
    uint32_t actual[4]{};
    require(cudaMemcpy(actual, children, sizeof(actual), cudaMemcpyDeviceToHost));
    ok = ok && actual[0] == 1 && actual[1] == 0 && actual[2] == 1 && actual[3] == UINT32_MAX;
    require(cudaFree(rows));
    require(cudaFree(arena));
    require(cudaFree(slots));
    require(cudaFree(children));
    return ok;
}

bool resolve_owner_tile_keeps_incoming_labels() {
    const uint8_t states[] = {3, 9, 3, 0};
    const uint8_t children[] = {9, 3};
    const GenericRouteRecord records[] = {
        {0, 6, 0, 1, 0, 0}, {0, 6, 0, 2, 0, 0}, {0, 6, 0, 3, 0, 0}, {}
    };
    const uint32_t queued = 3, zero = 0;
    auto* candidates = buffer(states, 4);
    auto* arena = buffer(children, 2);
    auto* meta = buffer(records, 4);
    auto* counts = buffer(&queued, 1);
    auto* error = buffer(&zero, 1);
    auto* slots = buffer<uint64_t>(nullptr, 8);
    auto* output = buffer<uint32_t>(nullptr, 4);
    require(cudaMemset(slots, 255, 8 * sizeof(uint64_t)));
    require(static_cast<cudaError_t>(
        mgbfs_generic_seed_shared_u8_wide(1, arena, 2, 2, slots, 8, 0, 0, error, nullptr)));
    require(static_cast<cudaError_t>(mgbfs_generic_resolve_incoming_edges_v1(
        1, 1, candidates, candidates, meta, meta, counts, counts, 0, 1, 0, 1, 4,
        slots, 8, arena, 2, 0, 2, output, 4, error, nullptr)));
    uint32_t ids[4]{}, status = 0;
    require(cudaMemcpy(ids, output, sizeof(ids), cudaMemcpyDeviceToHost));
    require(cudaMemcpy(&status, error, sizeof(status), cudaMemcpyDeviceToHost));
    const bool ok = status == 0 && ids[0] == 1 && ids[1] == 0 && ids[2] == 1 && ids[3] == UINT32_MAX;
    require(static_cast<cudaError_t>(mgbfs_generic_resolve_incoming_edges_v2(
        1, 1, candidates, candidates, meta, meta, counts, counts, 0, 1, 0, 1, 4,
        slots, 8, arena, 2, 0, 2, output, 2, 1, 2, error, nullptr)));
    require(cudaMemcpy(ids, output, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    require(cudaMemcpy(&status, error, sizeof(status), cudaMemcpyDeviceToHost));
    const bool tile_ok = status == 0 && ids[0] == 0 && ids[1] == 1;
    require(cudaFree(candidates));
    require(cudaFree(arena));
    require(cudaFree(meta));
    require(cudaFree(counts));
    require(cudaFree(error));
    require(cudaFree(slots));
    require(cudaFree(output));
    return ok && tile_ok;
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) return 77;
    auto* best = buffer<uint64_t>(nullptr, 1);
    const uint8_t rows[] = {0, 3, 4, 0, 0, 1, 4, 0, 0, 2, 1, 0};
    const uint32_t perms[] = {0, 1, 2, 1, 2, 0};
    const int64_t goals[] = {1, 2, 3, 4, 4, 1};
    auto* states = buffer(rows, 12);
    auto* perm = buffer(perms, 6);
    auto* targets = buffer(goals, 6);
    MgbfsPathScanV1 request{1, 0, 3, 3, 1, 2, 2, 2, 0, 1, 4,
                           states, perm, nullptr, nullptr, targets};
    if (!scan(request, best, 3)) return 1;
    request.target_count = 1;
    if (!scan(request, best, UINT64_MAX)) return 2;
    request.transitions = 1;
    if (!scan(request, best, 1)) return 3;
    if (!scan(request, best, 1, true)) return 8;
    const int64_t tied_goals[] = {0, 0, 0, 3, 1, 2};
    require(cudaMemcpy(targets, tied_goals, sizeof(tied_goals), cudaMemcpyHostToDevice));
    request.transitions = 0;
    request.target_count = 2;
    if (!scan(request, best, 1) || !scan(request, best, 2, true)) return 9;
    request.count = 3;
    if (!scan(request, best, 1) || !scan(request, best, 2, true)) return 10;
    require(cudaFree(states));
    require(cudaFree(perm));
    require(cudaFree(targets));

    const int64_t source[] = {3}, matrix[] = {int64_t(1) << 62}, wanted[] = {1};
    const uint32_t modulo[] = {5};
    auto* wide = buffer(source, 1);
    auto* mx = buffer(matrix, 1);
    auto* mod = buffer(modulo, 1);
    auto* target = buffer(wanted, 1);
    request = MgbfsPathScanV1{8, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
                              wide, nullptr, mx, mod, target};
    if (!scan(request, best, 0)) return 4;
    require(cudaFree(wide));
    require(cudaFree(mx));
    require(cudaFree(mod));
    require(cudaFree(target));
    if (!meet_with_collisions(best)) return 5;
    if (!resolve_keeps_all_incoming_edges()) return 6;
    if (!resolve_owner_tile_keeps_incoming_labels()) return 7;
    if (!targets_with_collisions(best)) return 11;
    if (!byte_targets_use_shared_bucket(best)) return 12;
    if (!bounded_target_probes(best)) return 13;
    require(cudaFree(best));
}
