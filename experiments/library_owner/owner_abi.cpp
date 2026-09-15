#include "owner_abi.h"
#include "cudf_owner.hpp"
#include <limits>
#include <vector>
#include <map>
#include <mutex>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

// RED scaffolds for the allocation-free native AoS boundary.
extern "C" int mgbfs_library_candidates_from_aos_v1(const void*, uint32_t,
    uint32_t, void*, uint64_t, void*, MgbfsLibraryCandidatesV1* result) {
  if (result) *result = {};
  return -1;
}
extern "C" int mgbfs_library_keys_to_aos_v1(MgbfsLibraryKeysV1, void*, uint32_t, void*) {
  return -1;
}

namespace {
using FixedPool = rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>;
struct PoolHandle {
  int device;
  decltype(rmm::mr::get_current_device_resource_ref()) previous;
  rmm::mr::cuda_memory_resource upstream;
  FixedPool pool;
  rmm::mr::statistics_resource_adaptor<FixedPool> stats;
  PoolHandle(int id, size_t bytes)
      : device(id), previous(rmm::mr::get_current_device_resource_ref()),
        pool(&upstream, bytes, bytes), stats(&pool) {}
};
std::mutex pool_mutex;
std::map<int, PoolHandle*> device_pools;
}

extern "C" int mgbfs_library_pool_create_v1(uint64_t bytes, uint64_t reserve, void** output) {
  if (!output) return -1;
  *output = nullptr;
  try {
    if (!bytes || bytes % 256 || reserve < (1ULL << 30) ||
        bytes > std::numeric_limits<size_t>::max()) return -1;
    int device;
    size_t free, total;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaMemGetInfo(&free, &total) != cudaSuccess) return -1;
    if (reserve > free || bytes > free - reserve) return -1;
    std::lock_guard<std::mutex> lock(pool_mutex);
    if (device_pools.count(device)) return -1;
    auto handle = std::make_unique<PoolHandle>(device, static_cast<size_t>(bytes));
    // Map insertion may throw: do it before exposing the new current resource.
    device_pools.emplace(device, handle.get());
    try { rmm::mr::set_current_device_resource_ref(handle->stats); }
    catch (...) { device_pools.erase(device); throw; }
    *output = handle.release();
    return 0;
  } catch (...) { return -1; }
}
extern "C" int mgbfs_library_pool_destroy_v1(void* pool) {
  if (!pool) return -1;
  try {
    int device;
    if (cudaGetDevice(&device) != cudaSuccess) return -1;
    std::lock_guard<std::mutex> lock(pool_mutex);
    auto found = device_pools.find(device);
    if (found == device_pools.end() || found->second != pool) return -1;
    auto* handle = found->second;
    if (handle->stats.get_bytes_counter().value != 0 ||
        rmm::mr::get_current_device_resource_ref() !=
            decltype(handle->previous){handle->stats}) return -1;
    rmm::mr::set_current_device_resource_ref(handle->previous);
    device_pools.erase(found);
    delete handle;
    return 0;
  } catch (...) { return -1; }
}

namespace {
struct Handle {
  mgbfs::CudfOwner owner;
  uint64_t epoch{0}, last_epoch{0};
  bool pending{false}, has_last{false}, poisoned{false};
  Handle(cudf::table_view history, int32_t capacity, rmm::cuda_stream_view stream)
      : owner(history, capacity, stream) {}
};

int32_t count(uint32_t rows) {
  if (rows > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
    throw std::runtime_error("OWNER_COUNT_BOUND");
  return static_cast<int32_t>(rows);
}
std::vector<cudf::column_view> columns(MgbfsLibraryKeysV1 input) {
  if (input.reserved) throw std::runtime_error("OWNER_RESERVED");
  auto rows = count(input.rows);
  std::vector<cudf::column_view> out;
  for (auto ptr : input.words) {
    if (rows && !ptr) throw std::runtime_error("OWNER_NULL_KEYS");
    out.emplace_back(cudf::data_type{cudf::type_id::UINT32}, rows, ptr, nullptr, 0, 0);
  }
  return out;
}
int fail(Handle* handle) {
  if (handle) handle->poisoned = true;
  return -1;
}
}

extern "C" int mgbfs_library_owner_create_v1(MgbfsLibraryKeysV1 history, uint32_t capacity,
                                             void* stream, void** owner) {
  if (!owner) return -1;
  *owner = nullptr;
  try {
    auto keys = columns(history);
    *owner = new Handle(cudf::table_view(keys), count(capacity),
                        rmm::cuda_stream_view{static_cast<cudaStream_t>(stream)});
    return 0;
  } catch (...) { return -1; }
}
extern "C" int mgbfs_library_owner_compare_v1(void* owner, uint64_t epoch,
    MgbfsLibraryCandidatesV1 input, MgbfsLibrarySurvivorsV1* result) {
  if (result) *result = {};
  auto* handle = static_cast<Handle*>(owner);
  if (!handle || !result) return fail(handle);
  try {
    if (handle->poisoned || handle->pending ||
        (handle->has_last && epoch <= handle->last_epoch)) return fail(handle);
    auto cols = columns(input.keys);
    if (input.keys.rows && !input.source_indices) return fail(handle);
    cols.emplace_back(cudf::data_type{cudf::type_id::UINT32}, count(input.keys.rows),
                      input.source_indices, nullptr, 0, 0);
    auto staged = handle->owner.compare(cudf::table_view(cols));
    result->source_indices = staged.column(4).data<uint32_t>();
    result->rows = static_cast<uint32_t>(staged.num_rows());
    result->epoch = epoch;
    handle->epoch = epoch;
    handle->pending = true;
    return 0;
  } catch (...) { return fail(handle); }
}
extern "C" int mgbfs_library_owner_commit_v1(void* owner, uint64_t epoch, uint32_t granted) {
  auto* handle = static_cast<Handle*>(owner);
  if (!handle) return -1;
  try {
    if (handle->poisoned || !handle->pending || epoch != handle->epoch) return fail(handle);
    handle->owner.commit(count(granted));
    handle->last_epoch = epoch;
    handle->has_last = true;
    handle->pending = false;
    return 0;
  } catch (...) { return fail(handle); }
}
extern "C" void mgbfs_library_owner_destroy_v1(void* owner) {
  delete static_cast<Handle*>(owner);
}
extern "C" int mgbfs_library_owner_export_v1(void* owner, MgbfsLibraryKeysV1* keys) {
  if (keys) *keys = {};
  auto* handle = static_cast<Handle*>(owner);
  if (!handle || !keys) return fail(handle);
  try {
    if (handle->poisoned || handle->pending) return fail(handle);
    auto view = handle->owner.export_committed();
    MgbfsLibraryKeysV1 result{};
    result.rows = static_cast<uint32_t>(view.num_rows());
    for (int c = 0; c < 4; ++c) result.words[c] = view.column(c).data<uint32_t>();
    *keys = result;
    return 0;
  } catch (...) { return fail(handle); }
}
