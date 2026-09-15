#include "owner_abi.h"
#include "cudf_owner.hpp"
#include <limits>
#include <vector>

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
