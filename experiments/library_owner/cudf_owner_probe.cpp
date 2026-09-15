// Characterization gate for the proposed libcudf owner, NOT a BFS backend.
// Hand-checked fixture: preserve all 128 key bits and original source ordinal;
// reuse the history index; include pending next-layer keys in duplicate checks.
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <set>
#include "cudf_owner.hpp"
#include "owner_abi.h"

void check(bool ok, char const* msg) { if (!ok) throw std::runtime_error(msg); }
void cuda_check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

using Row = std::array<uint32_t, 5>;
std::unique_ptr<cudf::table> upload(std::vector<Row> const& rows,
                                  rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (int c = 0; c < 5; ++c) {
    auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT32},
        static_cast<cudf::size_type>(rows.size()), cudf::mask_state::UNALLOCATED, stream);
    std::vector<uint32_t> host;
    for (auto const& row : rows) host.push_back(row[c]);
    if (!host.empty()) {
      cuda_check(cudaMemcpyAsync(col->mutable_view().data<uint32_t>(), host.data(),
          host.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream.value()));
      // Fixture upload only: host vector must outlive the asynchronous copy.
      stream.synchronize();
    }
    columns.push_back(std::move(col));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table> probe(cudf::table_view incoming,
                                 cudf::filtered_join const& history,
                                 rmm::cuda_stream_view stream) {
  auto unique = cudf::stable_distinct(incoming, {0, 1, 2, 3},
      cudf::duplicate_keep_option::KEEP_FIRST, cudf::null_equality::EQUAL,
      cudf::nan_equality::ALL_EQUAL, stream);
  auto indices = history.anti_join(unique->view().select({0, 1, 2, 3}), stream);
  cudf::column_view map{cudf::data_type{cudf::type_id::INT32},
      static_cast<cudf::size_type>(indices->size()), indices->data(), nullptr, 0, 0, {}};
  return cudf::gather(unique->view(), map, cudf::out_of_bounds_policy::DONT_CHECK, stream);
}

void expect_indices(cudf::table_view result, std::vector<uint32_t> expected,
                    rmm::cuda_stream_view stream) {
  std::vector<uint32_t> actual(result.num_rows());
  if (!actual.empty()) cuda_check(cudaMemcpyAsync(actual.data(),
      result.column(4).data<uint32_t>(), actual.size() * sizeof(uint32_t),
      cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  std::sort(actual.begin(), actual.end());
  check(actual == expected, "Hash128 survivor/source-index mismatch");
}

void fixture(rmm::cuda_stream_view stream) {
  auto old = upload({Row{7, 8, 9, 10, 0}}, stream);
  cudf::filtered_join history(old->view().select({0, 1, 2, 3}),
      cudf::null_equality::EQUAL, cudf::set_as_build_table::RIGHT, stream);
  auto input = upload({Row{7,8,9,10,0}, Row{7,8,9,11,1}, Row{7,8,9,11,2},
                       Row{7,8,10,10,3}, Row{7,9,9,10,4}, Row{8,8,9,10,5}}, stream);
  auto survivors = probe(input->view(), history, stream);
  expect_indices(survivors->view(), {1,3,4,5}, stream);
  // Repeated probes must leave the cached history intact.
  auto again = probe(input->view(), history, stream);
  expect_indices(again->view(), {1,3,4,5}, stream);
  // Only after external capacity reservation may these keys be committed.
  // The test models that moment by retaining the owned survivor columns.
  cudf::filtered_join accepted(survivors->view().select({0,1,2,3}),
      cudf::null_equality::EQUAL, cudf::set_as_build_table::RIGHT, stream);
  auto after_next = probe(again->view(), accepted, stream);
  expect_indices(after_next->view(), {}, stream);
  auto empty = upload({}, stream);
  auto empty_result = probe(empty->view(), history, stream);
  expect_indices(empty_result->view(), {}, stream);
}

void owner_fixture(rmm::cuda_stream_view stream) {
  auto old = upload({Row{7,8,9,10,0}}, stream);
  auto input = upload({Row{7,8,9,10,0}, Row{7,8,9,11,1}, Row{7,8,9,11,2},
                       Row{8,8,9,10,3}}, stream);
  mgbfs::CudfOwner owner(old->view().select({0,1,2,3}), 2, stream);
  expect_indices(owner.compare(input->view()), {1,3}, stream);
  check(owner.accepted_count() == 0, "Compare published accepted keys");
  owner.commit(2);
  check(owner.accepted_count() == 2, "Commit did not append keys");
  expect_indices(owner.compare(input->view()), {}, stream);
  owner.commit(0);
  check(owner.accepted_count() == 2, "Empty commit changed accepted keys");

  mgbfs::CudfOwner denied(old->view().select({0,1,2,3}), 2, stream);
  expect_indices(denied.compare(input->view()), {1,3}, stream);
  bool rejected = false;
  try { denied.commit(1); } catch (std::runtime_error const&) { rejected = true; }
  check(rejected && denied.accepted_count() == 0, "Insufficient credit published keys");
  rejected = false;
  try { denied.commit(2); } catch (std::runtime_error const&) { rejected = true; }
  check(rejected, "Poisoned owner permitted retry");
}

void owner_multibatch_fixture(rmm::cuda_stream_view stream) {
  // Independent host membership oracle; GPU backend never reads this set.
  std::set<uint32_t> visited;
  std::vector<Row> history;
  auto key = [](uint32_t id, uint32_t origin) {
    return Row{id % 97, id / 97, 0xdeadbeefU, 0x12345678U, origin};
  };
  for (uint32_t id = 0; id < 4096; ++id) {
    visited.insert(id);
    history.push_back(key(id, id));
  }
  auto old = upload(history, stream);
  mgbfs::CudfOwner owner(old->view().select({0,1,2,3}), 16384, stream);
  for (uint32_t batch = 0; batch < 48; ++batch) {
    std::vector<Row> incoming;
    std::vector<uint32_t> expected;
    std::set<uint32_t> unique;
    for (uint32_t i = 0; i < 1024; ++i) {
      // Every adjacent pair is a duplicate; batches overlap both old and next.
      auto id = (batch * 521 + (i / 2) * 17) % 20000;
      incoming.push_back(key(id, i));
      if (!visited.count(id) && unique.insert(id).second) expected.push_back(i);
    }
    auto input = upload(incoming, stream);
    auto result = owner.compare(input->view());
    check(owner.accepted_count() == static_cast<int>(visited.size() - 4096),
          "Compare changed cumulative accepted count");
    expect_indices(result, expected, stream);
    owner.commit(static_cast<cudf::size_type>(expected.size()));
    visited.insert(unique.begin(), unique.end());
    check(owner.accepted_count() == static_cast<int>(visited.size() - 4096),
          "Multi-batch cumulative accepted mismatch");
  }
  // Ensure this exercised a populated index, not an all-old or empty fixture.
  check(visited.size() > 16000, "Multi-batch fixture has insufficient coverage");
  stream.synchronize();
}

void abi_fixture(rmm::cuda_stream_view stream) {
  auto old = upload({Row{7,8,9,10,0}}, stream);
  auto input = upload({Row{7,8,9,10,0}, Row{7,8,9,11,1}, Row{7,8,9,11,2}}, stream);
  auto keys = [](cudf::table_view t) {
    MgbfsLibraryKeysV1 k{};
    k.rows = static_cast<uint32_t>(t.num_rows());
    for (int i = 0; i < 4; ++i) k.words[i] = t.column(i).data<uint32_t>();
    return k;
  };
  void* owner = nullptr;
  check(mgbfs_library_owner_create_v1(keys(old->view()), 1, stream.value(), &owner) == 0,
        "OWNER_ABI_CREATE");
  // Drain even after a failed assertion before releasing external history/pool.
  try {
    MgbfsLibraryCandidatesV1 candidates{keys(input->view()),
        input->view().column(4).data<uint32_t>()};
    MgbfsLibrarySurvivorsV1 out{};
    check(mgbfs_library_owner_compare_v1(owner, 5, candidates, &out) == 0,
          "OWNER_ABI_COMPARE");
    check(out.rows == 1 && out.epoch == 5 && out.reserved == 0, "OWNER_ABI_RESULT");
    uint32_t origin = 999;
    cuda_check(cudaMemcpyAsync(&origin, out.source_indices, sizeof(origin),
                              cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();
    check(origin == 1, "OWNER_ABI_PROVENANCE");
    check(mgbfs_library_owner_commit_v1(owner, 5, 1) == 0, "OWNER_ABI_COMMIT");
    MgbfsLibraryKeysV1 committed{};
    check(mgbfs_library_owner_export_v1(owner, &committed) == 0,
          "OWNER_ABI_EXPORT");
    check(committed.rows == 1 && committed.reserved == 0, "OWNER_ABI_EXPORT_COUNT");
    std::array<uint32_t, 4> exported{};
    for (int c = 0; c < 4; ++c)
      cuda_check(cudaMemcpyAsync(&exported[c], committed.words[c], sizeof(uint32_t),
                                cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();
    check(exported == std::array<uint32_t, 4>{7,8,9,11}, "OWNER_ABI_EXPORT_KEY");
    check(mgbfs_library_owner_compare_v1(owner, 6, candidates, &out) == 0,
          "OWNER_ABI_NEXT_COMPARE");
    check(out.rows == 0, "OWNER_ABI_REACCEPTED_DUPLICATE");
    check(mgbfs_library_owner_commit_v1(owner, 5, 0) != 0, "OWNER_ABI_STALE_EPOCH");
    check(mgbfs_library_owner_commit_v1(owner, 6, 0) != 0, "OWNER_ABI_POISON_RETRY");
    check(mgbfs_library_owner_export_v1(owner, &committed) != 0 &&
          committed.rows == 0 && committed.reserved == 0 &&
          std::all_of(std::begin(committed.words), std::end(committed.words),
                      [](auto p) { return p == nullptr; }), "OWNER_ABI_POISON_EXPORT");
    stream.synchronize();
  } catch (...) {
    stream.synchronize();
    mgbfs_library_owner_destroy_v1(owner);
    throw;
  }
  mgbfs_library_owner_destroy_v1(owner);
  owner = nullptr;
  check(mgbfs_library_owner_create_v1(keys(old->view()), 1, stream.value(), &owner) == 0,
        "OWNER_ABI_PENDING_CREATE");
  try {
    MgbfsLibraryCandidatesV1 candidates{keys(input->view()),
        input->view().column(4).data<uint32_t>()};
    MgbfsLibrarySurvivorsV1 out{};
    check(mgbfs_library_owner_compare_v1(owner, 1, candidates, &out) == 0,
          "OWNER_ABI_PENDING_COMPARE");
    MgbfsLibraryKeysV1 committed{};
    check(mgbfs_library_owner_export_v1(owner, &committed) != 0 && committed.rows == 0,
          "OWNER_ABI_EXPORTED_UNCOMMITTED");
    stream.synchronize();
  } catch (...) {
    stream.synchronize();
    mgbfs_library_owner_destroy_v1(owner);
    throw;
  }
  mgbfs_library_owner_destroy_v1(owner);
}

void pool_abi_fixture(rmm::cuda_stream_view stream) {
  constexpr uint64_t bytes = 64ULL << 20;
  auto previous = rmm::mr::get_current_device_resource_ref();
  void* pool = nullptr;
  check(mgbfs_library_pool_create_v1(bytes, 1ULL << 30, &pool) == 0 && pool,
        "POOL_ABI_CREATE");
  try {
    void* nested = nullptr;
    check(mgbfs_library_pool_create_v1(bytes, 1ULL << 30, &nested) != 0 && !nested,
          "POOL_ABI_NESTED");
    {
      rmm::device_buffer live(256, stream);
      check(mgbfs_library_pool_destroy_v1(pool) != 0, "POOL_ABI_FREED_LIVE_STORAGE");
      cuda_check(cudaMemsetAsync(live.data(), 0x5a, live.size(), stream.value()));
      stream.synchronize();
    }
    stream.synchronize();
    bool exhausted = false;
    try { rmm::device_buffer impossible(bytes + 256, stream); }
    catch (rmm::out_of_memory const&) { exhausted = true; }
    check(exhausted, "POOL_ABI_GREW");
    // Exercise a real owner while this ABI-installed resource is active.
    abi_fixture(stream);
    stream.synchronize();
  } catch (...) {
    stream.synchronize();
    mgbfs_library_pool_destroy_v1(pool);
    throw;
  }
  check(mgbfs_library_pool_destroy_v1(pool) == 0, "POOL_ABI_DESTROY");
  check(rmm::mr::get_current_device_resource_ref() == previous, "POOL_ABI_RESTORE");
  for (uint64_t invalid : {uint64_t{0}, bytes + 1}) {
    pool = nullptr;
    check(mgbfs_library_pool_create_v1(invalid, 1ULL << 30, &pool) != 0 && !pool,
          "POOL_ABI_INVALID_CAPACITY");
  }
  check(mgbfs_library_pool_create_v1(bytes, 0, &pool) != 0 && !pool,
        "POOL_ABI_MISSING_RESERVE");
}

void layout_fixture(rmm::cuda_stream_view stream) {
  constexpr uint32_t rows = 65;
  constexpr size_t stride = 512, bytes = 5 * stride;
  std::vector<uint32_t> original(rows * 4);
  for (uint32_t i = 0; i < rows; ++i)
    for (uint32_t c = 0; c < 4; ++c) original[i * 4 + c] = 0xff000000u + i * 19 + c;
  rmm::device_buffer input(original.data(), original.size() * 4, stream);
  rmm::device_buffer scratch(bytes + 256, stream), output(original.size() * 4, stream);
  cuda_check(cudaMemsetAsync(scratch.data(), 0xa5, scratch.size(), stream.value()));
  MgbfsLibraryCandidatesV1 converted{};
  check(mgbfs_library_candidates_from_aos_v1(input.data(), rows, rows, scratch.data(),
        bytes, stream.value(), &converted) == 0, "LAYOUT_AOS_TO_SOA");
  check(converted.keys.rows == rows && converted.keys.reserved == 0, "LAYOUT_METADATA");
  for (size_t c = 0; c < 4; ++c)
    check(reinterpret_cast<const char*>(converted.keys.words[c]) ==
          static_cast<const char*>(scratch.data()) + c * stride, "LAYOUT_PLANE_OFFSET");
  check(reinterpret_cast<const char*>(converted.source_indices) ==
        static_cast<const char*>(scratch.data()) + 4 * stride, "LAYOUT_ORDINAL_OFFSET");
  check(mgbfs_library_keys_to_aos_v1(converted.keys, output.data(), rows,
        stream.value()) == 0, "LAYOUT_SOA_TO_AOS");
  std::vector<uint32_t> planes((bytes + 256) / 4), roundtrip(original.size());
  cuda_check(cudaMemcpyAsync(planes.data(), scratch.data(), scratch.size(),
                            cudaMemcpyDeviceToHost, stream.value()));
  cuda_check(cudaMemcpyAsync(roundtrip.data(), output.data(), output.size(),
                            cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  check(roundtrip == original, "LAYOUT_ROUNDTRIP");
  for (size_t c = 0; c < 5; ++c) {
    for (uint32_t i = 0; i < rows; ++i)
      check(planes[c * stride / 4 + i] == (c == 4 ? i : original[i * 4 + c]),
            "LAYOUT_PLANE_VALUE");
    for (size_t i = rows; i < stride / 4; ++i)
      check(planes[c * stride / 4 + i] == 0xa5a5a5a5, "LAYOUT_PADDING_WRITE");
  }
  for (size_t i = bytes / 4; i < planes.size(); ++i)
    check(planes[i] == 0xa5a5a5a5, "LAYOUT_GUARD_WRITE");
  check(mgbfs_library_candidates_from_aos_v1(input.data(), rows, rows, scratch.data(),
        bytes - 1, stream.value(), &converted) != 0 && converted.keys.rows == 0 &&
        converted.source_indices == nullptr, "LAYOUT_SHORT_BUFFER");
}

int main() {
  try {
    cuda_check(cudaSetDevice(0));
    rmm::cuda_stream stream;
    pool_abi_fixture(stream.view());
    constexpr std::size_t bytes = 64ULL << 20;
    std::size_t free = 0, total = 0;
    cuda_check(cudaMemGetInfo(&free, &total));
    check(free >= bytes + (1ULL << 30), "Insufficient free VRAM/reserve");
    rmm::mr::cuda_memory_resource upstream;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool(&upstream, bytes, bytes);
    rmm::mr::statistics_resource_adaptor<decltype(pool)> stats(&pool);
    auto previous = rmm::mr::set_current_device_resource_ref(stats);
    try {
      fixture(stream.view());
      layout_fixture(stream.view());
      owner_fixture(stream.view());
      owner_multibatch_fixture(stream.view());
      abi_fixture(stream.view());
      stream.synchronize();
      check(stats.get_bytes_counter().value == 0, "Library allocation escaped fixture lifetime");
      bool exhausted = false;
      try { rmm::device_buffer impossible(bytes + 256, stream.view()); }
      catch (rmm::out_of_memory const&) { exhausted = true; }
      check(exhausted, "Fixed pool unexpectedly grew");
      check(pool.pool_size() == bytes, "Reserved pool size changed");
      std::cout << "{\"status\":\"PASS\",\"kind\":\"library_characterization\","
                << "\"pool_reserved_bytes\":" << pool.pool_size()
                << ",\"suballocation_peak_bytes\":" << stats.get_bytes_counter().peak << "}\n";
    } catch (...) {
      rmm::mr::set_current_device_resource_ref(previous);
      throw;
    }
    rmm::mr::set_current_device_resource_ref(previous);
    return 0;
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
