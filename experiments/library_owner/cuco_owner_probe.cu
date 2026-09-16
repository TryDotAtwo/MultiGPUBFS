// Real indexed owner contract: stable committed keys, KEEP_FIRST and fail-fast.
#include "cuco_owner.cuh"
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <array>
#include <vector>
#include <iostream>
#include <stdexcept>

namespace {
using Key = std::array<uint32_t, 4>;
void require(bool value, char const* message) {
  if (!value) throw std::runtime_error(message);
}
void check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
template<class F> void rejects(F function, char const* message) {
  bool rejected = false;
  try { function(); } catch (std::runtime_error const&) { rejected = true; }
  require(rejected, message);
}
struct Input {
  rmm::device_buffer words, sources;
  rmm::cuda_stream_view stream;
  uint32_t capacity, rows{0};
  Input(uint32_t size, rmm::cuda_stream_view s, rmm::device_async_resource_ref resource)
      : words(size * 16, s, resource), sources(size * 4, s, resource), stream(s), capacity(size) {}
  void upload(std::vector<Key> const& keys, std::vector<uint32_t> const& ids) {
    require(keys.size() == ids.size() && keys.size() <= capacity, "FIXTURE_INPUT");
    rows = static_cast<uint32_t>(keys.size());
    std::vector<uint32_t> plane(rows);
    for (unsigned c = 0; c < 4; ++c) {
      for (unsigned row = 0; row < rows; ++row) plane[row] = keys[row][c];
      check(cudaMemcpyAsync(static_cast<uint32_t*>(words.data()) + c * capacity,
          plane.data(), rows * 4, cudaMemcpyHostToDevice, stream.value()));
      stream.synchronize();
    }
    check(cudaMemcpyAsync(sources.data(), ids.data(), rows * 4,
        cudaMemcpyHostToDevice, stream.value()));
    stream.synchronize();
  }
  MgbfsLibraryKeysV1 keys() const {
    MgbfsLibraryKeysV1 view{};
    view.rows = rows;
    for (unsigned c = 0; c < 4; ++c)
      view.words[c] = static_cast<uint32_t const*>(words.data()) + c * capacity;
    return view;
  }
  MgbfsLibraryCandidatesV1 candidates() const {
    return {keys(), static_cast<uint32_t const*>(sources.data())};
  }
};
std::vector<uint32_t> read(MgbfsLibrarySurvivorsV1 result, rmm::cuda_stream_view stream) {
  std::vector<uint32_t> host(result.rows);
  if (result.rows) check(cudaMemcpyAsync(host.data(), result.source_indices,
      result.rows * 4, cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  return host;
}
std::vector<Key> read_keys(MgbfsLibraryKeysV1 keys, rmm::cuda_stream_view stream) {
  std::vector<Key> host(keys.rows);
  std::vector<uint32_t> plane(keys.rows);
  for (unsigned c = 0; c < 4; ++c) {
    if (keys.rows) check(cudaMemcpyAsync(plane.data(), keys.words[c], keys.rows * 4,
        cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();
    for (unsigned i = 0; i < keys.rows; ++i) host[i][c] = plane[i];
  }
  return host;
}
}
int main() {
  rmm::cuda_stream stream;
  constexpr size_t pool_bytes = 64 << 20;
  rmm::mr::cuda_memory_resource upstream;
  rmm::mr::pool_memory_resource pool{&upstream, pool_bytes, pool_bytes};
  rmm::mr::statistics_resource_adaptor stats{&pool};
  rmm::device_async_resource_ref resource{stats};
  {
    Input previous(1, stream.view(), resource), current(1, stream.view(), resource);
    Input input(16, stream.view(), resource);
    Key a{10,20,30,40}, b{11,20,30,40}, c{10,21,30,40};
    Key d{10,20,31,40}, e{10,20,30,41}, f{99,88,77,66};
    previous.upload({a}, {0});
    current.upload({b}, {0});
    {
      mgbfs::CucoOwner owner(previous.keys(), current.keys(), 8, 16, stream.view(), resource);
      stream.synchronize();
      auto const allocated = stats.get_bytes_counter().value;
      auto const peak = stats.get_bytes_counter().peak;
      input.upload({a,c,c,d,b,e}, {500,90,1,80,3,70});
      auto first = owner.compare(1, input.candidates());
      require(read(first, stream.view()) == std::vector<uint32_t>{90,80,70}, "KEEP_FIRST_ROW");
      owner.commit(1, 3);
      stream.synchronize();
      // Recycle the exact same caller slot. A persistent candidate index would
      // now change identity and lose C from membership.
      input.upload({c,f,f}, {4,55,2});
      require(read(owner.compare(2, input.candidates()), stream.view()) ==
              std::vector<uint32_t>{55}, "COMMITTED_KEYS_SURVIVE_SLOT_REUSE");
      owner.commit(2, 1);
      stream.synchronize();
      input.upload({a,b,c,d,e,f}, {0,1,2,3,4,5});
      require(owner.compare(3, input.candidates()).rows == 0, "ALL_HISTORY_WINDOWS");
      owner.commit(3, 0);
      stream.synchronize();
      require(read_keys(owner.export_committed(), stream.view()) ==
              std::vector<Key>{c,d,e,f}, "APPEND_ORDER_KEYS");
      require(stats.get_bytes_counter().value == allocated &&
              stats.get_bytes_counter().peak == peak, "NO_STEADY_STATE_ALLOCATION");
      owner.seal();
      previous.upload({f}, {0});
      current.upload({f}, {0});
      require(read_keys(owner.export_committed(), stream.view()) ==
              std::vector<Key>{c,d,e,f}, "SEALED_EXPORT_OWNS_KEYS");
    }
    {
      mgbfs::CucoOwner owner({}, {}, 1, 16, stream.view(), resource);
      input.upload({a,b}, {1,2});
      rejects([&] { owner.compare(1, input.candidates()); }, "ACCEPTED_CAPACITY");
      rejects([&] { owner.export_committed(); }, "CAPACITY_POISONS");
      stream.synchronize();
    }
    {
      mgbfs::CucoOwner owner({}, {}, 8, 16, stream.view(), resource);
      input.upload({a,b}, {1,2});
      owner.compare(1, input.candidates());
      rejects([&] { owner.commit(1, 1); }, "REAL_COMMIT_CREDITS");
      rejects([&] { owner.compare(2, input.candidates()); }, "CREDIT_POISONS");
      stream.synchronize();
    }
    {
      mgbfs::CucoOwner owner({}, {}, 8, 1, stream.view(), resource);
      input.upload({a,b}, {1,2});
      rejects([&] { owner.compare(1, input.candidates()); }, "INPUT_CAPACITY");
      stream.synchronize();
    }
    {
      mgbfs::CucoOwner owner({}, {}, 8, 16, stream.view(), resource);
      require(owner.compare(9, {}).rows == 0, "EMPTY_INPUT");
      owner.commit(9, 0);
      rejects([&] { owner.compare(9, {}); }, "EPOCH_REPLAY");
      stream.synchronize();
    }
  }
  stream.synchronize();
  require(stats.get_bytes_counter().value == 0, "OWNER_POOL_LEAK");
  require(pool.pool_size() == pool_bytes, "OWNER_POOL_GROWTH");
  std::cout << "{\"status\":\"PASS\",\"scope\":\"cuco_indexed_owner_contract\"}\n";
}
