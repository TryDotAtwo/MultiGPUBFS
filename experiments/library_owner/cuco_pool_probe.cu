// Actual cuco/RMM binding gate, not an indexed Hash128 owner or speed result.
#include "cuco_pool_allocator.hpp"
#include "cuco_index.hpp"
#include <cuco/static_set.cuh>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <cuda/std/functional>
#include <cstdint>
#include <iostream>
#include <stdexcept>

void require(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}
int main() {
  constexpr std::size_t bytes = 64 << 20;
  rmm::cuda_stream stream;
  rmm::mr::cuda_memory_resource upstream;
  rmm::mr::pool_memory_resource pool{&upstream, bytes, bytes};
  rmm::mr::statistics_resource_adaptor stats{&pool};
  rmm::device_async_resource_ref resource{stats};
  using Allocator = mgbfs::CucoPoolAllocator<std::uint64_t,
      rmm::device_async_resource_ref, rmm::cuda_stream_view>;
  {
    Allocator allocator{resource};
    cuda::stream_ref cu_stream{stream.value()};
    auto set = cuco::static_set{cuco::extent<std::size_t>{128},
        cuco::empty_key<std::uint64_t>{UINT64_MAX},
        cuda::std::equal_to<std::uint64_t>{},
        cuco::linear_probing<1, cuco::default_hash_function<std::uint64_t>>{},
        cuco::cuda_thread_scope<cuda::thread_scope_device>{},
        cuco::storage<1>{}, allocator, cu_stream};
    std::uint64_t host[] = {0, 7, 7, 9, 0};
    rmm::device_buffer input{host, sizeof(host), stream.view(), resource};
    auto* keys = static_cast<std::uint64_t*>(input.data());
    require(set.insert(keys, keys + 5, cu_stream) == 3, "CUCO_INSERT_COUNT");
    require(set.insert(keys, keys + 5, cu_stream) == 0, "CUCO_DUPLICATE_COUNT");
    require(set.size(cu_stream) == 3, "CUCO_SIZE");
    set.clear(cu_stream);
    require(set.size(cu_stream) == 0, "CUCO_CLEAR");
    require(set.insert(keys, keys + 5, cu_stream) == 3, "CUCO_REUSE");
    stream.synchronize();
  }
  {
    std::uint32_t words[4][5] = {{11,12,11,11,11}, {22,22,23,22,22},
                                 {33,33,33,34,33}, {44,44,44,44,45}};
    rmm::device_buffer data{words, sizeof(words), stream.view(), resource};
    auto* base = static_cast<std::uint32_t*>(data.data());
    mgbfs::IndexKeyViews views{};
    for (unsigned storage = 0; storage < 4; ++storage)
      for (unsigned word = 0; word < 4; ++word)
        views.planes[storage][word] = base + word * 5;
    auto indexed = cuco::static_set{cuco::extent<std::size_t>{128},
        cuco::empty_key<std::uint64_t>{mgbfs::empty_index},
        mgbfs::IndexKeyEqual{views},
        cuco::linear_probing<1, mgbfs::IndexKeyHasher>{mgbfs::IndexKeyHasher{views}},
        cuco::cuda_thread_scope<cuda::thread_scope_device>{},
        cuco::storage<1>{}, Allocator{resource}, cuda::stream_ref{stream.value()}};
    std::uint64_t indices[20];
    for (unsigned storage = 0; storage < 4; ++storage)
      for (unsigned row = 0; row < 5; ++row)
        indices[storage * 5 + row] = mgbfs::key_index(storage, row);
    rmm::device_buffer input{indices, sizeof(indices), stream.view(), resource};
    auto* keys = static_cast<std::uint64_t*>(input.data());
    cuda::stream_ref cu_stream{stream.value()};
    require(indexed.insert(keys, keys + 20, cu_stream) == 5, "CUCO_HASH128_IDENTITY");
    require(indexed.insert(keys, keys + 20, cu_stream) == 0, "CUCO_HASH128_REPROBE");
    require(indexed.size(cu_stream) == 5, "CUCO_HASH128_SIZE");
    stream.synchronize();
  }
  stream.synchronize();
  require(stats.get_bytes_counter().value == 0, "CUCO_POOL_LEAK");
  require(pool.pool_size() == bytes, "CUCO_POOL_GROWTH");
  std::cout << "{\"status\":\"PASS\",\"scope\":\"cuco_pool_binding_only\","
            << "\"pool_reserved_bytes\":" << bytes << ",\"suballocation_peak_bytes\":"
            << stats.get_bytes_counter().peak << "}\n";
}
