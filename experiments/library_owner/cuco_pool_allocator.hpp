#pragma once

#include <cstddef>
#include <exception>
#include <limits>
#include <new>
#include <utility>

namespace mgbfs {

// ResourceRef must refer to the admitted fixed pool and outlive this allocator,
// every container copy, and every queued access. There is no default resource.
// Production binding: ResourceRef=rmm::device_async_resource_ref,
// ResourceStream=rmm::cuda_stream_view. cuco supplies cuda::stream_ref.
template <class T, class ResourceRef, class ResourceStream>
class CucoPoolAllocator {
 public:
  using value_type = T;

  explicit CucoPoolAllocator(ResourceRef resource) : resource_(std::move(resource)) {}

  template <class Stream>
  T* allocate(std::size_t count, Stream stream) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length{};
    }
    return static_cast<T*>(resource_.allocate(ResourceStream{stream.get()}, count * sizeof(T)));
  }

  template <class Stream>
  void deallocate(T* pointer, std::size_t count, Stream stream) noexcept {
    // A mismatched/overflowed deallocation is a fatal ownership violation.
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) std::terminate();
    resource_.deallocate(ResourceStream{stream.get()}, pointer, count * sizeof(T));
  }

 private:
  ResourceRef resource_;
};

}  // namespace mgbfs
