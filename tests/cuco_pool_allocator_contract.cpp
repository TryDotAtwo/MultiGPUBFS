#include "../experiments/library_owner/cuco_pool_allocator.hpp"
#include <cassert>
#include <cstdint>
#include <limits>
#include <new>

struct Stream { int id; int get() const { return id; } };
struct View { int id; explicit View(int value) : id(value) {} };
struct Trace {
  int stream = -1;
  std::size_t bytes = 0;
  int allocations = 0;
  int releases = 0;
  bool full = false;
  alignas(256) unsigned char storage[256];
};
struct Resource {
  Trace* trace;
  void* allocate(View stream, std::size_t bytes) {
    ++trace->allocations;
    if (trace->full) throw std::bad_alloc{};
    trace->stream = stream.id;
    trace->bytes = bytes;
    return trace->storage;
  }
  void deallocate(View stream, void* pointer, std::size_t bytes) noexcept {
    assert(pointer == trace->storage);
    trace->stream = stream.id;
    trace->bytes = bytes;
    ++trace->releases;
  }
};

int main() {
  Trace first{}, other{};
  using Alloc = mgbfs::CucoPoolAllocator<std::uint64_t, Resource, View>;
  Resource selected{&first};
  Alloc allocator{selected};
  selected = Resource{&other}; // Resource selection must be captured, not looked up later.
  auto* pointer = allocator.allocate(3, Stream{17});
  assert(first.allocations == 1 && other.allocations == 0);
  assert(first.bytes == 24 && first.stream == 17);
  auto copied = allocator;
  copied.deallocate(pointer, 3, Stream{19});
  assert(first.releases == 1 && first.bytes == 24 && first.stream == 19);
  mgbfs::CucoPoolAllocator<char, Resource, View> rebound{allocator};
  auto* scratch = rebound.allocate(13, Stream{23});
  assert(first.bytes == 13 && first.stream == 23 && other.allocations == 0);
  rebound.deallocate(scratch, 13, Stream{23});
  try {
    allocator.allocate(std::numeric_limits<std::size_t>::max(), Stream{17});
    return 1;
  } catch (std::bad_array_new_length const&) {}
  assert(first.allocations == 2);
  first.full = true;
  try { allocator.allocate(1, Stream{17}); return 2; }
  catch (std::bad_alloc const&) {}
  assert(first.allocations == 3 && other.allocations == 0);
}
