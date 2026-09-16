#include "../experiments/library_owner/cuco_index.hpp"
#include <cassert>
#include <cstdint>

int main() {
  // Row 0 repeated in another storage; rows 1..4 differ in exactly one word.
  std::uint32_t planes[4][5] = {{11,12,11,11,11}, {22,22,23,22,22},
                               {33,33,33,34,33}, {44,44,44,44,45}};
  mgbfs::IndexKeyViews views{};
  for (int storage = 0; storage < 4; ++storage)
    for (int word = 0; word < 4; ++word) views.planes[storage][word] = planes[word];
  mgbfs::IndexKeyEqual equal{views};
  mgbfs::IndexKeyHasher hash{views};
  auto first = mgbfs::key_index(0, 0);
  for (unsigned storage = 0; storage < 4; ++storage) {
    auto alias = mgbfs::key_index(storage, 0);
    assert(equal(first, alias));
    assert(hash(first) == hash(alias));
    for (unsigned row = 1; row < 5; ++row)
      assert(!equal(first, mgbfs::key_index(storage, row)));
  }
  // An empty slot must not dereference the sentinel as an array index.
  assert(equal(mgbfs::empty_index, mgbfs::empty_index));
  assert(!equal(first, mgbfs::empty_index));
  assert(!equal(mgbfs::empty_index, first));
  mgbfs::IndexKeyEqual empty{};
  assert(empty(mgbfs::empty_index, mgbfs::empty_index));
}
