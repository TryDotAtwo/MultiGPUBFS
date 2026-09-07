#include "dense_frame_layout.h"
#include <cassert>
#include <cstdio>

int main() {
  uint32_t header[16];
  for (uint32_t i = 0; i < 16; ++i) header[i] = i + 5;
  for (uint32_t i = 0; i < 64; ++i)
    assert(mgbfs_frame_prefix_word(header, i) == (i < 16 ? i + 5 : 0));
  {
    MgbfsDenseFrameLayout layout{};
    assert(mgbfs_dense_frame_layout(3, 16, &layout) == 0);
    const uint32_t hashes[] = {99,99,99,99, 1,2,3,4, 5,6,7,8, 9,10,11,12};
    const uint64_t refs[] = {99, 3, 0, 2};
    const uint32_t states[] = {10,11,12,13, 20,21,22,23, 30,31,32,33, 40,41,42,43};
    const uint32_t expected_states[] = {40,41,42,43, 10,11,12,13, 30,31,32,33};
    for (uint64_t word = 0; word < 192; ++word) {
      uint32_t value = UINT32_MAX;
      assert(mgbfs_dense_frame_word(layout, 16, 1, 4, hashes, refs, states, word, &value));
      const uint32_t expected = word < 12 ? uint32_t(word + 1)
          : word >= 64 && word < 67 ? uint32_t(refs[word - 63])
          : word >= 128 && word < 140 ? expected_states[word - 128] : 0;
      assert(value == expected);
    }
    uint32_t value = 0;
    assert(!mgbfs_dense_frame_word(layout, 16, 1, 4, hashes, refs, states, 192, &value));
    assert(!mgbfs_dense_frame_word(layout, 16, 0, 4, hashes, refs, states, 64, &value));
    assert(!mgbfs_dense_frame_word(layout, 16, 0, 4, hashes, refs, states, 128, &value));
  }
  MgbfsDenseFrameLayout x{};
  assert(mgbfs_dense_frame_layout(17, 32, &x) == 0);
  assert(x.hash_bytes == 272 && x.ordinal_offset == 512);
  assert(x.ordinal_bytes == 68 && x.state_offset == 768);
  assert(x.state_bytes == 544 && x.total_bytes == 1536);
  assert(mgbfs_dense_frame_layout(0, 16, &x) == 0);
  assert(x.hash_bytes == 0 && x.ordinal_offset == 0 && x.ordinal_bytes == 0);
  assert(x.state_offset == 0 && x.state_bytes == 0 && x.total_bytes == 0);
  assert(mgbfs_dense_frame_layout(1, 16, &x) == 0);
  assert(x.total_bytes == 768 && x.state_offset == 512);
  assert(mgbfs_dense_frame_layout(1, 0, &x) != 0);
  assert(mgbfs_dense_frame_layout(1, 17, &x) != 0);
  assert(mgbfs_dense_frame_layout(UINT32_MAX, UINT32_MAX - 15, &x) != 0);
  assert(x.total_bytes == 0);
  assert(mgbfs_dense_frame_layout(1, 16, nullptr) != 0);
  std::puts("DENSE_FRAME_LAYOUT_PASS");
}
