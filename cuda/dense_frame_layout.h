#pragma once
#include <stdint.h>

struct MgbfsDenseFrameLayout {
  uint64_t hash_bytes, ordinal_offset, ordinal_bytes;
  uint64_t state_offset, state_bytes, total_bytes;
};

// CPU-side enqueue geometry; no allocation or CUDA calls.
inline int mgbfs_dense_frame_layout(uint32_t count, uint32_t stride,
                                   MgbfsDenseFrameLayout* out) {
  if (!out) return 1;
  *out = {};
  if (!stride || stride % 16) return 1;
  MgbfsDenseFrameLayout x{};
  x.hash_bytes = uint64_t(count) * 16;
  x.ordinal_bytes = uint64_t(count) * 4;
  x.state_bytes = uint64_t(count) * stride;
  x.ordinal_offset = (x.hash_bytes + 255) & ~uint64_t(255);
  const uint64_t ord_reserved = (x.ordinal_bytes + 255) & ~uint64_t(255);
  x.state_offset = x.ordinal_offset + ord_reserved;
  if (x.state_bytes > UINT64_MAX - 255) return 1;
  const uint64_t state_reserved = (x.state_bytes + 255) & ~uint64_t(255);
  if (state_reserved > UINT64_MAX - x.state_offset) return 1;
  x.total_bytes = x.state_offset + state_reserved;
  *out = x;
  return 0;
}

#ifdef __CUDACC__
#define MGBFS_FRAME_HD __host__ __device__
#else
#define MGBFS_FRAME_HD
#endif

// Shared scalar specification used by the CUDA gather and the CPU byte oracle.
MGBFS_FRAME_HD inline bool mgbfs_dense_frame_word(
    const MgbfsDenseFrameLayout& layout, uint32_t stride, uint32_t begin,
    uint32_t source_count, const uint32_t* hashes, const uint64_t* refs,
    const uint32_t* states, uint64_t word, uint32_t* value) {
  if (word >= layout.total_bytes / 4) return false;
  const uint64_t byte = word * 4;
  if (byte < layout.hash_bytes) {
    *value = hashes[uint64_t(begin) * 4 + word];
  } else if (byte >= layout.ordinal_offset &&
             byte < layout.ordinal_offset + layout.ordinal_bytes) {
    const uint64_t ref = refs[uint64_t(begin) + (byte - layout.ordinal_offset) / 4];
    if (ref >= source_count) return false;
    *value = uint32_t(ref);
  } else if (byte >= layout.state_offset &&
             byte < layout.state_offset + layout.state_bytes) {
    const uint64_t state_word = (byte - layout.state_offset) / 4;
    const uint32_t width = stride / 4;
    const uint64_t ref = refs[uint64_t(begin) + state_word / width];
    if (ref >= source_count) return false;
    *value = states[ref * width + state_word % width];
  } else {
    *value = 0;
  }
  return true;
}
MGBFS_FRAME_HD inline uint32_t mgbfs_frame_prefix_word(const uint32_t* header, uint32_t word) {
  return word < 16 ? header[word] : 0;
}
#undef MGBFS_FRAME_HD
