#pragma once
#include <cstdint>

#if defined(__CUDACC__)
#define MGBFS_CUCO_HD __host__ __device__
#else
#define MGBFS_CUCO_HD
#endif

namespace mgbfs {
inline constexpr std::uint64_t empty_index = UINT64_MAX;
// Callers validate storage < 4 and row <= INT32_MAX before publishing indices.
// 0/1/2/3 address previous/current/accepted/candidate immutable SoA views.
MGBFS_CUCO_HD constexpr std::uint64_t key_index(unsigned storage, std::uint32_t row) {
  return (std::uint64_t{storage} << 62) | row;
}
struct IndexKeyViews {
  std::uint32_t const* planes[4][4]{};
  MGBFS_CUCO_HD std::uint32_t word(std::uint64_t index, unsigned component) const {
    return planes[index >> 62][component][index & ((std::uint64_t{1} << 62) - 1)];
  }
};
struct IndexKeyEqual {
  IndexKeyViews keys;
  MGBFS_CUCO_HD bool operator()(std::uint64_t lhs, std::uint64_t rhs) const {
    if (lhs == empty_index || rhs == empty_index) return lhs == rhs;
    for (unsigned word = 0; word < 4; ++word)
      if (keys.word(lhs, word) != keys.word(rhs, word)) return false;
    return true;
  }
};
struct IndexKeyHasher {
  IndexKeyViews keys;
  MGBFS_CUCO_HD std::uint64_t operator()(std::uint64_t index) const {
    // Table placement hash only. Identity is still equality of all 128 bits.
    if (index == empty_index) return 0;
    std::uint64_t value = 0xcbf29ce484222325ULL;
    for (unsigned word = 0; word < 4; ++word) {
      value ^= keys.word(index, word);
      value *= 0x100000001b3ULL;
    }
    value ^= value >> 32;
    value *= 0xd6e8feb86659fd93ULL;
    return value ^ (value >> 32);
  }
};
}  // namespace mgbfs
#undef MGBFS_CUCO_HD
