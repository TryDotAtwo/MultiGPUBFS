#pragma once
#include <stdint.h>
#ifdef __CUDACC__
#define MGBFS_INDEX_HD __host__ __device__
#else
#define MGBFS_INDEX_HD
#endif

template<bool Packed>
MGBFS_INDEX_HD inline uint64_t mgbfs_state_source_row(const uint64_t* refs,
    uint32_t selected) {
  if constexpr (Packed) return selected;
  else return refs[selected];
}
template<bool Packed>
MGBFS_INDEX_HD inline bool mgbfs_state_source_index(const uint64_t* refs,
    uint32_t selected, uint32_t sorted, uint32_t candidates, uint64_t* row) {
  if (selected >= sorted) return false;
  const uint64_t source = mgbfs_state_source_row<Packed>(refs, selected);
  if (source >= candidates) return false;
  *row = source;
  return true;
}
#undef MGBFS_INDEX_HD
