#include "state_index.h"
#include <cstdio>
#include <cstdlib>
static void require(bool value) { if (!value) std::abort(); }
int main() {
  uint64_t row = 99;
  // Packed rows use the span-local index: no reference buffer exists.
  require(mgbfs_state_source_index<true>(nullptr, 2, 3, 3, &row) && row == 2);
  require(mgbfs_state_source_index<true>(nullptr, 0, 3, 3, &row) && row == 0);
  require(!mgbfs_state_source_index<true>(nullptr, 3, 3, 4, &row));
  require(!mgbfs_state_source_index<true>(nullptr, 2, 3, 2, &row));
  require(!mgbfs_state_source_index<true>(nullptr, 0, 0, 0, &row));
  const uint64_t refs[] = {3, 1, 0, UINT64_MAX};
  require(mgbfs_state_source_index<false>(refs, 0, 4, 4, &row) && row == 3);
  require(mgbfs_state_source_index<false>(refs, 2, 4, 4, &row) && row == 0);
  require(!mgbfs_state_source_index<false>(refs, 3, 4, 4, &row));
  // Must check sorted bounds before dereferencing even an absent ref array.
  require(!mgbfs_state_source_index<false>(nullptr, 0, 0, 4, &row));
  std::puts("STATE_INDEX_PASS");
}
