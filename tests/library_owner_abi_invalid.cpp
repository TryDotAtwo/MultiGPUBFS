// CPU-only invalid-handle boundary test, not a CUDA execution gate.
#include "owner_abi.h"
#include <cstdio>
int main() {
  uint32_t sentinel = 42;
  MgbfsLibrarySurvivorsV1 out{&sentinel, 7, 99, 1};
  MgbfsLibraryCandidatesV1 input{};
  int status = mgbfs_library_owner_compare_v1(nullptr, 0, input, &out);
  if (status == 0 || out.source_indices != nullptr || out.epoch != 0 ||
      out.rows != 0 || out.reserved != 0) {
    std::fputs("INVALID_HANDLE_LEFT_STALE_SURVIVORS\n", stderr);
    return 1;
  }
  return 0;
}
