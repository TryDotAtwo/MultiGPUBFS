#include "mgbfs_cuda.h"
#ifdef MGBFS_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

extern "C" int mgbfs_trace_ranges_available() {
#ifdef MGBFS_NVTX
  return 1;
#else
  return 0;
#endif
}
extern "C" void mgbfs_trace_range_push(const char* label) {
#ifdef MGBFS_NVTX
  nvtxRangePushA(label);
#else
  (void)label;
#endif
}
extern "C" void mgbfs_trace_range_pop() {
#ifdef MGBFS_NVTX
  nvtxRangePop();
#endif
}
