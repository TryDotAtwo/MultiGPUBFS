#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace mgbfs {
// Enqueued after CUB selection on the same stream. No host count is needed.
// Invalid control stays available for the existing host fatal check; never
// read selected tails or write a partial result for an invalid count.
static __global__ void gather_owner_sources(uint32_t const* selected,
    uint32_t const* control, uint32_t incoming_rows,
    uint32_t const* input, uint32_t* output) {
  uint32_t const count = control[0];
  if (control[1] || count > incoming_rows) return;
  for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < count;
       row += gridDim.x * blockDim.x)
    output[row] = input[selected[row]];
}
}
