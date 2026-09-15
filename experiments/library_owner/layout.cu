#include "owner_abi.h"
#include <cuda_runtime.h>
#include <climits>
#include <cstdint>

namespace {
__global__ void to_soa(const uint4* input, uint32_t rows, uint32_t* planes,
                       uint64_t stride) {
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  uint4 key = input[row];
  planes[row] = key.x;
  planes[stride + row] = key.y;
  planes[2 * stride + row] = key.z;
  planes[3 * stride + row] = key.w;
  planes[4 * stride + row] = row;
}
__global__ void to_aos(MgbfsLibraryKeysV1 keys, uint4* output) {
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= keys.rows) return;
  output[row] = make_uint4(keys.words[0][row], keys.words[1][row],
                          keys.words[2][row], keys.words[3][row]);
}
bool aligned(const void* ptr, uintptr_t alignment) {
  return ptr && reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}
}

extern "C" int mgbfs_library_candidates_from_aos_v1(const void* hashes, uint32_t rows,
    uint32_t capacity, void* scratch, uint64_t scratch_bytes, void* stream,
    MgbfsLibraryCandidatesV1* result) {
  if (!result) return -1;
  *result = {};
  if (!capacity || capacity > INT_MAX || rows > capacity ||
      !aligned(scratch, 256) || (rows && !aligned(hashes, 16))) return -1;
  uint64_t stride_bytes = (uint64_t(capacity) * 4 + 255) & ~uint64_t(255);
  if (scratch_bytes < 5 * stride_bytes) return -1;
  if (rows) {
    to_soa<<<(rows + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint4*>(hashes), rows, static_cast<uint32_t*>(scratch),
        stride_bytes / 4);
    if (cudaPeekAtLastError() != cudaSuccess) return -1;
  }
  for (int c = 0; c < 4; ++c)
    result->keys.words[c] = reinterpret_cast<uint32_t*>(
        static_cast<char*>(scratch) + c * stride_bytes);
  result->keys.rows = rows;
  result->source_indices = reinterpret_cast<uint32_t*>(
      static_cast<char*>(scratch) + 4 * stride_bytes);
  return 0;
}

extern "C" int mgbfs_library_keys_to_aos_v1(MgbfsLibraryKeysV1 keys, void* output,
    uint32_t capacity, void* stream) {
  if (keys.reserved || keys.rows > INT_MAX || capacity > INT_MAX ||
      keys.rows > capacity) return -1;
  if (!keys.rows) return 0;
  if (!aligned(output, 16)) return -1;
  for (auto ptr : keys.words) if (!aligned(ptr, 4)) return -1;
  to_aos<<<(keys.rows + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      keys, static_cast<uint4*>(output));
  return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}
