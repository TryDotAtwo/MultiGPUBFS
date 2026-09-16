#include "owner_source_gather.cuh"
#include <cuda_runtime.h>
#include <array>
#include <stdexcept>

static void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
int main() {
  uint32_t *selected, *input, *output, *control;
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  check(cudaMalloc(&selected, 16)); check(cudaMalloc(&input, 16));
  check(cudaMalloc(&output, 16)); check(cudaMalloc(&control, 8));
  std::array<uint32_t, 4> picks{2, 0, UINT32_MAX, UINT32_MAX};
  std::array<uint32_t, 4> sources{71, 19, 83, 5};
  check(cudaMemcpyAsync(selected, picks.data(), 16, cudaMemcpyHostToDevice, stream));
  check(cudaMemcpyAsync(input, sources.data(), 16, cudaMemcpyHostToDevice, stream));
  std::array<std::array<uint32_t, 2>, 4> cases{{{2, 0}, {0, 0}, {5, 0}, {2, 1}}};
  for (auto counters : cases) {
    check(cudaMemsetAsync(output, 0xff, 16, stream));
    check(cudaMemcpyAsync(control, counters.data(), 8, cudaMemcpyHostToDevice, stream));
    mgbfs::gather_owner_sources<<<1, 256, 0, stream>>>(selected, control, 4, input, output);
    check(cudaGetLastError());
    std::array<uint32_t, 4> actual{};
    check(cudaMemcpyAsync(actual.data(), output, 16, cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    auto expected = std::array<uint32_t, 4>{UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    if (counters[0] == 2 && counters[1] == 0) expected = {83, 71, UINT32_MAX, UINT32_MAX};
    if (actual != expected) throw std::runtime_error("SOURCE_GATHER_COUNT_OR_TAIL");
  }
  check(cudaFree(control)); check(cudaFree(output));
  check(cudaFree(input)); check(cudaFree(selected));
  check(cudaStreamDestroy(stream));
}
