#include "multigpubfs_cuda.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace {

void set_error(char* error, std::size_t capacity, const char* operation,
               cudaError_t status) {
  if (error != nullptr && capacity > 0) {
    std::snprintf(error, capacity, "%s failed: %s", operation,
                  cudaGetErrorString(status));
  }
}

__global__ void affine_kernel(const std::uint32_t* input, std::uint32_t* output,
                              std::size_t count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = input[index] * 3U + 1U;
  }
}

}  // namespace

extern "C" int mgbfs_cuda_device_info(char* name, std::size_t name_capacity,
                                       int* major, int* minor, char* error,
                                       std::size_t error_capacity) {
  cudaDeviceProp properties{};
  const auto status = cudaGetDeviceProperties(&properties, 0);
  if (status != cudaSuccess) {
    set_error(error, error_capacity, "cudaGetDeviceProperties", status);
    return 1;
  }
  if (name != nullptr && name_capacity > 0) {
    std::snprintf(name, name_capacity, "%s", properties.name);
  }
  if (major != nullptr) *major = properties.major;
  if (minor != nullptr) *minor = properties.minor;
  return 0;
}

extern "C" int mgbfs_cuda_affine(const std::uint32_t* host_input,
                                  std::uint32_t* host_output,
                                  std::size_t count, char* error,
                                  std::size_t error_capacity) {
  if ((host_input == nullptr || host_output == nullptr) && count != 0) {
    if (error != nullptr && error_capacity > 0) {
      std::snprintf(error, error_capacity, "null host buffer");
    }
    return 2;
  }
  const std::size_t bytes = count * sizeof(std::uint32_t);
  std::uint32_t* device_input = nullptr;
  std::uint32_t* device_output = nullptr;
  auto status = cudaMalloc(&device_input, bytes);
  if (status != cudaSuccess) {
    set_error(error, error_capacity, "cudaMalloc(input)", status);
    return 1;
  }
  status = cudaMalloc(&device_output, bytes);
  if (status != cudaSuccess) {
    set_error(error, error_capacity, "cudaMalloc(output)", status);
    cudaFree(device_input);
    return 1;
  }
  status = cudaMemcpy(device_input, host_input, bytes, cudaMemcpyHostToDevice);
  if (status == cudaSuccess && count != 0) {
    affine_kernel<<<(count + 255) / 256, 256>>>(device_input, device_output, count);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(host_output, device_output, bytes, cudaMemcpyDeviceToHost);
  }
  if (status != cudaSuccess) {
    set_error(error, error_capacity, "CUDA affine smoke", status);
  }
  cudaFree(device_output);
  cudaFree(device_input);
  return status == cudaSuccess ? 0 : 1;
}
