#include "multigpubfs_cuda.h"

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

struct SortUniqueContext {
  std::uint64_t universe_size{};
  std::size_t capacity{};
  std::size_t output_capacity{};
  std::size_t bitmap_words{};
  int end_bit{};
  std::uint32_t* bitmap{};
  std::uint32_t* input{};
  std::uint32_t* sorted{};
  std::uint32_t* unique{};
  std::uint32_t* output{};
  int* unique_count{};
  std::uint32_t* accepted{};
  std::uint32_t* overflow{};
  std::uint32_t* invalid{};
  void* temporary{};
  std::size_t temporary_bytes{};
  cudaEvent_t start{};
  cudaEvent_t after_sort{};
  cudaEvent_t after_unique{};
  cudaEvent_t stop{};
};

void set_error(char* error, std::size_t capacity, const char* operation,
               cudaError_t status) {
  if (error != nullptr && capacity > 0) {
    std::snprintf(error, capacity, "%s failed: %s", operation,
                  cudaGetErrorString(status));
  }
}

int fail(char* error, std::size_t capacity, const char* message) {
  if (error != nullptr && capacity > 0) {
    std::snprintf(error, capacity, "%s", message);
  }
  return 2;
}

int significant_bits(std::uint64_t universe_size) {
  int bits = 0;
  std::uint64_t maximum = universe_size - 1;
  do {
    ++bits;
    maximum >>= 1U;
  } while (maximum != 0);
  return bits;
}

__global__ void seed_kernel(std::uint32_t* bitmap,
                            const std::uint32_t* keys, std::size_t count,
                            std::uint64_t universe_size,
                            std::uint32_t* invalid) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const std::uint32_t key = keys[index];
  if (static_cast<std::uint64_t>(key) >= universe_size) {
    atomicExch(invalid, 1U);
    return;
  }
  atomicOr(&bitmap[key >> 5U], 1U << (key & 31U));
}

__global__ void claim_unique_kernel(
    std::uint32_t* bitmap, const std::uint32_t* unique,
    const int* unique_count, std::size_t launch_count,
    std::uint64_t universe_size, std::uint32_t* output,
    std::uint32_t output_capacity, std::uint32_t* accepted,
    std::uint32_t* overflow, std::uint32_t* invalid) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= launch_count || index >= static_cast<std::size_t>(*unique_count))
    return;
  const std::uint32_t key = unique[index];
  if (static_cast<std::uint64_t>(key) >= universe_size) {
    atomicExch(invalid, 1U);
    return;
  }
  const std::uint32_t mask = 1U << (key & 31U);
  const std::uint32_t previous = atomicOr(&bitmap[key >> 5U], mask);
  if ((previous & mask) != 0U) return;
  const std::uint32_t position = atomicAdd(accepted, 1U);
  if (position < output_capacity) {
    output[position] = key;
  } else {
    atomicExch(overflow, 1U);
  }
}

void release(SortUniqueContext* context) {
  if (context == nullptr) return;
  if (context->stop != nullptr) cudaEventDestroy(context->stop);
  if (context->after_unique != nullptr) cudaEventDestroy(context->after_unique);
  if (context->after_sort != nullptr) cudaEventDestroy(context->after_sort);
  if (context->start != nullptr) cudaEventDestroy(context->start);
  if (context->temporary != nullptr) cudaFree(context->temporary);
  if (context->invalid != nullptr) cudaFree(context->invalid);
  if (context->overflow != nullptr) cudaFree(context->overflow);
  if (context->accepted != nullptr) cudaFree(context->accepted);
  if (context->unique_count != nullptr) cudaFree(context->unique_count);
  if (context->output != nullptr) cudaFree(context->output);
  if (context->unique != nullptr) cudaFree(context->unique);
  if (context->sorted != nullptr) cudaFree(context->sorted);
  if (context->input != nullptr) cudaFree(context->input);
  if (context->bitmap != nullptr) cudaFree(context->bitmap);
  delete context;
}

}  // namespace

extern "C" int mgbfs_sort_unique_create(
    std::uint64_t universe_size, std::size_t candidate_capacity,
    std::size_t output_capacity, mgbfs_sort_unique_handle* handle, char* error,
    std::size_t error_capacity) {
  if (handle == nullptr || universe_size == 0 ||
      universe_size > (std::uint64_t{1} << 32U) || candidate_capacity == 0 ||
      candidate_capacity > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      output_capacity == 0 ||
      output_capacity > std::numeric_limits<std::uint32_t>::max()) {
    return fail(error, error_capacity, "invalid sort-unique context capacity");
  }
  *handle = nullptr;
  auto* context = new SortUniqueContext{};
  context->universe_size = universe_size;
  context->capacity = candidate_capacity;
  context->output_capacity = output_capacity;
  context->bitmap_words = static_cast<std::size_t>((universe_size + 31U) / 32U);
  context->end_bit = significant_bits(universe_size);

#define MGBFS_ALLOC(call, operation)                                            \
  do {                                                                          \
    const auto status = (call);                                                  \
    if (status != cudaSuccess) {                                                 \
      set_error(error, error_capacity, operation, status);                       \
      release(context);                                                          \
      return 1;                                                                  \
    }                                                                            \
  } while (false)

  MGBFS_ALLOC(cudaMalloc(&context->bitmap, context->bitmap_words * 4),
              "cudaMalloc(bitmap)");
  MGBFS_ALLOC(cudaMalloc(&context->input, candidate_capacity * 4),
              "cudaMalloc(input)");
  MGBFS_ALLOC(cudaMalloc(&context->sorted, candidate_capacity * 4),
              "cudaMalloc(sorted)");
  MGBFS_ALLOC(cudaMalloc(&context->unique, candidate_capacity * 4),
              "cudaMalloc(unique)");
  MGBFS_ALLOC(cudaMalloc(&context->output, output_capacity * 4),
              "cudaMalloc(output)");
  MGBFS_ALLOC(cudaMalloc(&context->unique_count, sizeof(int)),
              "cudaMalloc(unique_count)");
  MGBFS_ALLOC(cudaMalloc(&context->accepted, 4), "cudaMalloc(accepted)");
  MGBFS_ALLOC(cudaMalloc(&context->overflow, 4), "cudaMalloc(overflow)");
  MGBFS_ALLOC(cudaMalloc(&context->invalid, 4), "cudaMalloc(invalid)");

  std::size_t sort_bytes = 0;
  std::size_t unique_bytes = 0;
  MGBFS_ALLOC(cub::DeviceRadixSort::SortKeys(
                  nullptr, sort_bytes, context->input, context->sorted,
                  static_cast<int>(candidate_capacity), 0, context->end_bit),
              "query DeviceRadixSort::SortKeys");
  MGBFS_ALLOC(cub::DeviceSelect::Unique(
                  nullptr, unique_bytes, context->sorted, context->unique,
                  context->unique_count, static_cast<int>(candidate_capacity)),
              "query DeviceSelect::Unique");
  context->temporary_bytes = std::max(sort_bytes, unique_bytes);
  MGBFS_ALLOC(cudaMalloc(&context->temporary, context->temporary_bytes),
              "cudaMalloc(temporary)");
  MGBFS_ALLOC(cudaEventCreate(&context->start), "cudaEventCreate(start)");
  MGBFS_ALLOC(cudaEventCreate(&context->after_sort),
              "cudaEventCreate(after_sort)");
  MGBFS_ALLOC(cudaEventCreate(&context->after_unique),
              "cudaEventCreate(after_unique)");
  MGBFS_ALLOC(cudaEventCreate(&context->stop), "cudaEventCreate(stop)");
#undef MGBFS_ALLOC

  *handle = context;
  return 0;
}

extern "C" int mgbfs_sort_unique_seed(
    mgbfs_sort_unique_handle handle, const std::uint32_t* host_keys,
    std::size_t count, char* error, std::size_t error_capacity) {
  auto* context = static_cast<SortUniqueContext*>(handle);
  if (context == nullptr || count > context->capacity ||
      (host_keys == nullptr && count != 0)) {
    return fail(error, error_capacity, "invalid sort-unique seed arguments");
  }
  auto status = cudaMemset(context->bitmap, 0, context->bitmap_words * 4);
  if (status == cudaSuccess) status = cudaMemset(context->invalid, 0, 4);
  if (status == cudaSuccess && count != 0) {
    status = cudaMemcpy(context->input, host_keys, count * 4,
                        cudaMemcpyHostToDevice);
  }
  if (status == cudaSuccess && count != 0) {
    seed_kernel<<<(count + 255) / 256, 256>>>(
        context->bitmap, context->input, count, context->universe_size,
        context->invalid);
    status = cudaGetLastError();
  }
  std::uint32_t invalid = 0;
  if (status == cudaSuccess) {
    status = cudaMemcpy(&invalid, context->invalid, 4, cudaMemcpyDeviceToHost);
  }
  if (status != cudaSuccess) {
    set_error(error, error_capacity, "sort-unique seed", status);
    return 1;
  }
  return invalid == 0 ? 0 : fail(error, error_capacity, "seed key out of range");
}

extern "C" int mgbfs_sort_unique_upload(
    mgbfs_sort_unique_handle handle, const std::uint32_t* host_candidates,
    std::size_t count, char* error, std::size_t error_capacity) {
  auto* context = static_cast<SortUniqueContext*>(handle);
  if (context == nullptr || count > context->capacity ||
      (host_candidates == nullptr && count != 0)) {
    return fail(error, error_capacity, "invalid sort-unique upload arguments");
  }
  const auto status = count == 0
                          ? cudaSuccess
                          : cudaMemcpy(context->input, host_candidates, count * 4,
                                       cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    set_error(error, error_capacity, "sort-unique upload", status);
    return 1;
  }
  return 0;
}

extern "C" int mgbfs_sort_unique_run(
    mgbfs_sort_unique_handle handle, std::size_t count,
    std::uint32_t* host_output, std::size_t host_output_capacity,
    std::size_t* unique_count, std::size_t* accepted_count,
    std::size_t* output_written, int* overflow, float* sort_milliseconds,
    float* unique_milliseconds, float* claim_milliseconds,
    float* total_milliseconds, char* error, std::size_t error_capacity) {
  auto* context = static_cast<SortUniqueContext*>(handle);
  if (context == nullptr || count > context->capacity ||
      unique_count == nullptr || accepted_count == nullptr ||
      output_written == nullptr || overflow == nullptr ||
      sort_milliseconds == nullptr || unique_milliseconds == nullptr ||
      claim_milliseconds == nullptr || total_milliseconds == nullptr ||
      (host_output == nullptr && host_output_capacity != 0)) {
    return fail(error, error_capacity, "invalid sort-unique run arguments");
  }

  auto status = cudaMemset(context->accepted, 0, 4);
  if (status == cudaSuccess) status = cudaMemset(context->overflow, 0, 4);
  if (status == cudaSuccess) status = cudaMemset(context->invalid, 0, 4);
  if (status == cudaSuccess) status = cudaMemset(context->unique_count, 0, sizeof(int));
  if (status == cudaSuccess) status = cudaEventRecord(context->start);
  std::size_t temporary_bytes = context->temporary_bytes;
  if (status == cudaSuccess && count != 0) {
    status = cub::DeviceRadixSort::SortKeys(
        context->temporary, temporary_bytes, context->input, context->sorted,
        static_cast<int>(count), 0, context->end_bit);
  }
  if (status == cudaSuccess) status = cudaEventRecord(context->after_sort);
  temporary_bytes = context->temporary_bytes;
  if (status == cudaSuccess && count != 0) {
    status = cub::DeviceSelect::Unique(
        context->temporary, temporary_bytes, context->sorted, context->unique,
        context->unique_count, static_cast<int>(count));
  }
  if (status == cudaSuccess) status = cudaEventRecord(context->after_unique);
  if (status == cudaSuccess && count != 0) {
    claim_unique_kernel<<<(count + 255) / 256, 256>>>(
        context->bitmap, context->unique, context->unique_count, count,
        context->universe_size, context->output,
        static_cast<std::uint32_t>(context->output_capacity), context->accepted,
        context->overflow, context->invalid);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess) status = cudaEventRecord(context->stop);
  if (status == cudaSuccess) status = cudaEventSynchronize(context->stop);
  if (status == cudaSuccess)
    status = cudaEventElapsedTime(sort_milliseconds, context->start,
                                  context->after_sort);
  if (status == cudaSuccess)
    status = cudaEventElapsedTime(unique_milliseconds, context->after_sort,
                                  context->after_unique);
  if (status == cudaSuccess)
    status = cudaEventElapsedTime(claim_milliseconds, context->after_unique,
                                  context->stop);
  if (status == cudaSuccess)
    status = cudaEventElapsedTime(total_milliseconds, context->start,
                                  context->stop);

  int device_unique = 0;
  std::uint32_t accepted = 0;
  std::uint32_t device_overflow = 0;
  std::uint32_t invalid = 0;
  if (status == cudaSuccess)
    status = cudaMemcpy(&device_unique, context->unique_count, sizeof(int),
                        cudaMemcpyDeviceToHost);
  if (status == cudaSuccess)
    status = cudaMemcpy(&accepted, context->accepted, 4, cudaMemcpyDeviceToHost);
  if (status == cudaSuccess)
    status = cudaMemcpy(&device_overflow, context->overflow, 4,
                        cudaMemcpyDeviceToHost);
  if (status == cudaSuccess)
    status = cudaMemcpy(&invalid, context->invalid, 4, cudaMemcpyDeviceToHost);
  const std::size_t written =
      accepted < context->output_capacity ? accepted : context->output_capacity;
  const std::size_t copied =
      written < host_output_capacity ? written : host_output_capacity;
  if (status == cudaSuccess && copied != 0) {
    status = cudaMemcpy(host_output, context->output, copied * 4,
                        cudaMemcpyDeviceToHost);
  }
  if (status != cudaSuccess) {
    set_error(error, error_capacity, "sort-unique run", status);
    return 1;
  }
  *unique_count = static_cast<std::size_t>(device_unique);
  *accepted_count = accepted;
  *output_written = copied;
  *overflow = device_overflow != 0 || written > host_output_capacity;
  if (invalid != 0)
    return fail(error, error_capacity, "candidate key out of range");
  return 0;
}

extern "C" void mgbfs_sort_unique_destroy(mgbfs_sort_unique_handle handle) {
  release(static_cast<SortUniqueContext*>(handle));
}

extern "C" int mgbfs_sort_unique_memory(
    mgbfs_sort_unique_handle handle, std::size_t* temporary_bytes,
    std::size_t* allocated_bytes, char* error, std::size_t error_capacity) {
  auto* context = static_cast<SortUniqueContext*>(handle);
  if (context == nullptr || temporary_bytes == nullptr ||
      allocated_bytes == nullptr) {
    return fail(error, error_capacity, "invalid sort-unique memory arguments");
  }
  *temporary_bytes = context->temporary_bytes;
  *allocated_bytes = context->bitmap_words * sizeof(std::uint32_t) +
                     (3 * context->capacity + context->output_capacity) *
                         sizeof(std::uint32_t) +
                     sizeof(int) + 3 * sizeof(std::uint32_t) +
                     context->temporary_bytes;
  return 0;
}
