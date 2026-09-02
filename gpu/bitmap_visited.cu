#include "multigpubfs_cuda.h"

#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

struct BitmapContext {
  std::uint64_t universe_size{};
  std::size_t candidate_capacity{};
  std::size_t output_capacity{};
  std::size_t bitmap_words{};
  std::uint32_t* bitmap{};
  std::uint32_t* candidates{};
  std::uint32_t* output{};
  std::uint32_t* accepted{};
  std::uint32_t* overflow{};
  std::uint32_t* invalid{};
  cudaEvent_t start{};
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

template <bool WarpAggregate, bool BlockCompact>
__global__ void filter_kernel(
    std::uint32_t* bitmap, const std::uint32_t* candidates, std::size_t count,
    std::uint64_t universe_size, std::uint32_t* output,
    std::uint32_t output_capacity, std::uint32_t* accepted,
    std::uint32_t* overflow, std::uint32_t* invalid) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  std::uint32_t key = 0;
  bool valid = index < count;
  if (valid) {
    key = candidates[index];
    if (static_cast<std::uint64_t>(key) >= universe_size) {
      atomicExch(invalid, 1U);
      valid = false;
    }
  }

  bool claimant = valid;
  if constexpr (WarpAggregate) {
    const unsigned valid_lanes = __ballot_sync(0xffffffffU, valid);
    if (valid) {
      const unsigned equal_lanes = __match_any_sync(valid_lanes, key);
      claimant = static_cast<int>(threadIdx.x & 31U) == __ffs(equal_lanes) - 1;
    }
  }

  int accepted_flag = 0;
  if (claimant) {
    const std::uint32_t mask = 1U << (key & 31U);
    const std::uint32_t previous = atomicOr(&bitmap[key >> 5U], mask);
    accepted_flag = (previous & mask) == 0U;
  }

  if constexpr (BlockCompact) {
    using BlockScan = cub::BlockScan<int, 256>;
    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ std::uint32_t block_base;
    int local_position = 0;
    int block_total = 0;
    BlockScan(scan_storage).ExclusiveSum(accepted_flag, local_position,
                                         block_total);
    if (threadIdx.x == 0) {
      block_base = block_total == 0
                       ? 0
                       : atomicAdd(accepted, static_cast<std::uint32_t>(block_total));
      if (block_total != 0 &&
          static_cast<std::uint64_t>(block_base) + block_total > output_capacity) {
        atomicExch(overflow, 1U);
      }
    }
    __syncthreads();
    if (accepted_flag != 0) {
      const std::uint32_t position = block_base + local_position;
      if (position < output_capacity) output[position] = key;
    }
  } else if (accepted_flag != 0) {
    const std::uint32_t position = atomicAdd(accepted, 1U);
    if (position < output_capacity) {
      output[position] = key;
    } else {
      atomicExch(overflow, 1U);
    }
  }
}

cudaError_t launch_filter(BitmapContext* context, int variant,
                          std::size_t count) {
  const auto grid = static_cast<unsigned>((count + 255) / 256);
#define MGBFS_LAUNCH(warp, block)                                                \
  filter_kernel<warp, block><<<grid, 256>>>(                                    \
      context->bitmap, context->candidates, count, context->universe_size,       \
      context->output, static_cast<std::uint32_t>(context->output_capacity),     \
      context->accepted, context->overflow, context->invalid)
  switch (variant) {
    case 0:
      MGBFS_LAUNCH(false, false);
      break;
    case 1:
      MGBFS_LAUNCH(true, false);
      break;
    case 2:
      MGBFS_LAUNCH(false, true);
      break;
    case 3:
      MGBFS_LAUNCH(true, true);
      break;
    default:
      return cudaErrorInvalidValue;
  }
#undef MGBFS_LAUNCH
  return cudaGetLastError();
}

void release(BitmapContext* context) {
  if (context == nullptr) return;
  if (context->stop != nullptr) cudaEventDestroy(context->stop);
  if (context->start != nullptr) cudaEventDestroy(context->start);
  if (context->invalid != nullptr) cudaFree(context->invalid);
  if (context->overflow != nullptr) cudaFree(context->overflow);
  if (context->accepted != nullptr) cudaFree(context->accepted);
  if (context->output != nullptr) cudaFree(context->output);
  if (context->candidates != nullptr) cudaFree(context->candidates);
  if (context->bitmap != nullptr) cudaFree(context->bitmap);
  delete context;
}

}  // namespace

extern "C" int mgbfs_bitmap_create(
    std::uint64_t universe_size, std::size_t candidate_capacity,
    std::size_t output_capacity, mgbfs_bitmap_handle* handle, char* error,
    std::size_t error_capacity) {
  if (handle == nullptr || universe_size == 0 ||
      universe_size > (std::uint64_t{1} << 32U) || candidate_capacity == 0 ||
      output_capacity == 0 ||
      output_capacity > std::numeric_limits<std::uint32_t>::max()) {
    return fail(error, error_capacity, "invalid bitmap context capacity");
  }
  *handle = nullptr;
  auto* context = new BitmapContext{};
  context->universe_size = universe_size;
  context->candidate_capacity = candidate_capacity;
  context->output_capacity = output_capacity;
  context->bitmap_words = static_cast<std::size_t>((universe_size + 31U) / 32U);

#define MGBFS_ALLOC(call, operation)                                            \
  do {                                                                          \
    const auto status = (call);                                                  \
    if (status != cudaSuccess) {                                                 \
      set_error(error, error_capacity, operation, status);                       \
      release(context);                                                          \
      return 1;                                                                  \
    }                                                                            \
  } while (false)

  MGBFS_ALLOC(cudaMalloc(&context->bitmap,
                         context->bitmap_words * sizeof(std::uint32_t)),
              "cudaMalloc(bitmap)");
  MGBFS_ALLOC(cudaMalloc(&context->candidates,
                         candidate_capacity * sizeof(std::uint32_t)),
              "cudaMalloc(candidates)");
  MGBFS_ALLOC(cudaMalloc(&context->output,
                         output_capacity * sizeof(std::uint32_t)),
              "cudaMalloc(output)");
  MGBFS_ALLOC(cudaMalloc(&context->accepted, sizeof(std::uint32_t)),
              "cudaMalloc(accepted)");
  MGBFS_ALLOC(cudaMalloc(&context->overflow, sizeof(std::uint32_t)),
              "cudaMalloc(overflow)");
  MGBFS_ALLOC(cudaMalloc(&context->invalid, sizeof(std::uint32_t)),
              "cudaMalloc(invalid)");
  MGBFS_ALLOC(cudaEventCreate(&context->start), "cudaEventCreate(start)");
  MGBFS_ALLOC(cudaEventCreate(&context->stop), "cudaEventCreate(stop)");
#undef MGBFS_ALLOC

  *handle = context;
  return 0;
}

extern "C" int mgbfs_bitmap_seed(mgbfs_bitmap_handle handle,
                                  const std::uint32_t* host_keys,
                                  std::size_t count, char* error,
                                  std::size_t error_capacity) {
  auto* context = static_cast<BitmapContext*>(handle);
  if (context == nullptr || count > context->candidate_capacity ||
      (host_keys == nullptr && count != 0)) {
    return fail(error, error_capacity, "invalid bitmap seed arguments");
  }
  auto status = cudaMemset(context->bitmap, 0,
                           context->bitmap_words * sizeof(std::uint32_t));
  if (status == cudaSuccess) status = cudaMemset(context->invalid, 0, 4);
  if (status == cudaSuccess && count != 0) {
    status = cudaMemcpy(context->candidates, host_keys, count * 4,
                        cudaMemcpyHostToDevice);
  }
  if (status == cudaSuccess && count != 0) {
    seed_kernel<<<(count + 255) / 256, 256>>>(
        context->bitmap, context->candidates, count, context->universe_size,
        context->invalid);
    status = cudaGetLastError();
  }
  std::uint32_t invalid = 0;
  if (status == cudaSuccess) {
    status = cudaMemcpy(&invalid, context->invalid, 4, cudaMemcpyDeviceToHost);
  }
  if (status != cudaSuccess) {
    set_error(error, error_capacity, "bitmap seed", status);
    return 1;
  }
  return invalid == 0 ? 0 : fail(error, error_capacity, "seed key out of range");
}

extern "C" int mgbfs_bitmap_upload(
    mgbfs_bitmap_handle handle, const std::uint32_t* host_candidates,
    std::size_t count, char* error, std::size_t error_capacity) {
  auto* context = static_cast<BitmapContext*>(handle);
  if (context == nullptr || count > context->candidate_capacity ||
      (host_candidates == nullptr && count != 0)) {
    return fail(error, error_capacity, "invalid bitmap upload arguments");
  }
  const auto status = count == 0
                          ? cudaSuccess
                          : cudaMemcpy(context->candidates, host_candidates,
                                       count * 4, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    set_error(error, error_capacity, "bitmap upload", status);
    return 1;
  }
  return 0;
}

extern "C" int mgbfs_bitmap_run_variant(
    mgbfs_bitmap_handle handle, int variant, std::size_t count,
    std::uint32_t* host_output,
    std::size_t host_output_capacity, std::size_t* accepted_count,
    std::size_t* output_written, int* overflow, float* kernel_milliseconds,
    char* error, std::size_t error_capacity) {
  auto* context = static_cast<BitmapContext*>(handle);
  if (context == nullptr || count > context->candidate_capacity ||
      accepted_count == nullptr || output_written == nullptr ||
      overflow == nullptr || kernel_milliseconds == nullptr ||
      (host_output == nullptr && host_output_capacity != 0) || variant < 0 ||
      variant > 3) {
    return fail(error, error_capacity, "invalid bitmap run arguments");
  }
  auto status = cudaMemset(context->accepted, 0, 4);
  if (status == cudaSuccess) status = cudaMemset(context->overflow, 0, 4);
  if (status == cudaSuccess) status = cudaMemset(context->invalid, 0, 4);
  if (status == cudaSuccess) status = cudaEventRecord(context->start);
  if (status == cudaSuccess && count != 0) {
    status = launch_filter(context, variant, count);
  }
  if (status == cudaSuccess) status = cudaEventRecord(context->stop);
  if (status == cudaSuccess) status = cudaEventSynchronize(context->stop);
  if (status == cudaSuccess) {
    status = cudaEventElapsedTime(kernel_milliseconds, context->start,
                                  context->stop);
  }
  std::uint32_t accepted = 0;
  std::uint32_t device_overflow = 0;
  std::uint32_t invalid = 0;
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
    set_error(error, error_capacity, "bitmap run", status);
    return 1;
  }
  *accepted_count = accepted;
  *output_written = copied;
  *overflow = device_overflow != 0 || written > host_output_capacity;
  if (invalid != 0) return fail(error, error_capacity, "candidate key out of range");
  return 0;
}

extern "C" int mgbfs_bitmap_run(
    mgbfs_bitmap_handle handle, std::size_t count, std::uint32_t* host_output,
    std::size_t host_output_capacity, std::size_t* accepted_count,
    std::size_t* output_written, int* overflow, float* kernel_milliseconds,
    char* error, std::size_t error_capacity) {
  return mgbfs_bitmap_run_variant(
      handle, 0, count, host_output, host_output_capacity, accepted_count,
      output_written, overflow, kernel_milliseconds, error, error_capacity);
}

extern "C" void mgbfs_bitmap_destroy(mgbfs_bitmap_handle handle) {
  release(static_cast<BitmapContext*>(handle));
}
