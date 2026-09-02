#include "multigpubfs_cuda.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <limits>
#include <utility>

namespace {

struct CayleyContext {
  int n{};
  std::uint32_t state_count{};
  std::size_t frontier_capacity{};
  std::size_t bitmap_words{};
  std::uint32_t* bitmap{};
  std::uint64_t* frontier_a{};
  std::uint64_t* frontier_b{};
  std::uint64_t* current{};
  std::uint64_t* next{};
  std::uint32_t current_count{};
  std::uint32_t* next_count{};
  std::uint32_t* overflow{};
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

std::uint32_t factorial(int n) {
  std::uint32_t result = 1;
  for (int value = 2; value <= n; ++value) result *= value;
  return result;
}

std::uint64_t identity_state(int n) {
  std::uint64_t state = 0;
  for (int index = 0; index < n; ++index) {
    state |= static_cast<std::uint64_t>(index) << (4 * index);
  }
  return state;
}

template <int N>
__device__ __forceinline__ std::uint32_t rank_state(std::uint64_t state) {
  std::uint32_t result = 0;
#pragma unroll
  for (int index = 0; index < N; ++index) {
    const std::uint32_t value = (state >> (4 * index)) & 0xfU;
    std::uint32_t smaller = 0;
#pragma unroll
    for (int right = index + 1; right < N; ++right) {
      smaller += ((state >> (4 * right)) & 0xfU) < value;
    }
    result = result * (N - index) + smaller;
  }
  return result;
}

__device__ __forceinline__ std::uint64_t swap_adjacent(std::uint64_t state,
                                                        int generator) {
  const int left_shift = 4 * generator;
  const int right_shift = left_shift + 4;
  const std::uint64_t left = (state >> left_shift) & 0xfULL;
  const std::uint64_t right = (state >> right_shift) & 0xfULL;
  const std::uint64_t mask = (0xfULL << left_shift) | (0xfULL << right_shift);
  return (state & ~mask) | (left << right_shift) | (right << left_shift);
}

template <int N, bool WarpAggregate>
__global__ void expand_kernel(
    std::uint32_t* bitmap, const std::uint64_t* frontier,
    std::uint32_t frontier_count, int layout, std::uint64_t* next,
    std::uint32_t next_capacity, std::uint32_t* next_count,
    std::uint32_t* overflow) {
  const std::uint64_t candidate_index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t candidate_count =
      static_cast<std::uint64_t>(frontier_count) * (N - 1);
  const bool valid = candidate_index < candidate_count;
  std::uint64_t child = 0;
  std::uint32_t key = 0;
  if (valid) {
    const std::uint32_t parent_index =
        layout == 0 ? candidate_index / (N - 1) : candidate_index % frontier_count;
    const int generator =
        layout == 0 ? candidate_index % (N - 1) : candidate_index / frontier_count;
    child = swap_adjacent(frontier[parent_index], generator);
    key = rank_state<N>(child);
  }

  bool claimant = valid;
  if constexpr (WarpAggregate) {
    const unsigned lanes = __ballot_sync(0xffffffffU, valid);
    if (valid) {
      const unsigned equal = __match_any_sync(lanes, key);
      claimant = static_cast<int>(threadIdx.x & 31U) == __ffs(equal) - 1;
    }
  }
  if (!claimant) return;
  const std::uint32_t mask = 1U << (key & 31U);
  const std::uint32_t previous = atomicOr(&bitmap[key >> 5U], mask);
  if ((previous & mask) != 0U) return;
  const std::uint32_t position = atomicAdd(next_count, 1U);
  if (position < next_capacity) {
    next[position] = child;
  } else {
    atomicExch(overflow, 1U);
  }
}

cudaError_t launch(CayleyContext* context, int variant, int layout) {
  const std::uint64_t candidates =
      static_cast<std::uint64_t>(context->current_count) * (context->n - 1);
  const auto grid = static_cast<unsigned>((candidates + 255) / 256);
#define MGBFS_CAYLEY_LAUNCH(n, warp)                                            \
  expand_kernel<n, warp><<<grid, 256>>>(                                       \
      context->bitmap, context->current, context->current_count, layout,        \
      context->next, static_cast<std::uint32_t>(context->frontier_capacity),    \
      context->next_count, context->overflow)
#define MGBFS_CAYLEY_CASE(n)                                                     \
  case n:                                                                        \
    if (variant == 0) {                                                          \
      MGBFS_CAYLEY_LAUNCH(n, false);                                             \
    } else {                                                                     \
      MGBFS_CAYLEY_LAUNCH(n, true);                                              \
    }                                                                            \
    break
  switch (context->n) {
    MGBFS_CAYLEY_CASE(8);
    MGBFS_CAYLEY_CASE(9);
    MGBFS_CAYLEY_CASE(10);
    default:
      return cudaErrorInvalidValue;
  }
#undef MGBFS_CAYLEY_CASE
#undef MGBFS_CAYLEY_LAUNCH
  return cudaGetLastError();
}

void release(CayleyContext* context) {
  if (context == nullptr) return;
  if (context->stop != nullptr) cudaEventDestroy(context->stop);
  if (context->start != nullptr) cudaEventDestroy(context->start);
  if (context->overflow != nullptr) cudaFree(context->overflow);
  if (context->next_count != nullptr) cudaFree(context->next_count);
  if (context->frontier_b != nullptr) cudaFree(context->frontier_b);
  if (context->frontier_a != nullptr) cudaFree(context->frontier_a);
  if (context->bitmap != nullptr) cudaFree(context->bitmap);
  delete context;
}

}  // namespace

extern "C" int mgbfs_cayley_create(int n, std::size_t frontier_capacity,
                                     mgbfs_cayley_handle* handle, char* error,
                                     std::size_t error_capacity) {
  if (handle == nullptr || n < 8 || n > 10 || frontier_capacity == 0 ||
      frontier_capacity > std::numeric_limits<std::uint32_t>::max()) {
    return fail(error, error_capacity, "invalid Cayley context arguments");
  }
  *handle = nullptr;
  auto* context = new CayleyContext{};
  context->n = n;
  context->state_count = factorial(n);
  context->frontier_capacity = frontier_capacity;
  context->bitmap_words = (context->state_count + 31U) / 32U;

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
              "cudaMalloc(cayley bitmap)");
  MGBFS_ALLOC(cudaMalloc(&context->frontier_a, frontier_capacity * 8),
              "cudaMalloc(frontier_a)");
  MGBFS_ALLOC(cudaMalloc(&context->frontier_b, frontier_capacity * 8),
              "cudaMalloc(frontier_b)");
  MGBFS_ALLOC(cudaMalloc(&context->next_count, 4), "cudaMalloc(next_count)");
  MGBFS_ALLOC(cudaMalloc(&context->overflow, 4), "cudaMalloc(overflow)");
  MGBFS_ALLOC(cudaEventCreate(&context->start), "cudaEventCreate(start)");
  MGBFS_ALLOC(cudaEventCreate(&context->stop), "cudaEventCreate(stop)");
#undef MGBFS_ALLOC
  context->current = context->frontier_a;
  context->next = context->frontier_b;
  *handle = context;
  return 0;
}

extern "C" int mgbfs_cayley_reset(mgbfs_cayley_handle handle, char* error,
                                    std::size_t error_capacity) {
  auto* context = static_cast<CayleyContext*>(handle);
  if (context == nullptr) return fail(error, error_capacity, "null Cayley context");
  context->current = context->frontier_a;
  context->next = context->frontier_b;
  context->current_count = 1;
  const std::uint64_t identity = identity_state(context->n);
  std::uint32_t first_word = 1;
  auto status = cudaMemset(context->bitmap, 0, context->bitmap_words * 4);
  if (status == cudaSuccess)
    status = cudaMemcpy(context->bitmap, &first_word, 4, cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(context->current, &identity, 8, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    set_error(error, error_capacity, "Cayley reset", status);
    return 1;
  }
  return 0;
}

extern "C" int mgbfs_cayley_step(
    mgbfs_cayley_handle handle, int variant, int layout,
    std::size_t* next_frontier_count, float* kernel_milliseconds, char* error,
    std::size_t error_capacity) {
  auto* context = static_cast<CayleyContext*>(handle);
  if (context == nullptr || (variant != 0 && variant != 1) ||
      (layout != 0 && layout != 1) || next_frontier_count == nullptr ||
      kernel_milliseconds == nullptr) {
    return fail(error, error_capacity, "invalid Cayley step arguments");
  }
  auto status = cudaMemset(context->next_count, 0, 4);
  if (status == cudaSuccess) status = cudaMemset(context->overflow, 0, 4);
  if (status == cudaSuccess) status = cudaEventRecord(context->start);
  if (status == cudaSuccess && context->current_count != 0)
    status = launch(context, variant, layout);
  if (status == cudaSuccess) status = cudaEventRecord(context->stop);
  if (status == cudaSuccess) status = cudaEventSynchronize(context->stop);
  if (status == cudaSuccess)
    status = cudaEventElapsedTime(kernel_milliseconds, context->start,
                                  context->stop);
  std::uint32_t count = 0;
  std::uint32_t overflow = 0;
  if (status == cudaSuccess)
    status = cudaMemcpy(&count, context->next_count, 4, cudaMemcpyDeviceToHost);
  if (status == cudaSuccess)
    status = cudaMemcpy(&overflow, context->overflow, 4, cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    set_error(error, error_capacity, "Cayley step", status);
    return 1;
  }
  if (overflow != 0) return fail(error, error_capacity, "Cayley frontier overflow");
  context->current_count = count;
  std::swap(context->current, context->next);
  *next_frontier_count = count;
  return 0;
}

extern "C" int mgbfs_cayley_copy_frontier(
    mgbfs_cayley_handle handle, std::uint64_t* host_frontier,
    std::size_t host_capacity, std::size_t* copied, char* error,
    std::size_t error_capacity) {
  auto* context = static_cast<CayleyContext*>(handle);
  if (context == nullptr || host_capacity < context->current_count ||
      (host_frontier == nullptr && context->current_count != 0) ||
      copied == nullptr) {
    return fail(error, error_capacity, "invalid Cayley copy arguments");
  }
  const auto status = context->current_count == 0
                          ? cudaSuccess
                          : cudaMemcpy(host_frontier, context->current,
                                       context->current_count * 8,
                                       cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    set_error(error, error_capacity, "Cayley frontier copy", status);
    return 1;
  }
  *copied = context->current_count;
  return 0;
}

extern "C" void mgbfs_cayley_destroy(mgbfs_cayley_handle handle) {
  release(static_cast<CayleyContext*>(handle));
}
