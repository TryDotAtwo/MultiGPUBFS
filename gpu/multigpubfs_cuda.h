#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define MGBFS_EXPORT __declspec(dllexport)
#else
#define MGBFS_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
typedef void* mgbfs_bitmap_handle;
typedef void* mgbfs_sort_unique_handle;
typedef void* mgbfs_cayley_handle;

MGBFS_EXPORT int mgbfs_cuda_device_info(char* name, std::size_t name_capacity,
                                        int* major, int* minor, char* error,
                                        std::size_t error_capacity);
MGBFS_EXPORT int mgbfs_cuda_affine(const std::uint32_t* host_input,
                                   std::uint32_t* host_output,
                                   std::size_t count, char* error,
                                   std::size_t error_capacity);

MGBFS_EXPORT int mgbfs_bitmap_create(std::uint64_t universe_size,
                                     std::size_t candidate_capacity,
                                     std::size_t output_capacity,
                                     mgbfs_bitmap_handle* handle, char* error,
                                     std::size_t error_capacity);
MGBFS_EXPORT int mgbfs_bitmap_seed(mgbfs_bitmap_handle handle,
                                   const std::uint32_t* host_keys,
                                   std::size_t count, char* error,
                                   std::size_t error_capacity);
MGBFS_EXPORT int mgbfs_bitmap_upload(mgbfs_bitmap_handle handle,
                                     const std::uint32_t* host_candidates,
                                     std::size_t count, char* error,
                                     std::size_t error_capacity);
MGBFS_EXPORT int mgbfs_bitmap_run(
    mgbfs_bitmap_handle handle, std::size_t count, std::uint32_t* host_output,
    std::size_t host_output_capacity, std::size_t* accepted_count,
    std::size_t* output_written, int* overflow, float* kernel_milliseconds,
    char* error, std::size_t error_capacity);
MGBFS_EXPORT int mgbfs_bitmap_run_variant(
    mgbfs_bitmap_handle handle, int variant, std::size_t count,
    std::uint32_t* host_output, std::size_t host_output_capacity,
    std::size_t* accepted_count, std::size_t* output_written, int* overflow,
    float* kernel_milliseconds, char* error, std::size_t error_capacity);
MGBFS_EXPORT void mgbfs_bitmap_destroy(mgbfs_bitmap_handle handle);

MGBFS_EXPORT int mgbfs_sort_unique_create(
    std::uint64_t universe_size, std::size_t candidate_capacity,
    std::size_t output_capacity, mgbfs_sort_unique_handle* handle, char* error,
    std::size_t error_capacity);
MGBFS_EXPORT int mgbfs_sort_unique_seed(
    mgbfs_sort_unique_handle handle, const std::uint32_t* host_keys,
    std::size_t count, char* error, std::size_t error_capacity);
MGBFS_EXPORT int mgbfs_sort_unique_upload(
    mgbfs_sort_unique_handle handle, const std::uint32_t* host_candidates,
    std::size_t count, char* error, std::size_t error_capacity);
MGBFS_EXPORT int mgbfs_sort_unique_run(
    mgbfs_sort_unique_handle handle, std::size_t count,
    std::uint32_t* host_output, std::size_t host_output_capacity,
    std::size_t* unique_count, std::size_t* accepted_count,
    std::size_t* output_written, int* overflow, float* sort_milliseconds,
    float* unique_milliseconds, float* claim_milliseconds,
    float* total_milliseconds, char* error, std::size_t error_capacity);
MGBFS_EXPORT int mgbfs_sort_unique_memory(
    mgbfs_sort_unique_handle handle, std::size_t* temporary_bytes,
    std::size_t* allocated_bytes, char* error, std::size_t error_capacity);
MGBFS_EXPORT void mgbfs_sort_unique_destroy(mgbfs_sort_unique_handle handle);

MGBFS_EXPORT int mgbfs_cayley_create(int n, std::size_t frontier_capacity,
                                     mgbfs_cayley_handle* handle, char* error,
                                     std::size_t error_capacity);
MGBFS_EXPORT int mgbfs_cayley_reset(mgbfs_cayley_handle handle, char* error,
                                    std::size_t error_capacity);
MGBFS_EXPORT int mgbfs_cayley_step(
    mgbfs_cayley_handle handle, int variant, int layout,
    std::size_t* next_frontier_count, float* kernel_milliseconds, char* error,
    std::size_t error_capacity);
MGBFS_EXPORT int mgbfs_cayley_copy_frontier(
    mgbfs_cayley_handle handle, std::uint64_t* host_frontier,
    std::size_t host_capacity, std::size_t* copied, char* error,
    std::size_t error_capacity);
MGBFS_EXPORT void mgbfs_cayley_destroy(mgbfs_cayley_handle handle);
}
