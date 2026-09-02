cmake_minimum_required(VERSION 3.22)
project(multigpubfs_gpu LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_library(multigpubfs_cuda SHARED
  device_smoke.cu
  bitmap_visited.cu
  sort_unique_visited.cu
  cayley_bfs.cu
)
target_compile_features(multigpubfs_cuda PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(multigpubfs_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(multigpubfs_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
set_target_properties(multigpubfs_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
  POSITION_INDEPENDENT_CODE ON
)
