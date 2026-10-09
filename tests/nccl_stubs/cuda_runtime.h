#pragma once
using cudaStream_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess = 0;
inline int cudaSetDevice(int) { return 0; }
inline int cudaDeviceSynchronize() { return 0; }
inline const char* cudaGetErrorString(int) { return "injected CUDA error"; }

#include <cstdlib>
#include <cstddef>
constexpr unsigned cudaHostAllocMapped = 2;
inline int cudaHostAlloc(void** out,std::size_t bytes,unsigned){*out=std::malloc(bytes);return *out?0:1;}
inline int cudaHostGetDevicePointer(void** out,void* host,unsigned){*out=host;return 0;}
inline int cudaFreeHost(void* host){std::free(host);return 0;}
