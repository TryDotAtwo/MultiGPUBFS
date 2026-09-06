#pragma once
using cudaStream_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess = 0;
inline int cudaSetDevice(int) { return 0; }
inline const char* cudaGetErrorString(int) { return "injected CUDA error"; }
