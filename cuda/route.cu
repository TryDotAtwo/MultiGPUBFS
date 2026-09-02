#include <cstdint>
#include "mgbfs_cuda.h"
#include <cstddef>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cuda/std/tuple>
struct alignas(16) Key128 {uint32_t words[4];};
static_assert(sizeof(Key128)==16);
struct Decompose {
  __host__ __device__ auto operator()(Key128& k) const {
    return cuda::std::tie(k.words[3],k.words[2],k.words[1],k.words[0]);
  }
};
struct RoutePlan {
  uint32_t capacity{};Key128* sorted{};uint64_t* refs{};
  uint32_t* indices{};uint32_t* selected{};uint8_t* flags{};
  void* scratch{};size_t scratch_bytes{};
  ~RoutePlan(){cudaFree(scratch);cudaFree(flags);cudaFree(selected);cudaFree(indices);cudaFree(refs);cudaFree(sorted);}
};
static void checked(cudaError_t r){if(r!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(r));}
__global__ void mark_unique(const Key128* keys,uint8_t* flags,uint32_t* indices,uint32_t n){
  const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  bool same=i!=0;
  if(i)for(int j=0;j<4;++j)same=same&&(keys[i].words[j]==keys[i-1].words[j]);
  flags[i]=!same;indices[i]=i;
}
__global__ void compact(const Key128* keys,const uint64_t* refs,const uint32_t* indices,const uint32_t* count,Key128* output,uint64_t* outrefs,uint32_t launch_count){
  const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=launch_count||i>=*count)return;
  const uint32_t source=indices[i];output[i]=keys[source];outrefs[i]=refs[source];
}
__global__ void publish_count(uint32_t* output,uint32_t count){*output=count;}
extern "C" int mgbfs_route_create(uint32_t capacity,void** out,char* error,size_t error_capacity){
  if(!out)return 1;*out=nullptr;
  try{
    if(capacity==0||capacity>INT32_MAX)throw std::runtime_error("ROUTE_CAPACITY");
    auto p=std::make_unique<RoutePlan>();p->capacity=capacity;
    checked(cudaMalloc(&p->sorted,size_t(capacity)*16));checked(cudaMalloc(&p->refs,size_t(capacity)*8));
    checked(cudaMalloc(&p->indices,size_t(capacity)*4));checked(cudaMalloc(&p->selected,size_t(capacity)*4));checked(cudaMalloc(&p->flags,capacity));
    size_t sort_bytes=0,select_bytes=0;
    checked(cub::DeviceRadixSort::SortPairs(nullptr,sort_bytes,static_cast<const Key128*>(nullptr),p->sorted,static_cast<const uint64_t*>(nullptr),p->refs,int(capacity),Decompose{},0,128));
    checked(cub::DeviceSelect::Flagged(nullptr,select_bytes,p->indices,p->flags,p->selected,static_cast<uint32_t*>(nullptr),int(capacity)));
    p->scratch_bytes=std::max(sort_bytes,select_bytes);checked(cudaMalloc(&p->scratch,p->scratch_bytes));
    *out=p.release();return 0;
  }catch(const std::exception& e){if(error&&error_capacity)std::snprintf(error,error_capacity,"%s",e.what());return 1;}
}
extern "C" int mgbfs_route_run(void* plan,const void* hashes,const uint64_t* refs,void* output,uint64_t* outrefs,uint32_t* output_count,uint32_t count,int pre_dedup,void* raw_stream){
  auto* p=static_cast<RoutePlan*>(plan);
  if(!p||!hashes||!refs||!output||!outrefs||!output_count||count>p->capacity||(pre_dedup!=0&&pre_dedup!=1))return 1;
  auto stream=static_cast<cudaStream_t>(raw_stream);
  if(count==0){publish_count<<<1,1,0,stream>>>(output_count,0);return cudaGetLastError()==cudaSuccess?0:2;}
  size_t bytes=p->scratch_bytes;
  if(cub::DeviceRadixSort::SortPairs(p->scratch,bytes,static_cast<const Key128*>(hashes),p->sorted,refs,p->refs,int(count),Decompose{},0,128,stream)!=cudaSuccess)return 3;
  if(!pre_dedup){
    if(cudaMemcpyAsync(output,p->sorted,size_t(count)*16,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return 4;
    if(cudaMemcpyAsync(outrefs,p->refs,size_t(count)*8,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return 4;
    publish_count<<<1,1,0,stream>>>(output_count,count);
  }else{
    mark_unique<<<(count+255)/256,256,0,stream>>>(p->sorted,p->flags,p->indices,count);
    if(cudaGetLastError()!=cudaSuccess)return 5;
    bytes=p->scratch_bytes;
    if(cub::DeviceSelect::Flagged(p->scratch,bytes,p->indices,p->flags,p->selected,output_count,int(count),stream)!=cudaSuccess)return 6;
    compact<<<(count+255)/256,256,0,stream>>>(p->sorted,p->refs,p->selected,output_count,static_cast<Key128*>(output),outrefs,count);
  }
  return cudaGetLastError()==cudaSuccess?0:7;
}
extern "C" void mgbfs_route_destroy(void* p){delete static_cast<RoutePlan*>(p);}
