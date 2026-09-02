#include "mgbfs_cuda.h"
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <climits>
static_assert(sizeof(MgbfsFrontierState)==8);
struct MaterializePlan {
  uint32_t stride{}, capacity{}, frontier{};
  uint64_t *keys{}, *sorted{};
  uint32_t *indices{}, *order{};
  void* scratch{}; size_t scratch_bytes{};
  ~MaterializePlan(){cudaFree(scratch);cudaFree(order);cudaFree(indices);cudaFree(sorted);cudaFree(keys);}
};
static void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void check_append(const uint32_t* count,MgbfsFrontierState* state,uint32_t capacity,uint32_t frontier){
  if(state->fatal)return;
  if(*count>capacity||state->count>frontier||*count>frontier-state->count)state->fatal=1;
}
__global__ void prepare_requests(const uint64_t* refs,const uint32_t* count,MgbfsFrontierState* state,
 uint64_t* keys,uint32_t* indices,uint32_t capacity,uint32_t source_count){
  uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=capacity)return;
  // Capacity was checked by a preceding kernel; never read past request spans.
  bool active=*count<=capacity&&i<*count;
  uint64_t key=active?refs[i]:UINT64_MAX;
  if(active&&key>=source_count)atomicCAS(&state->fatal,0u,2u);
  keys[i]=key;indices[i]=i;
}
__global__ void append_states(const uint4* source,const uint4* hashes,const uint64_t* sorted,const uint32_t* order,
 const uint32_t* count,uint4* states,uint4* out_hashes,const MgbfsFrontierState* state,uint32_t capacity,uint32_t chunks){
  if(state->fatal)return;
  // Adjacent lanes copy consecutive 16-byte chunks. Sorted requests make source
  // rows monotonic; gaps remain possible and need transaction profiling.
  for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<uint64_t(*count)*chunks;i+=uint64_t(gridDim.x)*blockDim.x){
    uint32_t row=uint32_t(i/chunks), chunk=uint32_t(i%chunks);
    states[(uint64_t(state->count)+row)*chunks+chunk]=source[sorted[row]*chunks+chunk];
    if(chunk==0)out_hashes[state->count+row]=hashes[order[row]];
  }
}
__global__ void publish_append(const uint32_t* count,MgbfsFrontierState* state){if(!state->fatal)state->count+=*count;}
extern "C" int mgbfs_materialize_create(uint32_t stride,uint32_t capacity,uint32_t frontier,void** out,char* error,size_t error_capacity){
  if(!out)return 1;*out=nullptr;
  try{
    if(!stride||stride%16||!capacity||capacity>INT32_MAX||!frontier)throw std::runtime_error("MATERIALIZE_CAPACITY_OR_STRIDE");
    auto p=std::make_unique<MaterializePlan>();p->stride=stride;p->capacity=capacity;p->frontier=frontier;
    checked(cudaMalloc(&p->keys,size_t(capacity)*8));checked(cudaMalloc(&p->sorted,size_t(capacity)*8));
    checked(cudaMalloc(&p->indices,size_t(capacity)*4));checked(cudaMalloc(&p->order,size_t(capacity)*4));
    checked(cub::DeviceRadixSort::SortPairs(nullptr,p->scratch_bytes,p->keys,p->sorted,p->indices,p->order,int(capacity)));
    checked(cudaMalloc(&p->scratch,p->scratch_bytes));*out=p.release();return 0;
  }catch(const std::exception& e){if(error&&error_capacity)std::snprintf(error,error_capacity,"%s",e.what());return 1;}
}
extern "C" int mgbfs_materialize_run(void* plan,const uint8_t* source,uint32_t source_count,const void* hashes,const uint64_t* refs,const uint32_t* count,uint8_t* states,void* out_hashes,MgbfsFrontierState* state,void* raw_stream){
  auto p=static_cast<MaterializePlan*>(plan);
  if(!p||!source||!hashes||!refs||!count||!states||!out_hashes||!state)return 1;
  auto stream=static_cast<cudaStream_t>(raw_stream);
  check_append<<<1,1,0,stream>>>(count,state,p->capacity,p->frontier);
  if(cudaGetLastError()!=cudaSuccess)return 2;
  prepare_requests<<<(p->capacity+255)/256,256,0,stream>>>(refs,count,state,p->keys,p->indices,p->capacity,source_count);
  if(cudaGetLastError()!=cudaSuccess)return 2;
  size_t bytes=p->scratch_bytes;
  if(cub::DeviceRadixSort::SortPairs(p->scratch,bytes,p->keys,p->sorted,p->indices,p->order,int(p->capacity),0,64,stream)!=cudaSuccess)return 3;
  append_states<<<(p->capacity+255)/256,256,0,stream>>>(reinterpret_cast<const uint4*>(source),static_cast<const uint4*>(hashes),p->sorted,p->order,count,reinterpret_cast<uint4*>(states),static_cast<uint4*>(out_hashes),state,p->capacity,p->stride/16);
  if(cudaGetLastError()!=cudaSuccess)return 2;
  publish_append<<<1,1,0,stream>>>(count,state);
  return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" void mgbfs_materialize_destroy(void* p){delete static_cast<MaterializePlan*>(p);}
