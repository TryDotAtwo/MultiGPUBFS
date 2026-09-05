#include "archive_pack.h"
#include <cuda_runtime.h>

namespace {
__global__ void pack(uint32_t n,uint32_t stride,const uint8_t* states,uint8_t* out,
                     MgbfsStateRingControl* ring){
  __shared__ uint32_t columns[8];
  if(threadIdx.x<8)columns[threadIdx.x]=0;
  __syncthreads();
  uint32_t row=threadIdx.x;
  if(row<n){
    const uint8_t* matrix=states+size_t(blockIdx.x)*stride;
    uint32_t selected=0,ones=0;
    for(uint32_t column=0;column<n;++column){
      uint8_t value=matrix[size_t(row)*n+column];
      if(value==1){selected=column;++ones;}
      else if(value!=0)atomicCAS(&ring->fatal,0u,18u);
    }
    if(ones!=1)atomicCAS(&ring->fatal,0u,18u);
    else {
      uint32_t mask=1u<<(selected&31),old=atomicOr(&columns[selected>>5],mask);
      if(old&mask)atomicCAS(&ring->fatal,0u,18u);
      out[size_t(blockIdx.x)*n+row]=uint8_t(selected);
    }
  }
}
}

extern "C" int mgbfs_archive_pack_permutation_u8(uint32_t n,uint32_t stride,
    const uint8_t* states,uint32_t count,uint8_t* permutations,
    MgbfsStateRingControl* ring,void* raw_stream){
  if(!n||n>255||stride<uint64_t(n)*n||!states||!permutations||!ring)return 1;
  if(!count)return 0;
  pack<<<count,256,0,static_cast<cudaStream_t>(raw_stream)>>>(n,stride,states,permutations,ring);
  return cudaGetLastError()==cudaSuccess?0:2;
}
