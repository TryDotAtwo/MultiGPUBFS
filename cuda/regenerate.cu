#include "regenerate.h"
#include <cuda_runtime.h>
#include <climits>
#include <cstddef>
static_assert(sizeof(MgbfsRegenerateOrigin)==16);
static_assert(offsetof(MgbfsRegenerateOrigin,parent)==8);

namespace {
__global__ void validate(const MgbfsRegenerateOrigin* requests,const uint32_t* count,
 uint32_t capacity,uint32_t source,uint32_t moves,uint64_t begin,uint32_t parents,uint32_t* fatal){
  if(*count>capacity){if(blockIdx.x==0&&threadIdx.x==0)atomicCAS(fatal,0u,1u);return;}
  for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<*count;i+=uint64_t(gridDim.x)*blockDim.x){
    const auto r=requests[i];
    if(r.source!=source||r.move>=moves||r.reserved||r.parent<begin||r.parent-begin>=parents)
      atomicCAS(fatal,0u,2u);
  }
}
__global__ void apply(uint32_t n,uint32_t modulus,uint32_t stride,uint64_t begin,
 const uint8_t* parents,const uint8_t* generators,const MgbfsRegenerateOrigin* requests,
 const uint32_t* count,uint8_t* output,const uint32_t* fatal){
  if(*fatal)return;
  // Dense response order is request order. Adjacent lanes write adjacent bytes;
  // source grouping/sorting belongs to the caller, which retains origin leases.
  for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<uint64_t(*count)*stride;i+=uint64_t(gridDim.x)*blockDim.x){
    const uint32_t row=uint32_t(i/stride),j=uint32_t(i%stride);
    uint32_t sum=0;
    if(j<uint64_t(n)*n){
      const auto r=requests[row];
      const auto p=parents+(r.parent-begin)*stride;
      const auto g=generators+uint64_t(r.move)*n*n;
      for(uint32_t k=0;k<n;++k)sum+=uint32_t(g[uint64_t(j/n)*n+k])*p[uint64_t(k)*n+j%n];
    }
    output[i]=uint8_t(sum%modulus);
  }
}
}
extern "C" int mgbfs_regenerate_selected(uint32_t n,uint32_t moves,uint32_t modulus,
 uint32_t stride,uint32_t capacity,uint32_t source,uint64_t begin,uint32_t parent_count,
 const uint8_t* parents,const uint8_t* generators,const MgbfsRegenerateOrigin* requests,
 const uint32_t* count,uint8_t* output,uint32_t* fatal,void* raw_stream){
  if(!n||n>uint32_t(INT_MAX)/65025||!moves||moves>65536||modulus<2||modulus>256||
     uint64_t(n)*n>stride||stride%16||!capacity||!parents||!generators||!requests||!count||!output||!fatal)return 1;
  auto stream=static_cast<cudaStream_t>(raw_stream);
  const uint32_t blocks=uint32_t((uint64_t(capacity)+255)/256>4096?4096:(uint64_t(capacity)+255)/256);
  validate<<<blocks,256,0,stream>>>(requests,count,capacity,source,moves,begin,parent_count,fatal);
  if(cudaGetLastError()!=cudaSuccess)return 2;
  apply<<<blocks,256,0,stream>>>(n,modulus,stride,begin,parents,generators,requests,count,output,fatal);
  return cudaGetLastError()==cudaSuccess?0:2;
}
