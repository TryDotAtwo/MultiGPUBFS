#include "mgbfs_cuda.h"
#include <cuda_runtime.h>
#include <cstdint>
namespace {
struct alignas(16) Key { uint32_t w[4]; };
__global__ void split(const Key* keys,uint32_t count,uint32_t* output){
  if(threadIdx.x||blockIdx.x)return;uint32_t lo=0,hi=count;
  while(lo<hi){uint32_t mid=lo+(hi-lo)/2;if((keys[mid].w[3]>>31)==0)lo=mid+1;else hi=mid;}
  output[0]=lo;output[1]=count-lo;
}
__global__ void gather(const uint4* source,const uint64_t* refs,uint4* output,uint32_t count,uint32_t chunks,uint32_t source_count,uint32_t* owner_counts){
  uint64_t p=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(p>=uint64_t(count)*chunks)return;uint32_t row=p/chunks,chunk=p%chunks;uint64_t ref=refs[row];if(ref>=source_count){atomicExch(owner_counts,UINT32_MAX);return;}output[uint64_t(row)*chunks+chunk]=source[ref*chunks+chunk];
}
}
extern "C" int mgbfs_exchange_pack(uint32_t stride,uint32_t capacity,const uint8_t* source_states,uint32_t source_count,const void* sorted_hashes,const uint64_t* sorted_refs,uint32_t count,uint8_t* packed_states,uint32_t* owner_counts,void* raw_stream){
  if(!stride||stride%16||!capacity||count>capacity||!source_states||!sorted_hashes||!sorted_refs||!packed_states||!owner_counts)return 1;auto stream=static_cast<cudaStream_t>(raw_stream);split<<<1,1,0,stream>>>(static_cast<const Key*>(sorted_hashes),count,owner_counts);if(count)gather<<<(uint64_t(count)*(stride/16)+255)/256,256,0,stream>>>(reinterpret_cast<const uint4*>(source_states),sorted_refs,reinterpret_cast<uint4*>(packed_states),count,stride/16,source_count,owner_counts);return cudaGetLastError()==cudaSuccess?0:2;
}
