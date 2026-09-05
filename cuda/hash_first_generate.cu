#include "hash_first_generate.h"
#include <cuda_runtime.h>
#include <cstdint>
namespace {
constexpr uint64_t prime=4294967291ULL;
__global__ void validate(uint32_t moves,uint32_t parents,uint32_t capacity,uint64_t begin,
 const uint32_t* count,uint32_t* output_count,uint32_t* fatal){
 *output_count=0;
 if(*fatal)return;
 const uint64_t size=uint64_t(*count)*moves;
 if(*count>parents||size>capacity){*fatal=1;return;}
 if(*count&&begin>UINT64_MAX-uint64_t(*count-1)){*fatal=2;return;}
 *output_count=uint32_t(size);
}
__global__ void generate(uint32_t n,uint32_t moves,uint32_t modulus,uint32_t stride,
 uint32_t source,uint64_t begin,const uint8_t* parents,const uint8_t* generators,
 const uint32_t* coefficients,const uint32_t* offsets,const uint32_t* count,
 uint32_t* hashes,MgbfsRegenerateOrigin* origins){
 const unsigned lane=threadIdx.x%32;
 const uint64_t warp=(uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
 const uint64_t warp_stride=uint64_t(gridDim.x)*blockDim.x/32;
 for(uint64_t row=warp;row<*count;row+=warp_stride){
   const uint64_t parent=row/moves;
   const uint32_t move=uint32_t(row%moves);
   const uint8_t* p=parents+parent*stride;
   const uint8_t* g=generators+uint64_t(move)*n*n;
   uint64_t sums[4]={0,0,0,0};
   for(uint32_t j=lane;j<n*n;j+=32){
     uint32_t product=0;
     for(uint32_t k=0;k<n;++k)product+=uint32_t(g[(j/n)*n+k])*p[k*n+j%n];
     const uint32_t child=product%modulus;
     #pragma unroll
     for(unsigned h=0;h<4;++h)sums[h]+=uint64_t(child)*coefficients[uint64_t(j)*4+h];
   }
   // Width <=33025: whole-state sum <=33025*255*(p-1), safely below 2^64.
   for(unsigned delta=16;delta;delta/=2){
     #pragma unroll
     for(unsigned h=0;h<4;++h)sums[h]+=__shfl_down_sync(0xffffffffu,sums[h],delta);
   }
   // Broadcast each completed reduction before coalesced four-word stores.
   #pragma unroll
   for(unsigned h=0;h<4;++h)sums[h]=__shfl_sync(0xffffffffu,sums[h],0);
   if(lane<4)hashes[row*4+lane]=uint32_t((sums[lane]+offsets[lane])%prime);
   if(lane==0)origins[row]={source,uint16_t(move),0,begin+parent};
 }
}
}
extern "C" int mgbfs_generate_hash_only(
 uint32_t n,uint32_t moves,uint32_t modulus,uint32_t stride,uint32_t parent_capacity,
 uint32_t capacity,uint32_t source,uint64_t begin,const uint8_t* parents,
 const uint8_t* generators,const uint32_t* coefficients,const uint32_t* offsets,
 const uint32_t* parent_count,uint32_t* hashes,MgbfsRegenerateOrigin* origins,
 uint32_t* candidate_count,uint32_t* fatal,void* raw_stream){
 if(!n||uint64_t(n)*n>33025||!moves||moves>65536||modulus<2||modulus>256||
    uint64_t(n)*n>stride||stride%16||!parent_capacity||!capacity||!parents||
    !generators||!coefficients||!offsets||!parent_count||!hashes||!origins||
    !candidate_count||!fatal)return 1;
 auto stream=static_cast<cudaStream_t>(raw_stream);
 validate<<<1,1,0,stream>>>(moves,parent_capacity,capacity,begin,parent_count,candidate_count,fatal);
 if(cudaGetLastError()!=cudaSuccess)return 2;
 const uint32_t blocks=uint32_t((uint64_t(capacity)+7)/8>4096?4096:(uint64_t(capacity)+7)/8);
 generate<<<blocks,256,0,stream>>>(n,moves,modulus,stride,source,begin,parents,generators,
   coefficients,offsets,candidate_count,hashes,origins);
 return cudaGetLastError()==cudaSuccess?0:2;
}
