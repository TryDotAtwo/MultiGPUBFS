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
template<bool Tensor>
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
   if constexpr(Tensor){
     // PTX m8n8k16: lane/4 chooses A row / B column, lane%4 a
     // four-byte K fragment; each lane owns two adjacent result columns.
     // Every warp executes every MMA, including padded rows/columns/K.
     for(uint32_t r0=0;r0<n;r0+=8)for(uint32_t c0=0;c0<n;c0+=8){
       int32_t d0=0,d1=0;
       for(uint32_t k0=0;k0<n;k0+=16){
         uint32_t a=0,b=0;
         #pragma unroll
         for(unsigned i=0;i<4;++i){
           const uint32_t k=k0+(lane%4)*4+i,r=r0+lane/4,c=c0+lane/4;
           if(r<n&&k<n)a|=uint32_t(g[r*n+k])<<(8*i);
           if(c<n&&k<n)b|=uint32_t(p[k*n+c])<<(8*i);
         }
         #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
         asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
                      : "+r"(d0), "+r"(d1) : "r"(a), "r"(b));
         #else
         asm volatile("trap;");
         #endif
       }
       #pragma unroll
       for(unsigned i=0;i<2;++i){
         const uint32_t r=r0+lane/4,c=c0+(lane%4)*2+i;
         if(r<n&&c<n){
           const uint32_t child=uint32_t(i?d1:d0)%modulus;
           #pragma unroll
           for(unsigned h=0;h<4;++h)sums[h]+=uint64_t(child)*coefficients[uint64_t(r*n+c)*4+h];
         }
       }
     }
   }else{
   for(uint32_t j=lane;j<n*n;j+=32){
     uint32_t product=0;
     for(uint32_t k=0;k<n;++k)product+=uint32_t(g[(j/n)*n+k])*p[k*n+j%n];
     const uint32_t child=product%modulus;
     #pragma unroll
     for(unsigned h=0;h<4;++h)sums[h]+=uint64_t(child)*coefficients[uint64_t(j)*4+h];
   }
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
 generate<false><<<blocks,256,0,stream>>>(n,moves,modulus,stride,source,begin,parents,generators,
   coefficients,offsets,candidate_count,hashes,origins);
 return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_generate_hash_only_tc(
 uint32_t n,uint32_t moves,uint32_t modulus,uint32_t stride,uint32_t parent_capacity,
 uint32_t capacity,uint32_t source,uint64_t begin,const uint8_t* parents,
 const uint8_t* generators,const uint32_t* coefficients,const uint32_t* offsets,
 const uint32_t* parent_count,uint32_t* hashes,MgbfsRegenerateOrigin* origins,
 uint32_t* candidate_count,uint32_t* fatal,void* raw_stream){
 if(!n||uint64_t(n)*n>33025||!moves||moves>65536||modulus<2||modulus>256||
    uint64_t(n)*n>stride||stride%16||!parent_capacity||!capacity||!parents||
    !generators||!coefficients||!offsets||!parent_count||!hashes||!origins||
    !candidate_count||!fatal)return 1;
 // Explicit experimental SM75 policy; no scalar fallback on other devices.
 int device=0;cudaDeviceProp properties{};
 if(cudaGetDevice(&device)!=cudaSuccess||cudaGetDeviceProperties(&properties,device)!=cudaSuccess)return 2;
 if(properties.major!=7||properties.minor!=5)return 3;
 auto stream=static_cast<cudaStream_t>(raw_stream);
 validate<<<1,1,0,stream>>>(moves,parent_capacity,capacity,begin,parent_count,candidate_count,fatal);
 if(cudaGetLastError()!=cudaSuccess)return 2;
 const uint32_t blocks=uint32_t((uint64_t(capacity)+7)/8>4096?4096:(uint64_t(capacity)+7)/8);
 generate<true><<<blocks,256,0,stream>>>(n,moves,modulus,stride,source,begin,parents,generators,
   coefficients,offsets,candidate_count,hashes,origins);
 return cudaGetLastError()==cudaSuccess?0:2;
}
