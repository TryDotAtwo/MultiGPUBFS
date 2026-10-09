#pragma once
#include <cuda_runtime.h>
#include <cstdint>
static __device__ __forceinline__ uint64_t mix64(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
static __device__ __forceinline__ uint64_t finish_hash(uint64_t h,uint32_t bits){h=mix64(h);return bits==64?h:(bits? h&((uint64_t(1)<<bits)-1):0);}
struct Action {
 uint32_t kind,elements,n,m,generators,count,stride;
 const int64_t* parents;const uint32_t* permutations;const int64_t* matrices;const uint32_t* moduli;
 __device__ int64_t value(uint32_t child,uint32_t element)const {
  const uint32_t parent=child/generators,g=child%generators;
  if(kind==0)return parents[uint64_t(permutations[uint64_t(g)*elements+element])*stride+parent];
  const uint32_t row=element/m,col=element%m;const int64_t* matrix=matrices+uint64_t(g)*n*n;
  uint64_t sum=0;for(uint32_t k=0;k<n;k++)sum+=uint64_t(matrix[uint64_t(row)*n+k])*uint64_t(parents[uint64_t(k*m+col)*stride+parent]);
  int64_t v=static_cast<int64_t>(sum);uint32_t mod=moduli[g];if(mod){v%=int64_t(mod);if(v<0)v+=mod;}return v;
 }
 __device__ uint64_t hash(uint32_t child,uint64_t seed,uint32_t bits)const {
  uint64_t h=seed;for(uint32_t e=0;e<elements;e++)h=mix64(h^uint64_t(value(child,e))^uint64_t(e));return finish_hash(h,bits);
 }
};
