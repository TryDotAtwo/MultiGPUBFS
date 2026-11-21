#pragma once
#include <stdint.h>

// A permutation of [0,total): full stripes are rotated among owners; the
// final incomplete stripe keeps its original cyclic ordering. Each bit and
// stored state still has one exact ID, and per-owner bitmap lengths stay put.
static __device__ __forceinline__ uint64_t mgbfs_bitmap_page_mix(uint64_t x){
 x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);
}
static __device__ __forceinline__ uint64_t mgbfs_bitmap_permute_owner_id(
 uint64_t id,uint32_t world,uint32_t page,uint64_t boundary){
 if(!page||id>=boundary)return id;
 const unsigned shift=31u-__clz(page);
 const uint64_t page_index=id>>shift,stripe=page_index/world;
 const uint64_t slot=page_index%world,owner=(slot+(mgbfs_bitmap_page_mix(stripe)%world))%world;
 const uint64_t local=(stripe<<shift)+(id&(uint64_t(page)-1));
 return local*world+owner;
}
