#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// One bitmap owns 64 aligned pages. All shards share these regions.
// A run owns 1,2,4,...64 pages and one descriptor at its first page.
// No per-state pointers and no device locks; failed CAS is retryable pressure.
struct GenericSortedRunDescriptor {
 unsigned long long lease; // generation:32, retiring:1, published:1, publishing:1, refs:29
 uint32_t size_class;
 uint32_t count;
};
struct GenericSortedRunPool {
 unsigned long long* occupied;
 GenericSortedRunDescriptor* descriptors;
 uint32_t regions;
 uint32_t page_entries;
};
struct GenericSortedRunPoolShape {
 uint32_t entries;
 uint64_t bitmap_bytes, descriptor_bytes, hash_bytes, row_bytes, total_bytes;
};
// Same arithmetic is used by host admission and device validation. This is
// storage for run keys/refs only; canonical state and sort workspace are extra.
__host__ __device__ inline bool generic_sorted_run_pool_shape(
 uint32_t regions,uint32_t page_entries,GenericSortedRunPoolShape* shape) {
 *shape={};
 if(!regions||!page_entries||uint64_t(regions)*64>0xffffffffull)return false;
 uint64_t slots=uint64_t(regions)*64,entries=slots*page_entries;
 if(entries>0xffffffffull)return false;
 shape->entries=uint32_t(entries);
 shape->bitmap_bytes=uint64_t(regions)*sizeof(unsigned long long);
 shape->descriptor_bytes=slots*sizeof(GenericSortedRunDescriptor);
 shape->hash_bytes=entries*sizeof(uint64_t);shape->row_bytes=entries*sizeof(uint32_t);
 shape->total_bytes=shape->bitmap_bytes+shape->descriptor_bytes+shape->hash_bytes+shape->row_bytes;
 return true;
}
struct GenericSortedRunToken { uint32_t slot; uint32_t generation; };
enum GenericSortedRunAcquire { SORTED_RUN_ACQUIRED=0, SORTED_RUN_PRESSURE=1, SORTED_RUN_INVALID=2 };
static constexpr unsigned long long SORTED_RUN_REFS=0x1fffffffull;
static constexpr unsigned long long SORTED_RUN_PUBLISHING=0x20000000ull;
static constexpr unsigned long long SORTED_RUN_PUBLISHED=0x40000000ull;
static constexpr unsigned long long SORTED_RUN_RETIRING=0x80000000ull;

__device__ unsigned long long generic_sorted_run_mask(uint32_t start,uint32_t size_class) {
 uint32_t pages=1u<<size_class;
 return pages==64 ? ~0ull : ((1ull<<pages)-1ull)<<start;
}
__device__ GenericSortedRunAcquire generic_sorted_run_allocate(
 GenericSortedRunPool pool,uint32_t size_class,uint32_t first_region,
 GenericSortedRunToken* result,uint32_t* error) {
 GenericSortedRunPoolShape shape;
 if(size_class>6 || !generic_sorted_run_pool_shape(pool.regions,pool.page_entries,&shape)) {
  atomicOr(error,32u);return SORTED_RUN_INVALID;
 }
 uint32_t pages=1u<<size_class;
 for(uint32_t step=0;step<pool.regions;++step) {
  uint32_t region=uint32_t((uint64_t(first_region)+step)%pool.regions);
  for(uint32_t retry=0;retry<32;++retry) {
   unsigned long long observed=atomicCAS(pool.occupied+region,0ull,0ull);
   uint32_t start=64;
   for(uint32_t candidate=0;candidate<64;candidate+=pages) {
    if(!(observed&generic_sorted_run_mask(candidate,size_class))) {start=candidate;break;}
   }
   if(start==64)break;
   auto mask=generic_sorted_run_mask(start,size_class);
   if(atomicCAS(pool.occupied+region,observed,observed|mask)!=observed)continue;
   uint32_t slot=region*64+start;
   auto* descriptor=pool.descriptors+slot;
   unsigned long long prior=atomicCAS(&descriptor->lease,0ull,0ull);
   // The final reader releases the bitmap only after lease refs reach zero.
   if((prior&0xffffffffull)!=0 || uint32_t(prior>>32)==0xffffffffu) {
    atomicOr(error,512u);
    atomicAnd(pool.occupied+region,~mask);
    return SORTED_RUN_INVALID;
   }
   uint32_t generation=uint32_t(prior>>32)+1;
   descriptor->size_class=size_class;descriptor->count=0;
   __threadfence();
   atomicExch(&descriptor->lease,(uint64_t(generation)<<32)|1ull);
   *result={slot,generation};return SORTED_RUN_ACQUIRED;
  }
 }
 return SORTED_RUN_PRESSURE;
}
__device__ bool generic_sorted_run_publish(GenericSortedRunPool pool,
 GenericSortedRunToken token,uint32_t count,uint32_t* error) {
 if(token.slot>=uint64_t(pool.regions)*64) {atomicOr(error,8u);return false;}
 auto* d=pool.descriptors+token.slot;
 auto expected=(uint64_t(token.generation)<<32)|1ull;
 auto publishing=expected|SORTED_RUN_PUBLISHING;
 // Claim publication before writing metadata: a stale token cannot corrupt a reused slot.
 if(atomicCAS(&d->lease,expected,publishing)!=expected) {atomicOr(error,512u);return false;}
 if(d->size_class>6 || uint64_t(count)>uint64_t(pool.page_entries)*(1ull<<d->size_class)) {
  atomicOr(error,32u);atomicExch(&d->lease,expected);return false;
 }
 // Caller has completed all writes in this stream before publication.
 d->count=count;__threadfence();
 atomicExch(&d->lease,expected|SORTED_RUN_PUBLISHED);
 return true;
}
__device__ bool generic_sorted_run_read_acquire(GenericSortedRunPool pool,
 GenericSortedRunToken token,uint32_t* error) {
 if(token.slot>=uint64_t(pool.regions)*64) {atomicOr(error,8u);return false;}
 auto* d=pool.descriptors+token.slot;
 auto observed=atomicCAS(&d->lease,0ull,0ull);
 for(;;) {
  auto refs=observed&SORTED_RUN_REFS;
  if(uint32_t(observed>>32)!=token.generation || !(observed&SORTED_RUN_PUBLISHED)
    || (observed&SORTED_RUN_RETIRING) || !refs)return false;
  if(refs==SORTED_RUN_REFS) {atomicOr(error,512u);return false;}
  auto previous=atomicCAS(&d->lease,observed,observed+1ull);
  if(previous==observed) {__threadfence();return true;}
  observed=previous;
 }
}
__device__ bool generic_sorted_run_release(GenericSortedRunPool pool,
 GenericSortedRunToken token,bool owner,uint32_t* error) {
 if(token.slot>=uint64_t(pool.regions)*64) {atomicOr(error,8u);return false;}
 auto* d=pool.descriptors+token.slot;
 auto observed=atomicCAS(&d->lease,0ull,0ull);
 for(;;) {
  auto refs=observed&SORTED_RUN_REFS;
  if(uint32_t(observed>>32)!=token.generation || !refs) {atomicOr(error,512u);return false;}
  if((observed&SORTED_RUN_PUBLISHING) || (!owner && (!(observed&SORTED_RUN_PUBLISHED) || (refs==1 && !(observed&SORTED_RUN_RETIRING)))) || (owner && (observed&SORTED_RUN_RETIRING))) {atomicOr(error,512u);return false;}
  // An unpublished allocation may be aborted by its sole owner.
  auto desired=(observed-1ull)|(owner?SORTED_RUN_RETIRING:0ull);
  if(refs==1)desired=uint64_t(token.generation)<<32;
  auto previous=atomicCAS(&d->lease,observed,desired);
  if(previous!=observed) {observed=previous;continue;}
  if(refs==1) {
   uint32_t region=token.slot/64,start=token.slot%64;
   auto mask=generic_sorted_run_mask(start,d->size_class);
   __threadfence();atomicAnd(pool.occupied+region,~mask);
  }
  return true;
 }
}
