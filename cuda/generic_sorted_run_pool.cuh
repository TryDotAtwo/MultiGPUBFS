#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// One bitmap owns 64 aligned pages. All shards share these regions.
// A run owns an aligned power-of-two page group, including multiple regions,
// and one descriptor at its first page.
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

__device__ inline unsigned long long generic_sorted_run_mask(uint32_t start,uint32_t size_class) {
 uint32_t pages=1u<<size_class;
 return pages==64 ? ~0ull : ((1ull<<pages)-1ull)<<start;
}
__device__ inline void generic_sorted_run_free_pages(GenericSortedRunPool pool,uint32_t slot,uint32_t size_class){
 uint32_t region=slot/64,start=slot%64;
 if(size_class<=6)atomicAnd(pool.occupied+region,~generic_sorted_run_mask(start,size_class));
 else{uint32_t regions=1u<<(size_class-6);for(uint32_t i=0;i<regions;++i)atomicExch(pool.occupied+region+i,0ull);}
}
__device__ inline GenericSortedRunAcquire generic_sorted_run_claimed_descriptor(
 GenericSortedRunPool pool,uint32_t slot,uint32_t size_class,
 GenericSortedRunToken* result,uint32_t* error){
 auto* descriptor=pool.descriptors+slot;
 unsigned long long prior=atomicCAS(&descriptor->lease,0ull,0ull);
 if((prior&0xffffffffull)!=0 || uint32_t(prior>>32)==0xffffffffu){
  atomicOr(error,512u);generic_sorted_run_free_pages(pool,slot,size_class);return SORTED_RUN_INVALID;
 }
 uint32_t generation=uint32_t(prior>>32)+1;
 descriptor->size_class=size_class;descriptor->count=0;__threadfence();
 atomicExch(&descriptor->lease,(uint64_t(generation)<<32)|1ull);
 *result={slot,generation};return SORTED_RUN_ACQUIRED;
}
__device__ inline GenericSortedRunAcquire generic_sorted_run_allocate(
 GenericSortedRunPool pool,uint32_t size_class,uint32_t first_region,
 GenericSortedRunToken* result,uint32_t* error) {
 GenericSortedRunPoolShape shape;
 if(size_class>31 || !generic_sorted_run_pool_shape(pool.regions,pool.page_entries,&shape)){
  atomicOr(error,32u);return SORTED_RUN_INVALID;
 }
 if(size_class>6){
  uint32_t needed=1u<<(size_class-6),groups=pool.regions/needed;
  if(!groups)return SORTED_RUN_PRESSURE;
  uint32_t first=(first_region/needed)%groups;
  for(uint32_t step=0;step<groups;++step){
   uint32_t base=uint32_t((uint64_t(first)+step)%groups)*needed,claimed=0;
   // No spin lock. An unsuccessful partial claim releases ONLY its own words.
   // Ascending, aligned groups can overlap in size, but cannot steal live credits.
   for(;claimed<needed;++claimed)
    if(atomicCAS(pool.occupied+base+claimed,0ull,~0ull)!=0ull)break;
   if(claimed==needed)
    return generic_sorted_run_claimed_descriptor(pool,base*64,size_class,result,error);
   for(uint32_t i=0;i<claimed;++i)atomicExch(pool.occupied+base+i,0ull);
  }
  return SORTED_RUN_PRESSURE;
 }
 uint32_t pages=1u<<size_class;
 for(uint32_t step=0;step<pool.regions;++step){
  uint32_t region=uint32_t((uint64_t(first_region)+step)%pool.regions);
  for(uint32_t retry=0;retry<32;++retry){
   unsigned long long observed=atomicCAS(pool.occupied+region,0ull,0ull);
   uint32_t start=64;
   for(uint32_t candidate=0;candidate<64;candidate+=pages)
    if(!(observed&generic_sorted_run_mask(candidate,size_class))){start=candidate;break;}
   if(start==64)break;
   auto mask=generic_sorted_run_mask(start,size_class);
   if(atomicCAS(pool.occupied+region,observed,observed|mask)!=observed)continue;
   return generic_sorted_run_claimed_descriptor(pool,region*64+start,size_class,result,error);
  }
 }
 return SORTED_RUN_PRESSURE;
}
// Only the sole unpublished writer may return unused tail credit. The prefix
// already contains the unique output; no hash/ref/state payload is copied.
__device__ inline bool generic_sorted_run_trim(GenericSortedRunPool pool,
 GenericSortedRunToken token,uint32_t count,uint32_t* error){
 if(!count||token.slot>=uint64_t(pool.regions)*64){atomicOr(error,32u);return false;}
 auto* d=pool.descriptors+token.slot;auto expected=(uint64_t(token.generation)<<32)|1ull;
 if(atomicCAS(&d->lease,expected,expected|SORTED_RUN_PUBLISHING)!=expected){atomicOr(error,512u);return false;}
 uint32_t prior=d->size_class;
 uint64_t needed=(uint64_t(count)+pool.page_entries-1)/pool.page_entries;
 uint32_t cls=needed<=1?0:32-__clz(uint32_t(needed-1));
 if(prior>31||cls>prior||uint64_t(count)>uint64_t(pool.page_entries)*(1ull<<prior)){
  atomicOr(error,32u);atomicExch(&d->lease,expected);return false;
 }
 uint32_t region=token.slot/64,start=token.slot%64;
 if(cls<prior){
  if(prior<=6){
   auto give_back=generic_sorted_run_mask(start,prior)&~generic_sorted_run_mask(start,cls);
   atomicAnd(pool.occupied+region,~give_back);
  }else{
   uint32_t old_regions=1u<<(prior-6),keep_regions=cls>6?1u<<(cls-6):1;
   if(cls<6)atomicAnd(pool.occupied+region,generic_sorted_run_mask(0,cls));
   for(uint32_t i=keep_regions;i<old_regions;++i)atomicExch(pool.occupied+region+i,0ull);
  }
  d->size_class=cls;
 }
 __threadfence();atomicExch(&d->lease,expected);return true;
}
__device__ inline bool generic_sorted_run_publish(GenericSortedRunPool pool,
 GenericSortedRunToken token,uint32_t count,uint32_t* error) {
 if(token.slot>=uint64_t(pool.regions)*64) {atomicOr(error,8u);return false;}
 auto* d=pool.descriptors+token.slot;
 auto expected=(uint64_t(token.generation)<<32)|1ull;
 auto publishing=expected|SORTED_RUN_PUBLISHING;
 // Claim publication before writing metadata: a stale token cannot corrupt a reused slot.
 if(atomicCAS(&d->lease,expected,publishing)!=expected) {atomicOr(error,512u);return false;}
 if(d->size_class>31 || uint64_t(count)>uint64_t(pool.page_entries)*(1ull<<d->size_class)) {
  atomicOr(error,32u);atomicExch(&d->lease,expected);return false;
 }
 // Caller has completed all writes in this stream before publication.
 d->count=count;__threadfence();
 atomicExch(&d->lease,expected|SORTED_RUN_PUBLISHED);
 return true;
}
__device__ inline bool generic_sorted_run_read_acquire(GenericSortedRunPool pool,
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
__device__ inline bool generic_sorted_run_release(GenericSortedRunPool pool,
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
   uint32_t size_class=d->size_class;
   __threadfence();generic_sorted_run_free_pages(pool,token.slot,size_class);
  }
  return true;
 }
}
