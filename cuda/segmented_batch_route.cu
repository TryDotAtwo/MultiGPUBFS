// Bounded batch route prototype. No full-layer children, allocation, or CPU
// readback. A completed tile is scattered only after every queue fits.
#include "generic_route.h"
#include "generic_action.cuh"
#include "permutation_bitmap.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <limits>

namespace {
constexpr size_t kAlignment=256;
struct BatchWorkspace {
 size_t keys_in,keys_out,origins_in,origins_out,identity,begins,ends,temp,total,temp_bytes;
};
bool advance(size_t &cursor,size_t bytes,size_t *offset){
 if(cursor>std::numeric_limits<size_t>::max()-(kAlignment-1))return false;
 cursor=(cursor+kAlignment-1)&~(kAlignment-1);*offset=cursor;
 if(bytes>std::numeric_limits<size_t>::max()-cursor)return false;
 cursor+=bytes;return true;
}
int bits_needed(uint32_t n){int b=0;uint32_t v=n-1;while(v){v>>=1;++b;}return b?b:1;}
cudaError_t layout(uint32_t children,uint32_t buckets,BatchWorkspace *w){
 if(!w||!buckets||buckets>128u*4096u||children>=0x7fffffffU)return cudaErrorInvalidValue;
 *w={};if(!children)return cudaSuccess;
 size_t temp=0;
 cudaError_t rc=cub::DeviceRadixSort::SortPairs(nullptr,temp,
  static_cast<const uint32_t*>(nullptr),static_cast<uint32_t*>(nullptr),
  static_cast<const uint32_t*>(nullptr),static_cast<uint32_t*>(nullptr),
  int(children),0,bits_needed(buckets));
 if(rc!=cudaSuccess)return rc;
 size_t cursor=0;
 if(!advance(cursor,size_t(children)*4,&w->keys_in)||
    !advance(cursor,size_t(children)*4,&w->keys_out)||
    !advance(cursor,size_t(children)*4,&w->origins_in)||
    !advance(cursor,size_t(children)*4,&w->origins_out)||
    !advance(cursor,size_t(children)*8,&w->identity)||
    !advance(cursor,size_t(buckets)*4,&w->begins)||
    !advance(cursor,size_t(buckets)*4,&w->ends)||
    !advance(cursor,temp,&w->temp))return cudaErrorInvalidValue;
 w->temp_bytes=temp;w->total=cursor;return cudaSuccess;
}
template<class T,bool Bitmap>
__global__ void prepare(ActionT<T> action,BitmapDeviceDomain domain,uint32_t word_shard,
 uint64_t seed,uint32_t bits,uint32_t world,uint32_t shards,const uint32_t* map,
 const uint64_t* cuts,uint32_t* keys,uint32_t* origins,uint64_t* identity,uint32_t* error){
 uint32_t count=action.count*action.generators;
 for(uint32_t child=blockIdx.x*blockDim.x+threadIdx.x;child<count;child+=blockDim.x*gridDim.x){
  uint64_t id=0;uint32_t owner=0,shard=0,valid=1;
  if constexpr(Bitmap){
   if(!bitmap_rank(domain,action,child,&id)){atomicOr(error,8u);valid=0;}
   if(valid){
    owner=uint32_t(id%world);
    uint64_t key=word_shard?((id/world)>>6):id;
    shard=uint32_t((uint64_t(uint32_t(mix64(key)))*shards)>>32);
   }
  }else{
   id=action.hash(child,seed,bits);
   uint32_t high=uint32_t(id>>32);
   if(cuts){
    if(cuts[0]!=0||cuts[world]!=(uint64_t(1)<<32)){atomicOr(error,4u);valid=0;}
    if(valid){
     uint32_t low=0,upper=world;
     while(low+1<upper){uint32_t mid=low+(upper-low)/2;if(uint64_t(high)<cuts[mid])upper=mid;else low=mid;}
     owner=low;if(!(cuts[owner]<=high&&high<cuts[owner+1])){atomicOr(error,4u);valid=0;}
    }
   }else owner=uint32_t((uint64_t(high)*world)>>32);
   shard=uint32_t((uint64_t(uint32_t(id))*shards)>>32);
  }
  uint32_t rank=valid?(map?map[owner]:owner):0;
  if(rank>=world){atomicOr(error,2u);valid=0;rank=0;}
  keys[child]=valid?rank*shards+shard:0;origins[child]=child;identity[child]=id;
 }
}
__global__ void mark_bounds(const uint32_t* sorted,uint32_t n,uint32_t buckets,
 uint32_t* begins,uint32_t* ends,const uint32_t* error){
 if(*error)return;
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
  uint32_t bucket=sorted[i];if(bucket>=buckets)continue;
  if(!i||sorted[i-1]!=bucket)begins[bucket]=i;
  if(i+1==n||sorted[i+1]!=bucket)ends[bucket]=i+1;
 }
}
__global__ void finish_counts(const uint32_t* begins,const uint32_t* ends,
 uint32_t buckets,uint32_t capacity,uint32_t* counts,uint32_t* error){
 for(uint32_t q=blockIdx.x*blockDim.x+threadIdx.x;q<buckets;q+=blockDim.x*gridDim.x){
  uint32_t count=ends[q]-begins[q];counts[q]=count;
  if(count>capacity)atomicOr(error,1u);
 }
}
// Origin records carry shard rather than queue; avoid relying on launch geometry.
template<class T,bool Packed>
__global__ void scatter_exact(ActionT<T> action,uint32_t source,uint32_t shards,uint32_t capacity,uint64_t begin,
 const uint32_t* sorted_keys,const uint32_t* sorted_origins,const uint64_t* identity,
 const uint32_t* begins,uint32_t n,GenericRouteRecord* output,T* states,const uint32_t* error){
 if(*error)return;
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
  uint32_t queue=sorted_keys[i],child=sorted_origins[i],row=i-begins[queue];
  if constexpr(Packed){
   uint64_t lo=0,hi=0,tail=0;
   for(uint32_t e=0;e<action.elements;e++){
    uint64_t v=uint64_t(action.value(child,e));
    if(e<8)lo|=v<<(8*e);else if(e<16)hi|=v<<(8*(e-8));else tail|=v<<(8*(e-16));
   }
   output[uint64_t(queue)*capacity+row]={identity[child],lo,uint32_t(hi),uint32_t(hi>>32),uint32_t(tail),uint32_t(tail>>32)};
  }else{
   output[uint64_t(queue)*capacity+row]={identity[child],begin+child/action.generators,source,child%action.generators,queue%shards,0};
   for(uint32_t e=0;e<action.elements;e++)states[(uint64_t(queue)*action.elements+e)*capacity+row]=T(action.value(child,e));
  }
 }
}
template<class T,bool Bitmap,bool Packed>
int launch(ActionT<T> action,BitmapDeviceDomain domain,uint32_t word_shard,
 uint64_t seed,uint32_t bits,uint32_t world,uint32_t source,uint32_t shards,uint32_t capacity,
 uint64_t begin,const uint32_t* map,const uint64_t* cuts,GenericRouteRecord* queues,
 uint32_t* counts,uint32_t* error,void* states,void* scratch,const BatchWorkspace& w,cudaStream_t stream){
 uint32_t n=action.count*action.generators,buckets=world*shards;
 cudaError_t rc=cudaMemsetAsync(counts,0,size_t(buckets)*4,stream);if(rc!=cudaSuccess)return int(rc);
 if(!n)return int(cudaSuccess);
 char* base=static_cast<char*>(scratch);
 auto* ki=reinterpret_cast<uint32_t*>(base+w.keys_in);auto* ko=reinterpret_cast<uint32_t*>(base+w.keys_out);
 auto* oi=reinterpret_cast<uint32_t*>(base+w.origins_in);auto* oo=reinterpret_cast<uint32_t*>(base+w.origins_out);
 auto* ids=reinterpret_cast<uint64_t*>(base+w.identity);auto* first=reinterpret_cast<uint32_t*>(base+w.begins);
 auto* last=reinterpret_cast<uint32_t*>(base+w.ends);
 rc=cudaMemsetAsync(first,0,size_t(buckets)*4,stream);if(rc!=cudaSuccess)return int(rc);
 rc=cudaMemsetAsync(last,0,size_t(buckets)*4,stream);if(rc!=cudaSuccess)return int(rc);
 uint32_t blocks=(n+255)/256;if(blocks>65535)blocks=65535;
 prepare<T,Bitmap><<<blocks,256,0,stream>>>(action,domain,word_shard,seed,bits,world,shards,map,cuts,ki,oi,ids,error);
 if((rc=cudaGetLastError())!=cudaSuccess)return int(rc);
 size_t temp_bytes=w.temp_bytes;
 rc=cub::DeviceRadixSort::SortPairs(base+w.temp,temp_bytes,ki,ko,oi,oo,int(n),0,bits_needed(buckets),stream);
 if(rc!=cudaSuccess)return int(rc);
 mark_bounds<<<blocks,256,0,stream>>>(ko,n,buckets,first,last,error);
 if((rc=cudaGetLastError())!=cudaSuccess)return int(rc);
 uint32_t qblocks=(buckets+255)/256;if(qblocks>65535)qblocks=65535;
 finish_counts<<<qblocks,256,0,stream>>>(first,last,buckets,capacity,counts,error);
 if((rc=cudaGetLastError())!=cudaSuccess)return int(rc);
 scatter_exact<T,Packed><<<blocks,256,0,stream>>>(action,source,shards,capacity,begin,ko,oo,ids,first,n,queues,static_cast<T*>(states),error);
 return int(cudaGetLastError());
}
}
extern "C" int mgbfs_segmented_batch_route_scratch_bytes(uint32_t children,uint32_t buckets,size_t* bytes){
 if(!bytes)return int(cudaErrorInvalidValue);BatchWorkspace w{};cudaError_t rc=layout(children,buckets,&w);
 if(rc==cudaSuccess)*bytes=w.total;return int(rc);
}
extern "C" int mgbfs_segmented_batch_route(uint32_t bitmap,uint32_t word_shard,BitmapDeviceDomain domain,
 uint32_t state_bytes,uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const void* parents,uint32_t count,uint64_t stride,const uint32_t* permutations,const int64_t* matrices,
 const uint32_t* moduli,uint64_t seed,uint32_t bits,uint32_t world,uint32_t source,uint32_t shards,
 uint32_t capacity,uint64_t begin,const uint32_t* map,const uint64_t* cuts,GenericRouteRecord* queues,
 uint32_t* counts,uint32_t* error,void* states,void* scratch,size_t scratch_bytes,void* opaque_stream){
 if(bitmap>1||word_shard>1||(word_shard&&!bitmap)||(state_bytes!=1&&state_bytes!=8)||
  kind>1||(state_bytes==1&&kind!=0)||!elements||!n||!m||!generators||count>stride||
  uint64_t(count)*generators>=0x7fffffffULL||bits>64||!world||world>128||source>=world||
  !shards||shards>4096||!capacity||begin>UINT64_MAX-count||!parents||!queues||!counts||!error||
  ((state_bytes!=1||elements>24)&&!states)||
  (kind==0&&(!permutations||elements!=n||m!=1))||
  (kind==1&&(!matrices||!moduli||uint64_t(n)*m!=elements))||
  (bitmap&&(kind!=0||domain.elements!=elements||!domain.symbols||!domain.counts||!domain.total))||
  uint64_t(world)*shards>SIZE_MAX/sizeof(GenericRouteRecord)/capacity)return int(cudaErrorInvalidValue);
 uint32_t children=count*generators;BatchWorkspace w{};cudaError_t rc=layout(children,world*shards,&w);
 if(rc!=cudaSuccess)return int(rc);
 if(children&&(!scratch||uintptr_t(scratch)%kAlignment||scratch_bytes<w.total))return int(cudaErrorInvalidValue);
 cudaStream_t st=static_cast<cudaStream_t>(opaque_stream);
 if(state_bytes==1){
  ActionT<uint8_t> a{kind,elements,n,m,generators,count,stride,static_cast<const uint8_t*>(parents),permutations,matrices,moduli};
  if(bitmap){
   if(elements<=24)return launch<uint8_t,true,true>(a,domain,word_shard,seed,bits,world,source,shards,capacity,begin,map,cuts,queues,counts,error,states,scratch,w,st);
   return launch<uint8_t,true,false>(a,domain,word_shard,seed,bits,world,source,shards,capacity,begin,map,cuts,queues,counts,error,states,scratch,w,st);
  }
  if(elements<=24)return launch<uint8_t,false,true>(a,domain,word_shard,seed,bits,world,source,shards,capacity,begin,map,cuts,queues,counts,error,states,scratch,w,st);
  return launch<uint8_t,false,false>(a,domain,word_shard,seed,bits,world,source,shards,capacity,begin,map,cuts,queues,counts,error,states,scratch,w,st);
 }
 ActionT<int64_t> a{kind,elements,n,m,generators,count,stride,static_cast<const int64_t*>(parents),permutations,matrices,moduli};
 if(bitmap)return launch<int64_t,true,false>(a,domain,word_shard,seed,bits,world,source,shards,capacity,begin,map,cuts,queues,counts,error,states,scratch,w,st);
 return launch<int64_t,false,false>(a,domain,word_shard,seed,bits,world,source,shards,capacity,begin,map,cuts,queues,counts,error,states,scratch,w,st);
}

// Translate compact ordinals to the original source*q+row references.  The
// offsets belong to one shard and are stable until all of its tiles drain.
// Returning an invalid physical reference on bad metadata makes the existing
// incoming-action validation report a fatal error rather than reading memory.
namespace {
__global__ void dense_origins(const uint32_t* offsets,uint32_t world,uint32_t q,
 uint32_t begin,uint32_t n,uint32_t* out){
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
  uint32_t ordinal=begin+i;
  uint32_t low=0,high=world+1;
  while(low<high){uint32_t mid=low+(high-low)/2;
   if(offsets[mid]<=ordinal)low=mid+1;else high=mid;
  }
  if(!low||low>world||ordinal>=offsets[world]){out[i]=UINT32_MAX;continue;}
  uint32_t source=low-1,row=ordinal-offsets[source];
  out[i]=row<q?source*q+row:UINT32_MAX;
 }
}
}
extern "C" int mgbfs_segmented_dense_origins(const uint32_t* offsets,uint32_t world,uint32_t q,
 uint32_t begin,uint32_t selected,uint32_t* out,void* opaque_stream){
 if(!offsets||!out||!world||world>128||!q||uint64_t(world)*q>=0x80000000ULL||
    uint64_t(begin)+selected>uint64_t(world)*q)return int(cudaErrorInvalidValue);
 if(!selected)return int(cudaSuccess);
 uint32_t blocks=(selected+255)/256;if(blocks>65535)blocks=65535;
 dense_origins<<<blocks,256,0,static_cast<cudaStream_t>(opaque_stream)>>>(offsets,world,q,begin,selected,out);
 return int(cudaGetLastError());
}
