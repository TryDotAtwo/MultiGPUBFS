// Bounded batch route prototype. No full-layer children, allocation, or CPU
// readback. A completed tile is scattered only after every queue fits.
#include "generic_route.h"
#include "generic_action.cuh"
#include "packed_route_record.cuh"
#include "permutation_bitmap.cuh"
#include "bitmap_owner_page.cuh"
#include "bitmap_pack4_action.cuh"
#include "bitmap_pack4_route.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace {
constexpr size_t kAlignment=256;
struct BatchWorkspace {
 size_t keys_in,keys_out,origins_in,origins_out,identity,bounds,temp,total,temp_bytes;
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
    !advance(cursor,size_t(buckets)*2*sizeof(uint32_t),&w->bounds)||
    !advance(cursor,temp,&w->temp))return cudaErrorInvalidValue;
 w->temp_bytes=temp;w->total=cursor;return cudaSuccess;
}
static __device__ __constant__ uint32_t batch_bitmap_owner_page=0;
static __device__ __constant__ uint64_t batch_bitmap_owner_boundary=0;
extern "C" int mgbfs_bitmap_batch_set_owner_page(uint32_t page,uint64_t boundary){
 cudaError_t code=cudaMemcpyToSymbol(batch_bitmap_owner_page,&page,sizeof(page));if(code!=cudaSuccess)return int(code);
 return int(cudaMemcpyToSymbol(batch_bitmap_owner_boundary,&boundary,sizeof(boundary)));
}

// Read the parent before applying any generator. Each parent rank is
// reusable across every child in the tile; the usual route retains its state payload.
template<class T>
struct ParentCoordinates {
 const T* states;
 uint64_t stride;
 __device__ int64_t value(uint32_t row,uint32_t element) const {
  return states[uint64_t(element)*stride+row];
 }
};

template<class T>
__global__ void rank_tile_parents(const T* states,uint64_t stride,uint32_t count,
                                  BitmapDeviceDomain domain,uint32_t* ranks,uint32_t* error){
 const ParentCoordinates<T> parent{states,stride};
 for(uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;row<count;row+=blockDim.x*gridDim.x){
  uint64_t id=0;
  if(!bitmap_rank(domain,parent,row,&id)||id>UINT32_MAX){
   atomicOr(error,8u);
   id=0;
  }
  ranks[row]=uint32_t(id);
 }
}

template<class Action,bool Bitmap,bool Immutable>
__global__ void prepare(Action action,BitmapDeviceDomain domain,uint32_t word_shard,
 uint64_t seed,uint32_t bits,uint32_t world,uint32_t shards,const uint32_t* map,
 const uint64_t* cuts,uint32_t* keys,uint32_t* origins,uint64_t* identity,uint32_t* error,uint32_t layout_page,uint64_t layout_boundary,const uint32_t* parent_ranks,const uint64_t* xor_masks){
 uint32_t count=action.count*action.generators;
 for(uint32_t child=blockIdx.x*blockDim.x+threadIdx.x;child<count;child+=blockDim.x*gridDim.x){
  uint64_t id=0;uint32_t owner=0,shard=0,valid=1;
  if constexpr(Bitmap){
   if constexpr(std::is_same_v<Action,BitmapPack4Action>){
    if(!action.rank_candidate(domain,child,&id)){atomicOr(error,8u);valid=0;}
   }else if(parent_ranks){
    id=uint64_t(parent_ranks[child/action.generators])^xor_masks[child%action.generators];
    if(id>=domain.total){atomicOr(error,8u);valid=0;}
   }else if(!bitmap_rank(domain,action,child,&id)){atomicOr(error,8u);valid=0;}
   if(valid){
    if constexpr(Immutable)id=mgbfs_bitmap_permute_owner_id(id,world,layout_page,layout_boundary);
    else id=mgbfs_bitmap_permute_owner_id(id,world,batch_bitmap_owner_page,batch_bitmap_owner_boundary);
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
template<class T,bool Packed,bool Compact,class Action>
__global__ void scatter_exact(Action action,uint32_t source,uint32_t shards,uint32_t capacity,uint64_t begin,
 const uint32_t* sorted_keys,const uint32_t* sorted_origins,const uint64_t* identity,
 const uint32_t* begins,const uint32_t* ends,uint32_t n,GenericRouteRecord* output,T* states,const uint32_t* error){
 if(*error)return;
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
  uint32_t queue=sorted_keys[i],child=sorted_origins[i],row=i-begins[queue];
  const uint64_t output_row=Compact?i:uint64_t(queue)*capacity+row;
  if constexpr(Packed){
   output[output_row]=pack_route_state(action,child,identity[child]);
  }else{
   output[output_row]={identity[child],begin+child/action.generators,source,child%action.generators,queue%shards,0};
   for(uint32_t e=0;e<action.elements;e++){
    const uint64_t state_row=Compact?uint64_t(begins[queue])*action.elements+uint64_t(e)*(ends[queue]-begins[queue])+row:
     (uint64_t(queue)*action.elements+e)*capacity+row;
    states[state_row]=T(action.value(child,e));
   }
  }
 }
}
template<class T,bool Bitmap,bool Packed,bool Immutable,bool Compact,class Action>
int launch(Action action,BitmapDeviceDomain domain,uint32_t word_shard,
 uint64_t seed,uint32_t bits,uint32_t world,uint32_t source,uint32_t shards,uint32_t capacity,
 uint64_t begin,const uint32_t* map,const uint64_t* cuts,GenericRouteRecord* queues,
 uint32_t* counts,uint32_t* error,void* states,void* scratch,const BatchWorkspace& w,cudaStream_t stream,uint32_t layout_page,uint64_t layout_boundary,const uint64_t* xor_masks=nullptr){
 uint32_t n=action.count*action.generators,buckets=world*shards;
 cudaError_t rc=cudaMemsetAsync(counts,0,size_t(buckets)*4,stream);if(rc!=cudaSuccess)return int(rc);
 if(!n)return int(cudaSuccess);
 char* base=static_cast<char*>(scratch);
 auto* ki=reinterpret_cast<uint32_t*>(base+w.keys_in);auto* ko=reinterpret_cast<uint32_t*>(base+w.keys_out);
 auto* oi=reinterpret_cast<uint32_t*>(base+w.origins_in);auto* oo=reinterpret_cast<uint32_t*>(base+w.origins_out);
 auto* ids=reinterpret_cast<uint64_t*>(base+w.identity);auto* first=reinterpret_cast<uint32_t*>(base+w.bounds);
 auto* last=first+buckets;
 rc=cudaMemsetAsync(first,0,size_t(buckets)*2*sizeof(uint32_t),stream);if(rc!=cudaSuccess)return int(rc);
 uint32_t blocks=(n+255)/256;if(blocks>65535)blocks=65535;
 const uint32_t* parent_ranks=nullptr;
 if constexpr(Bitmap&&std::is_same_v<Action,ActionT<T>>){
  if(xor_masks){
   // The radix sort writes origins_out only after prepare has consumed the ranks.
   rank_tile_parents<<<(action.count+255)/256,256,0,stream>>>(action.parents,action.stride,action.count,domain,oo,error);
   if((rc=cudaGetLastError())!=cudaSuccess)return int(rc);
   parent_ranks=oo;
  }
 }
 prepare<Action,Bitmap,Immutable><<<blocks,256,0,stream>>>(action,domain,word_shard,seed,bits,world,shards,map,cuts,ki,oi,ids,error,layout_page,layout_boundary,parent_ranks,xor_masks);
 if((rc=cudaGetLastError())!=cudaSuccess)return int(rc);
 size_t temp_bytes=w.temp_bytes;
 rc=cub::DeviceRadixSort::SortPairs(base+w.temp,temp_bytes,ki,ko,oi,oo,int(n),0,bits_needed(buckets),stream);
 if(rc!=cudaSuccess)return int(rc);
 mark_bounds<<<blocks,256,0,stream>>>(ko,n,buckets,first,last,error);
 if((rc=cudaGetLastError())!=cudaSuccess)return int(rc);
 uint32_t qblocks=(buckets+255)/256;if(qblocks>65535)qblocks=65535;
 finish_counts<<<qblocks,256,0,stream>>>(first,last,buckets,capacity,counts,error);
 if((rc=cudaGetLastError())!=cudaSuccess)return int(rc);
 scatter_exact<T,Packed,Compact><<<blocks,256,0,stream>>>(action,source,shards,capacity,begin,ko,oo,ids,first,last,n,queues,static_cast<T*>(states),error);
 return int(cudaGetLastError());
}
}
extern "C" int mgbfs_segmented_batch_route_scratch_bytes(uint32_t children,uint32_t buckets,size_t* bytes){
 if(!bytes)return int(cudaErrorInvalidValue);BatchWorkspace w{};cudaError_t rc=layout(children,buckets,&w);
 if(rc==cudaSuccess)*bytes=w.total;return int(rc);
}
template<bool Immutable,bool Compact=false>
static int mgbfs_segmented_batch_route_impl(uint32_t bitmap,uint32_t word_shard,BitmapDeviceDomain domain,
 uint32_t state_bytes,uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const void* parents,uint32_t count,uint64_t stride,const uint32_t* permutations,const int64_t* matrices,
 const uint32_t* moduli,uint64_t seed,uint32_t bits,uint32_t world,uint32_t source,uint32_t shards,
 uint32_t capacity,uint64_t begin,const uint32_t* map,const uint64_t* cuts,GenericRouteRecord* queues,
 uint32_t* counts,uint32_t* error,void* states,void* scratch,size_t scratch_bytes,void* opaque_stream,uint32_t layout_page,uint64_t layout_boundary,const uint64_t* xor_masks=nullptr){
 if constexpr(Immutable){
  if(bitmap&&((!layout_page&&layout_boundary)||(layout_page&&((layout_page&(layout_page-1))||layout_page>(1u<<20)||!world||layout_boundary!=domain.total-domain.total%(uint64_t(layout_page)*world)))))return int(cudaErrorInvalidValue);
 }
 if((xor_masks&&(!bitmap||domain.total>(uint64_t{1}<<32)))||bitmap>1||word_shard>1||(word_shard&&!bitmap)||(state_bytes!=1&&state_bytes!=8)||
  kind>1||(state_bytes==1&&kind!=0)||!elements||!n||!m||!generators||count>stride||
  uint64_t(count)*generators>=0x7fffffffULL||bits>64||!world||world>128||source>=world||
  !shards||shards>4096||!capacity||begin>UINT64_MAX-count||!parents||!queues||!counts||!error||
  ((state_bytes!=1||elements>packed_route_max_elements)&&!states)||
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
   if(elements<=packed_route_max_elements)return launch<uint8_t,true,true,Immutable,Compact>(a,domain,word_shard,seed,bits,world,source,shards,capacity,begin,map,cuts,queues,counts,error,states,scratch,w,st,layout_page,layout_boundary,xor_masks);
   return launch<uint8_t,true,false,Immutable,Compact>(a,domain,word_shard,seed,bits,world,source,shards,capacity,begin,map,cuts,queues,counts,error,states,scratch,w,st,layout_page,layout_boundary,xor_masks);
  }
  if(elements<=packed_route_max_elements)return launch<uint8_t,false,true,Immutable,Compact>(a,domain,word_shard,seed,bits,world,source,shards,capacity,begin,map,cuts,queues,counts,error,states,scratch,w,st,layout_page,layout_boundary,xor_masks);
  return launch<uint8_t,false,false,Immutable,Compact>(a,domain,word_shard,seed,bits,world,source,shards,capacity,begin,map,cuts,queues,counts,error,states,scratch,w,st,layout_page,layout_boundary,xor_masks);
 }
 ActionT<int64_t> a{kind,elements,n,m,generators,count,stride,static_cast<const int64_t*>(parents),permutations,matrices,moduli};
 if(bitmap)return launch<int64_t,true,false,Immutable,Compact>(a,domain,word_shard,seed,bits,world,source,shards,capacity,begin,map,cuts,queues,counts,error,states,scratch,w,st,layout_page,layout_boundary,xor_masks);
 return launch<int64_t,false,false,Immutable,Compact>(a,domain,word_shard,seed,bits,world,source,shards,capacity,begin,map,cuts,queues,counts,error,states,scratch,w,st,layout_page,layout_boundary,xor_masks);
}
extern "C" int mgbfs_segmented_batch_route(uint32_t bitmap,uint32_t word_shard,BitmapDeviceDomain domain,
 uint32_t state_bytes,uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const void* parents,uint32_t count,uint64_t stride,const uint32_t* permutations,const int64_t* matrices,
 const uint32_t* moduli,uint64_t seed,uint32_t bits,uint32_t world,uint32_t source,uint32_t shards,
 uint32_t capacity,uint64_t begin,const uint32_t* map,const uint64_t* cuts,GenericRouteRecord* queues,
 uint32_t* counts,uint32_t* error,void* states,void* scratch,size_t scratch_bytes,void* opaque_stream){return mgbfs_segmented_batch_route_impl<false>(bitmap,word_shard,domain,state_bytes,kind,elements,n,m,generators,parents,count,stride,permutations,matrices,moduli,seed,bits,world,source,shards,capacity,begin,map,cuts,queues,counts,error,states,scratch,scratch_bytes,opaque_stream,0,0);}
extern "C" int mgbfs_segmented_batch_route_v2(uint32_t bitmap,uint32_t word_shard,BitmapDeviceDomain domain,
 uint32_t state_bytes,uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const void* parents,uint32_t count,uint64_t stride,const uint32_t* permutations,const int64_t* matrices,
 const uint32_t* moduli,uint64_t seed,uint32_t bits,uint32_t world,uint32_t source,uint32_t shards,
 uint32_t capacity,uint64_t begin,const uint32_t* map,const uint64_t* cuts,GenericRouteRecord* queues,
 uint32_t* counts,uint32_t* error,void* states,void* scratch,size_t scratch_bytes,void* opaque_stream,uint32_t layout_page,uint64_t layout_boundary){return mgbfs_segmented_batch_route_impl<true>(bitmap,word_shard,domain,state_bytes,kind,elements,n,m,generators,parents,count,stride,permutations,matrices,moduli,seed,bits,world,source,shards,capacity,begin,map,cuts,queues,counts,error,states,scratch,scratch_bytes,opaque_stream,layout_page,layout_boundary);}

// Compact rows follow the sorted queue keys. A bucket's state planes use its
// actual row count as the stride and the prefix sum of the bucket counts as
// its starting row. Neither output allocation needs world*shards*capacity.
extern "C" int mgbfs_segmented_compact_route_v1(uint32_t bitmap,uint32_t word_shard,BitmapDeviceDomain domain,
 uint32_t state_bytes,uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const void* parents,uint32_t count,uint64_t stride,const uint32_t* permutations,const int64_t* matrices,
 const uint32_t* moduli,uint64_t seed,uint32_t bits,uint32_t world,uint32_t source,uint32_t shards,
 uint32_t capacity,uint64_t begin,const uint32_t* map,const uint64_t* cuts,GenericRouteRecord* queues,
 uint32_t* counts,uint32_t* error,void* states,void* scratch,size_t scratch_bytes,void* opaque_stream,
 uint32_t layout_page,uint64_t layout_boundary,uint32_t max_output_rows){
 if(uint64_t(count)*generators>max_output_rows)return int(cudaErrorInvalidValue);
 return mgbfs_segmented_batch_route_impl<true,true>(bitmap,word_shard,domain,state_bytes,kind,elements,n,m,generators,
  parents,count,stride,permutations,matrices,moduli,seed,bits,world,source,shards,capacity,begin,map,cuts,queues,
  counts,error,states,scratch,scratch_bytes,opaque_stream,layout_page,layout_boundary);
}

extern "C" int mgbfs_segmented_compact_route_xor_v1(uint32_t bitmap,uint32_t word_shard,BitmapDeviceDomain domain,
 uint32_t state_bytes,uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const void* parents,uint32_t count,uint64_t stride,const uint32_t* permutations,const int64_t* matrices,
 const uint32_t* moduli,uint64_t seed,uint32_t bits,uint32_t world,uint32_t source,uint32_t shards,
 uint32_t capacity,uint64_t begin,const uint32_t* map,const uint64_t* cuts,GenericRouteRecord* queues,
 uint32_t* counts,uint32_t* error,void* states,void* scratch,size_t scratch_bytes,void* opaque_stream,
 uint32_t layout_page,uint64_t layout_boundary,uint32_t max_output_rows,const uint64_t* xor_masks){
 if(!xor_masks||uint64_t(count)*generators>max_output_rows)return int(cudaErrorInvalidValue);
 return mgbfs_segmented_batch_route_impl<true,true>(bitmap,word_shard,domain,state_bytes,kind,elements,n,m,generators,
  parents,count,stride,permutations,matrices,moduli,seed,bits,world,source,shards,capacity,begin,map,cuts,queues,
  counts,error,states,scratch,scratch_bytes,opaque_stream,layout_page,layout_boundary,xor_masks);
}

extern "C" int mgbfs_pack4_compact_route_v1(MgbfsPack4RouteArgs a,void* scratch,size_t scratch_bytes,
 uint32_t max_output_rows,void* stream){
 const uint64_t children=uint64_t(a.count)*a.generators;
 const uint64_t page_span=uint64_t(a.owner_page)*a.world;
 if(!mgbfs_pack4_valid(a.dictionary)||a.domain.elements!=a.dictionary.elements||!a.domain.symbols||!a.domain.counts||
  !a.domain.total||!a.parents||!a.permutations||!a.generators||children>=0x7fffffffULL||children>max_output_rows||
  !a.world||a.world>128||a.source>=a.world||!a.shards||a.shards>4096||!a.capacity||
  (a.state_bytes!=1&&a.state_bytes!=8)||!a.records||!a.counts||!a.error||((a.state_bytes==8||a.dictionary.elements>packed_route_max_elements)&&!a.payload)||
  (!a.owner_page&&a.owner_boundary)||(a.owner_page&&((a.owner_page&(a.owner_page-1))||a.owner_page>(1u<<20)||
  !page_span||a.owner_boundary!=a.domain.total-a.domain.total%page_span)))return int(cudaErrorInvalidValue);
 if(a.state_bytes==1)for(uint32_t i=0;i<a.dictionary.distinct;i++)
  if(a.dictionary.symbols[i]<0||a.dictionary.symbols[i]>255)return int(cudaErrorInvalidValue);
 BatchWorkspace w{};
 const auto status=layout(uint32_t(children),a.world*a.shards,&w);
 if(status!=cudaSuccess)return int(status);
 if(children&&(!scratch||uintptr_t(scratch)%kAlignment||scratch_bytes<w.total))return int(cudaErrorInvalidValue);
 BitmapPack4Action action{a.dictionary.elements,a.generators,a.count,a.parents,a.permutations,a.dictionary};
 const auto st=static_cast<cudaStream_t>(stream);
 if(a.state_bytes==1&&a.dictionary.elements>packed_route_max_elements)return launch<uint8_t,true,false,true,true>(action,a.domain,0,0,64,a.world,a.source,a.shards,
  a.capacity,0,a.map,nullptr,a.records,a.counts,a.error,a.payload,scratch,w,st,a.owner_page,a.owner_boundary);
 if(a.state_bytes==1)return launch<uint8_t,true,true,true,true>(action,a.domain,0,0,64,a.world,a.source,a.shards,
  a.capacity,0,a.map,nullptr,a.records,a.counts,a.error,a.payload,scratch,w,st,a.owner_page,a.owner_boundary);
 return launch<int64_t,true,false,true,true>(action,a.domain,0,0,64,a.world,a.source,a.shards,
  a.capacity,0,a.map,nullptr,a.records,a.counts,a.error,a.payload,scratch,w,st,a.owner_page,a.owner_boundary);
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
