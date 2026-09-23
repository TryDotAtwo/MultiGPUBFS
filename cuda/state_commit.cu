#include "state_commit.h"
#include "state_index.h"
#include <cuda_runtime.h>
#include <climits>
static_assert(sizeof(MgbfsStateRingControl)==64&&sizeof(MgbfsStateExtent)==64);
namespace {
__device__ void fatal(MgbfsStateRingControl* r,MgbfsOwnerControl* o,unsigned code){
  atomicCAS(&o->error,0u,code);atomicCAS(&r->fatal,0u,code);
}
__device__ void reserve_checked(MgbfsStateRingControl* r,MgbfsOwnerControl* o,MgbfsStateExtent* e,uint64_t n){
  *e={};
  if(r->fatal||o->error){fatal(r,o,r->fatal?r->fatal:o->error);return;}
  if(o->stage!=1||!r->capacity||!r->descriptor_capacity||r->head>r->tail||
     r->descriptor_head>r->descriptor_tail||r->tail-r->head>r->capacity||
     r->descriptor_tail-r->descriptor_head>r->descriptor_capacity){fatal(r,o,10);return;}
  if(!n)return;
  if(n>r->capacity){fatal(r,o,11);return;}
  if(r->descriptor_tail-r->descriptor_head==r->descriptor_capacity){fatal(r,o,12);return;}
  uint64_t start=r->tail,remainder=r->capacity-start%r->capacity;
  if(n>remainder){if(start>UINT64_MAX-remainder){fatal(r,o,13);return;}start+=remainder;}
  if(start>UINT64_MAX-n||r->descriptor_tail==UINT64_MAX){fatal(r,o,13);return;}
  uint64_t end=start+n,head=r->head;
  // No live descriptors means no live state records; skip otherwise wasted
  // wrap padding exactly as the CPU StateRing contract does.
  if(r->descriptor_head==r->descriptor_tail)head=start;
  if(end-head>r->capacity){fatal(r,o,11);return;}
  e->sequence=start;e->begin=start%r->capacity;e->count=n;
  e->descriptor=r->descriptor_tail;e->granted_rows=unsigned(n);
  r->head=head;r->tail=end;++r->descriptor_tail;
}
__global__ void reserve(MgbfsStateRingControl* r,MgbfsOwnerControl* o,MgbfsStateExtent* e){
  reserve_checked(r,o,e,o->survivors);
}
__global__ void reserve_rank_batch(MgbfsStateRingControl* r,MgbfsOwnerControl* o,
 MgbfsStateExtent* e,const uint32_t* survivors,const uint32_t* accepted,
 const uint32_t* capacities,uint32_t shards,uint32_t* offsets,uint32_t* layer,
 uint32_t layer_capacity,uint32_t request_capacity,bool hash_first){
 *e={};
 if(r->fatal||o->error){fatal(r,o,r->fatal?r->fatal:o->error);return;}
 if(o->stage!=1){fatal(r,o,10);return;}
 uint64_t total=0;
 for(uint32_t i=0;i<shards;++i){
   uint32_t n=survivors[i];
   if(accepted[i]>capacities[i]||n>capacities[i]-accepted[i]||
      total>UINT32_MAX-n){fatal(r,o,19);return;}
   total+=n;
 }
 if(*layer>layer_capacity||total>layer_capacity-*layer||
    (hash_first&&total>request_capacity)){fatal(r,o,16);return;}
 reserve_checked(r,o,e,total);
 if(o->error)return;
 uint32_t prefix=0;offsets[0]=0;
 for(uint32_t i=0;i<shards;++i){prefix+=survivors[i];offsets[i+1]=prefix;}
 o->survivors=prefix;
 *layer+=prefix;
}
__global__ void validate_shard_selection(const uint32_t* high,const uint32_t* candidate_count,
 const uint32_t* selected,const uint32_t* selected_count,uint32_t capacity,
 uint32_t logical_owner,uint32_t shards,uint32_t shift,
 MgbfsStateRingControl* r,MgbfsOwnerControl* o){
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
 if(r->fatal||o->error){if(i==0)fatal(r,o,r->fatal?r->fatal:o->error);return;}
 if(i==0){
   if(*candidate_count>capacity||*selected_count>*candidate_count)fatal(r,o,20);
 }
 uint32_t n=*selected_count<capacity?*selected_count:capacity;
 for(uint32_t row=i;row<n;row+=gridDim.x*blockDim.x){
   uint32_t source=selected[row];
   if(source>=*candidate_count||source>=capacity){fatal(r,o,20);continue;}
   uint32_t word=high[source];
   if((word>>shift)/shards!=logical_owner){fatal(r,o,20);continue;}
   if(row){
     uint32_t previous=selected[row-1];
     if(previous>=*candidate_count||previous>=capacity||source<=previous||
        word<high[previous])fatal(r,o,20);
   }
 }
}
__global__ void shard_directory(const uint32_t* high,const uint32_t* selected,
 const uint32_t* selected_count,uint32_t logical_owner,uint32_t shards,
 uint32_t shift,uint32_t* counts,uint32_t* offsets,
 const MgbfsStateRingControl* r,const MgbfsOwnerControl* o){
 if(r->fatal||o->error)return;
 uint32_t b=threadIdx.x;
 if(b<=shards){
   uint32_t target=logical_owner*shards+b;
   uint32_t lo=0,hi=*selected_count;
   while(lo<hi){
     uint32_t mid=lo+(hi-lo)/2;
     if((high[selected[mid]]>>shift)<target)lo=mid+1;else hi=mid;
   }
   offsets[b]=lo;
 }
 __syncthreads();
 if(b<shards)counts[b]=offsets[b+1]-offsets[b];
}
__global__ void validate_extent(MgbfsStateRingControl* r,MgbfsOwnerControl* o,MgbfsStateExtent* e,
    unsigned capacity,unsigned stride){
  if(r->fatal||o->error){fatal(r,o,r->fatal?r->fatal:o->error);return;}
  if(o->stage!=2||e->ready||e->count!=o->survivors||e->count!=e->granted_rows||
     e->count>capacity||!r->capacity||r->capacity>UINT64_MAX/stride){fatal(r,o,14);return;}
  if(!e->count)return;
  if(e->begin>=r->capacity||e->count>r->capacity-e->begin||e->sequence<r->head||
     e->sequence>r->tail||e->count>r->tail-e->sequence||e->begin!=e->sequence%r->capacity||
     e->descriptor<r->descriptor_head||e->descriptor>=r->descriptor_tail){fatal(r,o,14);return;}
}
__global__ void gate_rows(MgbfsOwnerControl* o,const MgbfsStateExtent* e,uint64_t* count){*count=o->error?0:e->count;}
__global__ void validate_rank_batch_shape(const uint32_t* source_rows,
    uint32_t source_capacity,const uint32_t* selected_count,uint32_t selected_capacity,
    MgbfsStateRingControl* r,MgbfsOwnerControl* o,const MgbfsStateExtent* e){
  if(o->error||r->fatal)return;
  if(*source_rows>source_capacity||*selected_count>selected_capacity||
     *selected_count!=e->count)fatal(r,o,15);
}
__global__ void validate_rank_batch_indices(const uint32_t* selected,
    const uint32_t* source_rows,MgbfsStateRingControl* r,
    MgbfsOwnerControl* o,const MgbfsStateExtent* e){
  if(o->error||r->fatal)return;
  for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<e->padding[0];
      i+=uint64_t(gridDim.x)*blockDim.x)
    if(selected[i]>=*source_rows)fatal(r,o,15);
}
// No temporary allocation: validated count is carried in extent padding[0].
// The index kernel never reads error while other blocks may atomically set it.
template<bool Packed=false>
__global__ void validate_indices(const uint64_t* refs,unsigned sorted,const uint32_t* selected,unsigned candidates,
    MgbfsStateRingControl* r,MgbfsOwnerControl* o,const MgbfsStateExtent* e){
  for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<e->padding[0];i+=uint64_t(gridDim.x)*blockDim.x){
    uint64_t row;
    if(!mgbfs_state_source_index<Packed>(refs,selected[i],sorted,candidates,&row))fatal(r,o,15);
  }
}
template<bool Packed=false>
__global__ void copy_states(const uint4* input,const uint64_t* refs,const uint32_t* selected,unsigned words,
    uint4* output,const MgbfsOwnerControl* o,const MgbfsStateExtent* e){
  if(o->error)return;
  for(uint64_t x=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;x<e->count*words;x+=uint64_t(gridDim.x)*blockDim.x){
    uint64_t row=x/words,word=x%words;output[e->begin*words+x]=input[mgbfs_state_source_row<Packed>(refs,selected[row])*words+word];
  }
}
__global__ void publish_ready(const MgbfsOwnerControl* o,MgbfsStateExtent* e){if(!o->error)e->ready=1;}
__global__ void build_requests(const MgbfsRegenerateOrigin* origins,const uint64_t* refs,
 const uint32_t* selected,MgbfsRegenerateOrigin* requests,uint64_t* targets,
 const MgbfsOwnerControl* o,const MgbfsStateExtent* e){
  if(o->error)return;
  for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<e->count;i+=uint64_t(gridDim.x)*blockDim.x){
    requests[i]=origins[refs[selected[i]]];
    targets[i]=e->sequence+i;
  }
}
__global__ void publish_requests(const MgbfsOwnerControl* o,const MgbfsStateExtent* e,uint32_t* count){
  *count=o->error?0:uint32_t(e->count);
}
__global__ void guard_layer(MgbfsStateRingControl*r,MgbfsOwnerControl*o,const uint32_t*n,uint32_t cap){
 if(!o->error&&(*n>cap||o->survivors>cap-*n))fatal(r,o,16);
}
__global__ void count_layer(const MgbfsOwnerControl*o,uint32_t*n){if(!o->error)*n+=o->survivors;}
__device__ void retire_dense_prefix_impl(MgbfsStateRingControl* r,MgbfsStateExtent* e,uint64_t n){
  if(r->fatal)return;
  if(!r->capacity){atomicCAS(&r->fatal,0u,17u);return;}
  uint64_t gap=e->sequence>=r->head?e->sequence-r->head:UINT64_MAX;
  uint64_t wrap=r->capacity-r->head%r->capacity;
  bool fifo=e->sequence==r->head||(gap==wrap&&e->begin==0);
  if(!n||!e->ready||n>e->count||!fifo||
     e->descriptor!=r->descriptor_head||e->padding[1]<e->descriptor||
     e->padding[1]>=r->descriptor_tail||e->begin!=e->sequence%r->capacity){
    atomicCAS(&r->fatal,0u,17u);return;
  }
  uint64_t next=e->sequence+n;
  r->head=next;e->sequence=next;e->begin=next%r->capacity;e->count-=n;
  e->granted_rows=unsigned(e->count);
  if(!e->count){r->descriptor_head=e->padding[1]+1;e->ready=0;}
}
__global__ void retire_dense_prefix(MgbfsStateRingControl* r,MgbfsStateExtent* e,uint64_t n){
  retire_dense_prefix_impl(r,e,n);
}
__global__ void retire_dense_prefix_value(MgbfsStateRingControl* r,MgbfsStateExtent e,uint64_t n){
  retire_dense_prefix_impl(r,&e,n);
}
__global__ void publish_next_extent(MgbfsStateRingControl* r,MgbfsOwnerControl* o,
    const MgbfsStateExtent* e,uint32_t* count,MgbfsStateExtent* out,uint32_t cap){
  if(r->fatal||o->error)return;
  if(!e->count)return;
  if(!cap||*count>cap||!e->ready||e->count!=e->granted_rows||
     !r->capacity||e->begin>=r->capacity||e->count>r->capacity-e->begin||
     e->begin!=e->sequence%r->capacity||e->sequence<r->head||
     e->sequence>r->tail||e->count>r->tail-e->sequence||
     e->descriptor<r->descriptor_head||e->descriptor>=r->descriptor_tail){
    fatal(r,o,24);return;
  }
  uint32_t n=*count;
  MgbfsStateExtent value=*e;value.padding[1]=e->descriptor;
  if(n){
    MgbfsStateExtent last=out[n-1];
    if(last.sequence>UINT64_MAX-last.count||
       last.begin>r->capacity-last.count||
       last.padding[1]>=e->descriptor){fatal(r,o,24);return;}
    if(last.sequence+last.count==e->sequence&&
       last.begin+last.count==e->begin){
      if(last.count>UINT32_MAX-e->count){fatal(r,o,24);return;}
      last.count+=e->count;last.granted_rows=uint32_t(last.count);
      last.padding[1]=e->descriptor;out[n-1]=last;return;
    }
  }
  if(n==cap){fatal(r,o,24);return;}
  out[n]=value;*count=n+1;
}
}
extern "C" int mgbfs_state_reserve(MgbfsStateRingControl* r,MgbfsOwnerControl* o,MgbfsStateExtent* e,void* stream){
  if(!r||!o||!e)return 1;reserve<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(r,o,e);return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_state_reserve_layer(MgbfsStateRingControl*r,MgbfsOwnerControl*o,MgbfsStateExtent*e,uint32_t*n,uint32_t cap,void*stream){
 if(!r||!o||!e||!n)return 1;auto s=static_cast<cudaStream_t>(stream);
 guard_layer<<<1,1,0,s>>>(r,o,n,cap);reserve<<<1,1,0,s>>>(r,o,e);count_layer<<<1,1,0,s>>>(o,n);
 return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_state_reserve_rank_batch(MgbfsStateRingControl*r,MgbfsOwnerControl*o,
 MgbfsStateExtent*e,const uint32_t*survivors,const uint32_t*accepted,
 const uint32_t*capacities,uint32_t shards,uint32_t*offsets,uint32_t*layer,
 uint32_t layer_capacity,uint32_t request_capacity,uint32_t hash_first,void*stream){
 if(!r||!o||!e||!survivors||!accepted||!capacities||!shards||!offsets||!layer)return 1;
 reserve_rank_batch<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(r,o,e,survivors,
   accepted,capacities,shards,offsets,layer,layer_capacity,request_capacity,hash_first!=0);
 return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_owner_shard_counts(const uint32_t*high,
 const uint32_t*candidate_count,const uint32_t*selected,
 const uint32_t*selected_count,uint32_t capacity,uint32_t logical_owner,
 uint32_t world,uint32_t shards,uint32_t*counts,uint32_t*offsets,
 MgbfsStateRingControl*r,MgbfsOwnerControl*o,void*stream){
 if(!high||!candidate_count||!selected||!selected_count||!capacity||
    capacity>INT_MAX||!world||(world&(world-1))||world>128||
    logical_owner>=world||!shards||(shards&(shards-1))||shards>256||
    !counts||!offsets||!r||!o)return 1;
 uint32_t product=world*shards,shift=32;
 for(uint32_t n=product;n>1;n>>=1)--shift;
 auto s=static_cast<cudaStream_t>(stream);
 uint32_t blocks=(capacity+255)/256;if(blocks>4096)blocks=4096;
 validate_shard_selection<<<blocks,256,0,s>>>(high,candidate_count,selected,
     selected_count,capacity,logical_owner,shards,shift,r,o);
 uint32_t threads=1;while(threads<=shards)threads<<=1;
 shard_directory<<<1,threads,0,s>>>(high,selected,selected_count,logical_owner,
     shards,shift,counts,offsets,r,o);
 return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_state_retire_dense_prefix(MgbfsStateRingControl*r,MgbfsStateExtent*e,uint64_t n,void*stream){
 if(!r||!e||!n)return 1;retire_dense_prefix<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(r,e,n);
 return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_state_retire_dense_prefix_value(MgbfsStateRingControl*r,MgbfsStateExtent e,uint64_t n,void*stream){
 if(!r||!n)return 1;retire_dense_prefix_value<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(r,e,n);
 return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_state_publish_next_extent(MgbfsStateRingControl*r,
 MgbfsOwnerControl*o,const MgbfsStateExtent*e,uint32_t*count,
 MgbfsStateExtent*out,uint32_t capacity,void*stream){
 if(!r||!o||!e||!count||!out||!capacity)return 1;
 publish_next_extent<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(
     r,o,e,count,out,capacity);
 return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_state_materialize(const uint8_t* input,uint32_t candidates,const uint64_t* refs,uint32_t sorted,
    const uint32_t* selected,uint32_t capacity,uint32_t stride,uint8_t* output,MgbfsStateRingControl* r,
    MgbfsOwnerControl* o,MgbfsStateExtent* e,void* stream){
  if(!input||!refs||!selected||!output||!r||!o||!e||!capacity||capacity>INT_MAX||!stride||stride%16)return 1;
  auto s=static_cast<cudaStream_t>(stream);unsigned blocks=(capacity+255)/256;if(blocks>4096)blocks=4096;
  validate_extent<<<1,1,0,s>>>(r,o,e,capacity,stride);
  gate_rows<<<1,1,0,s>>>(o,e,&e->padding[0]);
  validate_indices<false><<<blocks,256,0,s>>>(refs,sorted,selected,candidates,r,o,e);
  copy_states<false><<<blocks,256,0,s>>>(reinterpret_cast<const uint4*>(input),refs,selected,stride/16,reinterpret_cast<uint4*>(output),o,e);
  publish_ready<<<1,1,0,s>>>(o,e);return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_state_materialize_packed(const uint8_t* input,uint32_t rows,
    const uint32_t* selected,uint32_t capacity,uint32_t stride,uint8_t* output,MgbfsStateRingControl* r,
    MgbfsOwnerControl* o,MgbfsStateExtent* e,void* stream){
  if(!input||!selected||!output||!r||!o||!e||!capacity||capacity>INT_MAX||!stride||stride%16)return 1;
  auto s=static_cast<cudaStream_t>(stream);unsigned blocks=(capacity+255)/256;if(blocks>4096)blocks=4096;
  validate_extent<<<1,1,0,s>>>(r,o,e,capacity,stride);
  gate_rows<<<1,1,0,s>>>(o,e,&e->padding[0]);
  validate_indices<true><<<blocks,256,0,s>>>(nullptr,rows,selected,rows,r,o,e);
  copy_states<true><<<blocks,256,0,s>>>(reinterpret_cast<const uint4*>(input),nullptr,selected,stride/16,reinterpret_cast<uint4*>(output),o,e);
  publish_ready<<<1,1,0,s>>>(o,e);return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_state_materialize_rank_batch(const uint8_t* input,
    const uint32_t* source_rows,uint32_t source_capacity,
    const uint32_t* source_indices,const uint32_t* selected_count,
    uint32_t selected_capacity,uint32_t stride,uint8_t* output,
    MgbfsStateRingControl* r,MgbfsOwnerControl* o,MgbfsStateExtent* e,void* stream){
  if(!input||!source_rows||!source_capacity||source_capacity>INT_MAX||
     !source_indices||!selected_count||!selected_capacity||selected_capacity>INT_MAX||
     !stride||stride%16||!output||!r||!o||!e)return 1;
  auto s=static_cast<cudaStream_t>(stream);
  unsigned blocks=(selected_capacity+255)/256;if(blocks>4096)blocks=4096;
  validate_extent<<<1,1,0,s>>>(r,o,e,selected_capacity,stride);
  gate_rows<<<1,1,0,s>>>(o,e,&e->padding[0]);
  validate_rank_batch_shape<<<1,1,0,s>>>(source_rows,source_capacity,
      selected_count,selected_capacity,r,o,e);
  validate_rank_batch_indices<<<blocks,256,0,s>>>(source_indices,source_rows,r,o,e);
  copy_states<true><<<blocks,256,0,s>>>(reinterpret_cast<const uint4*>(input),
      nullptr,source_indices,stride/16,reinterpret_cast<uint4*>(output),o,e);
  publish_ready<<<1,1,0,s>>>(o,e);
  return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_state_build_requests(const MgbfsRegenerateOrigin* origins,uint32_t candidates,
 const uint64_t* refs,uint32_t sorted,const uint32_t* selected,uint32_t capacity,
 MgbfsRegenerateOrigin* requests,uint64_t* targets,uint32_t* count,
 MgbfsStateRingControl* r,MgbfsOwnerControl* o,MgbfsStateExtent* e,void* stream){
  if(!origins||!refs||!selected||!requests||!targets||!count||!r||!o||!e||!capacity||capacity>INT_MAX)return 1;
  auto s=static_cast<cudaStream_t>(stream);unsigned blocks=(capacity+255)/256;if(blocks>4096)blocks=4096;
  validate_extent<<<1,1,0,s>>>(r,o,e,capacity,16);
  gate_rows<<<1,1,0,s>>>(o,e,&e->padding[0]);
  validate_indices<false><<<blocks,256,0,s>>>(refs,sorted,selected,candidates,r,o,e);
  build_requests<<<blocks,256,0,s>>>(origins,refs,selected,requests,targets,o,e);
  publish_requests<<<1,1,0,s>>>(o,e,count);
  return cudaGetLastError()==cudaSuccess?0:2;
}
