#include "state_commit.h"
#include <cuda_runtime.h>
#include <climits>
static_assert(sizeof(MgbfsStateRingControl)==64&&sizeof(MgbfsStateExtent)==64);
namespace {
__device__ void fatal(MgbfsStateRingControl* r,MgbfsOwnerControl* o,unsigned code){
  atomicCAS(&o->error,0u,code);atomicCAS(&r->fatal,0u,code);
}
__global__ void reserve(MgbfsStateRingControl* r,MgbfsOwnerControl* o,MgbfsStateExtent* e){
  *e={};
  if(r->fatal||o->error){fatal(r,o,r->fatal?r->fatal:o->error);return;}
  if(o->stage!=1||!r->capacity||!r->descriptor_capacity||r->head>r->tail||
     r->descriptor_head>r->descriptor_tail||r->tail-r->head>r->capacity||
     r->descriptor_tail-r->descriptor_head>r->descriptor_capacity){fatal(r,o,10);return;}
  uint64_t n=o->survivors;
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
// No temporary allocation: validated count is carried in extent padding[0].
// The index kernel never reads error while other blocks may atomically set it.
__global__ void validate_indices(const uint64_t* refs,unsigned sorted,const uint32_t* selected,unsigned candidates,
    MgbfsStateRingControl* r,MgbfsOwnerControl* o,const MgbfsStateExtent* e){
  for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<e->padding[0];i+=uint64_t(gridDim.x)*blockDim.x){
    unsigned index=selected[i];if(index>=sorted||refs[index]>=candidates)fatal(r,o,15);
  }
}
__global__ void copy_states(const uint4* input,const uint64_t* refs,const uint32_t* selected,unsigned words,
    uint4* output,const MgbfsOwnerControl* o,const MgbfsStateExtent* e){
  if(o->error)return;
  for(uint64_t x=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;x<e->count*words;x+=uint64_t(gridDim.x)*blockDim.x){
    uint64_t row=x/words,word=x%words;output[e->begin*words+x]=input[refs[selected[row]]*words+word];
  }
}
__global__ void publish_ready(const MgbfsOwnerControl* o,MgbfsStateExtent* e){if(!o->error)e->ready=1;}
__global__ void guard_layer(MgbfsStateRingControl*r,MgbfsOwnerControl*o,const uint32_t*n,uint32_t cap){
 if(!o->error&&(*n>cap||o->survivors>cap-*n))fatal(r,o,16);
}
__global__ void count_layer(const MgbfsOwnerControl*o,uint32_t*n){if(!o->error)*n+=o->survivors;}
__global__ void retire_dense_prefix(MgbfsStateRingControl* r,MgbfsStateExtent* e,uint64_t n){
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
}
extern "C" int mgbfs_state_reserve(MgbfsStateRingControl* r,MgbfsOwnerControl* o,MgbfsStateExtent* e,void* stream){
  if(!r||!o||!e)return 1;reserve<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(r,o,e);return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_state_reserve_layer(MgbfsStateRingControl*r,MgbfsOwnerControl*o,MgbfsStateExtent*e,uint32_t*n,uint32_t cap,void*stream){
 if(!r||!o||!e||!n)return 1;auto s=static_cast<cudaStream_t>(stream);
 guard_layer<<<1,1,0,s>>>(r,o,n,cap);reserve<<<1,1,0,s>>>(r,o,e);count_layer<<<1,1,0,s>>>(o,n);
 return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_state_retire_dense_prefix(MgbfsStateRingControl*r,MgbfsStateExtent*e,uint64_t n,void*stream){
 if(!r||!e||!n)return 1;retire_dense_prefix<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(r,e,n);
 return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_state_materialize(const uint8_t* input,uint32_t candidates,const uint64_t* refs,uint32_t sorted,
    const uint32_t* selected,uint32_t capacity,uint32_t stride,uint8_t* output,MgbfsStateRingControl* r,
    MgbfsOwnerControl* o,MgbfsStateExtent* e,void* stream){
  if(!input||!refs||!selected||!output||!r||!o||!e||!capacity||capacity>INT_MAX||!stride||stride%16)return 1;
  auto s=static_cast<cudaStream_t>(stream);unsigned blocks=(capacity+255)/256;if(blocks>4096)blocks=4096;
  validate_extent<<<1,1,0,s>>>(r,o,e,capacity,stride);
  gate_rows<<<1,1,0,s>>>(o,e,&e->padding[0]);
  validate_indices<<<blocks,256,0,s>>>(refs,sorted,selected,candidates,r,o,e);
  copy_states<<<blocks,256,0,s>>>(reinterpret_cast<const uint4*>(input),refs,selected,stride/16,reinterpret_cast<uint4*>(output),o,e);
  publish_ready<<<1,1,0,s>>>(o,e);return cudaGetLastError()==cudaSuccess?0:2;
}
