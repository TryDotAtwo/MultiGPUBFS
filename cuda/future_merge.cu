#include "mgbfs_cuda.h"
#include <cuda_runtime.h>
#include <cub/device/device_select.cuh>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>

namespace {
struct alignas(16) Key { uint32_t w[4]; };
static_assert(sizeof(Key)==16);
__device__ bool less(Key a,Key b){for(int i=3;i>=0;--i){if(a.w[i]<b.w[i])return true;if(a.w[i]>b.w[i])return false;}return false;}
__device__ bool equal(Key a,Key b){return !less(a,b)&&!less(b,a);}
__device__ uint32_t lower(const Key* a,uint32_t n,Key x){uint32_t l=0,r=n;while(l<r){uint32_t m=l+(r-l)/2;if(less(a[m],x))l=m+1;else r=m;}return l;}
__device__ uint32_t upper(const Key* a,uint32_t n,Key x){uint32_t l=0,r=n;while(l<r){uint32_t m=l+(r-l)/2;if(!less(x,a[m]))l=m+1;else r=m;}return l;}
struct Plan {
  uint32_t stride{},future{},incoming{},total{};
  Key *merged{},*unique{}; uint64_t *tags{},*unique_tags{};
  uint32_t *indices{},*selected{},*selected_count{}; uint8_t* flags{};
  uint8_t* states{}; MgbfsFrontierState* state{}; void* scratch{}; size_t scratch_bytes{};
  ~Plan(){cudaFree(scratch);cudaFree(state);cudaFree(states);cudaFree(flags);cudaFree(selected_count);cudaFree(selected);cudaFree(indices);cudaFree(unique_tags);cudaFree(tags);cudaFree(unique);cudaFree(merged);}
};
void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void begin(const MgbfsFrontierState* old,const uint32_t* incoming,uint32_t old_cap,uint32_t in_cap,uint32_t old_bound,uint32_t in_bound,MgbfsFrontierState* out){
  *out={}; if(old->fatal){out->fatal=old->fatal;return;} if(old_bound>old_cap||in_bound>in_cap||old->count>old_bound||*incoming>in_bound){out->fatal=1;return;}
}
__global__ void merge_runs(const Key* a,const MgbfsFrontierState* old,const Key* b,const uint64_t* br,const uint32_t* incoming,Key* out,uint64_t* tags,MgbfsFrontierState* state){
  uint32_t i=blockIdx.x*blockDim.x+threadIdx.x,an=old->count,bn=*incoming;if(state->fatal)return;
  if(i<an){Key x=a[i];uint32_t at=i+lower(b,bn,x);out[at]=x;tags[at]=i;}
  if(i<bn){Key x=b[i];uint32_t at=i+upper(a,an,x);out[at]=x;tags[at]=(uint64_t(1)<<63)|br[i];}
}
__global__ void mark(const Key* keys,const MgbfsFrontierState* old,const uint32_t* incoming,uint32_t cap,uint32_t* indices,uint8_t* flags,const MgbfsFrontierState* state){
  uint32_t i=blockIdx.x*blockDim.x+threadIdx.x,n=old->count+*incoming;if(i>=cap)return;indices[i]=i;flags[i]=!state->fatal&&i<n&&(i==0||!equal(keys[i],keys[i-1]));
}
__global__ void validate_refs(const uint64_t* tags,const uint32_t* selected,const uint32_t* count,const MgbfsFrontierState* old,uint32_t source,MgbfsFrontierState* state,uint32_t cap){
  uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=cap||i>=*count||state->fatal)return;uint64_t tag=tags[selected[i]];
  uint64_t row=tag&~(uint64_t(1)<<63);uint32_t bound=(tag>>63)?source:old->count;if(row>=bound)atomicCAS(&state->fatal,0u,2u);
}
__global__ void gather(const Key* merged,const uint64_t* tags,const uint32_t* selected,const uint32_t* count,const uint8_t* old_states,const uint8_t* source,uint32_t chunks,Key* keys,uint64_t* out_tags,uint8_t* states,const MgbfsFrontierState* state,uint32_t cap){
  uint64_t p=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(state->fatal||p>=uint64_t(cap)*chunks)return;uint32_t row=p/chunks;if(row>=*count)return;uint32_t chunk=p%chunks;uint32_t at=selected[row];uint64_t tag=tags[at];uint64_t src=tag&~(uint64_t(1)<<63);const uint4* input=reinterpret_cast<const uint4*>((tag>>63)?source:old_states);reinterpret_cast<uint4*>(states)[uint64_t(row)*chunks+chunk]=input[src*chunks+chunk];if(chunk==0){keys[row]=merged[at];out_tags[row]=tag;}
}
__global__ void copy_back(const Key* keys,const uint8_t* states,const uint32_t* count,Key* out_keys,uint8_t* out_states,MgbfsFrontierState* out,uint32_t chunks,uint32_t cap){
  uint64_t p=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(out->fatal||p>=uint64_t(cap)*chunks)return;uint32_t row=p/chunks;if(row>=*count)return;uint32_t chunk=p%chunks;reinterpret_cast<uint4*>(out_states)[uint64_t(row)*chunks+chunk]=reinterpret_cast<const uint4*>(states)[uint64_t(row)*chunks+chunk];if(chunk==0)out_keys[row]=keys[row];
}
__global__ void publish(MgbfsFrontierState* state,const uint32_t* count,uint32_t cap){if(!state->fatal){if(*count>cap)state->fatal=1;else state->count=*count;}}
}
extern "C" int mgbfs_future_merge_create(uint32_t stride,uint32_t future,uint32_t incoming,void** out,char* error,size_t error_capacity){
  if(!out)return 1;*out=nullptr;try{if(!stride||stride%16||!future||!incoming||future>INT32_MAX-incoming)throw std::runtime_error("FUTURE_MERGE_SHAPE");auto p=std::make_unique<Plan>();p->stride=stride;p->future=future;p->incoming=incoming;p->total=future+incoming;
    size_t n=p->total;checked(cudaMalloc(&p->merged,n*16));checked(cudaMalloc(&p->unique,size_t(future)*16));checked(cudaMalloc(&p->tags,n*8));checked(cudaMalloc(&p->unique_tags,size_t(future)*8));checked(cudaMalloc(&p->indices,n*4));checked(cudaMalloc(&p->selected,n*4));checked(cudaMalloc(&p->selected_count,4));checked(cudaMalloc(&p->flags,n));checked(cudaMalloc(&p->states,size_t(future)*stride));checked(cudaMalloc(&p->state,sizeof(MgbfsFrontierState)));checked(cub::DeviceSelect::Flagged(nullptr,p->scratch_bytes,p->indices,p->flags,p->selected,p->selected_count,int(n)));checked(cudaMalloc(&p->scratch,p->scratch_bytes));*out=p.release();return 0;
  }catch(const std::exception& e){if(error&&error_capacity)std::snprintf(error,error_capacity,"%s",e.what());return 1;}}
extern "C" int mgbfs_future_merge_run(void* raw,uint8_t* future_states,void* future_hashes,MgbfsFrontierState* future_state,const uint8_t* source_states,uint32_t source_count,const void* incoming_hashes,const uint64_t* incoming_refs,const uint32_t* incoming_count,void* raw_stream){
  auto*p=static_cast<Plan*>(raw);if(!p)return 1;
  return mgbfs_future_merge_run_bounded(raw,future_states,future_hashes,future_state,p->future,source_states,source_count,incoming_hashes,incoming_refs,incoming_count,p->incoming,raw_stream);
}
extern "C" int mgbfs_future_merge_run_bounded(void* raw,uint8_t* future_states,void* future_hashes,MgbfsFrontierState* future_state,uint32_t old_bound,const uint8_t* source_states,uint32_t source_count,const void* incoming_hashes,const uint64_t* incoming_refs,const uint32_t* incoming_count,uint32_t incoming_bound,void* raw_stream){
  auto*p=static_cast<Plan*>(raw);if(!p||!future_states||!future_hashes||!future_state||!source_states||!incoming_hashes||!incoming_refs||!incoming_count)return 1;auto s=static_cast<cudaStream_t>(raw_stream);auto* state=p->state;
  if(old_bound>p->future||incoming_bound>p->incoming||old_bound>UINT32_MAX-incoming_bound)return 1;uint32_t active=old_bound+incoming_bound;uint32_t output_bound=min(p->future,active);
  begin<<<1,1,0,s>>>(future_state,incoming_count,p->future,p->incoming,old_bound,incoming_bound,state);
  if(active){merge_runs<<<(max(old_bound,incoming_bound)+255)/256,256,0,s>>>(static_cast<Key*>(future_hashes),future_state,static_cast<const Key*>(incoming_hashes),incoming_refs,incoming_count,p->merged,p->tags,state);mark<<<(active+255)/256,256,0,s>>>(p->merged,future_state,incoming_count,active,p->indices,p->flags,state);}
  size_t bytes=p->scratch_bytes;if(cub::DeviceSelect::Flagged(p->scratch,bytes,p->indices,p->flags,p->selected,p->selected_count,int(active),s)!=cudaSuccess)return 3;
  if(output_bound){validate_refs<<<(output_bound+255)/256,256,0,s>>>(p->tags,p->selected,p->selected_count,future_state,source_count,state,output_bound);}
  publish<<<1,1,0,s>>>(state,p->selected_count,p->future);
  if(output_bound){gather<<<(uint64_t(output_bound)*(p->stride/16)+255)/256,256,0,s>>>(p->merged,p->tags,p->selected,p->selected_count,future_states,source_states,p->stride/16,p->unique,p->unique_tags,p->states,state,output_bound);copy_back<<<(uint64_t(output_bound)*(p->stride/16)+255)/256,256,0,s>>>(p->unique,p->states,p->selected_count,static_cast<Key*>(future_hashes),future_states,state,p->stride/16,output_bound);}
  return cudaMemcpyAsync(future_state,state,sizeof(*state),cudaMemcpyDeviceToDevice,s)==cudaSuccess?0:4;}
extern "C" void mgbfs_future_merge_destroy(void* raw){delete static_cast<Plan*>(raw);}
