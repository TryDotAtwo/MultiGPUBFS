#include "mgbfs_cuda.h"
#include <cuda_runtime.h>
#include <cub/device/device_select.cuh>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>

namespace {
struct alignas(16) Key { uint32_t word[4]; };
static_assert(sizeof(Key)==16 && sizeof(MgbfsMacroSettleState)==16);
__device__ bool less(Key a,Key b){for(int i=3;i>=0;--i){if(a.word[i]<b.word[i])return true;if(a.word[i]>b.word[i])return false;}return false;}
__device__ bool equal(Key a,Key b){return !less(a,b)&&!less(b,a);}
struct Plan {
  uint32_t candidates{},layers{},history_capacity{};
  uint32_t *indices{},*selected{},*selected_count{}; uint8_t* flags{}; void* scratch{}; size_t scratch_bytes{};
  ~Plan(){cudaFree(scratch);cudaFree(flags);cudaFree(selected_count);cudaFree(selected);cudaFree(indices);}
};
void check(cudaError_t value){if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
__global__ void begin(const uint32_t* count,const uint32_t* history_counts,uint32_t candidates,uint32_t layers,uint32_t history_capacity,
  MgbfsMacroSettleState* state,uint32_t* output,uint64_t epoch){
  *output=0;if(state->fatal)return;
  if(epoch==0 || (state->last_epoch && epoch<=state->last_epoch)){state->fatal=2;return;}
  if(*count>candidates){state->fatal=1;return;}
  for(uint32_t layer=0;layer<layers;++layer)if(history_counts[layer]>history_capacity){state->fatal=1;return;}
}
__global__ void validate_sorted(const Key* future,const uint32_t* count,const Key* history,const uint32_t* history_counts,
  uint32_t layers,uint32_t history_capacity,MgbfsMacroSettleState* state,uint32_t bound){
  const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=bound||state->fatal)return;
  if(i>0 && i<*count && less(future[i],future[i-1]))atomicCAS(&state->fatal,0u,3u);
  for(uint32_t layer=0;layer<layers;++layer){const uint32_t n=history_counts[layer];const Key* run=history+size_t(layer)*history_capacity;
    if(i>0 && i<n && less(run[i],run[i-1]))atomicCAS(&state->fatal,0u,3u);
  }
}
__device__ bool contains(const Key* values,uint32_t count,Key needle){
  uint32_t low=0,high=count;while(low<high){const uint32_t mid=low+(high-low)/2;if(less(values[mid],needle))low=mid+1;else high=mid;}
  return low<count&&equal(values[low],needle);
}
__global__ void flag(const Key* future,const uint32_t* count,const Key* history,const uint32_t* history_counts,
  uint32_t layers,uint32_t history_capacity,const MgbfsMacroSettleState* state,uint32_t capacity,uint32_t* indices,uint8_t* flags){
  const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=capacity)return;indices[i]=i;bool keep=false;
  if(!state->fatal && i<*count){const Key key=future[i];keep=i==0||!equal(key,future[i-1]);
    for(uint32_t layer=0;keep&&layer<layers;++layer)keep=!contains(history+size_t(layer)*history_capacity,history_counts[layer],key);
  }flags[i]=keep;
}
__global__ void gather(const Key* future,const uint64_t* refs,const uint32_t* selected,const uint32_t* count,
  Key* output,uint64_t* output_refs,const MgbfsMacroSettleState* state,uint32_t capacity){
  const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<capacity&&!state->fatal&&i<*count){const uint32_t source=selected[i];output[i]=future[source];output_refs[i]=refs[source];}
}
__global__ void publish(MgbfsMacroSettleState* state,const uint32_t* count,uint32_t* output,uint64_t epoch){
  if(state->fatal)return;state->last_epoch=epoch;state->count=*count;*output=*count;
}
}

extern "C" int mgbfs_macro_settle_query(uint32_t candidates,uint32_t layers,uint32_t history_capacity,MgbfsMacroSettleBytes* out){
  if(!out)return 1;*out={};if(!candidates||!layers||!history_capacity||candidates>INT32_MAX)return 1;
  MgbfsMacroSettleBytes q{};q.indices=uint64_t(candidates)*4;q.selected=uint64_t(candidates)*4;q.flags=candidates;q.count=4;
  size_t scratch=0;if(cub::DeviceSelect::Flagged(nullptr,scratch,(uint32_t*)nullptr,(uint8_t*)nullptr,(uint32_t*)nullptr,(uint32_t*)nullptr,int(candidates))!=cudaSuccess)return 2;
  q.scratch=scratch;*out=q;return 0;
}
extern "C" int mgbfs_macro_settle_create(uint32_t candidates,uint32_t layers,uint32_t history_capacity,void** out,char* error,size_t error_capacity){
  if(!out)return 1;*out=nullptr;try{MgbfsMacroSettleBytes q{};if(mgbfs_macro_settle_query(candidates,layers,history_capacity,&q))throw std::runtime_error("MACRO_SETTLE_SHAPE");
    auto p=std::make_unique<Plan>();p->candidates=candidates;p->layers=layers;p->history_capacity=history_capacity;p->scratch_bytes=q.scratch;
    check(cudaMalloc(&p->indices,q.indices));check(cudaMalloc(&p->selected,q.selected));check(cudaMalloc(&p->selected_count,q.count));check(cudaMalloc(&p->flags,q.flags));check(cudaMalloc(&p->scratch,q.scratch));*out=p.release();return 0;
  }catch(const std::exception& e){if(error&&error_capacity)std::snprintf(error,error_capacity,"%s",e.what());return 1;}
}
extern "C" int mgbfs_macro_settle_run(void* raw,const void* future,const uint64_t* refs,const uint32_t* count,const void* history,const uint32_t* history_counts,
  void* survivors,uint64_t* survivor_refs,uint32_t* survivor_count,MgbfsMacroSettleState* state,uint64_t epoch,void* raw_stream){
  auto* p=static_cast<Plan*>(raw);if(!p||!future||!refs||!count||!history||!history_counts||!survivors||!survivor_refs||!survivor_count||!state)return 1;
  auto stream=static_cast<cudaStream_t>(raw_stream);constexpr uint32_t threads=256;
  begin<<<1,1,0,stream>>>(count,history_counts,p->candidates,p->layers,p->history_capacity,state,survivor_count,epoch);
  validate_sorted<<<(max(p->candidates,p->history_capacity)+threads-1)/threads,threads,0,stream>>>(static_cast<const Key*>(future),count,static_cast<const Key*>(history),history_counts,p->layers,p->history_capacity,state,max(p->candidates,p->history_capacity));
  flag<<<(p->candidates+threads-1)/threads,threads,0,stream>>>(static_cast<const Key*>(future),count,static_cast<const Key*>(history),history_counts,p->layers,p->history_capacity,state,p->candidates,p->indices,p->flags);
  if(cudaGetLastError()!=cudaSuccess)return 2;size_t bytes=p->scratch_bytes;
  if(cub::DeviceSelect::Flagged(p->scratch,bytes,p->indices,p->flags,p->selected,p->selected_count,int(p->candidates),stream)!=cudaSuccess)return 3;
  gather<<<(p->candidates+threads-1)/threads,threads,0,stream>>>(static_cast<const Key*>(future),refs,p->selected,p->selected_count,static_cast<Key*>(survivors),survivor_refs,state,p->candidates);
  publish<<<1,1,0,stream>>>(state,p->selected_count,survivor_count,epoch);return cudaGetLastError()==cudaSuccess?0:4;
}
extern "C" void mgbfs_macro_settle_destroy(void* raw){delete static_cast<Plan*>(raw);}
