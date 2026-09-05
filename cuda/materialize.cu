#include "mgbfs_cuda.h"
#include "state_commit.h"
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <climits>
static_assert(sizeof(MgbfsFrontierState)==8);
struct MaterializePlan {
  uint32_t stride{}, capacity{}, frontier{};
  uint64_t *keys{}, *sorted{};
  uint32_t *indices{}, *order{};
  void* scratch{}; size_t scratch_bytes{};
  ~MaterializePlan(){cudaFree(scratch);cudaFree(order);cudaFree(indices);cudaFree(sorted);cudaFree(keys);}
};
static void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void check_append(const uint32_t* count,MgbfsFrontierState* state,uint32_t capacity,uint32_t frontier){
  if(state->fatal)return;
  if(*count>capacity||state->count>frontier||*count>frontier-state->count)state->fatal=1;
}
__global__ void prepare_requests(const uint64_t* refs,const uint32_t* count,MgbfsFrontierState* state,
 uint64_t* keys,uint32_t* indices,uint32_t capacity,uint32_t source_count){
  uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=capacity)return;
  // Capacity was checked by a preceding kernel; never read past request spans.
  bool active=*count<=capacity&&i<*count;
  uint64_t key=active?refs[i]:UINT64_MAX;
  if(active&&key>=source_count)atomicCAS(&state->fatal,0u,2u);
  keys[i]=key;indices[i]=i;
}
__global__ void append_states(const uint4* source,const uint4* hashes,const uint64_t* sorted,const uint32_t* order,
 const uint32_t* count,uint4* states,uint4* out_hashes,const MgbfsFrontierState* state,uint32_t capacity,uint32_t chunks){
  if(state->fatal)return;
  // Adjacent lanes copy consecutive 16-byte chunks. Sorted requests make source
  // rows monotonic; gaps remain possible and need transaction profiling.
  for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<uint64_t(*count)*chunks;i+=uint64_t(gridDim.x)*blockDim.x){
    uint32_t row=uint32_t(i/chunks), chunk=uint32_t(i%chunks);
    states[(uint64_t(state->count)+row)*chunks+chunk]=source[sorted[row]*chunks+chunk];
    if(chunk==0)out_hashes[state->count+row]=hashes[order[row]];
  }
}
__global__ void publish_append(const uint32_t* count,MgbfsFrontierState* state){if(!state->fatal)state->count+=*count;}
extern "C" int mgbfs_materialize_query(uint32_t stride,uint32_t capacity,uint32_t frontier,MgbfsMaterializeBytes* out){
  if(!out)return 1;*out={};
  if(!stride||stride%16||!capacity||capacity>INT32_MAX||!frontier)return 1;
  MgbfsMaterializeBytes q{};q.keys=uint64_t(capacity)*8;q.sorted=uint64_t(capacity)*8;
  q.indices=uint64_t(capacity)*4;q.order=uint64_t(capacity)*4;
  size_t scratch=0;if(cub::DeviceRadixSort::SortPairs(nullptr,scratch,(uint64_t*)nullptr,(uint64_t*)nullptr,(uint32_t*)nullptr,(uint32_t*)nullptr,int(capacity))!=cudaSuccess)return 2;
  q.scratch=scratch;*out=q;return 0;
}
extern "C" int mgbfs_materialize_create(uint32_t stride,uint32_t capacity,uint32_t frontier,void** out,char* error,size_t error_capacity){
  if(!out)return 1;*out=nullptr;
  try{
    MgbfsMaterializeBytes q{};if(mgbfs_materialize_query(stride,capacity,frontier,&q))throw std::runtime_error("MATERIALIZE_CAPACITY_OR_STRIDE");
    auto p=std::make_unique<MaterializePlan>();p->stride=stride;p->capacity=capacity;p->frontier=frontier;
    p->scratch_bytes=q.scratch;checked(cudaMalloc(&p->keys,q.keys));checked(cudaMalloc(&p->sorted,q.sorted));
    checked(cudaMalloc(&p->indices,q.indices));checked(cudaMalloc(&p->order,q.order));
    checked(cudaMalloc(&p->scratch,p->scratch_bytes));*out=p.release();return 0;
  }catch(const std::exception& e){if(error&&error_capacity)std::snprintf(error,error_capacity,"%s",e.what());return 1;}
}
extern "C" int mgbfs_materialize_run(void* plan,const uint8_t* source,uint32_t source_count,const void* hashes,const uint64_t* refs,const uint32_t* count,uint8_t* states,void* out_hashes,MgbfsFrontierState* state,void* raw_stream){
  auto p=static_cast<MaterializePlan*>(plan);
  if(!p||!source||!hashes||!refs||!count||!states||!out_hashes||!state)return 1;
  auto stream=static_cast<cudaStream_t>(raw_stream);
  check_append<<<1,1,0,stream>>>(count,state,p->capacity,p->frontier);
  if(cudaGetLastError()!=cudaSuccess)return 2;
  prepare_requests<<<(p->capacity+255)/256,256,0,stream>>>(refs,count,state,p->keys,p->indices,p->capacity,source_count);
  if(cudaGetLastError()!=cudaSuccess)return 2;
  size_t bytes=p->scratch_bytes;
  if(cub::DeviceRadixSort::SortPairs(p->scratch,bytes,p->keys,p->sorted,p->indices,p->order,int(p->capacity),0,64,stream)!=cudaSuccess)return 3;
  append_states<<<(p->capacity+255)/256,256,0,stream>>>(reinterpret_cast<const uint4*>(source),static_cast<const uint4*>(hashes),p->sorted,p->order,count,reinterpret_cast<uint4*>(states),static_cast<uint4*>(out_hashes),state,p->capacity,p->stride/16);
  if(cudaGetLastError()!=cudaSuccess)return 2;
  publish_append<<<1,1,0,stream>>>(count,state);
  return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" void mgbfs_materialize_destroy(void* p){delete static_cast<MaterializePlan*>(p);}

__global__ void origin_keys(uint32_t source,const MgbfsRegenerateOrigin* origins,
 const uint32_t* count,uint32_t capacity,uint64_t* keys,uint32_t* indices,uint32_t* fatal){
  const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=capacity)return;
  const bool active=*count<=capacity&&i<*count;
  uint64_t key=UINT64_MAX;
  if(i==0&&*count>capacity)atomicCAS(fatal,0u,1u);
  if(active){
    const auto origin=origins[i];
    if(origin.source!=source||origin.reserved)atomicCAS(fatal,0u,2u);
    key=origin.parent;
  }
  keys[i]=key;indices[i]=i;
}
__global__ void sorted_origin_records(const MgbfsRegenerateOrigin* origins,const uint64_t* targets,
 const uint32_t* order,const uint32_t* count,const uint32_t* fatal,
 MgbfsRegenerateOrigin* output,uint64_t* output_targets){
  if(*fatal)return;
  const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=*count)return;
  output[i]=origins[order[i]];output_targets[i]=targets[order[i]];
}
extern "C" int mgbfs_materialize_sort_origins(void* plan,uint32_t source,
 const MgbfsRegenerateOrigin* origins,const uint64_t* targets,const uint32_t* count,
 MgbfsRegenerateOrigin* output,uint64_t* output_targets,uint32_t* fatal,void* raw_stream){
  auto p=static_cast<MaterializePlan*>(plan);
  if(!p||!origins||!targets||!count||!output||!output_targets||!fatal)return 1;
  auto stream=static_cast<cudaStream_t>(raw_stream);
  const uint32_t blocks=(p->capacity+255)/256;
  origin_keys<<<blocks,256,0,stream>>>(source,origins,count,p->capacity,p->keys,p->indices,fatal);
  if(cudaGetLastError()!=cudaSuccess)return 2;
  size_t bytes=p->scratch_bytes;
  // Stable sort: valid UINT64_MAX parents remain ahead of inactive padding.
  if(cub::DeviceRadixSort::SortPairs(p->scratch,bytes,p->keys,p->sorted,p->indices,p->order,
       int(p->capacity),0,64,stream)!=cudaSuccess)return 3;
  sorted_origin_records<<<blocks,256,0,stream>>>(origins,targets,p->order,count,fatal,output,output_targets);
  return cudaGetLastError()==cudaSuccess?0:2;
}

__device__ void response_fatal(MgbfsStateRingControl* ring,MgbfsOwnerControl* owner){
  atomicCAS(&owner->error,0u,18u);atomicCAS(&ring->fatal,0u,18u);
}
__global__ void response_keys(const uint64_t* targets,const uint32_t* count,uint32_t capacity,
 const uint32_t* group_fatal,uint32_t offset,bool complete,uint64_t* keys,uint32_t* indices,MgbfsStateRingControl* ring,
 MgbfsOwnerControl* owner,const MgbfsStateExtent* extent){
  const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=capacity)return;
  if(i==0&&(*count>capacity||offset>*count||extent->count>uint64_t(*count)-offset||
      (complete&&*count!=extent->count)||*group_fatal))response_fatal(ring,owner);
  keys[i]=*count<=capacity&&i<*count?targets[i]:UINT64_MAX;
  indices[i]=i;
}
__global__ void response_mapping(const uint64_t* sorted,const uint32_t* order,const uint32_t* count,
 uint32_t capacity,uint32_t offset,uint64_t* refs,MgbfsStateRingControl* ring,MgbfsOwnerControl* owner,
 const MgbfsStateExtent* extent){
  const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=capacity)return;
  // No concurrent reads of error while other blocks can set it.
  const uint64_t index=uint64_t(offset)+i;
  if(*count<=capacity&&i<extent->count&&index<*count&&
     (extent->sequence>UINT64_MAX-i||sorted[index]!=extent->sequence+i))response_fatal(ring,owner);
  refs[i]=index<capacity?order[index]:0;
}
static int apply_response_span(void* plan,const uint8_t* responses,const uint64_t* targets,
 const uint32_t* count,uint32_t offset,bool complete,const uint32_t* group_fatal,uint8_t* states,MgbfsStateRingControl* ring,
 MgbfsOwnerControl* owner,MgbfsStateExtent* extent,void* raw_stream){
  auto p=static_cast<MaterializePlan*>(plan);
  if(!p||!responses||!targets||!count||!group_fatal||!states||!ring||!owner||!extent)return 1;
  auto stream=static_cast<cudaStream_t>(raw_stream);
  const uint32_t blocks=(p->capacity+255)/256;
  response_keys<<<blocks,256,0,stream>>>(targets,count,p->capacity,group_fatal,offset,complete,p->keys,p->indices,ring,owner,extent);
  if(cudaGetLastError()!=cudaSuccess)return 2;
  size_t bytes=p->scratch_bytes;
  if(cub::DeviceRadixSort::SortPairs(p->scratch,bytes,p->keys,p->sorted,p->indices,p->order,
       int(p->capacity),0,64,stream)!=cudaSuccess)return 3;
  response_mapping<<<blocks,256,0,stream>>>(p->sorted,p->order,count,p->capacity,offset,p->keys,ring,owner,extent);
  if(cudaGetLastError()!=cudaSuccess)return 2;
  return mgbfs_state_materialize(responses,p->capacity,p->keys,p->capacity,p->indices,p->capacity,
    p->stride,states,ring,owner,extent,raw_stream);
}
extern "C" int mgbfs_state_apply_responses(void* plan,const uint8_t* responses,const uint64_t* targets,
 const uint32_t* count,const uint32_t* group_fatal,uint8_t* states,MgbfsStateRingControl* ring,
 MgbfsOwnerControl* owner,MgbfsStateExtent* extent,void* stream){
 return apply_response_span(plan,responses,targets,count,0,true,group_fatal,states,ring,owner,extent,stream);
}
extern "C" int mgbfs_state_apply_response_span(void* plan,const uint8_t* responses,const uint64_t* targets,
 const uint32_t* count,uint32_t offset,const uint32_t* group_fatal,uint8_t* states,MgbfsStateRingControl* ring,
 MgbfsOwnerControl* owner,MgbfsStateExtent* extent,void* stream){
 return apply_response_span(plan,responses,targets,count,offset,false,group_fatal,states,ring,owner,extent,stream);
}
