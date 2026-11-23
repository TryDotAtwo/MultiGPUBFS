#pragma once
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdint>

// Job-owned scratch; called by every thread of one cooperatively launched grid.
// input, positions, output and metadata must remain live until the grid drains.
// counts/offsets are u64; each record is written once into its destination range.
// On a bad destination or insufficient output capacity no payload is published.
namespace mgbfs_device_batch {
constexpr uint32_t max_world=8;
struct Payload {uint64_t key,packed;};
struct Input {uint64_t key,packed;uint32_t destination,reserved;};
static_assert(sizeof(Payload)==16,"wire payload");
static_assert(sizeof(Input)==24,"raw input");
enum : uint32_t { BAD_WORLD=1, BAD_DESTINATION=2, CAPACITY=4, OFFSET_OVERFLOW=8 };

__device__ inline void partition(const Input* input,Payload* output,
    uint64_t count,uint64_t capacity,uint32_t world,uint64_t* positions,
    uint64_t* block_prefix,uint64_t* offsets,uint32_t* error) {
  const auto grid=cooperative_groups::this_grid();
  const uint32_t tid=threadIdx.x;
  __shared__ unsigned long long local_count[max_world];
  if(!blockIdx.x && !tid) *error=0;
  grid.sync();
  if(!world || world>max_world) {
    if(!blockIdx.x && !tid) *error=BAD_WORLD;
    grid.sync();
    return;
  }
  for(uint32_t d=tid;d<world;d+=blockDim.x)local_count[d]=0;
  __syncthreads();
  const uint64_t step=uint64_t(gridDim.x)*blockDim.x;
  for(uint64_t begin=uint64_t(blockIdx.x)*blockDim.x;begin<count;) {
    const uint64_t i=begin+tid;
    const bool present=i>=begin && i<count;
    uint32_t destination=0;
    if(present) {
      destination=input[i].destination;
      if(destination>=world)atomicOr(error,uint32_t(BAD_DESTINATION));
    }
    const unsigned valid=__ballot_sync(0xffffffffu,present&&destination<world);
    if(present && destination<world) {
      const unsigned members=__match_any_sync(valid,destination);
      const int leader=__ffs(int(members))-1;
      const int lane=int(tid&31);
      unsigned long long base=0;
      if(lane==leader)
        base=atomicAdd(&local_count[destination],static_cast<unsigned long long>(__popc(members)));
      base=__shfl_sync(members,base,leader);
      const unsigned earlier=lane?((1u<<lane)-1u):0u;
      positions[i]=uint64_t(base)+__popc(members&earlier);
    }
    if(count-begin<=step)break;
    begin+=step;
  }
  __syncthreads();
  for(uint32_t d=tid;d<world;d+=blockDim.x)
    block_prefix[uint64_t(blockIdx.x)*world+d]=uint64_t(local_count[d]);
  grid.sync();
  if(blockIdx.x==0 && tid<world) {
    const uint32_t d=tid;
    uint64_t total=0;
    for(uint32_t b=0;b<gridDim.x;++b) {
      const uint64_t i=uint64_t(b)*world+d;
      const uint64_t next=block_prefix[i];
      block_prefix[i]=total;
      if(UINT64_MAX-total<next)atomicOr(error,uint32_t(OFFSET_OVERFLOW));
      else total+=next;
    }
    offsets[d]=total;
  }
  grid.sync();
  if(blockIdx.x==0 && tid==0) {
    uint64_t total=0;
    for(uint32_t d=0;d<world;++d) {
      const uint64_t next=offsets[d];
      offsets[d]=total;
      if(UINT64_MAX-total<next)atomicOr(error,uint32_t(OFFSET_OVERFLOW));
      else total+=next;
    }
    offsets[world]=total;
    if(total>capacity)atomicOr(error,uint32_t(CAPACITY));
  }
  grid.sync();
  if(!*error) {
    for(uint64_t begin=uint64_t(blockIdx.x)*blockDim.x;begin<count;) {
      const uint64_t i=begin+tid;
      if(i>=begin && i<count) {
        const Input value=input[i];
        const uint64_t at=offsets[value.destination]+
            block_prefix[uint64_t(blockIdx.x)*world+value.destination]+positions[i];
        output[at]={value.key,value.packed};
      }
      if(count-begin<=step)break;
      begin+=step;
    }
  }
  grid.sync();
}
}
