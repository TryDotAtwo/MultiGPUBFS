#pragma once
#include "device_batch_partition.cuh"
#include "device_pack4_accept.cuh"
#include <cuda_runtime.h>
#include <cstdint>

// A bounded receive tile, with a grid-wide prepare/reserve/materialize boundary.
// Visited bits are tentative until the tile and the whole layer drain. On any
// error the caller must discard the unfinished job generation, never reuse its
// visited plane, and expose only the previous completed layer.
namespace mgbfs_device_batch {
struct BitmapSpan {uint64_t extent,ordinal,row,count;};
struct BitmapBatch {
  mgbfs_device_pack4::Extent* extents;
  uint32_t* owners;
  uint64_t extent_count;
  uint64_t* bits;
  uint64_t domain,owner_len;
  uint64_t* rank_count;
  uint64_t rank_limit;
  uint32_t world,owner,shards,bank;
  uint64_t* positions;
  uint64_t* shard_counts;
  uint64_t* shard_starts;
  BitmapSpan* spans;
  uint32_t* error;
};
enum : uint32_t {BITMAP_RESOURCE=1, BITMAP_FATAL=2};
__device__ inline void bitmap_accept(const Payload* input,uint64_t n,BitmapBatch a) {
  const auto grid=cooperative_groups::this_grid();
  const uint64_t tid=grid.thread_rank();
  if(!tid)*a.error=0;
  for(uint64_t s=tid;s<a.shards;s+=grid.size())a.shard_counts[s]=0;
  grid.sync();
  if(!a.world||a.owner>=a.world||!a.shards||a.bank>=3||!a.bits||
     !a.rank_count||!a.positions||!a.shard_counts||!a.shard_starts||
     !a.spans||(!a.extents&&a.extent_count)||(!a.owners&&a.extent_count)) {
    if(!tid)atomicOr(a.error,uint32_t(BITMAP_FATAL));
    grid.sync();return;
  }
  const uint64_t step=uint64_t(gridDim.x)*blockDim.x;
  const unsigned lane=threadIdx.x&31u;
  const unsigned lower=lane?((1u<<lane)-1u):0u;
  for(uint64_t begin=uint64_t(blockIdx.x)*blockDim.x;begin<n;) {
    const uint64_t i=begin+threadIdx.x;
    const bool present=i>=begin&&i<n;
    uint64_t id=0,local=0;
    bool valid=false;
    if(present) {
      id=input[i].key;
      valid=id<a.domain && id%a.world==a.owner;
      if(valid){local=id/a.world;valid=local<a.owner_len;}
      a.positions[i]=UINT64_MAX;
      if(!valid)atomicOr(a.error,uint32_t(BITMAP_FATAL));
    }
    const unsigned valid_mask=__ballot_sync(0xffffffffu,present&&valid);
    bool winner=false;
    if(present&&valid) {
      const unsigned word_group=__match_any_sync(valid_mask,local>>6);
      const unsigned id_group=__match_any_sync(valid_mask,id);
      const bool representative=int(lane)==__ffs(int(id_group))-1;
      const unsigned long long bit=1ull<<(local&63u);
      unsigned long long combined=0;
      for(unsigned rest=word_group;rest;rest&=rest-1) {
        int source=__ffs(int(rest))-1;
        combined|=__shfl_sync(word_group,representative?bit:0ull,source);
      }
      const int leader=__ffs(int(word_group))-1;
      unsigned long long old=0;
      if(int(lane)==leader)
        old=atomicOr(reinterpret_cast<unsigned long long*>(&a.bits[local>>6]),combined);
      old=__shfl_sync(word_group,old,leader);
      winner=representative && !(old&bit);
    }
    const unsigned winner_mask=__ballot_sync(0xffffffffu,winner);
    if(winner) {
      const uint32_t shard=uint32_t((uint64_t(uint32_t(mix64(id)))*a.shards)>>32);
      const unsigned group=__match_any_sync(winner_mask,shard);
      const int leader=__ffs(int(group))-1;
      unsigned long long base=0;
      if(int(lane)==leader)
        base=atomicAdd(reinterpret_cast<unsigned long long*>(&a.shard_counts[shard]),
                       static_cast<unsigned long long>(__popc(group)));
      base=__shfl_sync(group,base,leader);
      a.positions[i]=uint64_t(base)+__popc(group&lower);
    }
    if(n-begin<=step)break;
    begin+=step;
  }
  grid.sync();
  if(!tid) {
    uint64_t total=0;
    for(uint32_t s=0;s<a.shards;++s) {
      const uint64_t count=a.shard_counts[s];
      if(UINT64_MAX-total<count){*a.error|=BITMAP_FATAL;break;}
      total+=count;
    }
    if(UINT64_MAX-*a.rank_count<total)*a.error|=BITMAP_FATAL;
    else if(a.rank_limit && (total>a.rank_limit || *a.rank_count>a.rank_limit-total))
      *a.error|=BITMAP_RESOURCE;
    uint64_t used=0;
    for(uint32_t s=0;s<a.shards;++s) {
      a.shard_starts[s]=used;
      if(*a.error)continue;
      uint64_t remaining=a.shard_counts[s],ordinal=0;
      // One coordinator owns all metadata changes while the grid is stopped.
      // Existing extents are used first; free extents are then leased once.
      for(unsigned pass=0;pass<2 && remaining;++pass) {
        for(uint64_t e=0;e<a.extent_count && remaining;++e) {
          const uint32_t owner=a.owners[e];
          if((pass==0 && owner!=s)||(pass==1 && owner!=UINT32_MAX))continue;
          auto extent=a.extents[e];
          if(!extent.rows||!extent.counts||!extent.capacity||extent.capacity>(1ull<<32)||
             extent.banks!=3||extent.reserved) {*a.error|=BITMAP_FATAL;break;}
          if(pass==1) {
            if(extent.counts[0]||extent.counts[1]||extent.counts[2])
              {*a.error|=BITMAP_FATAL;break;}
            a.owners[e]=s;
          }
          const uint64_t current=extent.counts[a.bank];
          if(current>extent.capacity){*a.error|=BITMAP_FATAL;break;}
          const uint64_t available=extent.capacity-current;
          const uint64_t grant=remaining<available?remaining:available;
          if(!grant)continue;
          if(used>=a.extent_count){*a.error|=BITMAP_FATAL;break;}
          a.spans[used++]={e,ordinal,current,grant};
          extent.counts[a.bank]=current+grant;
          remaining-=grant;ordinal+=grant;
        }
      }
      if(remaining && !*a.error)*a.error|=BITMAP_RESOURCE;
    }
    a.shard_starts[a.shards]=used;
    if(!*a.error)*a.rank_count+=total;
  }
  grid.sync();
  if(!*a.error) {
    for(uint64_t i=tid;i<n;i+=grid.size()) {
      const uint64_t ordinal=a.positions[i];
      if(ordinal==UINT64_MAX)continue;
      const uint64_t id=input[i].key;
      const uint32_t shard=uint32_t((uint64_t(uint32_t(mix64(id)))*a.shards)>>32);
      uint64_t lo=a.shard_starts[shard],hi=a.shard_starts[shard+1];
      while(lo<hi) {
        const uint64_t mid=lo+(hi-lo)/2;
        const BitmapSpan span=a.spans[mid];
        if(span.ordinal+span.count<=ordinal)lo=mid+1;else hi=mid;
      }
      if(lo>=a.shard_starts[shard+1]){atomicOr(a.error,uint32_t(BITMAP_FATAL));continue;}
      const BitmapSpan span=a.spans[lo];
      if(ordinal<span.ordinal||ordinal-span.ordinal>=span.count||span.extent>=a.extent_count)
        {atomicOr(a.error,uint32_t(BITMAP_FATAL));continue;}
      const auto extent=a.extents[span.extent];
      extent.rows[uint64_t(a.bank)*extent.capacity+span.row+(ordinal-span.ordinal)]=input[i].packed;
    }
  }
  grid.sync();
}
}
