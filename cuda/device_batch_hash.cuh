#pragma once
#include "device_batch_partition.cuh"
#include "device_segmented_hash_accept.cuh"
#include "bitmap_pack4.cuh"
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cstdint>

// One cooperative grid groups a bounded receive tile into stable shard-owned
// ranges. One CTA owns each shard at a time; independent shards run in parallel.
// Inside a shard each 128-record sub-tile is stably sorted by the full
// (hash, packed exact state), deduplicated, searched against ALL live extents,
// reserved as a group, and materialized by the whole CTA. Claimed slots never
// become reusable until every probe for the tile has completed. On any fatal
// or capacity error the caller must discard the unfinished job generation.
namespace mgbfs_device_batch {
struct HashSpan { uint64_t extent,ordinal,row,count; };
struct HashCapture { uint64_t depth,packed; };
struct HashBatch {
  mgbfs_device_hash::Extent* extents;
  uint32_t* owners;
  uint64_t extent_count;
  // One CTA owns each shard during accept; only that CTA changes its list.
  uint64_t* shard_head;
  uint64_t* extent_next;
  uint64_t* free_cursor;
  MgbfsPack4Dictionary dictionary;
  uint64_t* rank_count;
  uint64_t rank_limit;
  uint32_t shards,bank,state_bytes;
  uint64_t* grouped_origins; // offsets into the still-leased receive buffer
  uint64_t* positions;
  uint64_t* shard_counts;
  uint64_t* shard_starts;
  uint64_t* accepted_shard_mask;
  uint32_t* error;
  HashCapture* capture;
  uint64_t capture_capacity,*capture_count,capture_depth;
};
enum : uint32_t { HASH_RESOURCE=1, HASH_FATAL=2 };
struct HashInput {
  MgbfsPack4Dictionary dictionary;
  uint64_t packed;
  __device__ int64_t value(uint32_t element) const {
    return dictionary.symbols[(packed>>(4*element))&15u];
  }
};
__device__ inline uint32_t hash_shard(uint64_t hash,uint32_t shards) {
  return uint32_t((uint64_t(uint32_t(hash))*shards)>>32);
}
template<class T>
__device__ void hash_accept_shard(const Payload* input,uint64_t input_count,
                                  const uint64_t* origins,uint64_t n,uint32_t shard,HashBatch a) {
  constexpr unsigned block_threads=128;
  using Sort=cub::BlockRadixSort<unsigned long long,block_threads,1,unsigned long long>;
  using Scan=cub::BlockScan<unsigned,block_threads>;
  __shared__ union Storage {
    typename Sort::TempStorage sort;
    typename Scan::TempStorage scan;
  } storage;
  __shared__ uint64_t hashes[block_threads],words[block_threads];
  __shared__ HashSpan spans[block_threads];
  __shared__ unsigned span_count,tile_error,tile_total,stopped;
  const unsigned tid=threadIdx.x;
  for(uint64_t begin=0;begin<n;begin+=block_threads) {
    // Finish any transaction already underway before observing another CTA's
    // stop. No partial tile can leak into a later lookup.
    if(!tid)stopped=atomicAdd(a.error,0u)!=0;
    __syncthreads();
    if(stopped)break;
    const unsigned length=unsigned(min(uint64_t(block_threads),n-begin));
    const bool present=tid<length;
    const uint64_t origin=present?origins[begin+tid]:0;
    const bool valid_origin=!present || origin<input_count;
    const Payload original=present&&valid_origin?input[origin]:Payload{UINT64_MAX,UINT64_MAX};
    unsigned long long keys[1]={original.packed},values[1]={original.key};
    Sort(storage.sort).Sort(keys,values);
    __syncthreads();
    unsigned long long tmp=keys[0];keys[0]=values[0];values[0]=tmp;
    Sort(storage.sort).Sort(keys,values);
    __syncthreads();
    hashes[tid]=keys[0];words[tid]=values[0];
    if(!tid){tile_error=0;span_count=0;tile_total=0;}
    __syncthreads();
    if(!valid_origin)atomicOr(&tile_error,uint32_t(HASH_FATAL));
    __syncthreads();
    bool fresh=present && (!tid || hashes[tid]!=hashes[tid-1] || words[tid]!=words[tid-1]);
    if(fresh) {
      if(!mgbfs_pack4_valid_word(a.dictionary,words[tid])) {
        atomicOr(&tile_error,uint32_t(HASH_FATAL));fresh=false;
      } else {
        HashInput word{a.dictionary,words[tid]};
        uint64_t e=a.shard_head[shard],traversed=0;
        while(e!=UINT64_MAX) {
          if(e>=a.extent_count || ++traversed>a.extent_count ||
             cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(a.owners[e])
               .load(cuda::memory_order_acquire)!=shard) {
            atomicOr(&tile_error,uint32_t(HASH_FATAL));fresh=false;break;
          }
          const auto extent=a.extents[e];
          if(!mgbfs_device_hash::valid_extent(extent,a.bank,a.dictionary.elements,sizeof(T))) {
            atomicOr(&tile_error,uint32_t(HASH_FATAL));fresh=false;break;
          }
          const auto p=mgbfs_device_hash::probe<T>(extent,a.bank,a.dictionary.elements,hashes[tid],word);
          if(!p.valid) {atomicOr(&tile_error,uint32_t(HASH_FATAL));fresh=false;break;}
          if(p.duplicate){fresh=false;break;}
          e=a.extent_next[e];
        }
      }
    }
    __syncthreads();
    unsigned ordinal=0,total=0;
    Scan(storage.scan).ExclusiveSum(unsigned(fresh),ordinal,total);
    __syncthreads();
    if(!tid) {
      tile_total=total;
      if(!tile_error && total) {
        // All reservations are for exact new keys; there are no temporary
        // per-row reservations to return after another shard drains.
        auto rank=cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(*a.rank_count);
        uint64_t old=rank.load(cuda::memory_order_acquire);
        for(;;) {
          if(UINT64_MAX-old<total){tile_error|=HASH_FATAL;break;}
          if(a.rank_limit && (old>a.rank_limit || uint64_t(total)>a.rank_limit-old)) {
            tile_error|=HASH_RESOURCE;break;
          }
          if(rank.compare_exchange_weak(old,old+total,cuda::memory_order_acq_rel,
                                       cuda::memory_order_acquire))break;
        }
        uint64_t left=total,position=0;
        for(unsigned pass=0;pass<2 && left && !tile_error;++pass) {
          const uint64_t start=pass?a.free_cursor[shard]:0;
          if(start>=a.extent_count){tile_error|=HASH_FATAL;break;}
          for(uint64_t step=0;step<a.extent_count && left;++step) {
            const uint64_t e=pass?(step<a.extent_count-start?start+step:step-(a.extent_count-start)):step;
            uint32_t owner=cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(a.owners[e])
                .load(cuda::memory_order_acquire);
            if(pass==0 && owner!=shard)continue;
            if(pass==1) {
              if(owner!=UINT32_MAX)continue;
              uint32_t free=UINT32_MAX;
              if(!cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(a.owners[e])
                   .compare_exchange_strong(free,shard,cuda::memory_order_acq_rel,
                                            cuda::memory_order_acquire))continue;
            }
            const auto x=a.extents[e];
            if(!mgbfs_device_hash::valid_extent(x,a.bank,a.dictionary.elements,sizeof(T))) {
              tile_error|=HASH_FATAL;break;
            }
            if(pass==1 && (x.counts[0]||x.counts[1]||x.counts[2])) {
              tile_error|=HASH_FATAL;break;
            }
            if(pass==1) {
              a.extent_next[e]=a.shard_head[shard];
              a.shard_head[shard]=e;
              a.free_cursor[shard]=(e+1<a.extent_count)?e+1:0;
            }
            const uint64_t row=x.counts[a.bank];
            if(row>x.segment.capacity){tile_error|=HASH_FATAL;break;}
            const uint64_t available=x.segment.capacity-row;
            const uint64_t grant=left<available?left:available;
            if(!grant)continue;
            if(span_count>=block_threads){tile_error|=HASH_FATAL;break;}
            spans[span_count++]={e,position,row,grant};
            left-=grant;position+=grant;
          }
        }
        if(left && !tile_error)tile_error|=HASH_RESOURCE;
      }
      if(tile_error)atomicOr(a.error,tile_error);
    }
    __syncthreads();
    if(fresh && !tile_error) {
      unsigned which=0;
      while(which<span_count && spans[which].ordinal+spans[which].count<=ordinal)++which;
      if(which>=span_count)atomicOr(&tile_error,uint32_t(HASH_FATAL));
      else {
        const HashSpan span=spans[which];
        const auto x=a.extents[span.extent];const auto s=x.segment;
        const uint64_t row=span.row+(uint64_t(ordinal)-span.ordinal);
        const uint64_t start=mgbfs_shared_table_bucket(hashes[tid],s.slots_count);
        uint64_t claimed=s.slots_count;
        for(uint64_t step=0;step<s.slots_count;++step) {
          const uint64_t slot=(start+step)&(s.slots_count-1);
          auto ref=cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(s.slots[slot]);
          uint64_t seen=ref.load(cuda::memory_order_acquire);
          if(seen!=mgbfs_segmented_exact::empty && seen!=mgbfs_segmented_exact::retired)continue;
          const uint64_t pending=mgbfs_segmented_exact::pending|uint64_t(tid+1);
          if(ref.compare_exchange_strong(seen,pending,cuda::memory_order_acq_rel,
                                        cuda::memory_order_acquire)){claimed=slot;break;}
        }
        if(row>=s.capacity)atomicOr(&tile_error,uint32_t(HASH_FATAL));
        else if(claimed==s.slots_count)atomicOr(&tile_error,uint32_t(HASH_RESOURCE));
        else {
          const uint64_t local=uint64_t(a.bank)*s.capacity+row,stride=s.capacity*s.banks;
          auto* states=static_cast<T*>(s.states);
          const HashInput word{a.dictionary,words[tid]};
          for(uint32_t e=0;e<a.dictionary.elements;++e) {
            const int64_t value=word.value(e);
            if(int64_t(static_cast<T>(value))!=value)atomicOr(&tile_error,uint32_t(HASH_FATAL));
            states[uint64_t(e)*stride+local]=static_cast<T>(value);
          }
          if(s.positions) {
            if(s.position_item_bytes==4)reinterpret_cast<uint32_t*>(s.positions)[local]=uint32_t(claimed);
            else s.positions[local]=claimed;
          }
          const uint64_t token=(hashes[tid]&mgbfs_segmented_exact::prefix)|(uint64_t(a.bank)<<32)|row;
          cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(s.slots[claimed])
            .store(token,cuda::memory_order_release);
          if(a.capture) {
            const uint64_t at=atomicAdd(reinterpret_cast<unsigned long long*>(a.capture_count),1ull);
            if(at>=a.capture_capacity)atomicOr(&tile_error,uint32_t(HASH_FATAL));
            else a.capture[at]={a.capture_depth,words[tid]};
          }
        }
      }
    }
    __syncthreads();
    if(!tid) {
      if(tile_error)atomicOr(a.error,tile_error);
      else if(tile_total) {
        for(unsigned i=0;i<span_count;++i) {
          const auto span=spans[i];
          auto x=a.extents[span.extent];
          if(x.counts[a.bank]!=span.row || span.count>x.segment.capacity-span.row) {
            atomicOr(a.error,uint32_t(HASH_FATAL));break;
          }
          cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(x.counts[a.bank])
            .store(span.row+span.count,cuda::memory_order_release);
        }
        if(a.accepted_shard_mask && shard<64)
          atomicOr(reinterpret_cast<unsigned long long*>(a.accepted_shard_mask),1ull<<shard);
      }
    }
    __syncthreads();
  }
}
__device__ inline void hash_accept(const Payload* input,uint64_t n,HashBatch a) {
  auto grid=cooperative_groups::this_grid();
  const uint64_t tid=grid.thread_rank();
  if(!tid)*a.error=0;
  for(uint64_t s=tid;s<a.shards;s+=grid.size())a.shard_counts[s]=0;
  grid.sync();
  if(blockDim.x!=128 || !a.shards || a.bank>=3 || !a.rank_count || !a.grouped_origins ||
     !a.positions || !a.shard_counts || !a.shard_starts ||
     ((!a.extents||!a.owners)&&a.extent_count) ||
     !a.extent_count || !a.shard_head || !a.extent_next || !a.free_cursor ||
     !(a.state_bytes==1 || a.state_bytes==8) || !mgbfs_pack4_valid(a.dictionary) ||
     bool(a.capture)!=bool(a.capture_count) || (a.capture && !a.capture_capacity)) {
    if(!tid)atomicOr(a.error,uint32_t(HASH_FATAL));
    grid.sync();return;
  }
  const uint64_t step=uint64_t(gridDim.x)*blockDim.x;
  const unsigned lane=threadIdx.x&31u;
  const unsigned earlier=lane?((1u<<lane)-1u):0u;
  for(uint64_t begin=uint64_t(blockIdx.x)*blockDim.x;begin<n;) {
    const uint64_t i=begin+threadIdx.x;
    const bool present=i>=begin&&i<n;
    const uint32_t shard=present?hash_shard(input[i].key,a.shards):0;
    const unsigned mask=__ballot_sync(0xffffffffu,present);
    if(present) {
      const unsigned group=__match_any_sync(mask,shard);
      const int leader=__ffs(int(group))-1;
      unsigned long long base=0;
      if(int(lane)==leader)
        base=atomicAdd(reinterpret_cast<unsigned long long*>(&a.shard_counts[shard]),
                       static_cast<unsigned long long>(__popc(group)));
      base=__shfl_sync(group,base,leader);
      a.positions[i]=uint64_t(base)+__popc(group&earlier);
    }
    if(n-begin<=step)break;
    begin+=step;
  }
  grid.sync();
  if(!tid) {
    uint64_t prefix=0;
    for(uint32_t s=0;s<a.shards;++s) {
      a.shard_starts[s]=prefix;
      if(UINT64_MAX-prefix<a.shard_counts[s]){*a.error|=HASH_FATAL;break;}
      prefix+=a.shard_counts[s];
    }
    a.shard_starts[a.shards]=prefix;
    if(prefix!=n)*a.error|=HASH_FATAL;
  }
  grid.sync();
  if(!*a.error) {
    for(uint64_t i=tid;i<n;i+=grid.size()) {
      const Payload x=input[i];const uint32_t shard=hash_shard(x.key,a.shards);
      const uint64_t at=a.shard_starts[shard]+a.positions[i];
      if(at>=n)atomicOr(a.error,uint32_t(HASH_FATAL));else a.grouped_origins[at]=i;
    }
  }
  grid.sync();
  __shared__ unsigned block_run;
  if(!threadIdx.x)block_run=atomicAdd(a.error,0u)==0;
  __syncthreads();
  if(block_run) {
    for(uint32_t shard=blockIdx.x;shard<a.shards;shard+=gridDim.x) {
      const uint64_t begin=a.shard_starts[shard],end=a.shard_starts[shard+1];
      if(a.state_bytes==1)hash_accept_shard<uint8_t>(input,n,a.grouped_origins+begin,end-begin,shard,a);
      else hash_accept_shard<int64_t>(input,n,a.grouped_origins+begin,end-begin,shard,a);
    }
  }
  grid.sync();
}
}
