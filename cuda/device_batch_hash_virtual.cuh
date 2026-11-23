// Included inside mgbfs_device_batch after HashBatch and hash_accept_shard.
// CTA ownership is by (physical shard, exact-key virtual bucket). A complete
// grid barrier separates every snapshot lookup from every table mutation.
__device__ inline uint32_t hash_virtual_group(uint64_t hash,uint64_t packed,
                                              uint32_t shards,uint32_t buckets) {
  uint64_t x=packed+0x9e3779b97f4a7c15ull;
  x=(x^(x>>30))*0xbf58476d1ce4e5b9ull;
  x=(x^(x>>27))*0x94d049bb133111ebull;
  x^=x>>31;
  return hash_shard(hash,shards)*buckets+uint32_t(x&(buckets-1));
}

template<class T>
__device__ void hash_accept_virtual(const Payload* input,uint64_t input_count,HashBatch a) {
  auto grid=cooperative_groups::this_grid();
  constexpr unsigned threads=128;
  using Sort=cub::BlockRadixSort<unsigned long long,threads,1,unsigned long long>;
  using Scan=cub::BlockScan<unsigned,threads>;
  __shared__ union SharedTemp {
    typename Sort::TempStorage sort;
    typename Scan::TempStorage scan;
  } temp;
  __shared__ uint64_t hashes[threads],words[threads];
  __shared__ HashSpan spans[threads];
  __shared__ unsigned span_count,tile_error,tile_total,left,position;
  __shared__ uint64_t max_tiles;
  const unsigned tid=threadIdx.x;
  const uint32_t groups=a.shards*a.virtual_buckets;
  for(uint64_t base=0;base<groups;base+=gridDim.x) {
    const uint64_t group=base+blockIdx.x;
    if(!tid) {
      uint64_t maximum=0;
      const uint64_t end=min(uint64_t(groups),base+gridDim.x);
      for(uint64_t b=base;b<end;++b) {
        const uint64_t len=a.shard_starts[b+1]-a.shard_starts[b];
        const uint64_t tiles=len?1+(len-1)/threads:0;
        if(tiles>maximum)maximum=tiles;
      }
      max_tiles=maximum;
    }
    __syncthreads();
    for(uint64_t tile=0;tile<max_tiles;++tile) {
      const uint64_t group_begin=group<groups?a.shard_starts[group]:0;
      const uint64_t group_end=group<groups?a.shard_starts[group+1]:0;
      const uint64_t offset=tile*threads;
      const uint64_t group_len=group_end-group_begin;
      const unsigned length=offset<group_len?unsigned(min(uint64_t(threads),group_len-offset)):0;
      const bool present=tid<length;
      const uint64_t origin=present?a.grouped_origins[group_begin+offset+tid]:0;
      const bool valid_origin=!present || origin<input_count;
      const Payload in=present&&valid_origin?input[origin]:Payload{UINT64_MAX,UINT64_MAX};
      unsigned long long keys[1]={in.packed},values[1]={in.key};
      Sort(temp.sort).Sort(keys,values);
      __syncthreads();
      auto tmp=keys[0];keys[0]=values[0];values[0]=tmp;
      Sort(temp.sort).Sort(keys,values);
      __syncthreads();
      hashes[tid]=keys[0];words[tid]=values[0];
      if(!tid){span_count=0;tile_error=0;tile_total=0;left=0;position=0;}
      __syncthreads();
      if(!valid_origin)atomicOr(&tile_error,uint32_t(HASH_FATAL));
      bool fresh=present && (!tid || hashes[tid]!=hashes[tid-1] || words[tid]!=words[tid-1]);
      const uint32_t shard=group<groups?uint32_t(group/a.virtual_buckets):0;
      if(fresh) {
        if(!mgbfs_pack4_valid_word(a.dictionary,words[tid])) {
          atomicOr(&tile_error,uint32_t(HASH_FATAL));fresh=false;
        } else {
          const HashInput word{a.dictionary,words[tid]};
          uint64_t e=cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(a.shard_head[shard])
                       .load(cuda::memory_order_acquire),traversed=0;
          while(e!=UINT64_MAX) {
            if(e>=a.extent_count || ++traversed>a.extent_count ||
               cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(a.owners[e])
                  .load(cuda::memory_order_acquire)!=shard) {
              atomicOr(&tile_error,uint32_t(HASH_FATAL));fresh=false;break;
            }
            const auto extent=a.extents[e];
            if(!mgbfs_device_hash::valid_extent(extent,a.bank,
                a.packed_states?1:a.dictionary.elements,a.packed_states?8:sizeof(T))) {
              atomicOr(&tile_error,uint32_t(HASH_FATAL));fresh=false;break;
            }
            const auto p=a.packed_states?
              mgbfs_device_hash::probe<T,HashInput,true>(extent,a.bank,a.dictionary.elements,hashes[tid],word):
              mgbfs_device_hash::probe<T>(extent,a.bank,a.dictionary.elements,hashes[tid],word);
            if(!p.valid){atomicOr(&tile_error,uint32_t(HASH_FATAL));fresh=false;break;}
            if(p.duplicate){fresh=false;break;}
            e=a.extent_next[e];
          }
        }
      }
      __syncthreads();
      unsigned ordinal=0,total=0;
      Scan(temp.scan).ExclusiveSum(unsigned(fresh),ordinal,total);
      __syncthreads();
      if(!tid) {
        tile_total=total;
        if(a.owner_stats&&length) {
          const uint32_t active=atomicAdd(a.owner_stats+1,1u)+1;
          atomicMax(a.owner_stats,active);
          const uint32_t same=atomicAdd(a.owner_stats+6+shard,1u)+1;
          atomicMax(a.owner_stats+5,same);
          unsigned sm=0;asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
          if(sm<96)atomicOr(a.owner_stats+2+sm/32,1u<<(sm%32));
        }
        if(tile_error)atomicOr(a.error,tile_error);
      }
      // No CTA may publish or claim a table slot while any peer CTA is still
      // comparing its snapshot. All claims happen after this grid barrier.
      grid.sync();

      if(!tid && total && !atomicAdd(a.error,0u)) {
        auto rank=cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(*a.rank_count);
        uint64_t old=rank.load(cuda::memory_order_acquire);
        for(;;) {
          if(UINT64_MAX-old<total){tile_error|=HASH_FATAL;break;}
          if(a.rank_limit&&(old>a.rank_limit||uint64_t(total)>a.rank_limit-old)){
            tile_error|=HASH_RESOURCE;break;
          }
          if(rank.compare_exchange_weak(old,old+total,cuda::memory_order_acq_rel,
                                       cuda::memory_order_acquire))break;
        }
        left=total;
        // Existing same-shard extents first, then unowned extents from the
        // shared rank pool. Each count reservation is bounded and 64-bit.
        for(unsigned pass=0;pass<2 && left && !tile_error;++pass) {
          uint64_t e=pass?0:cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(a.shard_head[shard])
                                .load(cuda::memory_order_acquire);
          uint64_t visited=0;
          while(left && visited<a.extent_count) {
            if(pass) {
              e=(uint64_t(blockIdx.x)+visited)%a.extent_count;
            } else if(e==UINT64_MAX)break;
            if(e>=a.extent_count){tile_error|=HASH_FATAL;break;}
            ++visited;
            auto owner=cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(a.owners[e]);
            uint32_t who=owner.load(cuda::memory_order_acquire);
            if(pass) {
              if(who!=UINT32_MAX)continue;
              uint32_t empty_owner=UINT32_MAX;
              if(!owner.compare_exchange_strong(empty_owner,shard,cuda::memory_order_acq_rel,
                                               cuda::memory_order_acquire))continue;
            } else if(who!=shard){tile_error|=HASH_FATAL;break;}
            auto x=a.extents[e];
            if(!mgbfs_device_hash::valid_extent(x,a.bank,
                a.packed_states?1:a.dictionary.elements,a.packed_states?8:sizeof(T))) {
              tile_error|=HASH_FATAL;break;
            }
            if(pass) {
              if(x.counts[0]||x.counts[1]||x.counts[2]){tile_error|=HASH_FATAL;break;}
              auto head=cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(a.shard_head[shard]);
              uint64_t old_head=head.load(cuda::memory_order_acquire);
              do {a.extent_next[e]=old_head;}
              while(!head.compare_exchange_weak(old_head,e,cuda::memory_order_acq_rel,
                                               cuda::memory_order_acquire));
            }
            auto count=cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(x.counts[a.bank]);
            uint64_t row=count.load(cuda::memory_order_acquire);
            while(left) {
              if(row>x.segment.capacity){tile_error|=HASH_FATAL;break;}
              const uint64_t grant=min(uint64_t(left),x.segment.capacity-row);
              if(!grant)break;
              if(count.compare_exchange_weak(row,row+grant,cuda::memory_order_acq_rel,
                                             cuda::memory_order_acquire)) {
                if(span_count>=threads){tile_error|=HASH_FATAL;break;}
                spans[span_count++]={e,position,row,grant};
                left-=unsigned(grant);position+=unsigned(grant);break;
              }
            }
            if(!pass)e=a.extent_next[e];
          }
        }
      }
      // All extents leased by any CTA are now linked. A temporary miss is
      // retried against the complete list before reporting a real shortage.
      grid.sync();
      if(!tid && left && !tile_error && !atomicAdd(a.error,0u)) {
        uint64_t e=cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(a.shard_head[shard])
                     .load(cuda::memory_order_acquire),visited=0;
        while(left && e!=UINT64_MAX && visited++<a.extent_count) {
          if(e>=a.extent_count){tile_error|=HASH_FATAL;break;}
          auto x=a.extents[e];
          auto count=cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(x.counts[a.bank]);
          uint64_t row=count.load(cuda::memory_order_acquire);
          while(left) {
            if(row>x.segment.capacity){tile_error|=HASH_FATAL;break;}
            const uint64_t grant=min(uint64_t(left),x.segment.capacity-row);
            if(!grant)break;
            if(count.compare_exchange_weak(row,row+grant,cuda::memory_order_acq_rel,
                                           cuda::memory_order_acquire)) {
              if(span_count>=threads){tile_error|=HASH_FATAL;break;}
              spans[span_count++]={e,position,row,grant};
              left-=unsigned(grant);position+=unsigned(grant);break;
            }
          }
          e=a.extent_next[e];
        }
        if(e!=UINT64_MAX && visited>=a.extent_count)tile_error|=HASH_FATAL;
        if(left&&!tile_error)tile_error|=HASH_RESOURCE;
      }
      if(!tid&&tile_error)atomicOr(a.error,tile_error);
      grid.sync();

      if(fresh&&!atomicAdd(a.error,0u)) {
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
            if(a.packed_states)static_cast<uint64_t*>(s.states)[local]=words[tid];
            else {
              auto* states=static_cast<T*>(s.states);
              const HashInput word{a.dictionary,words[tid]};
              for(uint32_t k=0;k<a.dictionary.elements;++k) {
                const int64_t value=word.value(k);
                if(int64_t(static_cast<T>(value))!=value)atomicOr(&tile_error,uint32_t(HASH_FATAL));
                states[uint64_t(k)*stride+local]=static_cast<T>(value);
              }
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
        if(tile_total&&a.accepted_shard_mask&&shard<64&&!tile_error)
          atomicOr(reinterpret_cast<unsigned long long*>(a.accepted_shard_mask),1ull<<shard);
      }
      // All inserted rows and slots complete before the next lookup snapshot.
      grid.sync();
      if(!tid&&a.owner_stats&&length) {
        atomicSub(a.owner_stats+6+shard,1u);
        atomicSub(a.owner_stats+1,1u);
      }
      grid.sync();
      if(atomicAdd(a.error,0u))return;
    }
  }
}
