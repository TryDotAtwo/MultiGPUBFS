#pragma once
#include "device_pack4_bfs.cuh"
#include "device_batch_window.cuh"
#include "device_batch_hash.cuh"
#include "device_extent_prefix.cuh"
#include <cooperative_groups.h>

// Opt-in experimental multi-CTA engine ABI. Buffers are private to a job and
// their full sizes, non-aliasing and lifetimes are validated by the host.
namespace mgbfs_resident_hash_batch {
using namespace mgbfs_device_batch;
struct Args {
  mgbfs_resident::Args base;
  uint64_t send_capacity,receive_capacity;
  Input* raw;
  uint64_t* positions;
  uint64_t* block_prefix;
  uint64_t* offsets;
  Payload* receive;
  uint64_t* grouped_origins;
  uint64_t* shard_counts;
  uint64_t* shard_starts;
  void* spans;
  uint64_t* parent_prefix;
  uint32_t* error;
};
#ifdef MGBFS_BATCH_HASH_BFS_IMPLEMENTATION
__global__ void run(ncclWindow_t window,ncclDevComm comm,Args args,mgbfs_resident::Report* report) {
  using namespace mgbfs_resident;
  auto grid=cooperative_groups::this_grid();
  const auto& a=args.base;
  auto* extents=reinterpret_cast<mgbfs_device_hash::Extent*>(a.extents);
  const auto* config=reinterpret_cast<const HashConfig*>(a.visited);
  const uint64_t tid=grid.thread_rank();
  Window* peers[mgbfs_device_batch::max_world];
  bool pointers_ok=comm.lsaRank==int(a.rank)&&comm.lsaSize==int(a.world);
  for(unsigned r=0;r<a.world;++r) {
    peers[r]=static_cast<Window*>(ncclGetLsaPointer(window,0,r));
    pointers_ok&=peers[r]!=nullptr;
  }
  // A failed mapping is a group-level bootstrap failure; the independent host
  // supervisor tears down a lost rank. The mapping was validated by preflight.
  if(!pointers_ok)return;
  Window* local=peers[a.rank];
  Window* coordinator=peers[0];
  if(!config->shard_head||!config->extent_next||!config->free_cursor||config->packed_states>1||!config->virtual_buckets) {
    if(!tid)fail(local,WINDOW_PROTOCOL);
    grid.sync();return;
  }
  const uint32_t state_elements=config->packed_states?1:a.dictionary.elements;
  const uint64_t state_size=config->packed_states?8:config->state_bytes;
  uint64_t depth=0,sequence=1;
  unsigned choice=mgbfs_device_epoch::ERROR;
  if(!tid) {
    uint64_t current=0;
    for(uint64_t e=0;e<a.extent_count;++e) {
      const auto x=extents[e];
      if(!mgbfs_device_hash::valid_extent(x,0,state_elements,state_size)||
         x.counts[0]>x.segment.capacity||UINT64_MAX-current<x.counts[0]) {
        fail(local,WINDOW_PROTOCOL);break;
      }
      current+=x.counts[0];
    }
    if(current>1)fail(local,WINDOW_PROTOCOL);
    local->current=current;local->committed_count=current;
    local->sequence=sequence;a.layer_sizes[0]=1;
    if(a.capture_count)*a.capture_count=0;
    release(local->committed_epoch,1);
  }
  grid.sync();
  for(;;) {
    const uint64_t epoch=depth+1;
    const uint32_t read_bank=uint32_t(depth%3),write_bank=uint32_t((depth+1)%3);
    // Retire the oldest history bank only at the global completed-layer
    // boundary. A slot is checked against its exact bank/row before retirement.
    for(uint64_t e=0;e<a.extent_count;++e) {
      const auto x=extents[e];const auto seg=x.segment;
      if(!mgbfs_device_hash::valid_extent(x,write_bank,state_elements,state_size)) {
        if(!tid)fail(local,WINDOW_PROTOCOL);
      } else {
        const uint64_t count=x.counts[write_bank];
        if(count>seg.capacity) {if(!tid)fail(local,WINDOW_PROTOCOL);}
        else for(uint64_t row=tid;row<count;row+=grid.size()) {
          const uint64_t pos=uint64_t(write_bank)*seg.capacity+row;
          const uint64_t slot=seg.position_item_bytes==4?
            uint64_t(reinterpret_cast<const uint32_t*>(seg.positions)[pos]):seg.positions[pos];
          if(slot>=seg.slots_count){fail(local,WINDOW_PROTOCOL);continue;}
          auto cell=cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(seg.slots[slot]);
          uint64_t value=cell.load(cuda::memory_order_acquire);
          const uint64_t expected=(uint64_t(write_bank)<<32)|row;
          if(value==mgbfs_segmented_exact::empty || value==mgbfs_segmented_exact::retired ||
             (value&mgbfs_segmented_exact::pending) ||
             (value&mgbfs_segmented_exact::payload_mask)!=expected ||
             !cell.compare_exchange_strong(value,mgbfs_segmented_exact::retired,
                         cuda::memory_order_acq_rel,cuda::memory_order_acquire))
            fail(local,WINDOW_PROTOCOL);
        }
      }
    }
    grid.sync();
    if(!tid) {
      // Obtain the same completed frontier sizes on every rank before deriving
      // the common number of rounds. Empty ranks participate with zero records.
      uint64_t max_current=0;
      for(unsigned r=0;r<a.world;++r) {
        while(acquire(peers[r]->committed_epoch)<epoch) {
          if(propagate_failure(local,peers,a.world,a.terminal))break;
          __nanosleep(32);
        }
        if(acquire(local->status.failure))break;
        if(acquire(peers[r]->committed_epoch)!=epoch){fail(local,WINDOW_PROTOCOL);break;}
        if(peers[r]->committed_count>max_current)max_current=peers[r]->committed_count;
      }
      if(max_current>UINT64_MAX/a.generators)fail(local,WINDOW_PROTOCOL);
      const uint64_t edges=max_current*a.generators;
      local->rounds=edges?1+(edges-1)/args.send_capacity:0;
      local->shard_mask=0;*a.rank_count=0;
      uint64_t prefix=0;args.parent_prefix[0]=0;
      // Rebuild after bank retirement, before any shard CTA can probe. The
      // index contains exactly the extents with live rows for each owner.
      for(uint32_t shard=0;shard<a.shards;++shard) {
        config->shard_head[shard]=UINT64_MAX;
        config->free_cursor[shard]=0;
      }
      for(uint64_t e=0;e<a.extent_count;++e) {
        auto x=extents[e];
        if(!mgbfs_device_hash::valid_extent(x,read_bank,state_elements,state_size)||
           x.counts[read_bank]>x.segment.capacity||UINT64_MAX-prefix<x.counts[read_bank]) {
          fail(local,WINDOW_PROTOCOL);break;
        }
        x.counts[write_bank]=0;
        if(!x.counts[0]&&!x.counts[1]&&!x.counts[2]) {
          a.extent_owner[e]=UINT32_MAX;
          config->extent_next[e]=UINT64_MAX;
        } else {
          const uint32_t owner=a.extent_owner[e];
          if(owner>=a.shards){fail(local,WINDOW_PROTOCOL);break;}
          config->extent_next[e]=config->shard_head[owner];
          config->shard_head[owner]=e;
        }
        prefix+=x.counts[read_bank];args.parent_prefix[e+1]=prefix;
      }
      if(prefix!=local->current)fail(local,WINDOW_PROTOCOL);
    }
    grid.sync();
    const uint64_t rounds=local->rounds;
    for(uint64_t round=0;round<rounds;++round) {
      if(!tid && sequence>1)wait_ack(local,peers,a.world,sequence-1,a.terminal);
      grid.sync();
      if(acquire(local->status.failure))break;
      if(local->current>UINT64_MAX/a.generators) {
        if(!tid)fail(local,WINDOW_PROTOCOL);
        grid.sync();break;
      }
      const uint64_t edges=local->current*a.generators;
      const uint64_t begin=round*args.send_capacity;
      const uint64_t n=begin<edges?min(args.send_capacity,edges-begin):0;
      // Each thread decodes a parent once, then emits only the generators
      // belonging to this bounded edge window (including partial endpoints).
      const uint64_t first_parent=begin/a.generators;
      const uint64_t parent_end=n?1+(begin+n-1)/a.generators:first_parent;
      for(uint64_t parent_offset=tid;parent_offset<parent_end-first_parent;parent_offset+=grid.size()) {
        const uint64_t parent=first_parent+parent_offset,edge_base=parent*a.generators;
        const uint32_t first_generator=begin>edge_base?uint32_t(begin-edge_base):0;
        const uint64_t available=begin+n-edge_base;
        const uint32_t end_generator=uint32_t(min(uint64_t(a.generators),available));
        uint64_t extent=0,row=0;
        if(!locate_parent_extent(args.parent_prefix,a.extent_count,parent,&extent,&row)) {
          fail(local,WINDOW_PROTOCOL);continue;
        }
        const auto x=extents[extent];const auto seg=x.segment;
        const uint64_t local_row=uint64_t(read_bank)*seg.capacity+row,stride=seg.capacity*seg.banks;
        uint64_t packed=0;bool valid=true;
        if(config->packed_states) {
          packed=static_cast<const uint64_t*>(seg.states)[local_row];
          valid=mgbfs_pack4_valid_word(a.dictionary,packed);
        } else for(uint32_t k=0;k<a.dictionary.elements;++k) {
          const uint64_t offset=uint64_t(k)*stride+local_row;
          const int64_t value=config->state_bytes==1?
            int64_t(static_cast<const uint8_t*>(seg.states)[offset]):
            static_cast<const int64_t*>(seg.states)[offset];
          unsigned ordinal=0;
          while(ordinal<a.dictionary.distinct && a.dictionary.symbols[ordinal]!=value)++ordinal;
          if(ordinal==a.dictionary.distinct){valid=false;break;}
          packed|=uint64_t(ordinal)<<(4*k);
        }
        BitmapPack4Action action{a.dictionary.elements,a.generators,1,&packed,a.permutations,a.dictionary};
        if(!valid){fail(local,WINDOW_PROTOCOL);continue;}
        for(uint32_t generator=first_generator;generator<end_generator;++generator) {
          uint32_t bad=0;uint64_t child=0;
          if(!action.valid(generator,&bad)) {fail(local,WINDOW_PROTOCOL);break;}
          for(uint32_t k=0;k<a.dictionary.elements;++k) {
            const unsigned source=a.permutations[uint64_t(generator)*a.dictionary.elements+k];
            child|=((packed>>(4*source))&15ull)<<(4*k);
          }
          const HashWord word{a.dictionary,child};
          const uint64_t key=mgbfs_device_hash_key::hash(word,*config);
          const unsigned logical=mgbfs_device_hash_key::owner(key,a,*config);
          if(logical>=a.world){fail(local,WINDOW_PROTOCOL);break;}
          const unsigned destination=a.physical_of_logical[logical];
          if(destination>=a.world){fail(local,WINDOW_PROTOCOL);break;}
          args.raw[edge_base+generator-begin]={key,child,destination,0};
        }
      }
      grid.sync();
      if(!tid)propagate_failure(local,peers,a.world,a.terminal);
      grid.sync();
      if(acquire(local->status.failure))break;
      partition(args.raw,payload(local),n,args.send_capacity,a.world,args.positions,
                args.block_prefix,args.offsets,args.error);
      if(!tid && *args.error)fail(local,WINDOW_PROTOCOL);
      grid.sync();
      if(acquire(local->status.failure))break;
      __threadfence_system();
      grid.sync();
      if(!tid) {
        publish(local,a.job_generation,round,sequence,args.offsets,a.world);
        local->sent+=n;
      }
      grid.sync();
      if(!tid)wait_ready(local,peers,a.world,a.job_generation,round,sequence,args.send_capacity,a.terminal);
      grid.sync();
      if(acquire(local->status.failure))break;
      uint64_t incoming=0;
      for(unsigned src=0;src<a.world;++src)
        incoming+=peers[src]->offsets[a.rank+1]-peers[src]->offsets[a.rank];
      for(uint64_t first=0;first<incoming;first+=args.receive_capacity) {
        const uint64_t count=min(args.receive_capacity,incoming-first);
        for(uint64_t j=tid;j<count;j+=grid.size()) {
          uint64_t index=first+j;unsigned src=0;
          for(;src<a.world;++src) {
            const uint64_t amount=peers[src]->offsets[a.rank+1]-peers[src]->offsets[a.rank];
            if(index<amount)break;index-=amount;
          }
          if(src>=a.world){fail(local,WINDOW_PROTOCOL);continue;}
          const auto candidate=payload(peers[src])[peers[src]->offsets[a.rank]+index];
          if(!mgbfs_pack4_valid_word(a.dictionary,candidate.packed)) {
            fail(local,WINDOW_PROTOCOL);continue;
          }
          const HashWord word{a.dictionary,candidate.packed};
          const uint64_t actual=mgbfs_device_hash_key::hash(word,*config);
          const unsigned logical=mgbfs_device_hash_key::owner(actual,a,*config);
          if(actual!=candidate.key || logical>=a.world ||
             a.physical_of_logical[logical]!=a.rank) {
            fail(local,WINDOW_PROTOCOL);continue;
          }
          args.receive[j]=candidate;
        }
        grid.sync();
        if(!tid)propagate_failure(local,peers,a.world,a.terminal);
        grid.sync();
        if(acquire(local->status.failure))break;
        HashBatch batch{extents,a.extent_owner,a.extent_count,
          config->shard_head,config->extent_next,config->free_cursor,a.dictionary,
          a.rank_count,a.rank_limit,a.shards,write_bank,config->state_bytes,config->packed_states,config->virtual_buckets,config->owner_peak,
          args.grouped_origins,args.positions,args.shard_counts,args.shard_starts,reinterpret_cast<uint64_t*>(&local->shard_mask),
          args.error,reinterpret_cast<HashCapture*>(a.capture),a.capture_capacity,a.capture_count,depth+1};
        const uint64_t before=*a.rank_count;
        hash_accept(args.receive,count,batch);
        if(!tid) {
          if(*args.error)fail(local,(*args.error&HASH_FATAL)?WINDOW_PROTOCOL:WINDOW_RESOURCE);
          else {
            const uint64_t accepted=*a.rank_count-before;
            if(accepted>count)fail(local,WINDOW_PROTOCOL);
            else {
              local->received+=count;local->duplicates+=count-accepted;
              if(a.rank_limit&&*a.rank_count>=a.rank_limit)local->duplicates_full+=count-accepted;
            }
          }
        }
        grid.sync();
        if(acquire(local->status.failure))break;
      }
      __threadfence_system();
      grid.sync();
      if(acquire(local->status.failure))break;
      if(!tid)acknowledge(peers,a.world,a.rank,sequence);
      grid.sync();
      ++sequence;
    }
    if(!tid && !acquire(local->status.failure) && sequence>1)
      wait_ack(local,peers,a.world,sequence-1,a.terminal);
    grid.sync();
    if(!tid) {
      const unsigned active=__popcll(local->shard_mask);
      if(active>local->max_active_shards)local->max_active_shards=active;
      if(!acquire(local->status.failure)) {
        local->done_count=*a.rank_count;release(local->done_epoch,epoch);
      }
      if(a.rank==0) {
        for(;;) {
          auto failures_now=failures(peers,a.world,a.terminal);
          bool all=true;uint64_t sum=0;
          for(unsigned r=0;r<a.world;++r) {
            if(acquire(peers[r]->done_epoch)!=epoch)all=false;
            else if(UINT64_MAX-sum<peers[r]->done_count) {
              fail(local,WINDOW_PROTOCOL);failures_now|=WINDOW_PROTOCOL;
            } else sum+=peers[r]->done_count;
          }
          if(failures_now||all) {
            coordinator->global_next=failures_now?0:sum;
            coordinator->stop_reason=failures_now?reason_capacity:reason_none;
            auto requested=failures_now?
              ((failures_now&~ControlWord(WINDOW_RESOURCE))?mgbfs_device_epoch::ERROR:mgbfs_device_epoch::INCOMPLETE):
              (sum?mgbfs_device_epoch::ADVANCE:mgbfs_device_epoch::COMPLETE);
            if(!failures_now && sum && depth+1>=a.layer_capacity) {
              requested=mgbfs_device_epoch::INCOMPLETE;coordinator->stop_reason=reason_log_capacity;
            }
            coordinator->decision.request=requested;
            release(coordinator->decision.request_epoch,epoch);
            break;
          }
          __nanosleep(32);
        }
      }
      while(acquire(coordinator->decision.request_epoch)<epoch) {
        if(a.terminal&&cuda::atomic_ref<uint32_t,cuda::thread_scope_system>(*a.terminal)
             .load(cuda::memory_order_acquire))fail(local,WINDOW_CANCELLED);
        __nanosleep(32);
      }
      mgbfs_device_epoch::publish_stopped(local->status,epoch);
      if(a.rank==0) {
        mgbfs_device_epoch::RankStatus* status[mgbfs_device_batch::max_world];
        for(unsigned r=0;r<a.world;++r)status[r]=&peers[r]->status;
        while(!mgbfs_device_epoch::try_finalize(coordinator->decision,status,a.world,epoch))
          __nanosleep(32);
      }
      while(acquire(coordinator->decision.final_epoch)<epoch)__nanosleep(32);
    }
    grid.sync();
    choice=unsigned(coordinator->decision.final);
    if(choice!=mgbfs_device_epoch::ADVANCE)break;
    if(!tid) {
      local->current=*a.rank_count;local->committed_count=local->current;
      a.layer_sizes[depth+1]=coordinator->global_next;
      release(local->committed_epoch,epoch+1);
    }
    grid.sync();
    ++depth;
  }
  if(!tid) {
    report->depth=depth;report->sent=local->sent;report->received=local->received;
    report->duplicates=local->duplicates;report->duplicates_full=local->duplicates_full;
    report->blocked=0;report->capture_count=a.capture_count?*a.capture_count:0;
    report->global_next=coordinator->global_next;report->status=choice;report->rank=a.rank;
    report->max_active_shards_first64=uint32_t(local->max_active_shards);
    report->reason=choice==mgbfs_device_epoch::ERROR?reason_protocol:
      choice==mgbfs_device_epoch::INCOMPLETE?unsigned(coordinator->stop_reason):reason_none;
    for(unsigned r=0;r<a.world;++r)a.rank_counts[r]=peers[r]->committed_count;
  }
}
#endif
}
