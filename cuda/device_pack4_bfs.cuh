#pragma once

// Internal device-resident exact BFS entry. The wrapper is compiled only with
// NCCL device API and packed bitmap support. Public selection is explicit and experimental.
// All pointers in Args are local-device allocations owned for the whole job.
// The symmetric window and all its peer readers must remain alive until every
// rank has stopped and the external supervisor has confirmed retirement.

#include "device_epoch_decision.cuh"
#include "device_mailbox.cuh"
#include "device_pack4_accept.cuh"
#include "device_segmented_hash_accept.cuh"
#include "bitmap_pack4_action.cuh"
#include <nccl_device.h>
#include <cuda_runtime.h>

namespace mgbfs_resident {
using Word=unsigned long long;
using SystemAtomic=cuda::atomic_ref<Word,cuda::thread_scope_system>;
using BlockAtomic=cuda::atomic_ref<Word,cuda::thread_scope_block>;
constexpr unsigned lanes=4; // legacy/default geometry
constexpr unsigned max_lanes=8;
constexpr unsigned max_world=8;
constexpr unsigned threads=32*(lanes+2);
constexpr unsigned max_packet_records=8;
constexpr uint32_t payload_format=1;

struct Payload { uint64_t packed; uint32_t source,reserved; };
struct Capture { uint64_t depth,packed; };
// Separate opt-in ABI: in hash jobs Args.visited points to this immutable
// device config and Args.extents points to mgbfs_device_hash::Extent records.
// Existing bitmap Args and entry points keep their exact original layout.
struct HashConfig {
  uint64_t seed;
  uint32_t bits,state_bytes;
  const uint64_t* owner_cuts;
};
struct HashWord {
  MgbfsPack4Dictionary dictionary;uint64_t packed;
  __device__ int64_t value(uint32_t element) const {
    return dictionary.symbols[(packed>>(4*element))&15u];
  }
};
struct alignas(64) Control {
  mgbfs_device_epoch::RankStatus status;
  mgbfs_device_epoch::Decision decision;
  Word done_count,done_epoch,committed_count,global_next,stop_reason;
};
static_assert(sizeof(Control)%64==0,"resident control alignment");
static_assert(sizeof(Payload)==16,"resident packet ABI");

// A single segment provides three banks of packed rows. All extents belong to
// a shared pool; an owner tag of UINT32_MAX means the entire segment is free.
// For any other tag, all three row ranges and all count words are exclusively
// leased to that shard. Host admission must prove there is no overlap between
// any row or count range and retain all allocations until the job is retired.
struct Args {
  uint32_t abi_version,struct_bytes,world,rank;
  uint32_t owner,shards,generators,banks;
  uint32_t owner_page,reserved0;
  uint64_t owner_boundary,owner_len,job_generation,rank_limit;
  BitmapDeviceDomain domain;
  MgbfsPack4Dictionary dictionary;
  MgbfsDeviceMailboxLayoutV1 mailbox;
  const uint32_t* permutations;
  const uint32_t* physical_of_logical;
  mgbfs_device_pack4::Extent* extents;
  uint32_t* extent_owner;
  uint64_t extent_count;
  uint32_t* shard_locks;
  uint64_t* visited;
  uint64_t* rank_count;
  uint64_t* layer_sizes;
  uint64_t layer_capacity;
  uint64_t* rank_counts;
  Capture* capture;
  uint64_t capture_capacity;
  uint64_t* capture_count;
  uint32_t* terminal;
};
struct Report {
  uint64_t depth,sent,received,duplicates,duplicates_full,blocked;
  uint64_t capture_count,global_next;
  uint32_t status,rank,reason,max_active_shards_first64;
};
enum : uint32_t { reason_none=0,reason_capacity=1,reason_log_capacity=2,
                  reason_cancelled=3,reason_protocol=4,reason_capture=5 };

#ifdef MGBFS_RESIDENT_IMPLEMENTATION
__device__ __forceinline__ Word load(Word& x) {
  return SystemAtomic(x).load(cuda::memory_order_acquire);
}
__device__ __forceinline__ void store(Word& x,Word v) {
  SystemAtomic(x).store(v,cuda::memory_order_release);
}
__device__ __forceinline__ Word add(Word& x,Word v=1) {
  return BlockAtomic(x).fetch_add(v,cuda::memory_order_acq_rel);
}
__device__ __forceinline__ Word local_load(Word& x) {
  return BlockAtomic(x).load(cuda::memory_order_acquire);
}
__device__ __forceinline__ void failure(Control& x,Word bit) {
  SystemAtomic(x.status.failure).fetch_or(bit,cuda::memory_order_acq_rel);
}
__device__ __forceinline__ bool terminal(const uint32_t* p) {
  if(!p)return false;
  auto& v=*const_cast<uint32_t*>(p);
  return cuda::atomic_ref<uint32_t,cuda::thread_scope_system>(v)
      .load(cuda::memory_order_acquire)!=0;
}
__device__ __forceinline__ unsigned char* mailbox_base(Control* c) {
  return reinterpret_cast<unsigned char*>(c)+sizeof(Control);
}
__device__ __forceinline__ mgbfs_packet_ring::View view(
    Control* c,const Args& a,unsigned lane,unsigned destination) {
  return mgbfs_device_mailbox::view(mailbox_base(c),a.mailbox,destination,lane);
}
__device__ __forceinline__ mgbfs_packet_ring::CloseSlot& close(
    Control* c,const Args& a,unsigned lane,unsigned destination) {
  return mgbfs_device_mailbox::close(mailbox_base(c),a.mailbox,destination,lane);
}

__global__ void initialize(Control* local,Args a) {
  if(blockIdx.x||threadIdx.x)return;
  local->status={0,0};local->decision={0,0,0,0};
  local->done_count=local->done_epoch=local->committed_count=local->global_next=local->stop_reason=0;
  for(unsigned lane=0;lane<a.mailbox.lanes;++lane)for(unsigned dst=0;dst<a.world;++dst) {
    auto ring=view(local,a,lane,dst);
    for(unsigned slot=0;slot<a.mailbox.slots_per_ring;++slot)
      mgbfs_packet_ring::init_slot(ring.slots[slot],slot);
    mgbfs_packet_ring::init_close(close(local,a,lane,dst));
  }
}


// Startup-only data+credit validation through the actual registered LSA
// window. Each rank sends twice over the single-slot ring to every rank and
// checks the payload from every source. The caller initializes this layout,
// synchronizes all ranks, and must synchronize all ranks again before reuse.
// Terminal cancellation is shared with the resident job supervisor.
__global__ void peer_smoke(ncclWindow_t window,ncclDevComm comm,Args a,uint32_t* out) {
  if(blockIdx.x)return;
  const unsigned tid=threadIdx.x;
  __shared__ Control* peers[max_world];
  __shared__ unsigned failed;
  if(tid==64) {
    failed=(comm.lsaRank!=int(a.rank)||comm.lsaSize!=int(a.world)||
            a.mailbox.slots_per_ring!=1);
    for(unsigned r=0;r<a.world;++r) {
      peers[r]=static_cast<Control*>(ncclGetLsaPointer(window,0,r));
      if(!peers[r])failed=1;
    }
  }
  __syncthreads();
  constexpr Word phase=UINT64_MAX-1;
  constexpr uint32_t format=0x736d6f6bu;
  if(!failed&&tid==0) {
    for(unsigned dst=0;dst<a.world&&!atomicAdd(&failed,0u);++dst) {
      auto ring=view(peers[a.rank],a,0,dst);
      for(Word ticket=0;ticket<2&&!atomicAdd(&failed,0u);++ticket) {
        Payload p{(Word(a.rank)<<40)|(Word(dst)<<32)|ticket,a.rank,dst};
        for(;;) {
          if(terminal(a.terminal)){atomicExch(&failed,1u);break;}
          auto r=mgbfs_packet_ring::try_send(ring,ticket,a.job_generation,
                     phase,format,reinterpret_cast<const unsigned char*>(&p),sizeof(p));
          if(r==mgbfs_packet_ring::Result::ok)break;
          if(r!=mgbfs_packet_ring::Result::not_ready){atomicExch(&failed,1u);break;}
          __nanosleep(32);
        }
      }
    }
    // The second credit confirms the receiver has completed both reads.
    for(unsigned dst=0;dst<a.world&&!atomicAdd(&failed,0u);++dst) {
      auto ring=view(peers[a.rank],a,0,dst);
      while(SystemAtomic(ring.slots[0].sequence).load(cuda::memory_order_acquire)!=4) {
        if(terminal(a.terminal)){atomicExch(&failed,1u);break;}
        __nanosleep(32);
      }
    }
  } else if(!failed&&tid==32) {
    Word ticket[max_world]={};
    unsigned remaining=a.world*2,cursor=0;
    while(remaining&&!atomicAdd(&failed,0u)) {
      if(terminal(a.terminal)){atomicExch(&failed,1u);break;}
      const unsigned src=cursor++%a.world;
      if(ticket[src]==2)continue;
      auto ring=view(peers[src],a,0,a.rank);
      mgbfs_packet_ring::PacketHeader header{};
      const unsigned char* bytes=nullptr;
      auto r=mgbfs_packet_ring::try_peek(ring,ticket[src],a.job_generation,phase,&header,&bytes);
      if(r==mgbfs_packet_ring::Result::not_ready)continue;
      if(r!=mgbfs_packet_ring::Result::ok||header.len!=sizeof(Payload)||header.format!=format) {
        atomicExch(&failed,1u);break;
      }
      Payload p{};auto* target=reinterpret_cast<unsigned char*>(&p);
      for(unsigned i=0;i<sizeof(p);++i)target[i]=bytes[i];
      if(p.source!=src||p.reserved!=a.rank||
         p.packed!=((Word(src)<<40)|(Word(a.rank)<<32)|ticket[src])) {
        atomicExch(&failed,1u);break;
      }
      if(mgbfs_packet_ring::try_ack(ring,ticket[src],a.job_generation,phase)
         !=mgbfs_packet_ring::Result::ok){atomicExch(&failed,1u);break;}
      ++ticket[src];--remaining;
    }
  }
  __syncthreads();
  if(tid==64)*out=failed;
}

// The producer keeps a bounded packet alive until its ring accepts it.
__device__ __forceinline__ bool send_packet(
    Control& local,Control& coordinator,const Args& a,unsigned lane,unsigned destination,
    Word epoch,const Payload* payload,unsigned count,Word& ticket,Word& sent,Word& blocked) {
  for(;;) {
    if(terminal(a.terminal)){failure(local,mgbfs_device_epoch::FATAL);return false;}
    if(load(coordinator.decision.request_epoch)==epoch)return false;
    auto result=mgbfs_packet_ring::try_send(view(&local,a,lane,destination),
        ticket,a.job_generation,epoch,payload_format,
        reinterpret_cast<const unsigned char*>(payload),count*sizeof(Payload));
    if(result==mgbfs_packet_ring::Result::ok){++ticket;add(sent);return true;}
    if(result!=mgbfs_packet_ring::Result::not_ready){failure(local,mgbfs_device_epoch::FATAL);return false;}
    add(blocked);__nanosleep(32);
  }
}

__device__ inline uint64_t hash_word(const HashWord& word,const HashConfig& config) {
  uint64_t hash=config.seed;
  for(uint32_t e=0;e<word.dictionary.elements;++e)hash=mix64(hash^uint64_t(word.value(e))^uint64_t(e));
  return finish_hash(hash,config.bits);
}
__device__ inline unsigned hash_owner(uint64_t hash,const Args& a,const HashConfig& config) {
  const uint64_t high=hash>>32;
  if(!config.owner_cuts)return unsigned((high*a.world)>>32);
  if(config.owner_cuts[0]!=0||config.owner_cuts[a.world]!=(uint64_t(1)<<32))return a.world;
  uint32_t low=0,upper=a.world;
  while(low+1<upper){const uint32_t mid=(low+upper)/2;if(high<config.owner_cuts[mid])upper=mid;else low=mid;}
  return (config.owner_cuts[low]<=high&&high<config.owner_cuts[low+1])?low:a.world;
}
// All producer/consumer roles share one resident CTA. A blocked producer
// cannot prevent the separate consumer warps from receiving and returning
// credits. This first implementation deliberately favors a simple proof over
// throughput; more CTAs require a separate residency/progress proof.
template<unsigned lane_capacity,bool Hash=false>
__global__ void run(ncclWindow_t window,ncclDevComm comm,Args a,Report* report) {
  if(blockIdx.x)return;
  const unsigned tid=threadIdx.x, active_lanes=a.mailbox.lanes;
  const HashConfig* hash_config=nullptr;
  const mgbfs_device_hash::Extent* hash_extents=nullptr;
  if constexpr(Hash) {
    hash_config=reinterpret_cast<const HashConfig*>(a.visited);
    hash_extents=reinterpret_cast<const mgbfs_device_hash::Extent*>(a.extents);
  }
  __shared__ Control* rank[max_world];
  __shared__ mgbfs_device_epoch::RankStatus* ranks[max_world];
  __shared__ Word current,next,producers_done,consumers_done;
  __shared__ Word sent,received,duplicates,duplicates_full,blocked,shards_this_layer;
  __shared__ uint32_t max_active_shards;
  __shared__ uint64_t depth;
  __shared__ unsigned choice,bootstrap_failed,retirement_bad;
  if(tid==32*(active_lanes+1)) {
    bool invalid=comm.lsaRank!=int(a.rank)||comm.lsaSize!=int(a.world);
    for(unsigned r=0;r<a.world;++r) {
      rank[r]=static_cast<Control*>(ncclGetLsaPointer(window,0,r));
      ranks[r]=rank[r]?&rank[r]->status:nullptr;
      invalid|=!rank[r];
    }
    current=next=producers_done=consumers_done=0;
    sent=received=duplicates=duplicates_full=blocked=0;
    shards_this_layer=0;max_active_shards=0;
    depth=choice=bootstrap_failed=0;
    for(uint64_t i=0;i<a.extent_count;++i) {
      if constexpr(Hash) {
        const auto& e=hash_extents[i];
        if(!hash_config||!(hash_config->state_bytes==1||hash_config->state_bytes==8)||
           hash_config->bits>64||!mgbfs_device_hash::valid_extent(e,0,a.dictionary.elements,hash_config->state_bytes)||
           e.segment.banks!=3||e.counts[0]>e.segment.capacity||
           current>UINT64_MAX-e.counts[0]){invalid=true;break;}
        current+=e.counts[0];
      } else {
        const auto& e=a.extents[i];
        if(!e.counts||!e.rows||e.banks!=3||!e.capacity||
           e.capacity>(uint64_t(1)<<32)||e.counts[0]>e.capacity||
           current>UINT64_MAX-e.counts[0]) {invalid=true;break;}
        current+=e.counts[0];
      }
    }
    if(current>1)invalid=true;
    if(!invalid) {
      store(rank[a.rank]->committed_count,current);
      a.layer_sizes[0]=1;
      if(a.capture_count)*a.capture_count=0;
    } else {
      bootstrap_failed=1;
      report->status=mgbfs_device_epoch::ERROR;
      report->reason=reason_protocol;
      report->rank=a.rank;
    }
  }
  __syncthreads();
  // This is unreachable after successful group-wide capability preflight.
  bool invalid=comm.lsaRank!=int(a.rank)||comm.lsaSize!=int(a.world);
  for(unsigned r=0;r<a.world;++r)invalid|=!rank[r];
  if(invalid||bootstrap_failed)return;
  Control& local=*rank[a.rank];
  Control& coordinator=*rank[0];
  Word send_ticket[max_world]={};
  Word receive_tickets[max_world*lane_capacity]={};
  for(;;) {
    const Word epoch=depth+1;
    const unsigned read_bank=unsigned(depth%3);
    const unsigned write_bank=unsigned((depth+1)%3);
    if constexpr(Hash) {
      if(tid==32*(active_lanes+1))retirement_bad=0;
      __syncthreads();
      // All prior-layer owners drained before ADVANCE. Retire the oldest
      // bank in parallel, while no peer or local owner can read this table.
      for(uint64_t i=0;i<a.extent_count;++i) {
        const auto& e=hash_extents[i];const auto& seg=e.segment;
        if(!mgbfs_device_hash::valid_extent(e,write_bank,a.dictionary.elements,hash_config->state_bytes)) {
          atomicExch(&retirement_bad,1u);failure(local,mgbfs_device_epoch::FATAL);continue;
        }
        const uint64_t count=e.counts[write_bank];
        if(count>seg.capacity){atomicExch(&retirement_bad,1u);failure(local,mgbfs_device_epoch::FATAL);continue;}
        for(uint64_t row=tid;row<count;row+=blockDim.x) {
          const uint64_t pos=uint64_t(write_bank)*seg.capacity+row;
          const uint64_t slot=seg.position_item_bytes==4?uint64_t(reinterpret_cast<const uint32_t*>(seg.positions)[pos]):seg.positions[pos];
          if(slot>=seg.slots_count){atomicExch(&retirement_bad,1u);failure(local,mgbfs_device_epoch::FATAL);continue;}
          auto cell=cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(seg.slots[slot]);
          uint64_t seen=cell.load(cuda::memory_order_acquire);
          const uint64_t payload=(uint64_t(write_bank)<<32)|row;
          if(seen==mgbfs_segmented_exact::empty||seen==mgbfs_segmented_exact::retired||
             (seen&mgbfs_segmented_exact::pending)||(seen&mgbfs_segmented_exact::payload_mask)!=payload||
             !cell.compare_exchange_strong(seen,mgbfs_segmented_exact::retired,
                                           cuda::memory_order_acq_rel,cuda::memory_order_acquire))
            {atomicExch(&retirement_bad,1u);failure(local,mgbfs_device_epoch::FATAL);}
        }
      }
      __syncthreads();
    }
    if(tid==32*(active_lanes+1)) {
      *a.rank_count=0;shards_this_layer=0;
      for(unsigned s=0;s<a.shards;++s)a.shard_locks[s]=0;
      for(uint64_t i=0;i<a.extent_count;++i) {
        if constexpr(Hash) {
          const auto& e=hash_extents[i];
          if(retirement_bad)continue;
          e.counts[write_bank]=0;
          if(!e.counts[0]&&!e.counts[1]&&!e.counts[2])a.extent_owner[i]=UINT32_MAX;
        } else {
          auto& e=a.extents[i];
          e.counts[write_bank]=0;
          if(!e.counts[0]&&!e.counts[1]&&!e.counts[2])a.extent_owner[i]=UINT32_MAX;
        }
      }
    }
    __syncthreads();
    if(tid<active_lanes) {
      bool stopped=Hash&&retirement_bad;
      Payload pending[max_world][max_packet_records];
      unsigned pending_count[max_world]={};
      const unsigned packet_records=a.mailbox.requested_payload_bytes/sizeof(Payload);
      for(Word parent=tid;parent<current&&!stopped;parent+=active_lanes) {
        uint64_t remaining=parent,packed=0;bool found=false;
        for(uint64_t i=0;i<a.extent_count;++i) {
          if constexpr(Hash) {
            const auto& e=hash_extents[i];const uint64_t n=e.counts[read_bank];
            if(remaining<n) {
              const auto& seg=e.segment;
              const uint64_t local=uint64_t(read_bank)*seg.capacity+remaining,stride=seg.capacity*seg.banks;
              bool ok=true;
              for(unsigned k=0;k<a.dictionary.elements;++k) {
                const uint64_t offset=uint64_t(k)*stride+local;
                const int64_t value=hash_config->state_bytes==1?
                  int64_t(static_cast<const uint8_t*>(seg.states)[offset]):
                  static_cast<const int64_t*>(seg.states)[offset];
                unsigned ordinal=0;while(ordinal<a.dictionary.distinct&&a.dictionary.symbols[ordinal]!=value)++ordinal;
                if(ordinal==a.dictionary.distinct){ok=false;break;}
                packed|=uint64_t(ordinal)<<(4*k);
              }
              found=ok;break;
            }
            remaining-=n;
          } else {
            const auto& e=a.extents[i];const uint64_t n=e.counts[read_bank];
            if(remaining<n) {packed=e.rows[uint64_t(read_bank)*e.capacity+remaining];found=true;break;}
            remaining-=n;
          }
        }
        if(!found){failure(local,mgbfs_device_epoch::FATAL);break;}
        for(unsigned edge=0;edge<a.generators&&!stopped;++edge) {
          BitmapPack4Action action{a.dictionary.elements,a.generators,1,&packed,a.permutations,a.dictionary};
          uint32_t bad=0;
          if(!action.valid(edge,&bad)){failure(local,mgbfs_device_epoch::FATAL);break;}
          uint64_t child=0;
          for(unsigned e=0;e<a.dictionary.elements;++e) {
            const unsigned source=a.permutations[uint64_t(edge)*a.dictionary.elements+e];
            child|=((packed>>(4*source))&15ull)<<(4*e);
          }
          unsigned logical=a.world;
          if constexpr(Hash) {
            const HashWord word{a.dictionary,child};
            logical=hash_owner(hash_word(word,*hash_config),a,*hash_config);
          } else {
            uint64_t id=0;
            if(!bitmap_rank(a.domain,action,edge,&id)||id>=a.domain.total) {
              failure(local,mgbfs_device_epoch::FATAL);break;
            }
            id=mgbfs_bitmap_permute_owner_id(id,a.world,a.owner_page,a.owner_boundary);
            logical=unsigned(id%a.world);
          }
          if(logical>=a.world){failure(local,mgbfs_device_epoch::FATAL);break;}
          const unsigned destination=a.physical_of_logical[logical];
          if(destination>=a.world){failure(local,mgbfs_device_epoch::FATAL);break;}
          pending[destination][pending_count[destination]++]={child,a.rank,0};
          if(pending_count[destination]==packet_records) {
            if(!send_packet(local,coordinator,a,tid,destination,epoch,pending[destination],
                            pending_count[destination],send_ticket[destination],sent,blocked))
              stopped=true;
            else pending_count[destination]=0;
          }
        }
      }
      // Flush each partial destination packet before publishing its close.
      if(!stopped)for(unsigned dst=0;dst<a.world&&!stopped;++dst) {
        if(pending_count[dst]&&!send_packet(local,coordinator,a,tid,dst,epoch,
              pending[dst],pending_count[dst],send_ticket[dst],sent,blocked)) {
          stopped=true;break;
        }
        for(;;) {
          if(terminal(a.terminal)){failure(local,mgbfs_device_epoch::FATAL);stopped=true;break;}
          if(load(coordinator.decision.request_epoch)==epoch){stopped=true;break;}
          const auto result=mgbfs_packet_ring::try_close(close(&local,a,tid,dst),
              depth,a.job_generation,epoch,send_ticket[dst]);
          if(result==mgbfs_packet_ring::Result::ok)break;
          if(result!=mgbfs_packet_ring::Result::not_ready){failure(local,mgbfs_device_epoch::FATAL);stopped=true;break;}
          __nanosleep(32);
        }
      }
      add(producers_done);
    } else if(tid>=32 && tid<32*(active_lanes+1) && tid%32==0) {
      // One receiver warp per producer lane preserves single-consumer rings.
      // A busy shard is retried without waiting, allowing the other warps to
      // make progress on different shards and return their producer credits.
      const unsigned receiver=tid/32-1;
      bool closed[max_world*lane_capacity]={};
      unsigned packet_offset[max_world*lane_capacity]={};
      unsigned active=a.world,cursor=0;
      while(active) {
        if constexpr(Hash) {if(retirement_bad)break;}
        if(terminal(a.terminal)){failure(local,mgbfs_device_epoch::FATAL);break;}
        if(load(coordinator.decision.request_epoch)==epoch)break;
        const unsigned channel=(cursor++%a.world)*active_lanes+receiver;
        if(closed[channel])continue;
        const unsigned src=channel/active_lanes,lane=channel%active_lanes;
        Word& ticket=receive_tickets[channel];
        mgbfs_packet_ring::PacketHeader header{};const unsigned char* bytes=nullptr;
        auto result=mgbfs_packet_ring::try_peek(view(rank[src],a,lane,a.rank),
            ticket,a.job_generation,epoch,&header,&bytes);
        if(result==mgbfs_packet_ring::Result::ok) {
          if(!header.len||header.len%sizeof(Payload)||
             header.len>a.mailbox.requested_payload_bytes||header.format!=payload_format||
             packet_offset[channel]>=header.len/sizeof(Payload)) {
            failure(local,mgbfs_device_epoch::FATAL);break;
          }
          Payload payload{};auto* dst=reinterpret_cast<unsigned char*>(&payload);
          const unsigned byte_offset=packet_offset[channel]*sizeof(Payload);
          for(unsigned i=0;i<sizeof(payload);++i)dst[i]=bytes[byte_offset+i];
          if(payload.source!=src||payload.reserved||!mgbfs_pack4_valid_word(a.dictionary,payload.packed)) {
            failure(local,mgbfs_device_epoch::FATAL);break;
          }
          uint64_t id=0;unsigned shard=0;
          if constexpr(Hash) {
            const HashWord word{a.dictionary,payload.packed};
            id=hash_word(word,*hash_config);
            const unsigned logical=hash_owner(id,a,*hash_config);
            if(logical>=a.world||a.physical_of_logical[logical]!=a.rank){failure(local,mgbfs_device_epoch::FATAL);break;}
            shard=unsigned((uint64_t(uint32_t(id))*a.shards)>>32);
          } else {
            mgbfs_device_pack4::WordAccessor word{a.dictionary,payload.packed};
            if(!bitmap_rank(a.domain,word,0,&id)||id>=a.domain.total) {
              failure(local,mgbfs_device_epoch::FATAL);break;
            }
            id=mgbfs_bitmap_permute_owner_id(id,a.world,a.owner_page,a.owner_boundary);
            if(id%a.world!=a.owner||a.physical_of_logical[a.owner]!=a.rank) {
              failure(local,mgbfs_device_epoch::FATAL);break;
            }
            shard=unsigned((uint64_t(uint32_t(mix64(id)))*a.shards)>>32);
          }
          // The first hash implementation serializes shard transactions.
          // This makes a pool-full decision stable; competing producers and
          // receivers still make progress because the lease is nonblocking.
          cuda::atomic_ref<uint32_t,cuda::thread_scope_device> lease(a.shard_locks[Hash?0:shard]);
          uint32_t free=0;
          if(!lease.compare_exchange_strong(free,1,cuda::memory_order_acquire,
                                            cuda::memory_order_relaxed))continue;
          bool accepted_duplicate=false,accepted_new=false,accepted_resource=false;
          if constexpr(Hash) {
            mgbfs_device_hash::Owner owner{};
            owner.extents=hash_extents;owner.extent_count=a.extent_count;owner.extent_owner=a.extent_owner;
            owner.rank_count=a.rank_limit?a.rank_count:nullptr;owner.rank_limit=a.rank_limit;
            owner.elements=a.dictionary.elements;owner.shards=a.shards;owner.shard=shard;owner.bank=write_bank;
            const HashWord word{a.dictionary,payload.packed};
            const auto result=hash_config->state_bytes==1?
              mgbfs_device_hash::accept<uint8_t>(owner,id,word):mgbfs_device_hash::accept<int64_t>(owner,id,word);
            accepted_duplicate=result.status==mgbfs_device_hash::Status::duplicate;
            accepted_new=result.status==mgbfs_device_hash::Status::accepted;
            // A table-full response despite a free row violates the admitted
            // slot geometry; it is a protocol error, not a memory boundary.
            accepted_resource=result.status==mgbfs_device_hash::Status::extent_full||
              result.status==mgbfs_device_hash::Status::cap_full;
          } else {
            mgbfs_device_pack4::Owner owner{};
            owner.domain=a.domain;owner.dictionary=a.dictionary;
            owner.extents=a.extents;owner.extent_count=a.extent_count;
            owner.extent_owner=a.extent_owner;owner.bits=a.visited;
            owner.owner_len=a.owner_len;owner.rank_count=a.rank_limit?a.rank_count:nullptr;
            owner.rank_limit=a.rank_limit;owner.owner_boundary=a.owner_boundary;
            owner.world=a.world;owner.owner=a.owner;owner.shards=a.shards;
            owner.shard=shard;owner.bank=write_bank;owner.owner_page=a.owner_page;
            const auto result=mgbfs_device_pack4::accept_prevalidated(owner,id,payload.packed);
            accepted_duplicate=result.status==mgbfs_device_pack4::Status::duplicate;
            accepted_new=result.status==mgbfs_device_pack4::Status::accepted;
            accepted_resource=result.status==mgbfs_device_pack4::Status::extent_full||
              result.status==mgbfs_device_pack4::Status::cap_full;
          }
          lease.store(0,cuda::memory_order_release);
          bool failed=false;
          if(accepted_duplicate) {
            add(duplicates);
            if(a.rank_limit&&cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(*a.rank_count)
                 .load(cuda::memory_order_acquire)>=a.rank_limit)add(duplicates_full);
          } else if(accepted_new) {
            if(shard<64)atomicOr(&shards_this_layer,Word(1)<<shard);
            add(next);
            if(a.capture) {
              const uint64_t index=atomicAdd(reinterpret_cast<unsigned long long*>(a.capture_count),1ull);
              if(index>=a.capture_capacity){failure(local,mgbfs_device_epoch::FATAL);failed=true;}
              else a.capture[index]={depth+1,payload.packed};
            }
          } else if(accepted_resource) {
            failure(local,mgbfs_device_epoch::RESOURCE);failed=true;
          } else {failure(local,mgbfs_device_epoch::FATAL);failed=true;}
          if(failed)break;
          // Credit follows durable publication of every record in this packet.
          if(++packet_offset[channel]==header.len/sizeof(Payload)) {
            if(mgbfs_packet_ring::try_ack(view(rank[src],a,lane,a.rank),
                ticket,a.job_generation,epoch)!=mgbfs_packet_ring::Result::ok) {
              failure(local,mgbfs_device_epoch::FATAL);break;
            }
            packet_offset[channel]=0;
            ++ticket;add(received);
          }
        } else if(result!=mgbfs_packet_ring::Result::not_ready&&
                  result!=mgbfs_packet_ring::Result::future_phase) {
          failure(local,mgbfs_device_epoch::FATAL);break;
        } else {
          if(packet_offset[channel]){failure(local,mgbfs_device_epoch::FATAL);break;}
          Word tail=0;
          auto c=mgbfs_packet_ring::try_get_close(close(rank[src],a,lane,a.rank),
              depth,a.job_generation,epoch,&tail);
          if(c==mgbfs_packet_ring::Result::ok) {
            if(ticket>tail){failure(local,mgbfs_device_epoch::FATAL);break;}
            if(ticket==tail) {
              if(mgbfs_packet_ring::try_ack_close(close(rank[src],a,lane,a.rank),
                 depth,a.job_generation,epoch,ticket)!=mgbfs_packet_ring::Result::ok) {
                failure(local,mgbfs_device_epoch::FATAL);break;
              }
              closed[channel]=true;--active;
            }
          } else if(c!=mgbfs_packet_ring::Result::not_ready&&
                    c!=mgbfs_packet_ring::Result::future_phase) {
            failure(local,mgbfs_device_epoch::FATAL);break;
          }
        }
      }
      add(consumers_done);
    } else if(tid==32*(active_lanes+1)) {
      bool published=false;
      for(;;) {
        if(terminal(a.terminal))failure(local,mgbfs_device_epoch::FATAL);
        if(a.rank==0&&load(coordinator.decision.request_epoch)!=epoch) {
          Word failures=0;for(unsigned r=0;r<a.world;++r)failures|=load(rank[r]->status.failure);
          if(failures) {
            coordinator.global_next=0; // partial next is never a completed layer
            coordinator.stop_reason=(failures&~Word(mgbfs_device_epoch::RESOURCE))?
                reason_protocol:reason_capacity;
            coordinator.decision.request=(failures&~Word(mgbfs_device_epoch::RESOURCE))?
                mgbfs_device_epoch::ERROR:mgbfs_device_epoch::INCOMPLETE;
            store(coordinator.decision.request_epoch,epoch);
          }
        }
        if(!published&&!load(local.status.failure)&&local_load(producers_done)==active_lanes&&
           local_load(consumers_done)==active_lanes) {
          local.done_count=local_load(next);store(local.done_epoch,epoch);published=true;
        }
        if(a.rank==0&&load(coordinator.decision.request_epoch)!=epoch) {
          bool all=true;Word sum=0,failures=0;
          for(unsigned r=0;r<a.world;++r) {
            failures|=load(rank[r]->status.failure);
            if(load(rank[r]->done_epoch)!=epoch)all=false;
            else if(sum>UINT64_MAX-rank[r]->done_count){failures|=mgbfs_device_epoch::FATAL;failure(local,mgbfs_device_epoch::FATAL);}
            else sum+=rank[r]->done_count;
          }
          if(all&&sum&&depth+1>=a.layer_capacity) {
            failure(local,mgbfs_device_epoch::RESOURCE);failures|=mgbfs_device_epoch::RESOURCE;
            coordinator.stop_reason=reason_log_capacity;
          }
          if(failures||all) {
            coordinator.global_next=failures?0:sum;
            if(failures && coordinator.stop_reason!=reason_log_capacity)
              coordinator.stop_reason=(failures&~Word(mgbfs_device_epoch::RESOURCE))?reason_protocol:reason_capacity;
            coordinator.decision.request=failures?((failures&~Word(mgbfs_device_epoch::RESOURCE))?
                mgbfs_device_epoch::ERROR:mgbfs_device_epoch::INCOMPLETE):
                (sum?mgbfs_device_epoch::ADVANCE:mgbfs_device_epoch::COMPLETE);
            store(coordinator.decision.request_epoch,epoch);
          }
        }
        if(load(coordinator.decision.request_epoch)==epoch){choice=unsigned(coordinator.decision.request);break;}
        __nanosleep(32);
      }
    }
    __syncthreads();
    if(tid==32*(active_lanes+1)) {
      max_active_shards=max(max_active_shards,unsigned(__popcll(shards_this_layer)));
      mgbfs_device_epoch::publish_stopped(local.status,epoch);
      if(a.rank==0)while(!mgbfs_device_epoch::try_finalize(coordinator.decision,ranks,a.world,epoch)) {
        if(terminal(a.terminal)) {choice=mgbfs_device_epoch::ERROR;break;}
        __nanosleep(32);
      }
      while(load(coordinator.decision.final_epoch)!=epoch) {
        if(terminal(a.terminal)) {choice=mgbfs_device_epoch::ERROR;break;}
        __nanosleep(32);
      }
      if(load(coordinator.decision.final_epoch)==epoch)choice=unsigned(coordinator.decision.final);
    }
    __syncthreads();
    if(choice!=mgbfs_device_epoch::ADVANCE)break;
    if(tid==32*(active_lanes+1)) {
      current=next;next=producers_done=consumers_done=0;
      ++depth;a.layer_sizes[depth]=coordinator.global_next;
      store(local.committed_count,current);
    }
    __syncthreads();
  }
  if(tid==32*(active_lanes+1)) {
    report->depth=depth;report->sent=sent;report->received=received;
    report->duplicates=duplicates;report->duplicates_full=duplicates_full;report->blocked=blocked;
    report->capture_count=a.capture_count?*a.capture_count:0;
    report->global_next=coordinator.global_next;report->status=choice;report->rank=a.rank;
    report->max_active_shards_first64=max_active_shards;
    // The request reason is published before request_epoch. The final vote
    // can upgrade a resource stop to ERROR if another rank reported a fatal.
    report->reason=choice==mgbfs_device_epoch::ERROR?reason_protocol:
      choice==mgbfs_device_epoch::INCOMPLETE?unsigned(coordinator.stop_reason):reason_none;
    for(unsigned r=0;r<a.world;++r)a.rank_counts[r]=load(rank[r]->committed_count);
  }
}
#endif // MGBFS_RESIDENT_IMPLEMENTATION
}
