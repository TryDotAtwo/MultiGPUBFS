#pragma once
#include "device_batch_window.cuh"
#include "device_segmented_hash_accept.cuh"
#include "device_hash_key.cuh"
#include <cooperative_groups.h>
#include <cstdint>

// Included after the batch Args declaration. The main cooperative grid has
// stopped all owner CTAs at the completed-layer boundary. Rebuild in place:
// first validate every live row's old slot, then clear selected tables, then
// insert each still-live row and rewrite its slot position. No peer reads
// local hash tables. A failure poisons the job; a partial table is never
// reused as a successful BFS result.
namespace mgbfs_resident_hash_batch {
struct RehashRow {
  MgbfsPack4Dictionary dictionary;
  const void* states;
  uint64_t packed,local,stride;
  uint32_t packed_states,state_bytes;
  __device__ int64_t value(uint32_t e) const {
    if(packed_states)return dictionary.symbols[(packed>>(4*e))&15u];
    if(state_bytes==1)return int64_t(static_cast<const uint8_t*>(states)[uint64_t(e)*stride+local]);
    return static_cast<const int64_t*>(states)[uint64_t(e)*stride+local];
  }
};
__device__ inline uint64_t rehash_threshold(uint64_t slots) {
  const uint64_t n=slots/4;
  return n?n:1;
}
__device__ inline uint64_t position_at(const MgbfsSegment& s,uint64_t local) {
  return s.position_item_bytes==4?
    uint64_t(reinterpret_cast<const uint32_t*>(s.positions)[local]):s.positions[local];
}
__device__ inline void set_position(const MgbfsSegment& s,uint64_t local,uint64_t slot) {
  if(s.position_item_bytes==4)reinterpret_cast<uint32_t*>(s.positions)[local]=uint32_t(slot);
  else s.positions[local]=slot;
}
__device__ inline RehashRow rehash_row(const mgbfs_resident::Args& a,
    const mgbfs_resident::HashConfig& config,const MgbfsSegment& s,uint32_t bank,uint64_t row) {
  const uint64_t local=uint64_t(bank)*s.capacity+row;
  const uint64_t stride=s.capacity*s.banks;
  const uint64_t packed=config.packed_states?static_cast<const uint64_t*>(s.states)[local]:0;
  return {a.dictionary,s.states,packed,local,stride,config.packed_states,config.state_bytes};
}
__device__ inline void rehash_completed_boundary(const Args& args,
    const mgbfs_resident::HashConfig& config,mgbfs_device_batch::Window* local,
    uint32_t retiring_bank) {
  using namespace mgbfs_device_batch;
  using namespace mgbfs_segmented_exact;
  auto grid=cooperative_groups::this_grid();
  const uint64_t tid=grid.thread_rank();
  const auto& a=args.base;
  auto* extents=reinterpret_cast<mgbfs_device_hash::Extent*>(a.extents);
  auto* retired_counts=reinterpret_cast<uint64_t*>(args.spans);
  auto* rebuild_counts=retired_counts+a.extent_count;

  // The slot list must still represent each live row before it is rebuilt.
  // Validation also rejects any unresolved PENDING entry in the whole table.
  for(uint64_t e=0;e<a.extent_count;++e) {
    const auto x=extents[e];const auto seg=x.segment;
    if(retired_counts[e]<rehash_threshold(seg.slots_count))continue;
    const uint32_t elements=config.packed_states?1:a.dictionary.elements;
    const uint64_t width=config.packed_states?8:config.state_bytes;
    if(!mgbfs_device_hash::valid_extent(x,retiring_bank,elements,width)||
       !(config.state_bytes==1||config.state_bytes==8)) {
      if(!tid)fail(local,WINDOW_PROTOCOL);continue;
    }
    for(uint64_t slot=tid;slot<seg.slots_count;slot+=grid.size()) {
      const uint64_t observed=seg.slots[slot];
      if(observed!=empty&&observed!=retired&&(observed&pending))fail(local,WINDOW_PROTOCOL);
    }
    for(uint32_t bank=0;bank<3;++bank)if(bank!=retiring_bank) {
      const uint64_t count=x.counts[bank];
      if(count>seg.capacity){if(!tid)fail(local,WINDOW_PROTOCOL);continue;}
      for(uint64_t row=tid;row<count;row+=grid.size()) {
        const auto value=rehash_row(a,config,seg,bank,row);
        if(config.packed_states&&!mgbfs_pack4_valid_word(a.dictionary,value.packed)) {
          fail(local,WINDOW_PROTOCOL);continue;
        }
        const uint64_t hash=mgbfs_device_hash_key::hash(value,config);
        const uint64_t slot=position_at(seg,value.local);
        const uint64_t expected=(hash&prefix)|(uint64_t(bank)<<32)|row;
        if(slot>=seg.slots_count||seg.slots[slot]!=expected)fail(local,WINDOW_PROTOCOL);
      }
    }
  }
  grid.sync();

  // CUDA cooperative-grid synchronization provides the writer rendezvous:
  // all old readers and validation finish before the table is cleared.
  if(!acquire(local->status.failure)) {
    for(uint64_t e=0;e<a.extent_count;++e) {
      const auto seg=extents[e].segment;
      if(retired_counts[e]<rehash_threshold(seg.slots_count))continue;
      for(uint64_t slot=tid;slot<seg.slots_count;slot+=grid.size())seg.slots[slot]=empty;
    }
  }
  grid.sync();

  if(!acquire(local->status.failure)) {
    for(uint64_t e=0;e<a.extent_count;++e) {
      const auto x=extents[e];const auto seg=x.segment;
      if(retired_counts[e]<rehash_threshold(seg.slots_count))continue;
      for(uint32_t bank=0;bank<3;++bank)if(bank!=retiring_bank) {
        const uint64_t count=x.counts[bank];
        for(uint64_t row=tid;row<count;row+=grid.size()) {
          const auto value=rehash_row(a,config,seg,bank,row);
          const uint64_t hash=mgbfs_device_hash_key::hash(value,config);
          const uint64_t token=(hash&prefix)|(uint64_t(bank)<<32)|row;
          const uint64_t start=mgbfs_shared_table_bucket(hash,seg.slots_count);
          bool done=false;
          for(uint64_t probe=0;probe<seg.slots_count;++probe) {
            const uint64_t slot=(start+probe)&(seg.slots_count-1);
            auto cell=cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(seg.slots[slot]);
            uint64_t expected=empty;
            if(cell.compare_exchange_strong(expected,token,cuda::memory_order_acq_rel,
                                            cuda::memory_order_acquire)) {
              set_position(seg,value.local,slot);done=true;break;
            }
          }
          if(!done)fail(local,WINDOW_PROTOCOL);
        }
      }
    }
  }
  grid.sync();
  if(!tid&&!acquire(local->status.failure)) {
    for(uint64_t e=0;e<a.extent_count;++e) {
      if(retired_counts[e]>=rehash_threshold(extents[e].segment.slots_count)) {
        retired_counts[e]=0;
        if(rebuild_counts[e]==UINT64_MAX)fail(local,WINDOW_PROTOCOL);
        else ++rebuild_counts[e];
      }
    }
  }
  grid.sync();
}
}
