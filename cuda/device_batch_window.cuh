#pragma once
#include "device_batch_partition.cuh"
#include "device_epoch_decision.cuh"
#include <cuda/atomic>
#include <cstdint>

// Symmetric source-owned one-slot window. One local grid publishes READY;
// each destination alone publishes its ACK. The caller must execute a local
// grid barrier after every CTA finishes consuming the payload and before ACK.
// Windows and peer mappings stay live until all grids have drained.
namespace mgbfs_device_batch {
using ControlWord=unsigned long long;
using SystemAtomic=cuda::atomic_ref<ControlWord,cuda::thread_scope_system>;
struct alignas(64) Ack { ControlWord sequence; };
struct alignas(64) Window {
  ControlWord generation,round;
  uint64_t offsets[max_world+1];
  alignas(64) ControlWord ready;
  alignas(64) mgbfs_device_epoch::RankStatus status;
  alignas(64) mgbfs_device_epoch::Decision decision;
  ControlWord committed_count,committed_epoch,done_count,done_epoch;
  ControlWord global_next,stop_reason;
  ControlWord rounds,current,sequence;
  ControlWord sent,received,duplicates,duplicates_full,shard_mask,max_active_shards;
  Ack ack[max_world];
};
static_assert(alignof(Window)==64,"window ABI alignment");
static_assert(sizeof(Ack)==64,"each ACK has an independent cache line");
enum : ControlWord {
  WINDOW_RESOURCE=mgbfs_device_epoch::RESOURCE,
  WINDOW_PROTOCOL=mgbfs_device_epoch::FATAL,
  WINDOW_CANCELLED=mgbfs_device_epoch::FATAL
};
__device__ inline ControlWord acquire(ControlWord& x) {
  return SystemAtomic(x).load(cuda::memory_order_acquire);
}
__device__ inline void release(ControlWord& x,ControlWord v) {
  SystemAtomic(x).store(v,cuda::memory_order_release);
}
__device__ inline void fail(Window* local,ControlWord reason) {
  SystemAtomic(local->status.failure).fetch_or(reason,cuda::memory_order_acq_rel);
}
__device__ inline ControlWord failures(Window* const* peers,uint32_t world,const uint32_t* cancel) {
  ControlWord result=0;
  if(cancel && cuda::atomic_ref<uint32_t,cuda::thread_scope_system>(
      *const_cast<uint32_t*>(cancel)).load(cuda::memory_order_acquire))result|=WINDOW_CANCELLED;
  for(uint32_t p=0;p<world;++p)result|=acquire(peers[p]->status.failure);
  return result;
}
__device__ inline bool propagate_failure(Window* local,Window* const* peers,uint32_t world,const uint32_t* cancel) {
  const auto reason=failures(peers,world,cancel);
  if(reason){fail(local,reason);return true;}
  return false;
}
// All preceding payload writers must have executed __threadfence_system()
// and reached a grid barrier before the single publisher calls this.
__device__ inline void publish(Window* local,uint64_t generation,uint64_t round,
    uint64_t sequence,const uint64_t* offsets,uint32_t world) {
  local->generation=generation;local->round=round;
  for(uint32_t d=0;d<=world;++d)local->offsets[d]=offsets[d];
  release(local->ready,sequence);
}
// A single coordinator thread polls, validates headers and then joins a grid
// barrier before the other CTAs read payloads. On failure every rank observes
// a system-scope failure bit; there is no silent backend switch or slot reuse.
__device__ inline bool wait_ready(Window* local,Window* const* peers,uint32_t world,
    uint64_t generation,uint64_t round,uint64_t sequence,uint64_t capacity,
    const uint32_t* cancel) {
  for(uint32_t p=0;p<world;++p) {
    ControlWord ready=0;
    while((ready=acquire(peers[p]->ready))<sequence) {
      if(propagate_failure(local,peers,world,cancel))return false;
      __nanosleep(32);
    }
    if(ready!=sequence || peers[p]->generation!=generation ||
       peers[p]->round!=round || peers[p]->offsets[0]!=0 ||
       peers[p]->offsets[world]>capacity) {
      fail(local,WINDOW_PROTOCOL);return false;
    }
    for(uint32_t d=0;d<world;++d)
      if(peers[p]->offsets[d]>peers[p]->offsets[d+1]) {
        fail(local,WINDOW_PROTOCOL);return false;
      }
  }
  if(propagate_failure(local,peers,world,cancel))return false;
  return true;
}
__device__ inline void acknowledge(Window* const* peers,uint32_t world,
    uint32_t destination,uint64_t sequence) {
  for(uint32_t p=0;p<world;++p)release(peers[p]->ack[destination].sequence,sequence);
}
__device__ inline bool wait_ack(Window* local,Window* const* peers,uint32_t world,
    uint64_t sequence,const uint32_t* cancel) {
  for(uint32_t d=0;d<world;++d) {
    ControlWord seen=0;
    while((seen=acquire(local->ack[d].sequence))<sequence) {
      if(propagate_failure(local,peers,world,cancel))return false;
      __nanosleep(32);
    }
    if(seen!=sequence){fail(local,WINDOW_PROTOCOL);return false;}
  }
  if(propagate_failure(local,peers,world,cancel))return false;
  return true;
}
__device__ inline Payload* payload(Window* control) {
  return reinterpret_cast<Payload*>(reinterpret_cast<unsigned char*>(control)+sizeof(Window));
}
}
