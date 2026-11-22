// Experimental SPSC slot protocol, NOT wired into the production BFS.
// A distinct ring is owned by exactly one producer and exactly one consumer.
// The underlying memory, atomic capabilities and publication visibility must
// be admitted for every participating GPU by a real transport preflight.
// No protocol operation spins or starts another GPU operation. The scheduler
// must keep the consumer runnable when the producer receives not_ready.
#pragma once
#include <cuda/atomic>
#include <cstdint>

namespace mgbfs_device_ring {
struct Payload {
  unsigned long long value;
  uint32_t kind;
  uint32_t source;
};
struct alignas(32) Slot {
  alignas(8) unsigned long long sequence;
  unsigned long long epoch;
  Payload payload;
};
enum class Result : uint32_t {
  ok, not_ready, wrong_epoch, invalid_capacity, sequence_exhausted
};
constexpr unsigned long long max_ticket=(~0ull)>>1;

// Only at admission/reset with all readers, writers and GPU streams quiescent.
__device__ __forceinline__ void init_slot(Slot& s, unsigned long long epoch,
                                          unsigned long long index) {
  s.epoch=epoch;
  s.sequence=2*index;
}
__device__ __forceinline__ Result validate(unsigned long long capacity,
                                           unsigned long long ticket) {
  if(!capacity) return Result::invalid_capacity;
  // The release needs 2*(ticket+capacity), with no 64-bit wrap/ABA.
  if(capacity>max_ticket || ticket>max_ticket-capacity)
    return Result::sequence_exhausted;
  return Result::ok;
}
// Writer publishes only after all payload bytes have been written. The system
// scope is required but is not by itself proof that a particular inter-GPU
// mapping is coherent or offers the required atomic support.
__device__ __forceinline__ Result try_write(Slot* ring, unsigned long long capacity,
    unsigned long long epoch, unsigned long long ticket, const Payload& value) {
  Result v=validate(capacity,ticket); if(v!=Result::ok) return v;
  Slot& s=ring[ticket%capacity];
  if(s.epoch!=epoch) return Result::wrong_epoch;
  cuda::atomic_ref<unsigned long long,cuda::thread_scope_system> sequence(s.sequence);
  if(sequence.load(cuda::memory_order_acquire)!=2*ticket) return Result::not_ready;
  s.payload=value;
  sequence.store(2*ticket+1,cuda::memory_order_release);
  return Result::ok;
}
// Read does not grant credit back: the consumer must first durably commit
// acceptance, deduplication, or a replayable record, then explicitly release.
__device__ __forceinline__ Result try_read(Slot* ring, unsigned long long capacity,
    unsigned long long epoch, unsigned long long ticket, Payload& value) {
  Result v=validate(capacity,ticket); if(v!=Result::ok) return v;
  Slot& s=ring[ticket%capacity];
  if(s.epoch!=epoch) return Result::wrong_epoch;
  cuda::atomic_ref<unsigned long long,cuda::thread_scope_system> sequence(s.sequence);
  if(sequence.load(cuda::memory_order_acquire)!=2*ticket+1) return Result::not_ready;
  value=s.payload;
  return Result::ok;
}
__device__ __forceinline__ Result try_release(Slot* ring, unsigned long long capacity,
    unsigned long long epoch, unsigned long long ticket) {
  Result v=validate(capacity,ticket); if(v!=Result::ok) return v;
  Slot& s=ring[ticket%capacity];
  if(s.epoch!=epoch) return Result::wrong_epoch;
  cuda::atomic_ref<unsigned long long,cuda::thread_scope_system> sequence(s.sequence);
  if(sequence.load(cuda::memory_order_acquire)!=2*ticket+1) return Result::not_ready;
  sequence.store(2*(ticket+capacity),cuda::memory_order_release);
  return Result::ok;
}
} // namespace mgbfs_device_ring
