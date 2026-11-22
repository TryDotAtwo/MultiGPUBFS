#ifndef MGBFS_DEVICE_PACKET_RING_CUH
#define MGBFS_DEVICE_PACKET_RING_CUH

#include <cuda/atomic>
#include <cstdint>

/* Bounded SPSC device mailbox primitive. It is not a BFS scheduler or a
 * transport capability check. One device thread owns all writes to a ring;
 * one device thread owns all reads/acks. A packet is published by the same
 * writer that filled its header and payload. Every slot, its payload and the
 * separate close slot must be in a peer-visible, suitably aligned allocation
 * admitted for system-scope native atomics and actually tested by the caller.
 * Admission checks capacity*slot_bytes in wide arithmetic and provides that
 * payload allocation. Source bytes must not overlap the destination slot.
 *
 * The producer advances its cursor only after ok. The consumer must make a
 * packet's effects durable or retain an exact retry record before ack.
 * Functions never wait; not_ready returns control to a scheduler which must
 * continue servicing inbound traffic. The close slot is independent of a
 * full data ring and has its own explicit acknowledgement/lease.
 *
 * Initialize only while all prior readers and writers are quiescent; publish
 * initialized storage to every device before launching the users. Tickets are
 * monotone within a job and are never allowed to wrap. Reset across jobs
 * requires a new job generation plus a proven group-wide drain.
 */
namespace mgbfs_packet_ring {
using U64 = unsigned long long;
using SystemAtomic = cuda::atomic_ref<U64, cuda::thread_scope_system>;

struct PacketHeader {
  U64 generation;
  U64 phase;
  U64 ticket;
  uint32_t format;
  uint32_t len;
};
struct alignas(64) PacketSlot {
  alignas(8) U64 sequence;
  PacketHeader header;
};
struct alignas(64) CloseSlot {
  alignas(8) U64 sequence;
  U64 generation;
  U64 phase;
  U64 final_tail;
};
struct View {
  PacketSlot* slots;
  unsigned char* payload;
  uint32_t capacity;
  uint32_t slot_bytes;
};
static_assert(sizeof(PacketHeader) == 32, "packet header ABI");
static_assert(sizeof(PacketSlot) == 64, "packet slot ABI");
static_assert(sizeof(CloseSlot) == 64, "close slot ABI");

enum class Result : uint32_t {
  ok, not_ready, invalid_argument, invalid_capacity, invalid_length,
  sequence_exhausted, wrong_generation, future_phase, stale_phase,
  wrong_ticket, not_drained, bad_cursor
};
constexpr U64 max_ticket = (~0ull) >> 1;

__device__ __forceinline__ Result validate_ticket(U64 ticket, uint32_t capacity) {
  if (!capacity) return Result::invalid_capacity;
  if (ticket > max_ticket - U64(capacity)) return Result::sequence_exhausted;
  return Result::ok;
}
__device__ __forceinline__ Result validate_view(View view, U64 ticket) {
  auto status = validate_ticket(ticket, view.capacity);
  if (status != Result::ok) return status;
  if (!view.slots || (view.slot_bytes && !view.payload)) return Result::invalid_argument;
  return Result::ok;
}
__device__ __forceinline__ Result check_epoch(U64 generation, U64 phase, U64 ticket,
    const PacketHeader& header) {
  if (header.generation != generation) return Result::wrong_generation;
  if (header.ticket != ticket) return Result::wrong_ticket;
  if (header.phase > phase) return Result::future_phase;
  if (header.phase < phase) return Result::stale_phase;
  return Result::ok;
}
__device__ __forceinline__ void init_slot(PacketSlot& slot, U64 index) {
  slot.sequence = 2 * index;
}
__device__ __forceinline__ void init_close(CloseSlot& close) {
  close.sequence = 0;
}
__device__ __forceinline__ Result try_send(View view, U64 ticket, U64 generation,
    U64 phase, uint32_t format, const unsigned char* source, uint32_t len) {
  auto status = validate_view(view, ticket);
  if (status != Result::ok) return status;
  if (len > view.slot_bytes) return Result::invalid_length;
  if (len && !source) return Result::invalid_argument;
  const U64 index = ticket % view.capacity;
  PacketSlot& slot = view.slots[index];
  SystemAtomic seq(slot.sequence);
  if (seq.load(cuda::memory_order_acquire) != 2 * ticket) return Result::not_ready;
  slot.header = PacketHeader{generation, phase, ticket, format, len};
  if (len) {
    unsigned char* dest = view.payload + index * U64(view.slot_bytes);
    for (uint32_t i = 0; i < len; ++i) dest[i] = source[i];
  }
  seq.store(2 * ticket + 1, cuda::memory_order_release);
  return Result::ok;
}
__device__ __forceinline__ Result try_peek(View view, U64 ticket, U64 generation,
    U64 phase, PacketHeader* header, const unsigned char** payload) {
  auto status = validate_view(view, ticket);
  if (status != Result::ok) return status;
  if (!header || !payload) return Result::invalid_argument;
  const U64 index = ticket % view.capacity;
  PacketSlot& slot = view.slots[index];
  SystemAtomic seq(slot.sequence);
  if (seq.load(cuda::memory_order_acquire) != 2 * ticket + 1) return Result::not_ready;
  PacketHeader value = slot.header;
  status = check_epoch(generation, phase, ticket, value);
  if (status != Result::ok) return status;
  if (value.len > view.slot_bytes) return Result::invalid_length;
  *header = value;
  *payload = value.len ? view.payload + index * U64(view.slot_bytes) : nullptr;
  return Result::ok;
}
/* Caller owns the currently visible packet and has durably committed it. */
__device__ __forceinline__ Result try_ack(View view, U64 ticket, U64 generation, U64 phase) {
  PacketHeader header{};
  const unsigned char* payload = nullptr;
  auto status = try_peek(view, ticket, generation, phase, &header, &payload);
  if (status != Result::ok) return status;
  PacketSlot& slot = view.slots[ticket % view.capacity];
  SystemAtomic(slot.sequence).store(2 * (ticket + view.capacity), cuda::memory_order_release);
  return Result::ok;
}

__device__ __forceinline__ Result try_close(CloseSlot& close, U64 close_ticket,
    U64 generation, U64 phase, U64 final_tail) {
  auto status = validate_ticket(close_ticket, 1);
  if (status != Result::ok) return status;
  SystemAtomic seq(close.sequence);
  if (seq.load(cuda::memory_order_acquire) != 2 * close_ticket) return Result::not_ready;
  close.generation = generation;
  close.phase = phase;
  close.final_tail = final_tail;
  seq.store(2 * close_ticket + 1, cuda::memory_order_release);
  return Result::ok;
}
__device__ __forceinline__ Result try_get_close(CloseSlot& close, U64 close_ticket,
    U64 generation, U64 phase, U64* final_tail) {
  auto status = validate_ticket(close_ticket, 1);
  if (status != Result::ok) return status;
  if (!final_tail) return Result::invalid_argument;
  SystemAtomic seq(close.sequence);
  if (seq.load(cuda::memory_order_acquire) != 2 * close_ticket + 1) return Result::not_ready;
  if (close.generation != generation) return Result::wrong_generation;
  if (close.phase > phase) return Result::future_phase;
  if (close.phase < phase) return Result::stale_phase;
  *final_tail = close.final_tail;
  return Result::ok;
}
/* durable_tail is the next data ticket, after every earlier packet committed.
 * A close is acknowledged only once its final data packet is durable. */
__device__ __forceinline__ Result try_ack_close(CloseSlot& close, U64 close_ticket,
    U64 generation, U64 phase, U64 durable_tail) {
  U64 final_tail = 0;
  auto status = try_get_close(close, close_ticket, generation, phase, &final_tail);
  if (status != Result::ok) return status;
  if (durable_tail < final_tail) return Result::not_drained;
  if (durable_tail > final_tail) return Result::bad_cursor;
  SystemAtomic(close.sequence).store(2 * (close_ticket + 1), cuda::memory_order_release);
  return Result::ok;
}
} // namespace mgbfs_packet_ring
#endif
