#include "device_packet_ring.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

using namespace mgbfs_packet_ring;
struct alignas(64) Fixture {
  PacketSlot slots[2];
  unsigned char payload[16];
  CloseSlot close;
  int failure;
};
#define VERIFY(x) do { if (!(x)) { d->failure = __LINE__; return; } } while (0)

__global__ void check(Fixture* d) {
  if (blockIdx.x || threadIdx.x) return;
  init_slot(d->slots[0], 0);
  init_slot(d->slots[1], 1);
  init_close(d->close);
  View v{d->slots, d->payload, 1, 8};
  unsigned char data[3] = {7, 11, 17};
  PacketHeader h{};
  const unsigned char* ptr = nullptr;
  unsigned long long tail = 0;

  VERIFY(try_send(v, 0, 42, 3, 9, data, 3) == Result::ok);
  VERIFY(try_send(v, 1, 42, 3, 9, data, 3) == Result::not_ready);
  // A full data ring must not prevent publishing or observing its close marker.
  VERIFY(try_close(d->close, 0, 42, 3, 1) == Result::ok);
  VERIFY(try_get_close(d->close, 0, 42, 3, &tail) == Result::ok && tail == 1);
  VERIFY(try_close(d->close, 1, 42, 4, 2) == Result::not_ready);
  VERIFY(try_get_close(d->close, 0, 41, 3, &tail) == Result::wrong_generation);
  VERIFY(try_get_close(d->close, 0, 42, 2, &tail) == Result::future_phase);
  VERIFY(try_get_close(d->close, 0, 42, 4, &tail) == Result::stale_phase);
  VERIFY(try_ack_close(d->close, 0, 42, 3, 0) == Result::not_drained);
  VERIFY(try_peek(v, 0, 42, 2, &h, &ptr) == Result::future_phase);
  VERIFY(try_peek(v, 0, 42, 4, &h, &ptr) == Result::stale_phase);
  VERIFY(try_peek(v, 0, 41, 3, &h, &ptr) == Result::wrong_generation);
  VERIFY(try_peek(v, 0, 42, 3, &h, &ptr) == Result::ok);
  VERIFY(h.ticket == 0 && h.len == 3 && h.format == 9);
  VERIFY(ptr[0] == 7 && ptr[1] == 11 && ptr[2] == 17);
  VERIFY(try_send(v, 1, 42, 3, 9, data, 3) == Result::not_ready);
  VERIFY(try_ack(v, 0, 42, 3) == Result::ok);
  VERIFY(try_ack_close(d->close, 0, 42, 3, 1) == Result::ok);
  VERIFY(try_send(v, 1, 42, 4, 5, data, 3) == Result::ok);
  VERIFY(try_ack(v, 0, 42, 3) == Result::not_ready);
  d->slots[0].header.len = 9;
  VERIFY(try_peek(v, 1, 42, 4, &h, &ptr) == Result::invalid_length);
  d->slots[0].header.len = 3;
  d->slots[0].header.ticket = 99;
  VERIFY(try_peek(v, 1, 42, 4, &h, &ptr) == Result::wrong_ticket);
  d->slots[0].header.ticket = 1;
  VERIFY(try_peek(v, 1, 42, 4, &h, &ptr) == Result::ok && h.format == 5);
  VERIFY(try_ack(v, 1, 42, 4) == Result::ok);
  VERIFY(try_close(d->close, 1, 42, 4, 2) == Result::ok);
  VERIFY(try_get_close(d->close, 1, 42, 4, &tail) == Result::ok && tail == 2);
  VERIFY(try_ack_close(d->close, 1, 42, 4, 3) == Result::bad_cursor);
  VERIFY(try_ack_close(d->close, 1, 42, 4, 2) == Result::ok);

  VERIFY(try_send(v, 2, 42, 5, 0, data, 9) == Result::invalid_length);
  VERIFY(try_send(v, 2, 42, 5, 0, nullptr, 1) == Result::invalid_argument);
  VERIFY(try_send(v, 2, 42, 5, 0, nullptr, 0) == Result::ok);
  VERIFY(try_peek(v, 2, 42, 5, &h, &ptr) == Result::ok && h.len == 0);
  VERIFY(try_ack(v, 2, 42, 5) == Result::ok);
  VERIFY(try_send(v, ~0ull, 42, 5, 0, data, 3) == Result::sequence_exhausted);
  View invalid{d->slots, d->payload, 0, 8};
  VERIFY(try_send(invalid, 0, 42, 5, 0, data, 3) == Result::invalid_capacity);

  init_slot(d->slots[0], 0);
  init_slot(d->slots[1], 1);
  View two{d->slots, d->payload, 2, 8};
  VERIFY(try_send(two, 0, 43, 1, 0, data, 3) == Result::ok);
  VERIFY(try_send(two, 1, 43, 1, 1, data, 3) == Result::ok);
  VERIFY(try_send(two, 2, 43, 1, 2, data, 3) == Result::not_ready);
  VERIFY(try_ack(two, 0, 43, 1) == Result::ok);
  VERIFY(try_send(two, 2, 43, 1, 2, data, 3) == Result::ok);
  VERIFY(try_peek(two, 1, 43, 1, &h, &ptr) == Result::ok && h.format == 1);
  VERIFY(try_ack(two, 1, 43, 1) == Result::ok);
  VERIFY(try_peek(two, 2, 43, 1, &h, &ptr) == Result::ok && h.format == 2);
  VERIFY(try_ack(two, 2, 43, 1) == Result::ok);
}

int main() {
  int count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&count);
  if (count_status == cudaErrorNoDevice || (count_status == cudaSuccess && !count)) return 77;
  if (count_status != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceCount: %s\n", cudaGetErrorString(count_status));
    return 2;
  }
  Fixture* d = nullptr; Fixture result{};
  if (cudaMalloc(&d, sizeof(Fixture)) != cudaSuccess ||
      cudaMemset(d, 0, sizeof(Fixture)) != cudaSuccess) return 2;
  check<<<1,1>>>(d);
  auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error)); return 3;
  }
  if (cudaMemcpy(&result, d, sizeof(result), cudaMemcpyDeviceToHost) != cudaSuccess) return 4;
  cudaFree(d);
  if (result.failure) {
    std::fprintf(stderr, "FAIL device ring unit line %d\n", result.failure); return 5;
  }
  std::puts("device packet ring GPU unit passed");
  return 0;
}
