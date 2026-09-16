#include "control_transfer.h"
#include <cuda_runtime_api.h>
#include <new>
namespace {
struct Staging {
  MgbfsOwnerControl upload;
  MgbfsControlSnapshotV1 snapshot;
};
static_assert(sizeof(Staging) == 264);
struct Transfer {
  Staging* host{};
  cudaStream_t stream{};
  int device{};
  bool pending{}, poisoned{};
};
bool device_matches(Transfer const& t) {
  int device = -1;
  return cudaGetDevice(&device) == cudaSuccess && device == t.device;
}
int fail(Transfer& t) { t.poisoned = true; return -1; }
}
extern "C" int mgbfs_control_transfer_create_v1(void* stream, void** out) {
  if (!out) return -1;
  *out = nullptr;
  auto* t = new (std::nothrow) Transfer;
  if (!t) return -1;
  t->stream = static_cast<cudaStream_t>(stream);
  if (cudaGetDevice(&t->device) != cudaSuccess ||
      cudaHostAlloc(reinterpret_cast<void**>(&t->host), sizeof(Staging), 0) != cudaSuccess) {
    delete t;
    return -1;
  }
  *t->host = {};
  *out = t;
  return 0;
}
extern "C" int mgbfs_control_transfer_destroy_v1(void* handle) {
  if (!handle) return -1;
  auto* t = static_cast<Transfer*>(handle);
  // On a failed drain preserve DMA storage until process exit, even if poisoned.
  if (!device_matches(*t) || cudaStreamSynchronize(t->stream) != cudaSuccess) return -1;
  if (cudaFreeHost(t->host) != cudaSuccess) return -1;
  delete t;
  return 0;
}
extern "C" int mgbfs_control_transfer_upload_v1(void* handle,
    const MgbfsOwnerControl* host, MgbfsOwnerControl* device) {
  if (!handle) return -1;
  auto& t = *static_cast<Transfer*>(handle);
  if (t.poisoned || t.pending || !host || !device || !device_matches(t)) return fail(t);
  t.host->upload = *host;
  t.pending = true;
  if (cudaMemcpyAsync(device, &t.host->upload, sizeof(*host), cudaMemcpyHostToDevice,
                      t.stream) != cudaSuccess) return fail(t);
  return 0;
}
extern "C" int mgbfs_control_transfer_read_v1(void* handle, const MgbfsOwnerControl* control,
    const MgbfsStateExtent* extent, const MgbfsStateRingControl* ring, const uint32_t* count,
    MgbfsControlSnapshotV1* out) {
  if (out) *out = {};
  if (!handle) return -1;
  auto& t = *static_cast<Transfer*>(handle);
  if (t.poisoned || !out || !control || !extent || !ring || !device_matches(t)) return fail(t);
  auto& s = t.host->snapshot;
  s.count = 0;
  s.reserved = 0;
  if (cudaMemcpyAsync(&s.control, control, sizeof(s.control), cudaMemcpyDeviceToHost,
                      t.stream) != cudaSuccess ||
      cudaMemcpyAsync(&s.extent, extent, sizeof(s.extent), cudaMemcpyDeviceToHost,
                      t.stream) != cudaSuccess ||
      cudaMemcpyAsync(&s.ring, ring, sizeof(s.ring), cudaMemcpyDeviceToHost,
                      t.stream) != cudaSuccess ||
      (count && cudaMemcpyAsync(&s.count, count, sizeof(s.count), cudaMemcpyDeviceToHost,
                                t.stream) != cudaSuccess) ||
      cudaStreamSynchronize(t.stream) != cudaSuccess) return fail(t);
  t.pending = false;
  *out = s;
  return 0;
}
