#include "device_topology_probe.h"
#include <cuda_runtime.h>
#include <nccl.h>

namespace {
struct NativeTopologyApi {
  int device_count(int* value) { return int(cudaGetDeviceCount(value)); }
  int runtime_version(int* value) { return int(cudaRuntimeGetVersion(value)); }
  int driver_version(int* value) { return int(cudaDriverGetVersion(value)); }
  int nccl_version(int* value) { return int(ncclGetVersion(value)); }
  bool lsa_compiled() const {
#ifdef MGBFS_NCCL_LSA
    return true;
#else
    return false;
#endif
  }
  int cooperative(int device, int* value) {
    return int(cudaDeviceGetAttribute(value, cudaDevAttrCooperativeLaunch, device));
  }
  int can_access_peer(int src, int dst, int* value) {
    return int(cudaDeviceCanAccessPeer(value, src, dst));
  }
  int native_atomics(int src, int dst, int* value) {
    return int(cudaDeviceGetP2PAttribute(value, cudaDevP2PAttrNativeAtomicSupported, src, dst));
  }
};
}
extern "C" int mgbfs_device_topology_v1(const int32_t* devices, uint32_t count,
    uint32_t size, MgbfsDeviceTopologyV1* out) {
  NativeTopologyApi api;
  return mgbfs_device_topology_probe_v1(api, devices, count, size, out);
}
