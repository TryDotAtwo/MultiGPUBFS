#include "device_topology_v1.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { if (!(x)) { \
  std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
  std::exit(1); \
} } while (0)
#define CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::fprintf(stderr, "CUDA failure %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  std::exit(1); \
} } while (0)

static void check_group(const std::vector<int32_t>& devices) {
  MgbfsDeviceTopologyV1 out{};
  int current_before = -1, current_after = -1;
  CUDA(cudaGetDevice(&current_before));
  CHECK(mgbfs_device_topology_v1(devices.data(), devices.size(), sizeof(out), &out) == 0);
  CUDA(cudaGetDevice(&current_after));
  CHECK(current_before == current_after);
  CHECK(out.abi_version == 1 && out.struct_size == sizeof(out));
  CHECK(out.rank_count == devices.size());
#ifdef MGBFS_EXPECT_LSA
  CHECK(out.lsa_compiled == 1u);
#else
  CHECK(out.lsa_compiled == 0u);
#endif
  if (!out.lsa_compiled) {
    CHECK(out.status == MGBFS_TOPOLOGY_LSA_NOT_COMPILED);
    std::printf("ranks=%zu: build has no LSA; topology was not accepted\n", devices.size());
    return;
  }
  if (out.nccl_version < 22900) {
    CHECK(out.status == MGBFS_TOPOLOGY_NCCL_TOO_OLD);
    return;
  }

  int expected = MGBFS_TOPOLOGY_PRELIMINARY_OK;
  int src = -1, dst = -1;
  for (unsigned i = 0; i < devices.size(); ++i) {
    int cooperative = 0;
    CUDA(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, devices[i]));
    if (!cooperative) {
      expected = MGBFS_TOPOLOGY_COOPERATIVE_UNSUPPORTED;
      src = i;
      break;
    }
  }
  if (expected == MGBFS_TOPOLOGY_PRELIMINARY_OK) {
    for (unsigned i = 0; i < devices.size() && expected == MGBFS_TOPOLOGY_PRELIMINARY_OK; ++i) {
      for (unsigned j = 0; j < devices.size(); ++j) {
        if (i == j) continue;
        int peer = 0, atomic = 0;
        CUDA(cudaDeviceCanAccessPeer(&peer, devices[i], devices[j]));
        if (!peer) {
          expected = MGBFS_TOPOLOGY_PEER_UNSUPPORTED;
          src = i; dst = j; break;
        }
        CUDA(cudaDeviceGetP2PAttribute(&atomic, cudaDevP2PAttrNativeAtomicSupported,
                                      devices[i], devices[j]));
        if (!atomic) {
          expected = MGBFS_TOPOLOGY_NATIVE_ATOMICS_UNSUPPORTED;
          src = i; dst = j; break;
        }
      }
    }
  }
  CHECK(out.status == static_cast<unsigned>(expected));
  CHECK(out.src_rank == src && out.dst_rank == dst);
  if (src >= 0) CHECK(out.src_device == devices[src]);
  if (dst >= 0) CHECK(out.dst_device == devices[dst]);
  std::printf("ranks=%zu devices=", devices.size());
  for (auto device : devices) std::printf("%d,", device);
  std::printf(" status=%u stage=%u src_rank=%d dst_rank=%d src_device=%d dst_device=%d"
              " runtime=%d driver=%d nccl=%d\n",
              out.status, out.stage, out.src_rank, out.dst_rank,
              out.src_device, out.dst_device, out.cuda_runtime_version,
              out.cuda_driver_version, out.nccl_version);
}

int main() {
  int count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&count);
  if (count_status == cudaErrorNoDevice || (count_status == cudaSuccess && count == 0)) {
    std::puts("SKIP no visible CUDA device");
    return 77;
  }
  if (count_status != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceCount: %s\n", cudaGetErrorString(count_status));
    return 2;
  }
  const int n = std::min(count, 8);
  for (int i = 0; i < n; ++i) {
    cudaDeviceProp props{};
    char bus[32]{};
    CUDA(cudaGetDeviceProperties(&props, i));
    CUDA(cudaDeviceGetPCIBusId(bus, sizeof(bus), i));
    std::printf("device=%d uuid=", i);
    for (unsigned char b : props.uuid.bytes) std::printf("%02x", unsigned(b));
    std::printf(" pci=%s\n", bus);
  }
  check_group({0});
  if (n >= 2) check_group({0, 1});
  if (n >= 4) {
    check_group({0, 1, 2, 3});
    check_group({3, 1, 0, 2});
  }
  if (n >= 8) {
    check_group({0, 1, 2, 3, 4, 5, 6, 7});
    check_group({7, 6, 5, 4, 3, 2, 1, 0});
  }
  MgbfsDeviceTopologyV1 out{};
  const int32_t invalid[] = {-1};
  CHECK(mgbfs_device_topology_v1(invalid, 1, sizeof(out), &out) == 0);
  CHECK(out.status == MGBFS_TOPOLOGY_INVALID_DEVICE && out.src_device == -1);
  if (count >= 2) {
    const int32_t duplicate[] = {0, 0};
    CHECK(mgbfs_device_topology_v1(duplicate, 2, sizeof(out), &out) == 0);
    CHECK(out.status == MGBFS_TOPOLOGY_DUPLICATE_DEVICE);
  }
  CHECK(mgbfs_device_topology_v1(nullptr, 1, sizeof(out), &out) == 0);
  CHECK(out.status == MGBFS_TOPOLOGY_INVALID_ARGUMENT);
  std::puts("device-topology physical queries passed; communicator and smoke are untested");
  return 0;
}
