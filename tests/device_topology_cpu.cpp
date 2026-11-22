#include "device_topology_probe.h"
#include <cstdint>
#include <initializer_list>
#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { if (!(x)) { \
  std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
  std::exit(1); \
} } while (0)

struct FakeApi {
  int count = 4, runtime = 12080, driver = 12080, nccl = 23007;
  bool compiled = true;
  int failing_stage = 0, fail_code = 71;
  int failed_src = -1, failed_dst = -1;
  int no_coop = -1, no_peer_src = -1, no_peer_dst = -1;
  int no_atomic_src = -1, no_atomic_dst = -1;
  int calls = 0;
  int device_count(int* value) {
    ++calls; if (failing_stage == MGBFS_TOPOLOGY_STAGE_DEVICE_COUNT) return fail_code;
    *value = count; return 0;
  }
  int runtime_version(int* value) {
    ++calls; if (failing_stage == MGBFS_TOPOLOGY_STAGE_RUNTIME_VERSION) return fail_code;
    *value = runtime; return 0;
  }
  int driver_version(int* value) {
    ++calls; if (failing_stage == MGBFS_TOPOLOGY_STAGE_DRIVER_VERSION) return fail_code;
    *value = driver; return 0;
  }
  int nccl_version(int* value) {
    ++calls; if (failing_stage == MGBFS_TOPOLOGY_STAGE_NCCL_VERSION) return fail_code;
    *value = nccl; return 0;
  }
  bool lsa_compiled() const { return compiled; }
  int cooperative(int device, int* value) {
    ++calls; if (failing_stage == MGBFS_TOPOLOGY_STAGE_COOPERATIVE && device == failed_src) return fail_code;
    *value = device != no_coop; return 0;
  }
  int can_access_peer(int src, int dst, int* value) {
    ++calls; if (failing_stage == MGBFS_TOPOLOGY_STAGE_PEER_ACCESS &&
                 src == failed_src && dst == failed_dst) return fail_code;
    *value = !(src == no_peer_src && dst == no_peer_dst); return 0;
  }
  int native_atomics(int src, int dst, int* value) {
    ++calls; if (failing_stage == MGBFS_TOPOLOGY_STAGE_NATIVE_ATOMICS &&
                 src == failed_src && dst == failed_dst) return fail_code;
    *value = !(src == no_atomic_src && dst == no_atomic_dst); return 0;
  }
};

static MgbfsDeviceTopologyV1 run(FakeApi& api, const int32_t* devices, uint32_t n) {
  MgbfsDeviceTopologyV1 result{};
  CHECK(mgbfs_device_topology_probe_v1(api, devices, n, sizeof(result), &result) == 0);
  CHECK(result.abi_version == 1);
  CHECK(result.struct_size == sizeof(result));
  CHECK(result.rank_count == n);
  return result;
}

int main() {
  const int32_t ordered[] = {3, 0, 2, 1};
  FakeApi api;
  MgbfsDeviceTopologyV1 result{};
  CHECK(mgbfs_device_topology_probe_v1(api, ordered, 4, sizeof(result), nullptr) == -1);
  CHECK(mgbfs_device_topology_probe_v1(api, ordered, 4, sizeof(result) - 1, &result) == -1);
  CHECK(api.calls == 0);

  result = run(api, ordered, 4);
  CHECK(result.status == MGBFS_TOPOLOGY_PRELIMINARY_OK);
  CHECK(result.stage == MGBFS_TOPOLOGY_STAGE_NONE);
  CHECK(result.src_rank == -1 && result.dst_rank == -1);
  CHECK(result.cuda_runtime_version == 12080 && result.cuda_driver_version == 12080);
  CHECK(result.nccl_version == 23007 && result.lsa_compiled == 1);

  api = FakeApi{};
  result = run(api, nullptr, 1);
  CHECK(result.status == MGBFS_TOPOLOGY_INVALID_ARGUMENT && api.calls == 0);
  result = run(api, ordered, 0);
  CHECK(result.status == MGBFS_TOPOLOGY_INVALID_ARGUMENT);
  result = run(api, ordered, 9);
  CHECK(result.status == MGBFS_TOPOLOGY_INVALID_ARGUMENT);

  api = FakeApi{};
  const int32_t duplicate[] = {2, 1, 2};
  result = run(api, duplicate, 3);
  CHECK(result.status == MGBFS_TOPOLOGY_DUPLICATE_DEVICE);
  CHECK(result.src_rank == 0 && result.dst_rank == 2);
  CHECK(result.src_device == 2 && result.dst_device == 2);
  const int32_t bad[] = {1, 4};
  result = run(api, bad, 2);
  CHECK(result.status == MGBFS_TOPOLOGY_INVALID_DEVICE);
  CHECK(result.src_rank == 1 && result.src_device == 4);

  api = FakeApi{}; api.compiled = false;
  result = run(api, ordered, 4);
  CHECK(result.status == MGBFS_TOPOLOGY_LSA_NOT_COMPILED);
  api = FakeApi{}; api.nccl = 22899;
  result = run(api, ordered, 4);
  CHECK(result.status == MGBFS_TOPOLOGY_NCCL_TOO_OLD);

  api = FakeApi{}; api.no_coop = 2;
  result = run(api, ordered, 4);
  CHECK(result.status == MGBFS_TOPOLOGY_COOPERATIVE_UNSUPPORTED);
  CHECK(result.stage == MGBFS_TOPOLOGY_STAGE_COOPERATIVE);
  CHECK(result.src_rank == 2 && result.src_device == 2);

  // Only the reverse direction of 3 <-> 0 fails. The ordered result reports
  // the precise source/destination ordinals and rank positions.
  api = FakeApi{}; api.no_peer_src = 0; api.no_peer_dst = 3;
  result = run(api, ordered, 4);
  CHECK(result.status == MGBFS_TOPOLOGY_PEER_UNSUPPORTED);
  CHECK(result.src_rank == 1 && result.dst_rank == 0);
  CHECK(result.src_device == 0 && result.dst_device == 3);

  api = FakeApi{}; api.no_atomic_src = 2; api.no_atomic_dst = 1;
  result = run(api, ordered, 4);
  CHECK(result.status == MGBFS_TOPOLOGY_NATIVE_ATOMICS_UNSUPPORTED);
  CHECK(result.src_rank == 2 && result.dst_rank == 3);
  CHECK(result.src_device == 2 && result.dst_device == 1);

  api = FakeApi{};
  api.failing_stage = MGBFS_TOPOLOGY_STAGE_NATIVE_ATOMICS;
  api.failed_src = 3; api.failed_dst = 2; api.fail_code = 777;
  result = run(api, ordered, 4);
  CHECK(result.status == MGBFS_TOPOLOGY_CUDA_ERROR);
  CHECK(result.stage == MGBFS_TOPOLOGY_STAGE_NATIVE_ATOMICS);
  CHECK(result.api_code == 777 && result.src_rank == 0 && result.dst_rank == 2);

  api = FakeApi{};
  api.failing_stage = MGBFS_TOPOLOGY_STAGE_NCCL_VERSION; api.fail_code = 19;
  result = run(api, ordered, 4);
  CHECK(result.status == MGBFS_TOPOLOGY_NCCL_ERROR);
  CHECK(result.stage == MGBFS_TOPOLOGY_STAGE_NCCL_VERSION && result.api_code == 19);

  for (int stage : {MGBFS_TOPOLOGY_STAGE_DEVICE_COUNT,
                    MGBFS_TOPOLOGY_STAGE_RUNTIME_VERSION,
                    MGBFS_TOPOLOGY_STAGE_DRIVER_VERSION}) {
    api = FakeApi{}; api.failing_stage = stage; api.fail_code = 123;
    result = run(api, ordered, 4);
    CHECK(result.status == MGBFS_TOPOLOGY_CUDA_ERROR);
    CHECK(result.stage == static_cast<unsigned>(stage) && result.api_code == 123);
  }
  api = FakeApi{};
  api.failing_stage = MGBFS_TOPOLOGY_STAGE_COOPERATIVE; api.failed_src = 0;
  result = run(api, ordered, 4);
  CHECK(result.status == MGBFS_TOPOLOGY_CUDA_ERROR);
  CHECK(result.src_rank == 1 && result.src_device == 0 && result.api_code == 71);
  api = FakeApi{};
  api.failing_stage = MGBFS_TOPOLOGY_STAGE_PEER_ACCESS;
  api.failed_src = 2; api.failed_dst = 0;
  result = run(api, ordered, 4);
  CHECK(result.status == MGBFS_TOPOLOGY_CUDA_ERROR);
  CHECK(result.stage == MGBFS_TOPOLOGY_STAGE_PEER_ACCESS);
  CHECK(result.src_rank == 2 && result.dst_rank == 1 && result.api_code == 71);

  api = FakeApi{}; api.count = 1;
  result = run(api, ordered, 4);
  CHECK(result.status == MGBFS_TOPOLOGY_INVALID_ARGUMENT);
  struct ResultAndCanary { MgbfsDeviceTopologyV1 out; uint32_t canary; } extended{};
  extended.canary = 0xfeedbabeu;
  api = FakeApi{};
  CHECK(mgbfs_device_topology_probe_v1(api, ordered, 4, sizeof(extended), &extended.out) == 0);
  CHECK(extended.canary == 0xfeedbabeu);
  std::puts("device-topology CPU decision tests passed");
  return 0;
}
