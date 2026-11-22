#ifndef MGBFS_DEVICE_TOPOLOGY_V1_H
#define MGBFS_DEVICE_TOPOLOGY_V1_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Current resident-transport bootstrap is designed for at most eight ranks.
 * This does not limit the existing host-driven transport or a future ABI. */
#define MGBFS_DEVICE_TOPOLOGY_MAX_RANKS 8u

typedef enum MgbfsDeviceTopologyStatusV1 {
  /* Preliminary topology only: a communicator, registered window and
   * bidirectional data/ack smoke test are still required before use. */
  MGBFS_TOPOLOGY_PRELIMINARY_OK = 0,
  MGBFS_TOPOLOGY_INVALID_ARGUMENT = 1,
  MGBFS_TOPOLOGY_INVALID_DEVICE = 2,
  MGBFS_TOPOLOGY_DUPLICATE_DEVICE = 3,
  MGBFS_TOPOLOGY_CUDA_ERROR = 4,
  MGBFS_TOPOLOGY_NCCL_ERROR = 5,
  MGBFS_TOPOLOGY_LSA_NOT_COMPILED = 6,
  MGBFS_TOPOLOGY_NCCL_TOO_OLD = 7,
  MGBFS_TOPOLOGY_COOPERATIVE_UNSUPPORTED = 8,
  MGBFS_TOPOLOGY_PEER_UNSUPPORTED = 9,
  MGBFS_TOPOLOGY_NATIVE_ATOMICS_UNSUPPORTED = 10
} MgbfsDeviceTopologyStatusV1;

typedef enum MgbfsDeviceTopologyStageV1 {
  MGBFS_TOPOLOGY_STAGE_NONE = 0,
  MGBFS_TOPOLOGY_STAGE_DEVICE_COUNT = 1,
  MGBFS_TOPOLOGY_STAGE_RUNTIME_VERSION = 2,
  MGBFS_TOPOLOGY_STAGE_DRIVER_VERSION = 3,
  MGBFS_TOPOLOGY_STAGE_NCCL_VERSION = 4,
  MGBFS_TOPOLOGY_STAGE_COOPERATIVE = 5,
  MGBFS_TOPOLOGY_STAGE_PEER_ACCESS = 6,
  MGBFS_TOPOLOGY_STAGE_NATIVE_ATOMICS = 7
} MgbfsDeviceTopologyStageV1;

/* Fixed-width host ABI; device ordinals use the calling process's
 * CUDA_VISIBLE_DEVICES mapping. A rank/device of -1 denotes no such rank.
 * api_code is the unmodified CUDA or NCCL status when status says *_ERROR.
 * For unsupported connectivity, src/dst identify the directed failing pair.
 * No explicit CUDA allocation, kernel launch, communicator creation, or
 * peer enabling; CUDA and NCCL may initialize their own runtimes.
 */
typedef struct MgbfsDeviceTopologyV1 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t status;
  uint32_t stage;
  int32_t api_code;
  int32_t src_rank;
  int32_t dst_rank;
  int32_t src_device;
  int32_t dst_device;
  int32_t cuda_runtime_version;
  int32_t cuda_driver_version;
  int32_t nccl_version;
  uint32_t lsa_compiled;
  uint32_t rank_count;
} MgbfsDeviceTopologyV1;

/* 0 means out was populated, including for unsupported topology and CUDA/NCCL
 * errors. -1 means out is null or out_size is too small; no output is written.
 * The caller must supply an array containing rank_count int32_t ordinals;
 * it must not overlap the output buffer.
 * PRELIMINARY_OK is not a transport capability or successful launch.
 * The caller must agree on rank/device identity and validate the actual NCCL
 * communicator, its window, and a bidirectional smoke test before use. */
int mgbfs_device_topology_v1(const int32_t* devices, uint32_t rank_count,
                             uint32_t out_size, MgbfsDeviceTopologyV1* out);

#ifdef __cplusplus
}
static_assert(sizeof(MgbfsDeviceTopologyV1) == 56, "device topology V1 ABI");
#endif
#endif
