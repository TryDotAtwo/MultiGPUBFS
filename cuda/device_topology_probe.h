#ifndef MGBFS_DEVICE_TOPOLOGY_PROBE_H
#define MGBFS_DEVICE_TOPOLOGY_PROBE_H

#include "device_topology_v1.h"
#include <cstring>

/* Stateless decision procedure shared by native queries and injected tests.
 * Api returns 0 on success and the original CUDA/NCCL code on failure. */
template<class Api>
int mgbfs_device_topology_probe_v1(Api& api, const int32_t* devices,
    uint32_t count, uint32_t out_size, MgbfsDeviceTopologyV1* out) {
  if (!out || out_size < sizeof(*out)) return -1;
  std::memset(out, 0, sizeof(*out));
  out->abi_version = 1;
  out->struct_size = sizeof(*out);
  out->src_rank = out->dst_rank = out->src_device = out->dst_device = -1;
  out->rank_count = count;
  if (!devices || !count || count > MGBFS_DEVICE_TOPOLOGY_MAX_RANKS) {
    out->status = MGBFS_TOPOLOGY_INVALID_ARGUMENT;
    return 0;
  }
  int visible = 0;
  auto cuda_error = [&](uint32_t stage, int code) {
    out->status = MGBFS_TOPOLOGY_CUDA_ERROR;
    out->stage = stage;
    out->api_code = code;
  };
  auto rank = [&](uint32_t src) {
    out->src_rank = static_cast<int32_t>(src);
    out->src_device = devices[src];
  };
  auto pair = [&](uint32_t src, uint32_t dst) {
    rank(src);
    out->dst_rank = static_cast<int32_t>(dst);
    out->dst_device = devices[dst];
  };
  int error = api.device_count(&visible);
  if (error) { cuda_error(MGBFS_TOPOLOGY_STAGE_DEVICE_COUNT, error); return 0; }
  if (visible < 0 || count > static_cast<uint32_t>(visible)) {
    out->status = MGBFS_TOPOLOGY_INVALID_ARGUMENT;
    return 0;
  }
  for (uint32_t i = 0; i < count; ++i) {
    if (devices[i] < 0 || devices[i] >= visible) {
      out->status = MGBFS_TOPOLOGY_INVALID_DEVICE; rank(i); return 0;
    }
    for (uint32_t j = 0; j < i; ++j) {
      if (devices[i] == devices[j]) {
        out->status = MGBFS_TOPOLOGY_DUPLICATE_DEVICE; pair(j, i); return 0;
      }
    }
  }
  int version = 0;
  error = api.runtime_version(&version);
  if (error) { cuda_error(MGBFS_TOPOLOGY_STAGE_RUNTIME_VERSION, error); return 0; }
  out->cuda_runtime_version = version;
  error = api.driver_version(&version);
  if (error) { cuda_error(MGBFS_TOPOLOGY_STAGE_DRIVER_VERSION, error); return 0; }
  out->cuda_driver_version = version;
  error = api.nccl_version(&version);
  if (error) {
    out->status = MGBFS_TOPOLOGY_NCCL_ERROR;
    out->stage = MGBFS_TOPOLOGY_STAGE_NCCL_VERSION;
    out->api_code = error;
    return 0;
  }
  out->nccl_version = version;
  out->lsa_compiled = api.lsa_compiled() ? 1u : 0u;
  if (!out->lsa_compiled) { out->status = MGBFS_TOPOLOGY_LSA_NOT_COMPILED; return 0; }
  if (version < 22900) { out->status = MGBFS_TOPOLOGY_NCCL_TOO_OLD; return 0; }
  for (uint32_t i = 0; i < count; ++i) {
    int supported = 0;
    error = api.cooperative(devices[i], &supported);
    if (error) {
      cuda_error(MGBFS_TOPOLOGY_STAGE_COOPERATIVE, error); rank(i); return 0;
    }
    if (!supported) {
      out->status = MGBFS_TOPOLOGY_COOPERATIVE_UNSUPPORTED;
      out->stage = MGBFS_TOPOLOGY_STAGE_COOPERATIVE;
      rank(i); return 0;
    }
  }
  for (uint32_t i = 0; i < count; ++i) {
    for (uint32_t j = 0; j < count; ++j) {
      if (i == j) continue;
      int supported = 0;
      error = api.can_access_peer(devices[i], devices[j], &supported);
      if (error) {
        cuda_error(MGBFS_TOPOLOGY_STAGE_PEER_ACCESS, error); pair(i, j); return 0;
      }
      if (!supported) {
        out->status = MGBFS_TOPOLOGY_PEER_UNSUPPORTED;
        out->stage = MGBFS_TOPOLOGY_STAGE_PEER_ACCESS; pair(i, j); return 0;
      }
      error = api.native_atomics(devices[i], devices[j], &supported);
      if (error) {
        cuda_error(MGBFS_TOPOLOGY_STAGE_NATIVE_ATOMICS, error); pair(i, j); return 0;
      }
      if (!supported) {
        out->status = MGBFS_TOPOLOGY_NATIVE_ATOMICS_UNSUPPORTED;
        out->stage = MGBFS_TOPOLOGY_STAGE_NATIVE_ATOMICS; pair(i, j); return 0;
      }
    }
  }
  out->status = MGBFS_TOPOLOGY_PRELIMINARY_OK;
  return 0;
}
#endif
