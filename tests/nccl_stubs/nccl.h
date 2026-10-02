#pragma once
#include <cstddef>
using ncclComm_t = void*;
using ncclResult_t = int;
struct ncclUniqueId { char bytes[128]; };
constexpr int ncclSuccess = 0, ncclUint8 = 1, ncclUint32 = 2, ncclMax = 3;
constexpr int ncclInvalidUsage = 5;
constexpr int ncclInProgress = 7;
struct ncclConfig_t { int blocking = 1; };
#define NCCL_CONFIG_INITIALIZER ncclConfig_t{}
inline const char* ncclGetErrorString(int) { return "injected NCCL error"; }
extern int fail_stage, group_depth, send_calls, recv_calls, end_calls;
extern int abort_calls, destroy_calls, finalize_calls;
extern int async_state, async_query_status;
extern int async_pending_queries;
extern int group_end_override;
extern int init_blocking, init_calls;
extern int last_send_peer, last_recv_peer;
inline int ncclCommGetAsyncError(ncclComm_t, ncclResult_t* state) {
  *state = async_state;
  if(async_pending_queries > 0 && --async_pending_queries == 0) async_state = ncclSuccess;
  return async_query_status;
}
inline int ncclGetUniqueId(ncclUniqueId*) { return 0; }
inline int ncclCommInitRank(ncclComm_t* p, int, ncclUniqueId, int) { *p = reinterpret_cast<void*>(1); return 0; }
inline int ncclCommInitRankConfig(ncclComm_t* p, int, ncclUniqueId, int,
    ncclConfig_t* config) {
  ++init_calls;
  init_blocking = config->blocking;
  *p = reinterpret_cast<void*>(1);
  return async_state == ncclInProgress ? ncclInProgress : ncclSuccess;
}
inline int ncclCommDestroy(ncclComm_t) { ++destroy_calls; return 0; }
inline int ncclCommFinalize(ncclComm_t) { ++finalize_calls; return 0; }
inline int ncclCommAbort(ncclComm_t) { ++abort_calls; return 0; }
inline int ncclGroupStart() { if (fail_stage == 1) return 1; ++group_depth; return 0; }
inline int ncclSend(const void*, std::size_t, int, int peer, ncclComm_t, void*) { ++send_calls; last_send_peer = peer; return fail_stage == 2; }
inline int ncclRecv(void*, std::size_t, int, int peer, ncclComm_t, void*) { ++recv_calls; last_recv_peer = peer; return fail_stage == 3; }
inline int ncclGroupEnd() { ++end_calls; --group_depth; return group_end_override >= 0 ? group_end_override : fail_stage == 4; }
inline int ncclAllGather(const void*, void*, std::size_t, int, ncclComm_t, void*) { return 0; }
inline int ncclAllReduce(const void*, void*, std::size_t, int, int, ncclComm_t, void*) { return 0; }
