#pragma once
#include <cstddef>
using ncclComm_t = void*;
using ncclResult_t = int;
struct ncclUniqueId { char bytes[128]; };
constexpr int ncclSuccess = 0, ncclUint8 = 1, ncclUint32 = 2, ncclMax = 3;
inline const char* ncclGetErrorString(int) { return "injected NCCL error"; }
extern int fail_stage, group_depth, send_calls, recv_calls, end_calls;
inline int ncclGetUniqueId(ncclUniqueId*) { return 0; }
inline int ncclCommInitRank(ncclComm_t* p, int, ncclUniqueId, int) { *p = reinterpret_cast<void*>(1); return 0; }
inline int ncclCommDestroy(ncclComm_t) { return 0; }
inline int ncclGroupStart() { if (fail_stage == 1) return 1; ++group_depth; return 0; }
inline int ncclSend(const void*, std::size_t, int, int, ncclComm_t, void*) { ++send_calls; return fail_stage == 2; }
inline int ncclRecv(void*, std::size_t, int, int, ncclComm_t, void*) { ++recv_calls; return fail_stage == 3; }
inline int ncclGroupEnd() { ++end_calls; --group_depth; return fail_stage == 4; }
inline int ncclAllGather(const void*, void*, std::size_t, int, ncclComm_t, void*) { return 0; }
inline int ncclAllReduce(const void*, void*, std::size_t, int, int, ncclComm_t, void*) { return 0; }
