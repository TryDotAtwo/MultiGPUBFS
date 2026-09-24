// Diagnostic only: two T4 ranks, one missing collective, all-rank abort.
// Characterizes the NCCL 2.29 nonblocking failure boundary independently of BFS.
#include <cuda_runtime_api.h>
#include <nccl.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

using Clock = std::chrono::steady_clock;

static bool ready(ncclComm_t comm, Clock::time_point deadline) {
  while (Clock::now() < deadline) {
    ncclResult_t state = ncclSuccess;
    if (ncclCommGetAsyncError(comm, &state) != ncclSuccess) return false;
    if (state == ncclSuccess) return true;
    if (state != ncclInProgress) return false;
    std::this_thread::yield();
  }
  return false;
}

int main() {
  ncclUniqueId id{};
  if (ncclGetUniqueId(&id) != ncclSuccess) return 2;
  std::atomic<bool> rank1_issued{false};
  std::atomic<bool> fatal{false};
  std::array<int, 2> result{3, 3};
  std::array<std::thread, 2> ranks;
  const auto deadline = Clock::now() + std::chrono::seconds(15);
  for (int rank = 0; rank < 2; ++rank) {
    ranks[rank] = std::thread([&, rank] {
      if (cudaSetDevice(rank) != cudaSuccess) {
        result[rank] = 4;
        return;
      }
      ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
      config.blocking = 0;
      ncclComm_t comm{};
      const auto init = ncclCommInitRankConfig(&comm, 2, id, rank, &config);
      if ((init != ncclSuccess && init != ncclInProgress) || !comm ||
          !ready(comm, deadline)) {
        std::fprintf(stderr, "rank=%d stage=init result=%s\n", rank,
                     ncclGetErrorString(init));
        if (comm) ncclCommAbort(comm);
        result[rank] = 5;
        return;
      }
      cudaStream_t stream{};
      unsigned int *send{}, *receive{};
      if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
          cudaMalloc(&send, sizeof(*send)) != cudaSuccess ||
          cudaMalloc(&receive, sizeof(*receive)) != cudaSuccess ||
          cudaMemsetAsync(send, rank + 1, sizeof(*send), stream) != cudaSuccess) {
        ncclCommAbort(comm);
        result[rank] = 6;
        return;
      }
      if (rank == 1) {
        const auto call = ncclAllReduce(send, receive, 1, ncclUint32,
                                        ncclMax, comm, stream);
        if (call != ncclSuccess && call != ncclInProgress) {
          std::fprintf(stderr, "rank=1 stage=collective result=%s\n",
                       ncclGetErrorString(call));
          ncclCommAbort(comm);
          result[rank] = 7;
          return;
        }
        rank1_issued.store(true, std::memory_order_release);
        while (!fatal.load(std::memory_order_acquire) && Clock::now() < deadline)
          std::this_thread::yield();
      } else {
        while (!rank1_issued.load(std::memory_order_acquire) && Clock::now() < deadline)
          std::this_thread::yield();
        fatal.store(true, std::memory_order_release);
      }
      const auto abort = ncclCommAbort(comm);
      if (abort != ncclSuccess || Clock::now() >= deadline) {
        std::fprintf(stderr, "rank=%d stage=abort result=%s\n", rank,
                     ncclGetErrorString(abort));
        result[rank] = 8;
        return;
      }
      std::fprintf(stderr, "rank=%d stage=abort result=PASS\n", rank);
      result[rank] = 0;
    });
  }
  for (auto& rank : ranks) rank.join();
  return result[0] ? result[0] : result[1];
}
