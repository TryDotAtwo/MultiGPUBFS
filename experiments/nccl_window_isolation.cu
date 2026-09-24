// Diagnostic only: isolate NCCL window registration from MultiGPUBFS.
#include <cuda_runtime_api.h>
#include <nccl.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

int main(int argc, char** argv) {
  const bool nonblocking = argc == 2 && std::strcmp(argv[1], "nonblocking") == 0;
  if (argc > 2 || (argc == 2 && !nonblocking)) return 1;
  ncclUniqueId id{};
  if (ncclGetUniqueId(&id) != ncclSuccess) return 2;
  std::array<int, 2> results{};
  std::array<std::thread, 2> ranks;
  for (int rank = 0; rank < 2; ++rank) {
    ranks[rank] = std::thread([&, rank] {
      auto cuda = cudaSetDevice(rank);
      if (cuda != cudaSuccess) {
        std::fprintf(stderr, "rank=%d stage=set_device cuda=%s\n", rank,
                     cudaGetErrorString(cuda));
        results[rank] = 3;
        return;
      }
      ncclComm_t comm{};
      ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
      config.blocking = 0;
      auto nccl = nonblocking
                      ? ncclCommInitRankConfig(&comm, 2, id, rank, &config)
                      : ncclCommInitRank(&comm, 2, id, rank);
      auto progress = [&](ncclResult_t result) {
        if (result != ncclInProgress) return result;
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::seconds(30);
        while (std::chrono::steady_clock::now() < deadline) {
          ncclResult_t state = ncclSuccess;
          const auto poll = ncclCommGetAsyncError(comm, &state);
          if (poll != ncclSuccess) return poll;
          if (state != ncclInProgress) return state;
          std::this_thread::yield();
        }
        return ncclSystemError;
      };
      nccl = comm ? progress(nccl) : nccl;
      if (nccl != ncclSuccess) {
        std::fprintf(stderr, "rank=%d stage=init nccl=%s\n", rank,
                     ncclGetErrorString(nccl));
        results[rank] = 4;
        return;
      }
      void* memory{};
      nccl = ncclMemAlloc(&memory, 4096);
      if (nccl != ncclSuccess) {
        std::fprintf(stderr, "rank=%d stage=alloc nccl=%s\n", rank,
                     ncclGetErrorString(nccl));
        results[rank] = 5;
        ncclCommAbort(comm);
        return;
      }
      cuda = cudaMemset(memory, 0, 4096);
      if (cuda != cudaSuccess) {
        std::fprintf(stderr, "rank=%d stage=memset cuda=%s\n", rank,
                     cudaGetErrorString(cuda));
        results[rank] = 6;
        ncclCommAbort(comm);
        ncclMemFree(memory);
        return;
      }
      ncclWindow_t window{};
      nccl = ncclCommWindowRegister(comm, memory, 4096, &window,
                                    NCCL_WIN_COLL_SYMMETRIC);
      nccl = progress(nccl);
      if (nccl != ncclSuccess) {
        std::fprintf(stderr, "rank=%d stage=window_register nccl=%s last=%s\n",
                     rank, ncclGetErrorString(nccl), ncclGetLastError(comm));
        results[rank] = 7;
        ncclCommAbort(comm);
        ncclMemFree(memory);
        return;
      }
      std::fprintf(stderr, "rank=%d mode=%s stage=window_register result=PASS\n",
                   rank, nonblocking ? "nonblocking" : "blocking");
      nccl = progress(ncclCommWindowDeregister(comm, window));
      if (nccl != ncclSuccess) {
        std::fprintf(stderr, "rank=%d stage=window_deregister nccl=%s\n", rank,
                     ncclGetErrorString(nccl));
        results[rank] = 8;
        ncclCommAbort(comm);
        ncclMemFree(memory);
        return;
      }
      ncclMemFree(memory);
      ncclCommDestroy(comm);
    });
  }
  for (auto& rank : ranks) rank.join();
  return results[0] != 0 ? results[0] : results[1];
}
