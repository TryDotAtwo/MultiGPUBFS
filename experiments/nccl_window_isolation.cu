// Diagnostic only: isolate NCCL window registration from MultiGPUBFS.
#include <cuda_runtime.h>
#include <nccl.h>

#include <array>
#include <cstdio>
#include <thread>

int main() {
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
      auto nccl = ncclCommInitRank(&comm, 2, id, rank);
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
      if (nccl != ncclSuccess) {
        std::fprintf(stderr, "rank=%d stage=window_register nccl=%s last=%s\n",
                     rank, ncclGetErrorString(nccl), ncclGetLastError(comm));
        results[rank] = 7;
        ncclCommAbort(comm);
        ncclMemFree(memory);
        return;
      }
      std::fprintf(stderr, "rank=%d stage=window_register result=PASS\n", rank);
      ncclCommWindowDeregister(comm, window);
      ncclMemFree(memory);
      ncclCommDestroy(comm);
    });
  }
  for (auto& rank : ranks) rank.join();
  return results[0] != 0 ? results[0] : results[1];
}
