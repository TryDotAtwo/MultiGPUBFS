// Diagnostic only: isolate NCCL window registration from MultiGPUBFS.
#include <cuda_runtime_api.h>
#include <nccl.h>
#include "nccl_probe_teardown.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <cstdlib>
#include <fstream>
#include <string>
#ifdef MGBFS_WINDOW_DEVICE_PROBE
#include <cuda_runtime.h>
#include <nccl_device.h>
#include <atomic>
__global__ void sample_window(ncclDevComm dev,ncclWindow_t window,int rank) {
  if(threadIdx.x || blockIdx.x)return;
  auto* own=static_cast<volatile uint32_t*>(ncclGetLsaPointer(window,0,dev.lsaRank));
  auto* peer=static_cast<volatile uint32_t*>(ncclGetLsaPointer(window,0,1-rank));
  own[1]=peer[0];
  peer[2]=100+rank;
  __threadfence_system();
}
#endif

int main(int argc, char** argv) {
  const bool zero_resources = argc == 2 && std::strcmp(argv[1], "device_comm_zero") == 0;
  const bool device_comm_only = zero_resources || (argc == 2 && std::strcmp(argv[1], "device_comm_only") == 0);
  const bool device_comm = device_comm_only || (argc == 2 && std::strcmp(argv[1], "device_comm") == 0);
  const bool nonblocking = device_comm || (argc == 2 && std::strcmp(argv[1], "nonblocking") == 0);
  const bool device_probe = argc == 2 && std::strcmp(argv[1], "read_write") == 0;
  if (argc > 2 || (argc == 2 && !nonblocking && !device_probe)) return 1;
  const char* rank_text=std::getenv("MGBFS_WINDOW_RANK");
  const char* bootstrap=std::getenv("MGBFS_WINDOW_BOOTSTRAP");
  int process_rank=-1;
  if(bool(rank_text)!=bool(bootstrap))return 1;
  if(rank_text) {
    if(std::strcmp(rank_text,"0")!=0&&std::strcmp(rank_text,"1")!=0)return 1;
    if(!*bootstrap||device_probe)return 1; // GPU sample uses thread-local rendezvous.
    process_rank=rank_text[0]-'0';
  }
#ifndef MGBFS_WINDOW_DEVICE_PROBE
  if(device_probe||device_comm)return 1;
#else
  std::atomic<int> arrived[2]{};
  std::atomic<bool> finished[2]{};
  auto rendezvous=[&](int rank,int phase) {
    arrived[rank].store(phase,std::memory_order_release);
    const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    while(arrived[1-rank].load(std::memory_order_acquire)<phase) {
      if(finished[1-rank].load(std::memory_order_acquire)||std::chrono::steady_clock::now()>end)return false;
      std::this_thread::yield();
    }
    return true;
  };
#endif
  ncclUniqueId id{};
  if(process_rank<0||process_rank==0) {
    if(ncclGetUniqueId(&id)!=ncclSuccess)return 2;
    if(process_rank==0) {
      // Diagnostic callers supply a fresh bootstrap path. Publish atomically,
      // so rank 1 never observes a partial unique ID.
      if(std::ifstream(bootstrap).good())return 2;
      const std::string tmp=std::string(bootstrap)+".tmp";
      std::ofstream out(tmp,std::ios::binary);
      out.write(reinterpret_cast<const char*>(&id),sizeof(id));
      out.close();
      if(!out||std::rename(tmp.c_str(),bootstrap)!=0)return 2;
    }
  } else {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    bool ready=false;
    while(std::chrono::steady_clock::now()<deadline) {
      std::ifstream in(bootstrap,std::ios::binary);
      if(in.good()) {
        in.read(reinterpret_cast<char*>(&id),sizeof(id));
        if(in.gcount()!=sizeof(id)||in.peek()!=EOF)return 2;
        ready=true;break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if(!ready)return 2;
  }
  std::array<int, 2> results{};
  std::array<std::thread, 2> ranks;
  for (int rank = 0; rank < 2; ++rank) {
    if(process_rank>=0&&rank!=process_rank)continue;
    ranks[rank] = std::thread([&, rank] {
#ifdef MGBFS_WINDOW_DEVICE_PROBE
      struct Exit {std::atomic<bool>& bit;~Exit(){bit.store(true,std::memory_order_release);}} exit{finished[rank]};
#endif
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
      std::fprintf(stderr, "rank=%d stage=init_begin\n", rank);
      auto nccl = (nonblocking || device_probe)
                      ? ncclCommInitRankConfig(&comm, 2, id, rank, &config)
                      : ncclCommInitRank(&comm, 2, id, rank);
      auto progress = [&](ncclResult_t result) {
        if (result != ncclInProgress && result != ncclSuccess) return result;
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
      std::fprintf(stderr, "rank=%d stage=init_complete\n", rank);
#ifdef MGBFS_WINDOW_DEVICE_PROBE
      if (device_comm_only) {
        // Deliberately no ncclMemAlloc or WindowRegister: isolate internal
        // device-communicator activation from the user-window registration.
        ncclDevComm dev{};
        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        reqs.lsaBarrierCount = zero_resources ? 0 : 16;
        const char* stage = zero_resources ? "device_comm_zero_create" : "device_comm_only_create";
        std::fprintf(stderr, "rank=%d stage=%s_begin barriers=%d\n", rank, stage, int(reqs.lsaBarrierCount));
        nccl = progress(ncclDevCommCreate(comm, &reqs, &dev));
        if (nccl != ncclSuccess || dev.lsaSize != 2) {
          std::fprintf(stderr, "rank=%d stage=%s nccl=%d lsa_size=%d last=%s\n",
              rank, stage, int(nccl), dev.lsaSize, ncclGetLastError(comm));
          results[rank] = 12;
          ncclCommAbort(comm);
          return;
        }
        std::fprintf(stderr, "rank=%d stage=%s result=PASS\n", rank, stage);
        nccl = progress(ncclDevCommDestroy(comm, &dev));
        if (nccl != ncclSuccess) {
          results[rank] = 13;
          ncclCommAbort(comm);
          return;
        }
        nccl = nccl_probe_teardown([&]{return ncclCommFinalize(comm);},
            progress, [&]{return ncclCommDestroy(comm);});
        if (nccl != ncclSuccess) { results[rank] = 14; return; }
        std::fprintf(stderr, "rank=%d stage=teardown_complete\n", rank);
        return;
      }
#endif
      void* memory{};
      std::fprintf(stderr, "rank=%d stage=alloc_begin\n", rank);
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
      std::fprintf(stderr, "rank=%d stage=window_register_begin\n", rank);
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
#ifdef MGBFS_WINDOW_DEVICE_PROBE
      if(device_comm) {
        ncclDevComm dev{};
        ncclDevCommRequirements reqs=NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        reqs.lsaBarrierCount=16; // Same requirement as the production LSA path.
        std::fprintf(stderr,"rank=%d stage=device_comm_create_begin\n",rank);
        nccl=progress(ncclDevCommCreate(comm,&reqs,&dev));
        if(nccl!=ncclSuccess||dev.lsaSize!=2) {
          std::fprintf(stderr,"rank=%d stage=device_comm_create nccl=%d lsa_size=%d last=%s\n",
              rank,int(nccl),dev.lsaSize,ncclGetLastError(comm));
          results[rank]=12;ncclCommAbort(comm);return;
        }
        std::fprintf(stderr,"rank=%d stage=device_comm_create result=PASS\n",rank);
        nccl=progress(ncclDevCommDestroy(comm,&dev));
        if(nccl!=ncclSuccess){results[rank]=13;ncclCommAbort(comm);return;}
      }
      if(device_probe) {
        ncclDevComm dev{};
        ncclDevCommRequirements reqs=NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        reqs.lsaBarrierCount=1;
        nccl=progress(ncclDevCommCreate(comm,&reqs,&dev));
        uint32_t initial=11+rank;
        cuda=cudaMemcpy(memory,&initial,4,cudaMemcpyHostToDevice);
        std::fprintf(stderr,"rank=%d stage=device_prepare nccl=%d cuda=%d lsa_size=%d\n",
            rank,int(nccl),int(cuda),dev.lsaSize);
        if(nccl!=ncclSuccess||cuda!=cudaSuccess
            ||!rendezvous(rank,1)){results[rank]=9;ncclCommAbort(comm);return;}
        sample_window<<<1,32>>>(dev,window,rank);
        cuda=cudaDeviceSynchronize();
        std::fprintf(stderr,"rank=%d stage=sample_complete cuda=%d\n",rank,int(cuda));
        if(cuda!=cudaSuccess||!rendezvous(rank,2)) {
          results[rank]=10;ncclCommAbort(comm);return;
        }
        uint32_t observed[3]{};
        cuda=cudaMemcpy(observed,memory,sizeof(observed),cudaMemcpyDeviceToHost);
        std::fprintf(stderr,"rank=%d stage=read_write cuda=%d local=%u peer_read=%u peer_write=%u\n",
            rank,int(cuda),observed[0],observed[1],observed[2]);
        if(cuda!=cudaSuccess||observed[0]!=uint32_t(11+rank)||
            observed[1]!=uint32_t(12-rank)||observed[2]!=uint32_t(101-rank))results[rank]=11;
        ncclDevCommDestroy(comm,&dev);
      }
#endif
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
      nccl = nccl_probe_teardown([&]{return ncclCommFinalize(comm);},
          progress, [&]{return ncclCommDestroy(comm);});
      if(nccl!=ncclSuccess){results[rank]=14;return;}
      std::fprintf(stderr, "rank=%d stage=teardown_complete\n", rank);
    });
  }
  for (auto& rank : ranks) if(rank.joinable())rank.join();
  return results[0] != 0 ? results[0] : results[1];
}
