/* Physical multi-process LSA test for the packet primitive only, not BFS.
 * One producer and one consumer thread per rank exchange bounded SPSC rings
 * with self, an active peer, and empty peers, across two epochs and repeated
 * jobs. CPU validates the complete packet contents after the kernel ends. */
#include "device_packet_ring.cuh"
#include "device_topology_v1.h"
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace mgbfs_packet_ring;
#ifndef MGBFS_PACKET_WORLD
#define MGBFS_PACKET_WORLD 2
#endif
constexpr unsigned WORLD = MGBFS_PACKET_WORLD, CAP_MAX = 2, BYTES = 96, N = 13, PHASES = 2;
static_assert(WORLD >= 2 && WORLD <= 8, "test supports 2 to 8 ranks");
constexpr U64 ITER_LIMIT = 100000000ull;

#define CUDA_OK(x) do { auto e=(x); if(e!=cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); return 2; } } while(0)
#define NCCL_OK(x) do { auto e=(x); if(e!=ncclSuccess) { \
  std::fprintf(stderr, "%s: %s (%d)\n", #x, ncclGetErrorString(e), int(e)); return 3; } } while(0)

struct alignas(64) Mailbox {
  PacketSlot slots[WORLD][CAP_MAX];
  unsigned char payload[WORLD][CAP_MAX][BYTES];
  CloseSlot closes[WORLD];
};
struct Audit {
  unsigned present, phase, format, len;
  unsigned char bytes[BYTES];
};
struct Output {
  unsigned error, sent, received, closes_sent, closes_received;
  U64 blocked, closed_while_full;
  Audit records[WORLD][PHASES*N];
};
__host__ __device__ unsigned length_for(unsigned ticket) {
  switch (ticket % 5) { case 0: return 0; case 1: return 1; case 2: return 7;
    case 3: return 65; default: return BYTES; }
}
__host__ __device__ unsigned char value_for(unsigned src, unsigned dst, unsigned ticket, unsigned i) {
  return static_cast<unsigned char>((src*31u + dst*17u + ticket*7u + i*13u) & 255u);
}
__host__ __device__ unsigned format_for(unsigned src, unsigned dst, unsigned ticket) {
  return (src*3u + dst*7u + ticket) % 5u;
}
__host__ __device__ bool sends(unsigned src, unsigned dst, unsigned pattern) {
  return pattern == 1 || dst == src || dst == ((src + 1) % WORLD);
}
__device__ View view_for(Mailbox* box, unsigned dst, unsigned cap) {
  return View{box->slots[dst], &box->payload[dst][0][0], cap, BYTES};
}
__global__ void initialize(Mailbox* box) {
  if(blockIdx.x || threadIdx.x) return;
  for(unsigned dst=0;dst<WORLD;++dst) {
    for(unsigned i=0;i<CAP_MAX;++i) init_slot(box->slots[dst][i],i);
    init_close(box->closes[dst]);
  }
}
__device__ void fail(Output* output, unsigned error) { atomicCAS(&output->error, 0u, error); }

__global__ void exchange(ncclWindow_t window,ncclDevComm comm,unsigned rank,unsigned capacity,
                         unsigned pattern,U64 generation,Output* output) {
  __shared__ Mailbox* boxes[WORLD];
  __shared__ unsigned local_error;
  if(threadIdx.x == 0) {
    local_error = 0;
    for(unsigned i=0;i<WORLD;++i) {
      boxes[i] = static_cast<Mailbox*>(ncclGetLsaPointer(window,0,i));
      if(!boxes[i]) local_error = 1;
    }
    if(comm.lsaRank != int(rank) || comm.lsaSize != int(WORLD)) local_error = 2;
    if(local_error) fail(output, local_error);
  }
  __syncthreads();
  if(local_error) return;
  Mailbox* local = boxes[rank];
  U64 sent[WORLD] = {}, seen[WORLD] = {};
  for(unsigned phase=0;phase<PHASES;++phase) {
    if(threadIdx.x == 0) {
      bool closed[WORLD] = {};
      U64 iterations = 0;
      unsigned closed_count = 0;
      while(closed_count < WORLD && atomicAdd(&output->error,0u)==0u && ++iterations < ITER_LIMIT) {
        for(unsigned dst=0;dst<WORLD;++dst) {
          const U64 goal = sends(rank,dst,pattern) ? U64((phase+1)*N) : 0;
          if(sent[dst] < goal) {
            const unsigned ticket = unsigned(sent[dst]), len = length_for(ticket);
            unsigned char bytes[BYTES];
            for(unsigned i=0;i<len;++i) bytes[i] = value_for(rank,dst,ticket,i);
            auto r=try_send(view_for(local,dst,capacity),sent[dst],generation,phase,
                            format_for(rank,dst,ticket),bytes,len);
            if(r==Result::ok) { ++sent[dst]; ++output->sent; }
            else if(r==Result::not_ready) ++output->blocked;
            else { fail(output,100u+unsigned(r)); break; }
          }
          if(sent[dst] == goal && !closed[dst]) {
            auto r=try_close(local->closes[dst],phase,generation,phase,sent[dst]);
            if(r==Result::ok) {
              if(goal) {
                PacketSlot& slot=local->slots[dst][sent[dst] % capacity];
                if(SystemAtomic(slot.sequence).load(cuda::memory_order_acquire)!=2*sent[dst])
                  ++output->closed_while_full;
              }
              closed[dst]=true; ++closed_count; ++output->closes_sent;
            } else if(r!=Result::not_ready) { fail(output,200u+unsigned(r)); break; }
          }
        }
      }
      if(iterations >= ITER_LIMIT) fail(output,301);
      // Keep the close lease until every receiver has acknowledged its final
      // durable packet. This does not occupy a data slot.
      iterations = 0;
      bool acknowledged = false;
      while(!acknowledged && atomicAdd(&output->error,0u)==0u && ++iterations < ITER_LIMIT) {
        acknowledged = true;
        for(unsigned dst=0;dst<WORLD;++dst)
          if(SystemAtomic(local->closes[dst].sequence).load(cuda::memory_order_acquire)
               != 2*U64(phase+1)) acknowledged = false;
      }
      if(iterations >= ITER_LIMIT) fail(output,302);
    }
    if(threadIdx.x == 32) {
      bool close_seen[WORLD] = {}, complete[WORLD] = {};
      U64 tail[WORLD] = {};
      unsigned complete_count = 0;
      U64 iterations = 0;
      // Rank 1 deliberately delays consumption to exercise bounded queues.
      if(rank == 1) for(volatile unsigned n=0;n<10000;++n) {}
      while(complete_count < WORLD && atomicAdd(&output->error,0u)==0u && ++iterations < ITER_LIMIT) {
        for(unsigned src=0;src<WORLD;++src) {
          if(complete[src]) continue;
          Mailbox* remote=boxes[src];
          if(!close_seen[src]) {
            auto r=try_get_close(remote->closes[rank],phase,generation,phase,&tail[src]);
            if(r==Result::ok) close_seen[src]=true;
            else if(r!=Result::not_ready && r!=Result::future_phase) { fail(output,400u+unsigned(r)); break; }
          }
          if(close_seen[src] && seen[src]==tail[src]) {
            auto r=try_ack_close(remote->closes[rank],phase,generation,phase,seen[src]);
            if(r!=Result::ok) { fail(output,500u+unsigned(r)); break; }
            complete[src]=true; ++complete_count; ++output->closes_received;
            continue;
          }
          if(close_seen[src] && seen[src]>tail[src]) { fail(output,601); break; }
          PacketHeader header{}; const unsigned char* bytes=nullptr;
          auto r=try_peek(view_for(remote,rank,capacity),seen[src],generation,phase,&header,&bytes);
          if(r==Result::ok) {
            const unsigned ticket=unsigned(seen[src]);
            if(ticket>=PHASES*N || !sends(src,rank,pattern) || header.len!=length_for(ticket) ||
               header.format!=format_for(src,rank,ticket)) { fail(output,602); break; }
            Audit& record=output->records[src][ticket];
            record.present=1; record.phase=phase; record.format=header.format; record.len=header.len;
            for(unsigned i=0;i<header.len;++i) record.bytes[i]=bytes[i];
            // The full packet has been copied to durable output before ACK.
            r=try_ack(view_for(remote,rank,capacity),seen[src],generation,phase);
            if(r!=Result::ok) { fail(output,700u+unsigned(r)); break; }
            ++seen[src]; ++output->received;
          } else if(r!=Result::not_ready && r!=Result::future_phase) { fail(output,800u+unsigned(r)); break; }
        }
      }
      if(iterations >= ITER_LIMIT) fail(output,303);
    }
    __syncthreads();
    if(threadIdx.x==0) local_error=output->error;
    __syncthreads();
    if(local_error) break;
  }
}
static std::string encode(const ncclUniqueId& id) {
  const char* digits="0123456789abcdef"; std::string out;
  for(unsigned char c:id.internal) { out.push_back(digits[c>>4]); out.push_back(digits[c&15]); }
  return out;
}
static bool decode(const char* text,ncclUniqueId& id) {
  if(std::strlen(text)!=2*sizeof(id.internal)) return false;
  auto digit=[](char c) { return c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:-1; };
  for(size_t i=0;i<sizeof(id.internal);++i) {
    int a=digit(text[2*i]), b=digit(text[2*i+1]);
    if(a<0||b<0)return false;
    id.internal[i]=char(a*16+b);
  }
  return true;
}
static int rank_main(int rank,const ncclUniqueId& id) {
  int count=0; auto count_status=cudaGetDeviceCount(&count);
  if(count_status==cudaErrorNoDevice) return 77;
  if(count_status!=cudaSuccess) {
    std::fprintf(stderr,"cudaGetDeviceCount: %s\n",cudaGetErrorString(count_status));return 2;
  }
  if(count<int(WORLD)) return 77;
  const bool swap=std::getenv("MGBFS_PACKET_SWAP")!=nullptr;
  int32_t devices[WORLD];
  for(unsigned r=0;r<WORLD;++r) devices[r]=swap?int(WORLD-1-r):int(r);
  MgbfsDeviceTopologyV1 topology{};
  if(mgbfs_device_topology_v1(devices,WORLD,sizeof(topology),&topology)!=0 ||
     topology.status!=MGBFS_TOPOLOGY_PRELIMINARY_OK) {
    std::fprintf(stderr,"rank=%d unsupported topology status=%u src=%d dst=%d\n",
                 rank,topology.status,topology.src_device,topology.dst_device);
    return 77;
  }
  CUDA_OK(cudaSetDevice(devices[rank]));
  for(unsigned peer=0;peer<WORLD;++peer) if(peer!=unsigned(rank)) {
    auto status=cudaDeviceEnablePeerAccess(devices[peer],0);
    if(status==cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
    else if(status!=cudaSuccess) return 2;
  }
  ncclComm_t comm{}; ncclWindow_t window{}; ncclDevComm device_comm{};
  ncclCommProperties_t properties=NCCL_COMM_PROPERTIES_INITIALIZER;
  Mailbox* box=nullptr; Output* output=nullptr; int* vote=nullptr; cudaStream_t stream{};
  NCCL_OK(ncclCommInitRank(&comm,WORLD,id,rank));
  NCCL_OK(ncclCommQueryProperties(comm,&properties));
  if(!properties.deviceApiSupport || properties.nLsaTeams!=1) return 77;
  NCCL_OK(ncclMemAlloc(reinterpret_cast<void**>(&box),sizeof(Mailbox)));
  CUDA_OK(cudaMalloc(&output,sizeof(Output))); CUDA_OK(cudaMalloc(&vote,sizeof(int)));
  CUDA_OK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  NCCL_OK(ncclCommWindowRegister(comm,box,sizeof(Mailbox),&window,NCCL_WIN_COLL_SYMMETRIC));
  ncclDevCommRequirements req=NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER; req.lsaBarrierCount=1;
  NCCL_OK(ncclDevCommCreate(comm,&req,&device_comm));
  bool all_ok=true;
  for(unsigned capacity : {1u,2u}) {
   for(unsigned trial=0;trial<2;++trial) {
    U64 generation=100+capacity*10+trial;
    unsigned pattern=trial;
    initialize<<<1,1,0,stream>>>(box);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemsetAsync(output,0,sizeof(Output),stream));
    CUDA_OK(cudaMemsetAsync(vote,0,sizeof(int),stream));
    NCCL_OK(ncclAllReduce(vote,vote,1,ncclInt,ncclSum,comm,stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    unsigned urank=rank;
    void* args[]={&window,&device_comm,&urank,&capacity,&pattern,&generation,&output};
    CUDA_OK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(exchange),dim3(1),dim3(96),args,0,stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    Output got{}; CUDA_OK(cudaMemcpy(&got,output,sizeof(got),cudaMemcpyDeviceToHost));
    unsigned active_peers=pattern?WORLD:2u;
    bool ok=got.error==0 && got.sent==active_peers*N*PHASES &&
            got.received==active_peers*N*PHASES &&
            got.closes_sent==WORLD*PHASES && got.closes_received==WORLD*PHASES;
    for(unsigned src=0;src<WORLD;++src) for(unsigned ticket=0;ticket<PHASES*N;++ticket) {
      const Audit& record=got.records[src][ticket];
      const bool expected=sends(src,rank,pattern);
      ok &= record.present == unsigned(expected);
      if(expected) {
        ok &= record.phase==ticket/N && record.len==length_for(ticket) &&
              record.format==format_for(src,rank,ticket);
        for(unsigned i=0;i<record.len && i<BYTES;++i)
          ok &= record.bytes[i]==value_for(src,rank,ticket,i);
      }
    }
    std::printf("%s rank=%d device=%d capacity=%u trial=%u pattern=%u sent=%u received=%u "
                "close=%u/%u blocked=%llu closed_full=%llu error=%u\n",
      ok?"PASS":"FAIL",rank,devices[rank],capacity,trial,pattern,got.sent,got.received,
      got.closes_sent,got.closes_received,got.blocked,got.closed_while_full,got.error);
    int bad=ok?0:1;
    CUDA_OK(cudaMemcpyAsync(vote,&bad,sizeof(bad),cudaMemcpyHostToDevice,stream));
    NCCL_OK(ncclAllReduce(vote,vote,1,ncclInt,ncclSum,comm,stream));
    CUDA_OK(cudaMemcpyAsync(&bad,vote,sizeof(bad),cudaMemcpyDeviceToHost,stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    if(bad) {all_ok=false;break;}
   }
   if(!all_ok) break;
  }
  NCCL_OK(ncclDevCommDestroy(comm,&device_comm));
  NCCL_OK(ncclCommWindowDeregister(comm,window));
  CUDA_OK(cudaStreamDestroy(stream)); CUDA_OK(cudaFree(output)); CUDA_OK(cudaFree(vote));
  NCCL_OK(ncclMemFree(box)); NCCL_OK(ncclCommDestroy(comm));
  return all_ok?0:4;
}
int main(int argc,char** argv) {
  setvbuf(stdout,nullptr,_IONBF,0);
  if(argc==4 && !std::strcmp(argv[1],"--rank")) {
    int rank=std::atoi(argv[2]); ncclUniqueId id{};
    if(rank<0 || rank>=int(WORLD) || !decode(argv[3],id)) return 6;
    return rank_main(rank,id);
  }
  if(argc!=1) return 6;
  ncclUniqueId id{}; NCCL_OK(ncclGetUniqueId(&id));
  std::string encoded=encode(id); std::vector<pid_t> children(WORLD);
  for(unsigned rank=0;rank<WORLD;++rank) {
    children[rank]=fork();
    if(children[rank]==0) {
      auto value=std::to_string(rank);
      execl(argv[0],argv[0],"--rank",value.c_str(),encoded.c_str(),nullptr);
      _exit(127);
    }
    if(children[rank]<0) return 7;
  }
  bool ok=true; unsigned skipped=0;
  for(unsigned rank=0;rank<WORLD;++rank) {
    int status=0;
    if(waitpid(children[rank],&status,0)<0) ok=false;
    int exit_code=WIFEXITED(status)?WEXITSTATUS(status):-1;
    std::printf("CHILD rank=%u exit=%d\n",rank,exit_code);
    if(exit_code==77) ++skipped;
    else if(exit_code) ok=false;
  }
  if(skipped==WORLD && ok) return 77;
  if(!ok || skipped) return 8;
  std::puts("PASS bounded packet LSA physical test; no BFS claim");
  return 0;
}
