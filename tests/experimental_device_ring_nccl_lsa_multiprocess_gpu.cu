// Two-device NCCL LSA leaf test of experimental_device_ring.cuh.
// Each GPU obtains pointers through a registered NCCL symmetric window.
// Two processes / two physical devices; this remains a transport-only test.
// No NCCL, no BFS, no multi-process acceptance. Each GPU runs one scheduler
// kernel which alternates sends and receives; no host submits per record.
#include "../cuda/experimental_device_ring.cuh"
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstring>
#include <string>

using namespace mgbfs_device_ring;
#define CUDA_OK(expr) do { auto e=(expr); if(e!=cudaSuccess) { \
  std::fprintf(stderr,"%s: %s\n",#expr,cudaGetErrorString(e)); return 2; } \
} while(0)
struct alignas(64) Mailbox {
  Slot slots[7];
  alignas(8) unsigned long long close;
};
struct Stats {
  unsigned long long sent, received, send_blocked, loops;
  int error;
};
static_assert(alignof(Mailbox)>=64, "control alignment");
__global__ void init_box(Mailbox* m, unsigned capacity, unsigned long long epoch) {
  if(blockIdx.x || threadIdx.x) return;
  for(unsigned i=0;i<capacity;i++) init_slot(m->slots[i],epoch,i);
  m->close=0;
}
__global__ void exchange(ncclWindow_t win, ncclDevComm dev, unsigned rank,
    unsigned capacity, unsigned long long epoch, unsigned send_count,
    unsigned receive_count, unsigned delayed_reads, Stats* result) {
  if(blockIdx.x || threadIdx.x) return;
  Mailbox* local=static_cast<Mailbox*>(ncclGetLsaPointer(win,0,dev.lsaRank));
  Mailbox* remote=static_cast<Mailbox*>(ncclGetLsaPointer(win,0,1-dev.lsaRank));
  if(!local || !remote || dev.lsaRank!=int(rank) || dev.lsaSize!=2) {
    result->error=6; return;
  }
  unsigned long long sent=0,received=0,blocked=0,loops=0;
  bool closed=false;
  int error=0;
  cuda::atomic_ref<unsigned long long,cuda::thread_scope_system> local_close(local->close);
  cuda::atomic_ref<unsigned long long,cuda::thread_scope_system> remote_close(remote->close);
  while(loops++<20000000ull) {
    bool progressed=false;
    if(sent<send_count) {
      Payload p{(static_cast<unsigned long long>(rank)<<56) ^ sent,rank,0x51u};
      const Result w=try_write(local->slots,capacity,epoch,sent,p);
      if(w==Result::ok) { sent++; progressed=true; }
      else if(w==Result::not_ready) blocked++;
      else {error=1;break;}
    }
    if(sent==send_count&&!closed) {
      local_close.store(sent+1,cuda::memory_order_release);
      closed=true;progressed=true;
    }
    if(received<receive_count && loops>delayed_reads) {
      Payload p{};
      const Result r=try_read(remote->slots,capacity,epoch,received,p);
      if(r==Result::ok) {
        if(p.value!=((static_cast<unsigned long long>(1-rank)<<56)^received)
           || p.kind!=1-rank || p.source!=0x51u) {error=2;break;}
        // The test consumes and validates the payload before acknowledging.
        if(try_release(remote->slots,capacity,epoch,received)!=Result::ok)
          {error=3;break;}
        received++;progressed=true;
      } else if(r!=Result::not_ready) {error=4;break;}
    }
    if(closed && received==receive_count &&
       remote_close.load(cuda::memory_order_acquire)==receive_count+1ull) break;
    if(!progressed) __nanosleep(128);
  }
  if(loops>=20000000ull && !error)error=5;
  result->sent=sent;result->received=received;result->send_blocked=blocked;
  result->loops=loops;result->error=error;
}



#define NCCL_OK(expr) do { auto e=(expr); if(e!=ncclSuccess) { \
  std::fprintf(stderr,"%s: %s (%d)\n",#expr,ncclGetErrorString(e),int(e)); return 3; } \
} while(0)
static std::string encode(const ncclUniqueId& id) {
  const char *hex="0123456789abcdef";std::string result;
  for(unsigned char c: id.internal) {result.push_back(hex[c>>4]);result.push_back(hex[c&15]);}
  return result;
}
static bool decode(const char* s,ncclUniqueId& id) {
  if(std::strlen(s)!=2*sizeof(id.internal))return false;
  auto digit=[](char c)->int {if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;return -1;};
  for(size_t i=0;i<sizeof(id.internal);i++) {
    int a=digit(s[2*i]),b=digit(s[2*i+1]);if(a<0||b<0)return false;
    id.internal[i]=char(a*16+b);
  }
  return true;
}
static int rank_main(int rank,const ncclUniqueId& id) {
  int n=0,version=0; CUDA_OK(cudaGetDeviceCount(&n));NCCL_OK(ncclGetVersion(&version));
  if(n!=2||version<22900)return 77;
  CUDA_OK(cudaSetDevice(rank));
  int can=0,atom=0,coop=0;
  CUDA_OK(cudaDeviceCanAccessPeer(&can,rank,1-rank));
  CUDA_OK(cudaDeviceGetP2PAttribute(&atom,cudaDevP2PAttrNativeAtomicSupported,rank,1-rank));
  CUDA_OK(cudaDeviceGetAttribute(&coop,cudaDevAttrCooperativeLaunch,rank));
  std::printf("CAPABILITY pid=%d rank=%d p2p=%d native_atomics=%d coop=%d nccl=%d\n",
              int(getpid()),rank,can,atom,coop,version);
  if(!can||!atom||!coop)return 77;
  auto e=cudaDeviceEnablePeerAccess(1-rank,0);
  if(e!=cudaSuccess&&e!=cudaErrorPeerAccessAlreadyEnabled) return 2;
  if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();
  ncclComm_t comm{};ncclWindow_t win{};ncclDevComm dev{};
  Mailbox* box=nullptr;Stats* stats=nullptr;int* barrier=nullptr;cudaStream_t stream{};
  std::printf("PHASE rank=%d nccl_init\n",rank);
  NCCL_OK(ncclCommInitRank(&comm,2,id,rank));
  ncclCommProperties_t props=NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCL_OK(ncclCommQueryProperties(comm,&props));
  std::printf("NCCL rank=%d device_api=%d lsa_teams=%d\n",rank,int(props.deviceApiSupport),props.nLsaTeams);
  if(!props.deviceApiSupport||props.nLsaTeams!=1)return 77;
  NCCL_OK(ncclMemAlloc(reinterpret_cast<void**>(&box),sizeof(Mailbox)));
  CUDA_OK(cudaMalloc(&stats,sizeof(Stats)));CUDA_OK(cudaMalloc(&barrier,sizeof(int)));
  CUDA_OK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  std::printf("PHASE rank=%d window_register\n",rank);
  NCCL_OK(ncclCommWindowRegister(comm,box,sizeof(Mailbox),&win,NCCL_WIN_COLL_SYMMETRIC));
  ncclDevCommRequirements req=NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;req.lsaBarrierCount=1;
  std::printf("PHASE rank=%d device_comm_create\n",rank);
  NCCL_OK(ncclDevCommCreate(comm,&req,&dev));
  std::printf("NCCL_DEVICE rank=%d lsa_rank=%d lsa_size=%d\n",rank,dev.lsaRank,dev.lsaSize);
  if(dev.lsaRank!=rank||dev.lsaSize!=2)return 77;
  struct Case {unsigned cap,a,b,delay;};
  for(const Case c: {Case{1,8192,8192,0},Case{7,8192,8192,0},
                     Case{1,2048,0,10000},Case{7,0,2048,0}}) {
    static unsigned long long epoch=301;epoch++;
    init_box<<<1,1,0,stream>>>(box,c.cap,epoch);
    CUDA_OK(cudaGetLastError());CUDA_OK(cudaMemsetAsync(stats,0,sizeof(Stats),stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    // A real collective separates reset from both kernels. It is not per message.
    CUDA_OK(cudaMemsetAsync(barrier,0,sizeof(int),stream));
    NCCL_OK(ncclAllReduce(barrier,barrier,1,ncclInt,ncclSum,comm,stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    exchange<<<1,1,0,stream>>>(win,dev,rank,c.cap,epoch,rank?c.b:c.a,rank?c.a:c.b,
                                rank?c.delay:0,stats);
    CUDA_OK(cudaGetLastError());CUDA_OK(cudaStreamSynchronize(stream));
    Stats host{};CUDA_OK(cudaMemcpy(&host,stats,sizeof(Stats),cudaMemcpyDeviceToHost));
    std::printf("CASE pid=%d epoch=%llu cap=%u rank=%d sent=%llu received=%llu blocked=%llu loops=%llu error=%d\n",
        int(getpid()),epoch,c.cap,rank,host.sent,host.received,host.send_blocked,host.loops,host.error);
    if(host.error||host.sent!=(rank?c.b:c.a)||host.received!=(rank?c.a:c.b))return 4;
    if(c.delay&&rank==0&&!host.send_blocked)return 5;
    NCCL_OK(ncclAllReduce(barrier,barrier,1,ncclInt,ncclSum,comm,stream));
    CUDA_OK(cudaStreamSynchronize(stream));
  }
  std::printf("PHASE rank=%d destroy\n",rank);
  NCCL_OK(ncclDevCommDestroy(comm,&dev));NCCL_OK(ncclCommWindowDeregister(comm,win));
  CUDA_OK(cudaStreamDestroy(stream));CUDA_OK(cudaFree(stats));CUDA_OK(cudaFree(barrier));
  NCCL_OK(ncclMemFree(box));NCCL_OK(ncclCommDestroy(comm));
  std::printf("PASS rank=%d NCCL LSA leaf\n",rank);return 0;
}
int main(int argc,char** argv) {
  setvbuf(stdout,nullptr,_IONBF,0);
  if(argc==4 && std::strcmp(argv[1],"--rank")==0) {
    int rank=std::atoi(argv[2]);ncclUniqueId id{};
    if(rank<0||rank>1||!decode(argv[3],id))return 6;
    return rank_main(rank,id);
  }
  if(argc!=1)return 6;
  ncclUniqueId id{};NCCL_OK(ncclGetUniqueId(&id));
  const std::string value=encode(id);pid_t pids[2]{};
  for(int i=0;i<2;i++) {
    pids[i]=fork();
    if(pids[i]==0) {
      const char* rank=i?"1":"0";
      execl(argv[0],argv[0],"--rank",rank,value.c_str(),nullptr);
      _exit(127);
    }
    if(pids[i]<0)return 7;
  }
  bool ok=true;
  for(int i=0;i<2;i++) {
    int status=0;if(waitpid(pids[i],&status,0)<0)ok=false;
    std::printf("CHILD rank=%d exited=%d exit=%d\n",i,int(WIFEXITED(status)),WIFEXITED(status)?WEXITSTATUS(status):-1);
    if(!WIFEXITED(status)||WEXITSTATUS(status)!=0)ok=false;
  }
  if(!ok)return 8;
  std::puts("PASS two-process two-GPU NCCL LSA transport; no BFS");return 0;
}
