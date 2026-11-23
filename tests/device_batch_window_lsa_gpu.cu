#include "../cuda/device_batch_window.cuh"
#include <nccl.h>
#include <nccl_device.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace mgbfs_device_batch;
#ifndef MGBFS_BATCH_WORLD
#define MGBFS_BATCH_WORLD 2
#endif
constexpr unsigned WORLD=MGBFS_BATCH_WORLD, B=521, C=23, ROUNDS=12;
constexpr uint64_t GENERATION=20261122;
static_assert(WORLD>=2 && WORLD<=8,"world");
#define CU(x) do{auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));return 2;}}while(0)
#define NC(x) do{auto e=(x);if(e!=ncclSuccess){std::fprintf(stderr,"%s: %s\n",#x,ncclGetErrorString(e));return 3;}}while(0)

__host__ __device__ uint32_t destination(unsigned rank,unsigned round,unsigned i) {
  // Odd rounds have an empty self-path; rank zero is completely empty on a
  // different set of rounds. Other rounds include severe single-peer skew.
  if(round%3==0)return (rank+1)%WORLD;
  if(round%3==1)return (rank+1+(i%(WORLD-1)))%WORLD;
  return (i*17u+rank)%WORLD;
}
__host__ __device__ uint64_t nrecords(unsigned rank,unsigned round) {
  if(rank==0 && round%4==0)return 0;
  return round%5==0?33:B;
}
__global__ void exchange(ncclWindow_t window,ncclDevComm comm,unsigned rank,
    Input* raw,uint64_t* positions,uint64_t* hist,uint64_t* offsets,uint32_t* error,
    Payload* receive,uint32_t* audit,uint32_t* smids,unsigned inject_failure) {
  auto grid=cooperative_groups::this_grid();
  if(!threadIdx.x) {
    unsigned sm;
    asm("mov.u32 %0, %%smid;" : "=r"(sm));
    smids[blockIdx.x]=sm;
  }
  Window* peers[WORLD];
  for(unsigned p=0;p<WORLD;++p)peers[p]=static_cast<Window*>(ncclGetLsaPointer(window,0,p));
  Window* local=peers[rank];
  if(!grid.thread_rank() && (comm.lsaRank!=int(rank)||comm.lsaSize!=int(WORLD)))
    fail(local,WINDOW_PROTOCOL);
  grid.sync();
  for(unsigned round=0;round<ROUNDS;++round) {
    const uint64_t seq=round+1,n=nrecords(rank,round);
    if(round && !grid.thread_rank())wait_ack(local,peers,WORLD,seq-1,nullptr);
    grid.sync();
    if(acquire(local->status.failure))break;
    for(uint64_t i=grid.thread_rank();i<n;i+=grid.size()) {
      const uint64_t key=(uint64_t(round)<<40)|(uint64_t(rank)<<32)|i;
      raw[i]={key,key^0x9182736455463728ull,destination(rank,round,unsigned(i)),0};
    }
    grid.sync();
    partition(raw,payload(local),n,B,WORLD,positions,hist,offsets,error);
    if(!grid.thread_rank() && *error)fail(local,WINDOW_PROTOCOL);
    grid.sync();
    if(acquire(local->status.failure))break;
    // Every CTA is a writer. Its system fence precedes the grid rendezvous and
    // the coordinator's system-release publication of the window sequence.
    __threadfence_system();
    grid.sync();
    if(!grid.thread_rank())publish(local,
        GENERATION+unsigned(inject_failure && rank==1 && round==3),
        round,seq,offsets,WORLD);
    grid.sync();
    // Delay one receiving rank after READY to exercise slot backpressure.
    if(rank==1 && round==3 && !grid.thread_rank()) {
      const uint64_t start=clock64();while(clock64()-start<20000000ull){}
    }
    if(!grid.thread_rank())wait_ready(local,peers,WORLD,GENERATION,round,seq,B,nullptr);
    grid.sync();
    if(acquire(local->status.failure))break;
    uint64_t total=0;
    for(unsigned src=0;src<WORLD;++src)
      total+=peers[src]->offsets[rank+1]-peers[src]->offsets[rank];
    for(uint64_t first=0;first<total;first+=C) {
      const uint64_t visible=min(uint64_t(C),total-first);
      for(uint64_t j=grid.thread_rank();j<visible;j+=grid.size()) {
        uint64_t k=first+j;unsigned src=0;
        for(;src<WORLD;++src) {
          uint64_t amount=peers[src]->offsets[rank+1]-peers[src]->offsets[rank];
          if(k<amount)break;k-=amount;
        }
        if(src==WORLD)fail(local,WINDOW_PROTOCOL);
        else receive[j]=payload(peers[src])[peers[src]->offsets[rank]+k];
      }
      grid.sync();
      for(uint64_t j=grid.thread_rank();j<visible;j+=grid.size()) {
        const auto item=receive[j];
        const unsigned r=unsigned(item.key>>40),src=unsigned((item.key>>32)&255u);
        const uint64_t index=item.key&0xffffffffull;
        if(r!=round||src>=WORLD||index>=nrecords(src,round)||
           item.packed!=(item.key^0x9182736455463728ull)||
           destination(src,round,unsigned(index))!=rank)fail(local,WINDOW_PROTOCOL);
        else atomicAdd(&audit[(uint64_t(round)*WORLD+src)*B+index],1u);
      }
      grid.sync();
    }
    __threadfence_system();
    grid.sync();
    if(acquire(local->status.failure))break;
    if(!grid.thread_rank())acknowledge(peers,WORLD,rank,seq);
    grid.sync();
  }
  if(!grid.thread_rank() && !acquire(local->status.failure))
    wait_ack(local,peers,WORLD,ROUNDS,nullptr);
  grid.sync();
}
static std::string encode(const ncclUniqueId& id) {
  const char* digits="0123456789abcdef";std::string out;
  for(unsigned char c:id.internal){out.push_back(digits[c>>4]);out.push_back(digits[c&15]);}
  return out;
}
static bool decode(const char* text,ncclUniqueId& id) {
  if(std::strlen(text)!=2*sizeof(id.internal))return false;
  auto digit=[](char c){return c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:-1;};
  for(size_t i=0;i<sizeof(id.internal);++i){int a=digit(text[2*i]),b=digit(text[2*i+1]);
    if(a<0||b<0)return false;id.internal[i]=char(a*16+b);}
  return true;
}
static int run_rank(unsigned rank,const ncclUniqueId& id) {
  int count=0;CU(cudaGetDeviceCount(&count));if(count<int(WORLD))return 77;
  CU(cudaSetDevice(rank));
  for(unsigned peer=0;peer<WORLD;++peer)if(peer!=rank) {
    int access=0,atom=0;
    CU(cudaDeviceCanAccessPeer(&access,rank,peer));
    CU(cudaDeviceGetP2PAttribute(&atom,cudaDevP2PAttrNativeAtomicSupported,rank,peer));
    if(!access||!atom)return 77;
    auto result=cudaDeviceEnablePeerAccess(peer,0);
    if(result==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();
    else CU(result);
  }
  ncclComm_t comm{};ncclWindow_t window{};ncclDevComm device{};
  ncclCommProperties_t properties=NCCL_COMM_PROPERTIES_INITIALIZER;
  NC(ncclCommInitRank(&comm,WORLD,id,rank));NC(ncclCommQueryProperties(comm,&properties));
  if(!properties.deviceApiSupport||properties.nLsaTeams!=1)return 77;
  const size_t bytes=sizeof(Window)+B*sizeof(Payload);
  Window* box=nullptr;Input* raw=nullptr;Payload* receive=nullptr;
  uint64_t *positions=nullptr,*hist=nullptr,*offsets=nullptr;uint32_t *error=nullptr,*audit=nullptr,*smids=nullptr;
  unsigned inject_failure=std::getenv("MGBFS_BATCH_INJECT_FAILURE")!=nullptr;
  NC(ncclMemAlloc((void**)&box,bytes));
  NC(ncclCommWindowRegister(comm,box,bytes,&window,NCCL_WIN_COLL_SYMMETRIC));
  ncclDevCommRequirements req=NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  NC(ncclDevCommCreate(comm,&req,&device));
  int sms=0,active=0;CU(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,rank));
  CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,exchange,128,0));
  int blocks=std::min(sms*active,32);if(blocks<2)return 77;
  CU(cudaMalloc(&raw,B*sizeof(Input)));CU(cudaMalloc(&receive,C*sizeof(Payload)));
  CU(cudaMalloc(&positions,B*sizeof(uint64_t)));CU(cudaMalloc(&hist,blocks*WORLD*sizeof(uint64_t)));
  CU(cudaMalloc(&offsets,(WORLD+1)*sizeof(uint64_t)));CU(cudaMalloc(&error,sizeof(uint32_t)));
  CU(cudaMalloc(&audit,uint64_t(ROUNDS)*WORLD*B*sizeof(uint32_t)));
  CU(cudaMalloc(&smids,blocks*sizeof(uint32_t)));
  CU(cudaMemset(box,0,bytes));CU(cudaMemset(audit,0,uint64_t(ROUNDS)*WORLD*B*sizeof(uint32_t)));
  int* vote=nullptr;CU(cudaMalloc(&vote,sizeof(int)));CU(cudaMemset(vote,0,sizeof(int)));
  NC(ncclAllReduce(vote,vote,1,ncclInt,ncclSum,comm,nullptr));CU(cudaStreamSynchronize(nullptr));
  void* args[]={&window,&device,&rank,&raw,&positions,&hist,&offsets,&error,&receive,&audit,&smids,&inject_failure};
  CU(cudaLaunchCooperativeKernel((void*)exchange,dim3(blocks),dim3(128),args,0,nullptr));
  CU(cudaDeviceSynchronize());
  Window control{};CU(cudaMemcpy(&control,box,sizeof(control),cudaMemcpyDeviceToHost));
  std::vector<uint32_t> seen(uint64_t(ROUNDS)*WORLD*B);
  CU(cudaMemcpy(seen.data(),audit,seen.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost));
  std::vector<uint32_t> observed_sm(blocks);
  CU(cudaMemcpy(observed_sm.data(),smids,blocks*sizeof(uint32_t),cudaMemcpyDeviceToHost));
  std::sort(observed_sm.begin(),observed_sm.end());
  auto unique_sm=std::unique(observed_sm.begin(),observed_sm.end())-observed_sm.begin();
  bool ok=inject_failure?control.status.failure!=0:control.status.failure==0;
  if(unique_sm<2)ok=false;
  if(!inject_failure)for(unsigned round=0;round<ROUNDS;++round)for(unsigned src=0;src<WORLD;++src)
    for(unsigned i=0;i<B;++i) {
      bool expect=i<nrecords(src,round) && destination(src,round,i)==rank;
      if(seen[(uint64_t(round)*WORLD+src)*B+i]!=unsigned(expect))ok=false;
    }
  std::printf("%s rank=%u blocks=%d rounds=%u receive_tile=%u sms=%zu injected=%u failure=%llu\n",
    ok?"PASS":"FAIL",rank,blocks,ROUNDS,C,size_t(unique_sm),inject_failure,control.status.failure);
  NC(ncclDevCommDestroy(comm,&device));NC(ncclCommWindowDeregister(comm,window));
  CU(cudaFree(vote));CU(cudaFree(smids));CU(cudaFree(audit));CU(cudaFree(error));CU(cudaFree(offsets));
  CU(cudaFree(hist));CU(cudaFree(positions));CU(cudaFree(receive));CU(cudaFree(raw));
  NC(ncclMemFree(box));NC(ncclCommDestroy(comm));return ok?0:4;
}
int main(int argc,char** argv) {
  setvbuf(stdout,nullptr,_IONBF,0);
  if(argc==4&&!std::strcmp(argv[1],"--rank")) {
    int rank=std::atoi(argv[2]);ncclUniqueId id{};
    if(rank<0||rank>=int(WORLD)||!decode(argv[3],id))return 6;
    return run_rank(unsigned(rank),id);
  }
  ncclUniqueId id{};NC(ncclGetUniqueId(&id));std::string encoded=encode(id);
  std::vector<pid_t> children(WORLD);
  for(unsigned rank=0;rank<WORLD;++rank) {
    children[rank]=fork();
    if(children[rank]==0) {
      std::string value=std::to_string(rank);
      execl(argv[0],argv[0],"--rank",value.c_str(),encoded.c_str(),nullptr);_exit(127);
    }
    if(children[rank]<0)return 7;
  }
  bool ok=true;unsigned skipped=0;
  for(unsigned rank=0;rank<WORLD;++rank) {
    int status=0;waitpid(children[rank],&status,0);
    int code=WIFEXITED(status)?WEXITSTATUS(status):-1;
    std::printf("CHILD rank=%u exit=%d\n",rank,code);
    if(code==77)++skipped;else if(code)ok=false;
  }
  if(skipped==WORLD&&ok)return 77;
  return ok&&!skipped?0:8;
}
