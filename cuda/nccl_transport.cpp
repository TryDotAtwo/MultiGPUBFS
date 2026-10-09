#include "mgbfs_cuda.h"
#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <memory>
#include <climits>
#include <chrono>
#include <thread>
#include <atomic>
#include <cstdlib>
struct Comm;
namespace { int await_nccl(Comm*,ncclResult_t,ncclResult_t* = nullptr); }
namespace { int terminal_abort(Comm*); }
#ifdef MGBFS_NCCL_LSA
#include <nccl_device.h>
#include <cuda/atomic>
#if NCCL_VERSION_CODE < 22900
#error "MGBFS_NCCL_LSA requires NCCL 2.29 or newer"
#endif
#endif
struct Comm {
  ncclComm_t value{};
  uint32_t rank{}, world{};
  int (*cancel_requested)(void*){};
  void* cancel_context{};
  int (*retirement_probe)(void*,int){};
  void* retirement_context{};
  bool terminal_started{}, retirement_failed{};
  std::atomic<uint32_t>* terminal_host{};
  uint32_t* terminal_device{};
#ifdef MGBFS_NCCL_LSA
  ncclDevComm device{};
  ncclWindow_t window{};
  unsigned char* symmetric{};
  uint32_t candidate_capacity{}, state_stride{};
  size_t states_offset{};
  bool window_ready{},device_ready{};
  bool lsa_used{}, readers_retired{};
#endif
  ~Comm(){
    if(retirement_failed)return;
#ifdef MGBFS_NCCL_LSA
    // Normal teardown requires the caller to drain all LSA stream consumers.
    // After ncclCommAbort the process is terminal; avoid using an invalid comm.
    // An unsuccessful terminal handshake does not authorize unmapping a peer's
    // window. The process supervisor must terminate the rank group instead.
    if(value && device_ready)ncclDevCommDestroy(value,&device);
    if(value && window_ready)ncclCommWindowDeregister(value,window);
    if(symmetric)ncclMemFree(symmetric);
#endif
    if(value){
      const auto result=ncclCommFinalize(value);
      if(await_nccl(this,result)==0 && value)ncclCommDestroy(value);
      else if(value)ncclCommAbort(value);
    }
    if(terminal_host)cudaFreeHost(terminal_host);
  }
};
namespace {
// Only the rank's NCCL-calling thread enters this function or aborts `value`.
// The sideband thread may change the atomic observed by cancel_requested,
// but never calls NCCL or touches Comm itself.
int await_nccl(Comm* p, ncclResult_t submitted, ncclResult_t* failure) {
  if(failure)*failure=ncclSuccess;
  // A failed external-reader retirement keeps the handle alive deliberately.
  // It does not authorize querying or progressing new communicator work.
  if(!p || !p->value || p->terminal_started)return 13;
  if(submitted!=ncclSuccess && submitted!=ncclInProgress) {
    if(failure)*failure=submitted;
    std::fprintf(stderr,"MGBFS_NCCL_SUBMIT_FATAL rank=%u code=%d detail=%s\n",
        p->rank,int(submitted),ncclGetErrorString(submitted));
    return 6;
  }
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(120);
  for(;;) {
    // Do not abort here: return first so the runtime publishes failure to the
    // TCP sideband before the sole dispatcher starts communicator retirement.
    if(p->terminal_host && (p->terminal_host->load(std::memory_order_acquire)&2u))return 14;
    if(p->cancel_requested && p->cancel_requested(p->cancel_context)) {
      terminal_abort(p);
      return 7;
    }
    ncclResult_t state=ncclSuccess;
    const auto queried=p->value?ncclCommGetAsyncError(p->value,&state):ncclInvalidUsage;
    if(queried!=ncclSuccess) {
      if(failure)*failure=queried;
      std::fprintf(stderr,"MGBFS_NCCL_QUERY_FATAL rank=%u code=%d detail=%s\n",
          p->rank,int(queried),ncclGetErrorString(queried));
      return 8;
    }
    if(state==ncclSuccess) return 0;
    if(state!=ncclInProgress) {
      if(failure)*failure=state;
      std::fprintf(stderr,"MGBFS_NCCL_ASYNC_FATAL rank=%u code=%d detail=%s\n",
          p->rank,int(state),ncclGetErrorString(state));
      return 9;
    }
    if(std::chrono::steady_clock::now()>=deadline) {
      terminal_abort(p);
      return 10;
    }
    std::this_thread::yield();
  }
}
int terminal_abort(Comm* p) {
  if(!p || !p->value)return p?0:1;
  if(p->terminal_started)return 13;
  p->terminal_started=true;
  if(p->terminal_host)p->terminal_host->fetch_or(1,std::memory_order_release);
#ifdef MGBFS_NCCL_LSA
  if(p->lsa_used && !p->readers_retired) {
    const bool trace=std::getenv("MGBFS_TRACE_FAILURE_TEARDOWN")!=nullptr;
    // Revoke cancels NCCL's internal operations without freeing registered
    // memory. Our external LSA rendezvous observes its own sticky terminal.
    if(trace)std::fprintf(stderr,"MGBFS_FAILURE_TEARDOWN rank=%u stage=revoke_begin\n",p->rank);
    auto status=ncclCommRevoke(p->value,0);
    if(trace)std::fprintf(stderr,"MGBFS_FAILURE_TEARDOWN rank=%u stage=revoke_submitted code=%d\n",p->rank,int(status));
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(120);
    while(status==ncclInProgress) {
      if(ncclCommGetAsyncError(p->value,&status)!=ncclSuccess ||
         std::chrono::steady_clock::now()>=deadline) {
        if(p->retirement_probe)p->retirement_probe(p->retirement_context,-1);
        p->retirement_failed=true;return 11;
      }
      std::this_thread::yield();
    }
    if(trace)std::fprintf(stderr,"MGBFS_FAILURE_TEARDOWN rank=%u stage=revoke_ready code=%d\n",p->rank,int(status));
    if(trace)std::fprintf(stderr,"MGBFS_FAILURE_TEARDOWN rank=%u stage=external_readers_drain_begin\n",p->rank);
    if(status!=ncclSuccess || cudaDeviceSynchronize()!=cudaSuccess ||
       (!p->retirement_probe && p->world!=1)) {
      if(p->retirement_probe)p->retirement_probe(p->retirement_context,-1);
      p->retirement_failed=true;return 12;
    }
    if(trace)std::fprintf(stderr,"MGBFS_FAILURE_TEARDOWN rank=%u stage=external_readers_drain_end\n",p->rank);
    if(p->retirement_probe) {
      int acknowledged=p->retirement_probe(p->retirement_context,1);
      while(acknowledged==0 && std::chrono::steady_clock::now()<deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        acknowledged=p->retirement_probe(p->retirement_context,0);
      }
      if(acknowledged!=1) {
        p->retirement_probe(p->retirement_context,-1);
        p->retirement_failed=true;return 13;
      }
    }
    // A one-rank communicator has no peer that can retain our window. The
    // successful device drain above is its complete reader-retirement proof.
    // Multi-rank communicators still require every peer's sideband ACK.
    p->readers_retired=true;
  }
#endif
  const auto value=p->value;
  p->value=nullptr;
  const auto status=ncclCommAbort(value);
  // Non-LSA errors still participate in the same sideband completion. Here
  // NCCL abort has no external LSA reader whose window could be unmapped.
  if(p->retirement_probe) {
#ifdef MGBFS_NCCL_LSA
    if(p->readers_retired)return status==ncclSuccess?0:2;
#endif
    if(status!=ncclSuccess || cudaDeviceSynchronize()!=cudaSuccess) {
      p->retirement_failed=true;
      p->retirement_probe(p->retirement_context,-1);return 14;
    }
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(120);
    int ready=p->retirement_probe(p->retirement_context,1);
    while(ready==0 && std::chrono::steady_clock::now()<deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      ready=p->retirement_probe(p->retirement_context,0);
    }
    if(ready!=1) {p->retirement_failed=true;p->retirement_probe(p->retirement_context,-1);return 13;}
  }
  return status==ncclSuccess?0:2;
}
}
extern "C" int mgbfs_nccl_unique_id(void* out){if(!out)return 1;static_assert(sizeof(ncclUniqueId)==128);return ncclGetUniqueId(static_cast<ncclUniqueId*>(out))==ncclSuccess?0:2;}
extern "C" int mgbfs_nccl_create(uint32_t rank,uint32_t world,uint32_t device,const void* raw_id,void** out,char* error,size_t error_capacity){
  return mgbfs_nccl_create_with_cancel(rank,world,device,raw_id,out,error,
      error_capacity,nullptr,nullptr);
}
extern "C" int mgbfs_nccl_create_with_cancel(uint32_t rank,uint32_t world,
    uint32_t device,const void* raw_id,void** out,char* error,size_t error_capacity,
    int (*probe)(void*),void* context){
  if(!out)return 1;
  *out=nullptr;
  if(!raw_id||!world||rank>=world||bool(probe)!=bool(context))return 1;
  auto p=std::make_unique<Comm>();
  p->rank=rank;p->world=world;
  p->cancel_requested=probe;p->cancel_context=context;
  if(probe&&probe(context))return 7;
  cudaError_t ce=cudaSetDevice(int(device));
  if(ce!=cudaSuccess){
    if(error&&error_capacity)std::snprintf(error,error_capacity,"%s",cudaGetErrorString(ce));
    return 2;
  }
  static_assert(sizeof(std::atomic<uint32_t>)==sizeof(uint32_t));
  static_assert(std::atomic<uint32_t>::is_always_lock_free);
  void* terminal=nullptr;
  if(cudaHostAlloc(&terminal,sizeof(uint32_t),cudaHostAllocMapped)!=cudaSuccess)return 6;
  p->terminal_host=new(terminal)std::atomic<uint32_t>(0);
  if(cudaHostGetDevicePointer(reinterpret_cast<void**>(&p->terminal_device),terminal,0)!=cudaSuccess)return 6;
  ncclUniqueId id;std::memcpy(&id,raw_id,sizeof(id));
  ncclConfig_t config=NCCL_CONFIG_INITIALIZER;config.blocking=0;
  ncclResult_t e=ncclCommInitRankConfig(&p->value,int(world),id,int(rank),&config);
  const int ready=await_nccl(p.get(),e);
  if(ready){
    if(error&&error_capacity)std::snprintf(error,error_capacity,
        "init: %s (phase %d)",ncclGetErrorString(e),ready);
    // await_nccl may already have aborted on cancellation. Never reuse its
    // invalid handle or delegate abort to the sideband thread.
    if(p->value){ncclCommAbort(p->value);p->value=nullptr;}
    return 3;
  }
  *out=p.release();return 0;
}
extern "C" int mgbfs_nccl_cancel_words(void* raw,uint32_t** host,uint32_t** device){
  auto* p=static_cast<Comm*>(raw);
  if(!p||!p->value||!host||!device||!p->terminal_host||!p->terminal_device)return 1;
  *host=reinterpret_cast<uint32_t*>(p->terminal_host);*device=p->terminal_device;
  return 0;
}
extern "C" int mgbfs_nccl_bind_cancel(void* raw,int (*probe)(void*),void* context){
  auto* p=static_cast<Comm*>(raw);
  if(!p||!p->value||!probe||!context)return 1;
  p->cancel_requested=probe;
  p->cancel_context=context;
  return 0;
}
extern "C" int mgbfs_nccl_session_park(void* raw){
  auto* p=static_cast<Comm*>(raw);
  if(!p||!p->value||p->terminal_started)return 1;
#ifdef MGBFS_NCCL_LSA
  // LSA window reuse needs its own collective generation-reset acceptance.
  if(p->device_ready)return 2;
#endif
  if(cudaDeviceSynchronize()!=cudaSuccess)return 3;
  ncclResult_t state=ncclSuccess;
  if(ncclCommGetAsyncError(p->value,&state)!=ncclSuccess||state!=ncclSuccess)return 4;
  p->cancel_requested=nullptr;p->cancel_context=nullptr;
  p->retirement_probe=nullptr;p->retirement_context=nullptr;
  return 0;
}
extern "C" int mgbfs_nccl_bind_retirement(void* raw,int (*probe)(void*,int),void* context){
  auto* p=static_cast<Comm*>(raw);
  if(!p||!p->value||!probe||!context)return 1;
  p->retirement_probe=probe;p->retirement_context=context;
  return 0;
}
namespace {
// The dispatcher is the only caller. Stack descriptors are consumed during
// submission; their device payloads retain the existing last-reader lease.
int send_recv_lanes(void* raw,const void* const* send,const uint64_t* send_bytes,
    void* const* recv,const uint64_t* recv_bytes,unsigned lanes,uint32_t peer,
    void* raw_stream) {
  auto* p=static_cast<Comm*>(raw);
  if(!p || !p->value || p->terminal_started || peer>=p->world) return 1;
  for(unsigned lane=0;lane<lanes;++lane)
    if((!send[lane] && send_bytes[lane]) || (!recv[lane] && recv_bytes[lane]))return 1;
  if(p->cancel_requested && p->cancel_requested(p->cancel_context))return 7;
  auto stream=static_cast<cudaStream_t>(raw_stream);
  if(ncclGroupStart()!=ncclSuccess)return 2;
  int status=0;
  // Zero-byte lanes still participate. Every rank issues hash then state;
  // no maximum-buffer padding and no extra payload allocation are introduced.
  for(unsigned lane=0;lane<lanes && !status;++lane) {
    const auto sent=ncclSend(send[lane],size_t(send_bytes[lane]),ncclUint8,
        int(peer),p->value,stream);
    if(sent!=ncclSuccess && sent!=ncclInProgress)status=3;
    else {
      const auto received=ncclRecv(recv[lane],size_t(recv_bytes[lane]),ncclUint8,
          int(peer),p->value,stream);
      if(received!=ncclSuccess && received!=ncclInProgress)status=4;
    }
  }
  // Always close an opened group and settle asynchronous submission before
  // returning to the dispatcher, even when a lane failed. Keep first error.
  const auto end=ncclGroupEnd();
  const int settled=(end==ncclSuccess || end==ncclInProgress)?await_nccl(p,end):5;
  return status?status:settled;
}
}
extern "C" int mgbfs_nccl_send_recv(void* raw,const void* send,uint64_t send_bytes,
    uint32_t peer,void* recv,uint64_t recv_bytes,void* stream) {
  const void* sends[]={send};void* receives[]={recv};
  return send_recv_lanes(raw,sends,&send_bytes,receives,&recv_bytes,1,peer,stream);
}
extern "C" int mgbfs_nccl_send_recv_pair(void* raw,const void* hashes,uint64_t hash_bytes,
    const void* states,uint64_t state_bytes,uint32_t peer,void* recv_hashes,
    uint64_t recv_hash_bytes,void* recv_states,uint64_t recv_state_bytes,void* stream) {
  const void* sends[]={hashes,states};void* receives[]={recv_hashes,recv_states};
  const uint64_t sent[]={hash_bytes,state_bytes},received[]={recv_hash_bytes,recv_state_bytes};
  return send_recv_lanes(raw,sends,sent,receives,received,2,peer,stream);
}
extern "C" int mgbfs_nccl_all_gather_u32(void* raw,const uint32_t* send,uint32_t* recv,void* raw_stream){auto*p=static_cast<Comm*>(raw);if(!p||!p->value||p->terminal_started||!send||!recv)return 1;if(p->cancel_requested&&p->cancel_requested(p->cancel_context))return 7;return await_nccl(p,ncclAllGather(send,recv,1,ncclUint32,p->value,static_cast<cudaStream_t>(raw_stream)));}
extern "C" int mgbfs_nccl_all_reduce_max_u32(void* raw,const uint32_t* send,uint32_t* recv,void* raw_stream){auto*p=static_cast<Comm*>(raw);if(!p||!p->value||p->terminal_started||!send||!recv)return 1;if(p->cancel_requested&&p->cancel_requested(p->cancel_context))return 7;return await_nccl(p,ncclAllReduce(send,recv,1,ncclUint32,ncclMax,p->value,static_cast<cudaStream_t>(raw_stream)));}
extern "C" void mgbfs_nccl_destroy(void* raw){delete static_cast<Comm*>(raw);}
extern "C" int mgbfs_nccl_abort(void* raw){
  auto* p = static_cast<Comm*>(raw);
  if(!p) return 1;
  if(!p->value) return 0;
  std::fprintf(stderr,"MGBFS_FAILURE_TEARDOWN rank=%u stage=nccl_abort_begin\n",p->rank);
  const auto status=terminal_abort(p);
  std::fprintf(stderr,"MGBFS_FAILURE_TEARDOWN rank=%u stage=nccl_abort_end code=%d\n",p->rank,int(status));
  return status;
}
extern "C" int mgbfs_nccl_poll(void* raw){
  auto* p = static_cast<Comm*>(raw);
  if(!p || !p->value || p->terminal_started) return 1;
  if(p->terminal_host && (p->terminal_host->load(std::memory_order_acquire)&2u))return 5;
  ncclResult_t state = ncclSuccess;
  if(ncclCommGetAsyncError(p->value, &state) != ncclSuccess) return 2;
  return state == ncclSuccess ? 0 : (state == ncclInProgress ? 4 : 3);
}
extern "C" int mgbfs_nccl_scatter(void* raw,uint32_t source,const void* send,uint64_t send_capacity,const uint64_t* sizes,void* recv,uint64_t recv_bytes,uint64_t recv_capacity,void* stream) {
  auto* p = static_cast<Comm*>(raw);
  if(!p || !p->value || p->terminal_started || source >= p->world) return 1;
  if(p->cancel_requested && p->cancel_requested(p->cancel_context)) return 7;
  if(p->rank == source) {
    if(!sizes) return 1;
    uint64_t total = 0;
    for(uint32_t rank=0; rank<p->world; ++rank) {
      if(sizes[rank] > send_capacity-total) return 1;
      total += sizes[rank];
    }
    if(total && !send) return 1;
  } else if(recv_bytes > recv_capacity || (recv_bytes && !recv)) return 1;
  if(ncclGroupStart() != ncclSuccess) return 2;
  int status = 0;
  auto s = static_cast<cudaStream_t>(stream);
  if(p->rank == source) {
    uint64_t offset = 0;
    for(uint32_t rank=0; rank<p->world; ++rank) {
      const void* ptr = offset ? static_cast<const char*>(send)+offset : send;
      const auto result=rank==source?ncclSuccess:
          ncclSend(ptr,size_t(sizes[rank]),ncclUint8,int(rank),p->value,s);
      if(result!=ncclSuccess && result!=ncclInProgress) {
        status = 3;
        break;
      }
      offset += sizes[rank];
    }
  } else {
    const auto result=ncclRecv(recv,size_t(recv_bytes),ncclUint8,int(source),p->value,s);
    if(result!=ncclSuccess && result!=ncclInProgress)status=4;
  }
  const auto end = ncclGroupEnd();
  const int settled = (end==ncclSuccess||end==ncclInProgress)
      ? await_nccl(p,end) : 5;
  return status ? status : settled;
}

#ifdef MGBFS_NCCL_LSA
namespace {
constexpr size_t lsa_control_bytes=256;
constexpr unsigned lsa_copy_ctas=16,lsa_copy_threads=256;
static_assert(64+lsa_copy_ctas*sizeof(unsigned long long)<=lsa_control_bytes);
// One counter per CTA in each rank's existing symmetric control plane. There
// is one writer, serialized by the exchange stream. Payload remains exact-sized.
// External ncclDevComm has no public abort flag: never use the SDK's uncancellable
// barrier or modify its private fields for cancellation.
__device__ bool lsa_rendezvous(ncclDevComm dev,ncclWindow_t win,
    const uint32_t* terminal) {
  __shared__ unsigned long long expected;
  __shared__ int finished;
  __syncthreads();
  if(threadIdx.x==0) {
    auto* counters=reinterpret_cast<unsigned long long*>(
        static_cast<unsigned char*>(ncclGetLsaPointer(win,0,dev.lsaRank))+64);
    cuda::atomic_ref<unsigned long long,cuda::thread_scope_system> own(counters[blockIdx.x]);
    expected=own.load(cuda::memory_order_relaxed)+1;
    finished=expected==0?0:1;
    if(!finished) {
      auto* control=static_cast<uint32_t*>(ncclGetLsaPointer(win,0,dev.lsaRank));
      cuda::atomic_ref<uint32_t,cuda::thread_scope_system> fatal(control[2]);
      fatal.store(1,cuda::memory_order_release);
    }
    if(finished)own.store(expected,cuda::memory_order_release);
    for(unsigned rank=0;finished && rank<unsigned(dev.nRanks);++rank) {
      auto* peer=reinterpret_cast<unsigned long long*>(
          static_cast<unsigned char*>(ncclGetLsaPointer(win,0,rank))+64);
      cuda::atomic_ref<unsigned long long,cuda::thread_scope_system> arrived(peer[blockIdx.x]);
      cuda::atomic_ref<const uint32_t,cuda::thread_scope_system> stopped(*terminal);
      while(arrived.load(cuda::memory_order_acquire)<expected) {
        if(stopped.load(cuda::memory_order_acquire)) {finished=0;break;}
        __nanosleep(128);
      }
      if(stopped.load(cuda::memory_order_acquire))finished=0;
    }
  }
  __syncthreads();
  return finished!=0;
}
// The symmetric slot has one writer for each field in each exchange round:
// [recv_count, local_fatal, global_fatal] followed by dense hash/state planes.
__global__ void lsa_publish_count(ncclDevComm dev,ncclWindow_t win,
    const uint32_t* counts,const uint32_t* group_fatal,
    uint32_t logical_owner,uint32_t peer,uint32_t cap,const uint32_t* terminal){
  auto* local=static_cast<uint32_t*>(ncclGetLsaPointer(win,0,dev.lsaRank));
  if(*group_fatal){
    if(threadIdx.x==0){local[0]=0;local[1]=1;local[2]=1;}
    return;
  }
  if(!lsa_rendezvous(dev,win,terminal))return;
  if(threadIdx.x==0){
    auto* remote=static_cast<uint32_t*>(ncclGetLsaPointer(win,0,peer));
    uint64_t total=0;
    for(unsigned owner=0;owner<unsigned(dev.nRanks);++owner)total+=counts[owner];
    // This one capacity covers both the sorted source slot and the peer's
    // symmetric receive slot. Do not read past the source when only the
    // outgoing partition itself fits.
    uint32_t bad=local[1]||local[2]||total>cap;
    local[1]=bad;
    remote[0]=bad?0:counts[logical_owner];
  }
  lsa_rendezvous(dev,win,terminal);
}
__global__ void lsa_copy_exact(ncclDevComm dev,ncclWindow_t win,
    const uint4* hashes,const uint4* states,const uint32_t* counts,
    const uint32_t* group_fatal,
    uint32_t logical_owner,uint32_t peer,uint32_t cap,uint32_t stride,
    size_t states_offset,const uint32_t* terminal){
  if(*group_fatal)return;
  if(!lsa_rendezvous(dev,win,terminal))return;
  auto* local=static_cast<uint32_t*>(ncclGetLsaPointer(win,0,dev.lsaRank));
  bool bad=false;
  for(unsigned rank=0;rank<unsigned(dev.nRanks);++rank){
    auto* control=static_cast<const uint32_t*>(ncclGetLsaPointer(win,0,rank));
    bad|=control[1]!=0;
  }
  if(blockIdx.x==0&&threadIdx.x==0&&bad)local[2]=1;
  if(!bad){
    uint64_t begin=0;
    for(unsigned owner=0;owner<logical_owner;++owner)begin+=counts[owner];
    const uint32_t rows=counts[logical_owner];
    auto* remote=static_cast<unsigned char*>(ncclGetLsaPointer(win,0,peer));
    auto* dest_hashes=reinterpret_cast<uint4*>(remote+lsa_control_bytes);
    auto* dest_states=reinterpret_cast<uint4*>(remote+states_offset);
    const uint64_t t=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const uint64_t step=uint64_t(gridDim.x)*blockDim.x;
    if(hashes)for(uint64_t i=t;i<rows;i+=step)dest_hashes[i]=hashes[begin+i];
    if(states){
      if(stride%16==0){const uint64_t words=uint64_t(rows)*(stride/16);
        for(uint64_t i=t;i<words;i+=step)dest_states[i]=states[begin*(stride/16)+i];
      }else{auto* dst=reinterpret_cast<uint32_t*>(dest_states);auto* src=reinterpret_cast<const uint32_t*>(states);
        const uint64_t words=uint64_t(rows)*(stride/4);for(uint64_t i=t;i<words;i+=step)dst[i]=src[begin*(stride/4)+i];}
    }
  }
  lsa_rendezvous(dev,win,terminal);
}
// Same device epoch order as payload, including empty ranks. No NCCL internal
// kernel can remain unmatched between a rank-local host failure and a later
// HASH_FIRST rendezvous. CTA0 publishes/reduces; all CTA counters participate.
__global__ void lsa_fatal_vote(ncclDevComm dev,ncclWindow_t win,
    const uint32_t* send,uint32_t* receive,uint32_t* terminal,
    MgbfsStateRingControl* ring,MgbfsOwnerControl* owner){
  auto* local=static_cast<uint32_t*>(ncclGetLsaPointer(win,0,dev.lsaRank));
  if(blockIdx.x==0&&threadIdx.x==0)local[4]=ring?(ring->fatal!=0||owner->error!=0):*send;
  if(!lsa_rendezvous(dev,win,terminal))return;
  if(blockIdx.x==0&&threadIdx.x==0){
    uint32_t bad=0;
    for(unsigned rank=0;rank<unsigned(dev.nRanks);++rank){
      auto* peer=static_cast<const uint32_t*>(ncclGetLsaPointer(win,0,rank));
      bad|=peer[4]!=0;
    }
    *receive=bad;
    if(ring&&bad){
      atomicCAS(&ring->fatal,0u,22u);
      atomicCAS(&owner->error,0u,22u);
    }
  }
  lsa_rendezvous(dev,win,terminal);
  // Only the owner vote publishes logical failure. Generic fatal reductions
  // may carry a nonzero schedule/count, not an error. The final rendezvous
  // precedes publication so a healthy peer cannot miss this vote's writes.
  // Reuse the preallocated mapped terminal word: 1 is host cancellation,
  // 2 is device logical failure. Concurrent publishers only write nonzero,
  // so no PCIe atomic RMW is needed and cancellation cannot be cleared.
  // No count D2H or successful-epoch callback.
  if(ring&&blockIdx.x==0&&threadIdx.x==0&&*receive){
    cuda::atomic_ref<uint32_t,cuda::thread_scope_system> stopped(*terminal);
    stopped.store(2u,cuda::memory_order_release);
  }
}
void lsa_error(char* error,size_t capacity,const char* where,ncclResult_t code){
  if(error&&capacity)std::snprintf(error,capacity,"%s: %s",where,ncclGetErrorString(code));
}
}
extern "C" int mgbfs_nccl_lsa_prepare(void* raw,uint32_t cap,uint32_t stride,
    char* error,size_t error_capacity){
  auto* p=static_cast<Comm*>(raw);
  if(!p||!p->value||p->terminal_started||p->symmetric||p->window_ready||p->device_ready||!cap||cap>INT_MAX||
     !stride||(stride&15u)||!p->world||p->world>8)return 1;
  int version=0;
  if(ncclGetVersion(&version)!=ncclSuccess||version<22900)return 2;
  ncclCommProperties_t props=NCCL_COMM_PROPERTIES_INITIALIZER;
  auto result=ncclCommQueryProperties(p->value,&props);
  if(result!=ncclSuccess){lsa_error(error,error_capacity,"query_properties",result);return 3;}
  if(!props.deviceApiSupport||props.nLsaTeams!=1)return 4;
  const uint64_t hash_bytes=uint64_t(cap)*16;
  const uint64_t state_bytes=uint64_t(cap)*stride;
  const uint64_t state_offset=lsa_control_bytes+hash_bytes;
  if(state_offset>SIZE_MAX||state_bytes>SIZE_MAX-state_offset)return 1;
  p->states_offset=size_t(state_offset);
  p->candidate_capacity=cap;
  p->state_stride=stride;
  result=ncclMemAlloc(reinterpret_cast<void**>(&p->symmetric),
                      size_t(state_offset+state_bytes));
  if(result!=ncclSuccess){lsa_error(error,error_capacity,"mem_alloc",result);return 5;}
  if(cudaMemset(p->symmetric,0,lsa_control_bytes)!=cudaSuccess)return 6;
  return 0;
}
extern "C" int mgbfs_nccl_lsa_activate(void* raw,char* error,size_t error_capacity){
  auto* p=static_cast<Comm*>(raw);
  if(!p||!p->value||p->terminal_started||!p->symmetric||p->window_ready||p->device_ready)return 1;
  auto result=ncclSuccess;
  const size_t slot_bytes=p->states_offset+
      size_t(p->candidate_capacity)*p->state_stride;
  result=ncclCommWindowRegister(p->value,p->symmetric,
      slot_bytes,&p->window,NCCL_WIN_COLL_SYMMETRIC);
  ncclResult_t failure=ncclSuccess;
  if(const int ready=await_nccl(p,result,&failure);ready!=0){
    if(error&&error_capacity)std::snprintf(error,error_capacity,
        "window_register: progress=%d submitted=%d terminal=%d detail=%s",
        ready,int(result),int(failure),failure==ncclSuccess?
        "cancelled or timed out without terminal NCCL result":ncclGetErrorString(failure));
    return 7;
  }
  p->window_ready=true;
  ncclDevCommRequirements reqs=NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount=lsa_copy_ctas;
  result=ncclDevCommCreate(p->value,&reqs,&p->device);
  if(const int ready=await_nccl(p,result,&failure);ready!=0){
    if(error&&error_capacity)std::snprintf(error,error_capacity,
        "device_comm_create: progress=%d submitted=%d terminal=%d detail=%s",
        ready,int(result),int(failure),failure==ncclSuccess?
        "cancelled or timed out without terminal NCCL result":ncclGetErrorString(failure));
    return 8;
  }
  p->device_ready=true;
  if(p->device.lsaSize!=int(p->world))return 9;
  return 0;
}
extern "C" int mgbfs_nccl_lsa_exchange_rows(void* raw,const void* sorted_hashes,
    const void* packed_states,const uint32_t* owner_counts,
    const uint32_t* group_fatal,uint32_t logical_owner,uint32_t peer,
    uint32_t row_stride,void* raw_stream){
  auto* p=static_cast<Comm*>(raw);
  if(!p||!p->value||p->terminal_started||!p->device_ready||!p->terminal_device||
     (!packed_states&&!sorted_hashes)||!owner_counts||!group_fatal||!row_stride||
     (row_stride&3u)||row_stride>p->state_stride||
     logical_owner>=p->world||peer>=p->world||peer==p->rank)return 1;
  if(p->cancel_requested&&p->cancel_requested(p->cancel_context))return 7;
  p->lsa_used=true;
  auto stream=static_cast<cudaStream_t>(raw_stream);
  lsa_publish_count<<<1,32,0,stream>>>(p->device,p->window,owner_counts,
      group_fatal,logical_owner,peer,p->candidate_capacity,p->terminal_device);
  if(cudaGetLastError()!=cudaSuccess)return 2;
  lsa_copy_exact<<<lsa_copy_ctas,lsa_copy_threads,0,stream>>>(
      p->device,p->window,static_cast<const uint4*>(sorted_hashes),
      static_cast<const uint4*>(packed_states),owner_counts,group_fatal,
      logical_owner,peer,p->candidate_capacity,row_stride,p->states_offset,p->terminal_device);
  return cudaGetLastError()==cudaSuccess?0:3;
}
extern "C" int mgbfs_nccl_lsa_exchange(void* raw,const void* sorted_hashes,
    const void* packed_states,const uint32_t* owner_counts,
    const uint32_t* group_fatal,uint32_t logical_owner,uint32_t peer,void* raw_stream){
  auto* p=static_cast<Comm*>(raw);
  if(!p||!sorted_hashes)return 1;
  return mgbfs_nccl_lsa_exchange_rows(raw,sorted_hashes,packed_states,owner_counts,
      group_fatal,logical_owner,peer,p->state_stride,raw_stream);
}
extern "C" int mgbfs_nccl_lsa_view(void* raw,const uint32_t** count,
    const uint32_t** fatal,const void** hashes,const void** states){
  auto* p=static_cast<Comm*>(raw);
  if(!p||!p->device_ready||!count||!fatal||!hashes||!states)return 1;
  *count=reinterpret_cast<const uint32_t*>(p->symmetric);
  *fatal=reinterpret_cast<const uint32_t*>(p->symmetric+8);
  *hashes=p->symmetric+lsa_control_bytes;
  *states=p->symmetric+p->states_offset;
  return 0;
}
extern "C" int mgbfs_nccl_lsa_cancel_word(void* raw,uint32_t** word){
  auto* p=static_cast<Comm*>(raw);
  if(!p||!word||!p->device_ready||!p->terminal_host)return 1;
  *word=reinterpret_cast<uint32_t*>(p->terminal_host);
  return 0;
}
extern "C" int mgbfs_nccl_lsa_fatal_vote(void* raw,const uint32_t* send,
    uint32_t* receive,void* raw_stream){
  auto* p=static_cast<Comm*>(raw);
  if(!p||!p->value||p->terminal_started||!p->device_ready||!p->terminal_device||
     !send||!receive)return 1;
  if(p->cancel_requested&&p->cancel_requested(p->cancel_context))return 7;
  p->lsa_used=true;
  lsa_fatal_vote<<<lsa_copy_ctas,lsa_copy_threads,0,static_cast<cudaStream_t>(raw_stream)>>>(
      p->device,p->window,send,receive,p->terminal_device,nullptr,nullptr);
  return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_nccl_lsa_owner_fatal_vote(void* raw,
    MgbfsStateRingControl* ring,MgbfsOwnerControl* owner,uint32_t* receive,void* raw_stream){
  auto* p=static_cast<Comm*>(raw);
  if(!p||!p->value||p->terminal_started||!p->device_ready||!p->terminal_device||
     !ring||!owner||!receive)return 1;
  if(p->cancel_requested&&p->cancel_requested(p->cancel_context))return 7;
  p->lsa_used=true;
  lsa_fatal_vote<<<lsa_copy_ctas,lsa_copy_threads,0,static_cast<cudaStream_t>(raw_stream)>>>(
      p->device,p->window,nullptr,receive,p->terminal_device,ring,owner);
  return cudaGetLastError()==cudaSuccess?0:2;
}
#else
extern "C" int mgbfs_nccl_lsa_prepare(void*,uint32_t,uint32_t,char*,size_t){return 7;}
extern "C" int mgbfs_nccl_lsa_activate(void*,char*,size_t){return 7;}
extern "C" int mgbfs_nccl_lsa_exchange(void*,const void*,const void*,const uint32_t*,
    const uint32_t*,uint32_t,uint32_t,void*){return 7;}
extern "C" int mgbfs_nccl_lsa_exchange_rows(void*,const void*,const void*,const uint32_t*,
    const uint32_t*,uint32_t,uint32_t,uint32_t,void*){return 7;}
extern "C" int mgbfs_nccl_lsa_view(void*,const uint32_t**,const uint32_t**,
    const void**,const void**){return 7;}
extern "C" int mgbfs_nccl_lsa_cancel_word(void*,uint32_t**){return 7;}
extern "C" int mgbfs_nccl_lsa_fatal_vote(void*,const uint32_t*,uint32_t*,void*){return 7;}
extern "C" int mgbfs_nccl_lsa_owner_fatal_vote(void*,MgbfsStateRingControl*,MgbfsOwnerControl*,uint32_t*,void*){return 7;}
#endif

extern "C" int mgbfs_nccl_all_reduce_sum_u32(void* raw,const uint32_t* send,uint32_t* recv,void* raw_stream){auto*p=static_cast<Comm*>(raw);if(!p||!p->value||!send||!recv)return 1;if(p->cancel_requested&&p->cancel_requested(p->cancel_context))return 7;return await_nccl(p,ncclAllReduce(send,recv,1,ncclUint32,ncclSum,p->value,static_cast<cudaStream_t>(raw_stream)));}

// General graph queues carry GPU count, origin metadata and regenerated
// payload as one ordered submission. Capacities are agreed at startup;
// fixed-capacity fallback avoids intermediate host count readback. It is
// not a throughput claim; device-count transport remains a separate backend.
extern "C" int mgbfs_nccl_send_recv_triplet(void* raw,const void* count,uint64_t count_bytes,
 const void* metadata,uint64_t metadata_bytes,const void* states,uint64_t state_bytes,uint32_t peer,
 void* recv_count,void* recv_metadata,void* recv_states,void* stream){
 const void* send[]={count,metadata,states};void* recv[]={recv_count,recv_metadata,recv_states};
 const uint64_t bytes[]={count_bytes,metadata_bytes,state_bytes};
 return send_recv_lanes(raw,send,bytes,recv,bytes,3,peer,stream);
}

extern "C" int mgbfs_nccl_all_reduce_sum_u64(void* raw,const uint64_t* send,uint64_t* recv,void* raw_stream){
 auto*p=static_cast<Comm*>(raw);if(!p||!p->value||!send||!recv)return 1;
 if(p->cancel_requested&&p->cancel_requested(p->cancel_context))return 7;
 return await_nccl(p,ncclAllReduce(send,recv,1,ncclUint64,ncclSum,p->value,static_cast<cudaStream_t>(raw_stream)));
}

// One matched group exchanges all peer lanes. Source/receive slices are indexed
// by destination/source rank respectively; every rank agrees on lane extents.
extern "C" int mgbfs_nccl_exchange_triplets(void* raw,uint32_t rank,uint32_t world,
 const void* counts,uint64_t count_bytes,const void* records,uint64_t record_bytes,
 const void* states,uint64_t state_bytes,void* recv_counts,void* recv_records,void* recv_states,void* raw_stream){
 auto*p=static_cast<Comm*>(raw);if(!p||!p->value||p->terminal_started||p->rank!=rank||p->world!=world||!world||world>128)return 1;
 const void* send[]={counts,records,states};void* recv[]={recv_counts,recv_records,recv_states};const uint64_t bytes[]={count_bytes,record_bytes,state_bytes};
 for(unsigned lane=0;lane<3;lane++)if((bytes[lane]&&(!send[lane]||!recv[lane]))||bytes[lane]>SIZE_MAX/world)return 1;
 if(world==1)return 0;if(p->cancel_requested&&p->cancel_requested(p->cancel_context))return 7;
 if(ncclGroupStart()!=ncclSuccess)return 2;int status=0;auto stream=static_cast<cudaStream_t>(raw_stream);
 for(uint32_t peer=0;peer<world&&!status;peer++){if(peer==rank)continue;
  for(unsigned lane=0;lane<3&&!status;lane++){
   if(!bytes[lane])continue;
   auto sent=ncclSend((bytes[lane]?static_cast<const unsigned char*>(send[lane])+peer*bytes[lane]:send[lane]),size_t(bytes[lane]),ncclUint8,int(peer),p->value,stream);
   if(sent!=ncclSuccess&&sent!=ncclInProgress){status=3;break;}
   auto received=ncclRecv((bytes[lane]?static_cast<unsigned char*>(recv[lane])+peer*bytes[lane]:recv[lane]),size_t(bytes[lane]),ncclUint8,int(peer),p->value,stream);
   if(received!=ncclSuccess&&received!=ncclInProgress)status=4;
  }
 }
 const auto end=ncclGroupEnd();const int settled=(end==ncclSuccess||end==ncclInProgress)?await_nccl(p,end):5;return status?status:settled;
}

extern "C" int mgbfs_nccl_all_gather_bytes(void* raw,const void* send,void* recv,uint64_t bytes,void* raw_stream){
 auto*p=static_cast<Comm*>(raw);if(!p||!p->value||p->terminal_started||!send||!recv||!bytes||bytes>SIZE_MAX/p->world)return 1;
 if(p->cancel_requested&&p->cancel_requested(p->cancel_context))return 7;
 return await_nccl(p,ncclAllGather(send,recv,size_t(bytes),ncclUint8,p->value,static_cast<cudaStream_t>(raw_stream)));
}

// Startup metadata only: no device allocation or CUDA work.
extern "C" int mgbfs_nccl_lsa_compiled(){
#if defined(MGBFS_NCCL_LSA)
  return 1;
#else
  return 0;
#endif
}
