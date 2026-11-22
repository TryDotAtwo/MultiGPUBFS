// Two-device CUDA P2P leaf test of experimental_device_ring.cuh.
// No NCCL, no BFS, no multi-process acceptance. Each GPU runs one scheduler
// kernel which alternates sends and receives; no host submits per record.
#include "../cuda/experimental_device_ring.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

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
__global__ void exchange(Mailbox* local, Mailbox* remote, unsigned rank,
    unsigned capacity, unsigned long long epoch, unsigned send_count,
    unsigned receive_count, unsigned delayed_reads, Stats* result) {
  if(blockIdx.x || threadIdx.x) return;
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

int main() {
  int count=0;CUDA_OK(cudaGetDeviceCount(&count));
  if(count!=2){std::fprintf(stderr,"need exactly 2 visible GPUs; got %d\n",count);return 77;}
  for(int src=0;src<2;src++){
    const int dst=1-src;int can=0,atom=0,coop=0;
    CUDA_OK(cudaDeviceCanAccessPeer(&can,src,dst));
    CUDA_OK(cudaDeviceGetP2PAttribute(&atom,cudaDevP2PAttrNativeAtomicSupported,src,dst));
    CUDA_OK(cudaDeviceGetAttribute(&coop,cudaDevAttrCooperativeLaunch,src));
    std::printf("CAPABILITY src=%d dst=%d p2p=%d native_atomics=%d coop=%d\n",src,dst,can,atom,coop);
    if(!can||!atom)return 77;
    CUDA_OK(cudaSetDevice(src));
    auto e=cudaDeviceEnablePeerAccess(dst,0);
    if(e!=cudaSuccess && e!=cudaErrorPeerAccessAlreadyEnabled) {
      std::fprintf(stderr,"enable peer %d->%d: %s\n",src,dst,cudaGetErrorString(e));return 2;
    }
    if(e==cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
  }
  Mailbox* box[2]{};Stats* stats[2]{};cudaStream_t stream[2]{};
  for(int i=0;i<2;i++){
    CUDA_OK(cudaSetDevice(i));CUDA_OK(cudaMalloc(&box[i],sizeof(Mailbox)));
    CUDA_OK(cudaMalloc(&stats[i],sizeof(Stats)));
    CUDA_OK(cudaStreamCreateWithFlags(&stream[i],cudaStreamNonBlocking));
  }
  struct Case{unsigned cap,a,b,delay;};
  for(const Case c: {Case{1,8192,8192,0},Case{7,8192,8192,0},
                      Case{1,2048,0,10000},Case{7,0,2048,0}}) {
    static unsigned long long epoch=101;epoch++;
    for(int i=0;i<2;i++){
      CUDA_OK(cudaSetDevice(i));init_box<<<1,1,0,stream[i]>>>(box[i],c.cap,epoch);
      CUDA_OK(cudaGetLastError());CUDA_OK(cudaMemsetAsync(stats[i],0,sizeof(Stats),stream[i]));
    }
    for(int i=0;i<2;i++){CUDA_OK(cudaSetDevice(i));CUDA_OK(cudaStreamSynchronize(stream[i]));}
    for(int i=0;i<2;i++){
      CUDA_OK(cudaSetDevice(i));
      exchange<<<1,1,0,stream[i]>>>(box[i],box[1-i],i,c.cap,epoch,
          i?c.b:c.a,i?c.a:c.b,i?c.delay:0,stats[i]);
      CUDA_OK(cudaGetLastError());
    }
    Stats host[2]{};
    for(int i=0;i<2;i++) {
      CUDA_OK(cudaSetDevice(i));CUDA_OK(cudaStreamSynchronize(stream[i]));
      CUDA_OK(cudaMemcpy(&host[i],stats[i],sizeof(Stats),cudaMemcpyDeviceToHost));
      std::printf("CASE epoch=%llu cap=%u rank=%d sent=%llu received=%llu blocked=%llu loops=%llu error=%d\n",
          epoch,c.cap,i,host[i].sent,host[i].received,host[i].send_blocked,host[i].loops,host[i].error);
      if(host[i].error || host[i].sent!=(i?c.b:c.a) || host[i].received!=(i?c.a:c.b))return 3;
    }
    if(c.delay && !host[0].send_blocked){std::fprintf(stderr,"missing forced backpressure\n");return 4;}
  }
  for(int i=0;i<2;i++){
    CUDA_OK(cudaSetDevice(i));CUDA_OK(cudaStreamDestroy(stream[i]));
    CUDA_OK(cudaFree(stats[i]));CUDA_OK(cudaFree(box[i]));
  }
  std::puts("PASS two GPU CUDA P2P mailbox; no NCCL or BFS");
  return 0;
}
