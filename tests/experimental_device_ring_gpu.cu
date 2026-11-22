#include "../cuda/experimental_device_ring.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
using namespace mgbfs_device_ring;

#define CUDA_OK(expr) do { auto e=(expr); if(e!=cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(e)); return 2; } \
} while(0)

__device__ void expect(bool v, int* err, int code) { if (!v && !*err) *err=code; }

__global__ void boundaries(Slot* ring, int* err) {
  if (blockIdx.x || threadIdx.x) return;
  const unsigned long long epoch=9, capacity=1;
  init_slot(ring[0],epoch,0);
  Payload a{17,2,5}, out{};
  expect(try_write(ring,capacity,epoch,0,a)==Result::ok,err,1);
  expect(try_write(ring,capacity,epoch,1,a)==Result::not_ready,err,2);
  expect(try_write(ring,capacity,epoch+1,1,a)==Result::wrong_epoch,err,3);
  expect(try_read(ring,capacity,epoch,0,out)==Result::ok,err,4);
  expect(out.value==17 && out.kind==2 && out.source==5,err,5);
  expect(try_read(ring,capacity,epoch,0,out)==Result::ok,err,6);
  expect(try_release(ring,capacity,epoch,1)==Result::not_ready,err,7);
  expect(try_release(ring,capacity,epoch,0)==Result::ok,err,8);
  a.value=29; a.kind=3;
  expect(try_write(ring,capacity,epoch,1,a)==Result::ok,err,9);
  expect(try_read(ring,capacity,epoch,1,out)==Result::ok && out.value==29 && out.kind==3,err,10);
  expect(try_release(ring,capacity,epoch,1)==Result::ok,err,11);
  expect(try_write(ring,capacity,epoch,~0ull>>1,a)==Result::sequence_exhausted,err,12);
  expect(try_write(ring,0,epoch,0,a)==Result::invalid_capacity,err,13);
  init_slot(ring[0],epoch+1,0);
  expect(try_write(ring,capacity,epoch,0,a)==Result::wrong_epoch,err,14);
  expect(try_write(ring,capacity,epoch+1,0,a)==Result::ok,err,15);
}
__global__ void initialize(Slot* ring, unsigned capacity, unsigned long long epoch) {
  if (blockIdx.x || threadIdx.x) return;
  for(unsigned i=0;i<capacity;i++) init_slot(ring[i],epoch,i);
}
__device__ void producer(Slot* ring, unsigned capacity, unsigned long long epoch,
                         unsigned count, int* err) {
  if (threadIdx.x) return;
  for(unsigned i=0;i<count;i++) {
    Payload p{0x9e3779b900000000ull ^ i, i%13, i%7};
    unsigned attempt=0;
    Result r;
    do { r=try_write(ring,capacity,epoch,i,p); }
    while(r==Result::not_ready && ++attempt<100000000);
    if(r!=Result::ok) { *err=21; return; }
  }
}
__device__ void consumer(Slot* ring, unsigned capacity, unsigned long long epoch,
                         unsigned count, int* err) {
  if (threadIdx.x) return;
  for(unsigned i=0;i<count;i++) {
    Payload p{}; unsigned attempt=0; Result r;
    do { r=try_read(ring,capacity,epoch,i,p); }
    while(r==Result::not_ready && ++attempt<100000000);
    if(r!=Result::ok) { *err=22; return; }
    if(p.value!=(0x9e3779b900000000ull ^ i) || p.kind!=i%13 || p.source!=i%7) {
      *err=23; return;
    }
    if(try_release(ring,capacity,epoch,i)!=Result::ok) { *err=24; return; }
  }
}
__global__ void pair_test(Slot* ring, unsigned capacity, unsigned long long epoch,
                          unsigned count, int* ep, int* ec) {
  if(blockIdx.x==0) producer(ring,capacity,epoch,count,ep);
  if(blockIdx.x==1) consumer(ring,capacity,epoch,count,ec);
}
int main(int argc, char** argv) {
  Slot* ring=nullptr; int *ep=nullptr,*ec=nullptr; cudaStream_t sp,sc;
  CUDA_OK(cudaMalloc(&ring, 7*sizeof(Slot)));
  CUDA_OK(cudaMalloc(&ep,sizeof(int))); CUDA_OK(cudaMalloc(&ec,sizeof(int)));
  CUDA_OK(cudaMemset(ep,0,sizeof(int))); CUDA_OK(cudaMemset(ec,0,sizeof(int)));
  boundaries<<<1,1>>>(ring,ep); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
  int error=0; CUDA_OK(cudaMemcpy(&error,ep,sizeof(error),cudaMemcpyDeviceToHost));
  if(error) { std::fprintf(stderr,"boundary=%d\n",error); return 3; }
  std::puts("PASS boundary, backpressure, epoch, overflow"); std::fflush(stdout);
  if(argc>1 && argv[1][0]=='b') return 0;
  CUDA_OK(cudaStreamCreateWithFlags(&sp,cudaStreamNonBlocking));
  CUDA_OK(cudaStreamCreateWithFlags(&sc,cudaStreamNonBlocking));
  for(unsigned capacity: {1u,7u}) {
    CUDA_OK(cudaMemset(ep,0,sizeof(int))); CUDA_OK(cudaMemset(ec,0,sizeof(int)));
    initialize<<<1,1>>>(ring,capacity,41); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
    pair_test<<<2,1,0,sp>>>(ring,capacity,41,8192,ep,ec);
    CUDA_OK(cudaGetLastError()); CUDA_OK(cudaStreamSynchronize(sp)); CUDA_OK(cudaStreamSynchronize(sc));
    int other=0;
    CUDA_OK(cudaMemcpy(&error,ep,sizeof(error),cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(&other,ec,sizeof(other),cudaMemcpyDeviceToHost));
    if(error || other) { std::fprintf(stderr,"capacity=%u producer=%d consumer=%d\n",capacity,error,other);return 4; }
    std::printf("PASS same-device ring capacity=%u records=8192\n",capacity);
  }
  CUDA_OK(cudaStreamDestroy(sp)); CUDA_OK(cudaStreamDestroy(sc));
  CUDA_OK(cudaFree(ring)); CUDA_OK(cudaFree(ep)); CUDA_OK(cudaFree(ec));
  std::puts("PASS boundary, backpressure, epoch, overflow; same device only");
  return 0;
}
