#include "archive_pack.h"
#include <cuda_runtime.h>
#include <cassert>
#include <vector>

int main() {
  constexpr uint32_t n=3,stride=16,count=2;
  const uint8_t host[count*stride]={
    1,0,0,0,1,0,0,0,1, 0,0,0,0,0,0,0,
    0,1,0,0,0,1,1,0,0, 0,0,0,0,0,0,0,
  };
  uint8_t *states=nullptr,*output=nullptr;
  MgbfsStateRingControl *ring=nullptr;
  assert(cudaMalloc(&states,sizeof(host))==cudaSuccess);
  assert(cudaMalloc(&output,count*n)==cudaSuccess);
  assert(cudaMalloc(&ring,sizeof(*ring))==cudaSuccess);
  assert(cudaMemcpy(states,host,sizeof(host),cudaMemcpyHostToDevice)==cudaSuccess);
  assert(cudaMemset(ring,0,sizeof(*ring))==cudaSuccess);
  assert(mgbfs_archive_pack_permutation_u8(n,stride,states,count,output,ring,nullptr)==0);
  assert(cudaDeviceSynchronize()==cudaSuccess);
  uint8_t encoded[count*n]={}; MgbfsStateRingControl state{};
  assert(cudaMemcpy(encoded,output,sizeof(encoded),cudaMemcpyDeviceToHost)==cudaSuccess);
  assert(cudaMemcpy(&state,ring,sizeof(state),cudaMemcpyDeviceToHost)==cudaSuccess);
  const uint8_t expected[count*n]={0,1,2,1,2,0};
  for(unsigned i=0;i<count*n;++i)assert(encoded[i]==expected[i]);
  assert(state.fatal==0);

  const uint8_t invalid[stride]={1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0};
  assert(cudaMemcpy(states,invalid,sizeof(invalid),cudaMemcpyHostToDevice)==cudaSuccess);
  assert(mgbfs_archive_pack_permutation_u8(n,stride,states,1,output,ring,nullptr)==0);
  assert(cudaDeviceSynchronize()==cudaSuccess);
  assert(cudaMemcpy(&state,ring,sizeof(state),cudaMemcpyDeviceToHost)==cudaSuccess);
  assert(state.fatal==18);
  cudaFree(ring);cudaFree(output);cudaFree(states);
}
