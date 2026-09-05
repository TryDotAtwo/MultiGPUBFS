#include "archive_pack.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

static void require(bool value){if(!value)std::abort();}

int main() {
  constexpr uint32_t n=3,stride=16,count=2;
  const uint8_t host[count*stride]={
    1,0,0,0,1,0,0,0,1, 0,0,0,0,0,0,0,
    0,1,0,0,0,1,1,0,0, 0,0,0,0,0,0,0,
  };
  uint8_t *states=nullptr,*output=nullptr;
  MgbfsStateRingControl *ring=nullptr;
  require(cudaMalloc(&states,sizeof(host))==cudaSuccess);
  require(cudaMalloc(&output,count*n)==cudaSuccess);
  require(cudaMalloc(&ring,sizeof(*ring))==cudaSuccess);
  require(cudaMemcpy(states,host,sizeof(host),cudaMemcpyHostToDevice)==cudaSuccess);
  require(cudaMemset(ring,0,sizeof(*ring))==cudaSuccess);
  require(mgbfs_archive_pack_permutation_u8(n,stride,states,count,output,ring,nullptr)==0);
  require(cudaDeviceSynchronize()==cudaSuccess);
  uint8_t encoded[count*n]={}; MgbfsStateRingControl state{};
  require(cudaMemcpy(encoded,output,sizeof(encoded),cudaMemcpyDeviceToHost)==cudaSuccess);
  require(cudaMemcpy(&state,ring,sizeof(state),cudaMemcpyDeviceToHost)==cudaSuccess);
  const uint8_t expected[count*n]={0,1,2,1,2,0};
  for(unsigned i=0;i<count*n;++i)require(encoded[i]==expected[i]);
  require(state.fatal==0);

  const uint8_t invalid[stride]={1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0};
  require(cudaMemcpy(states,invalid,sizeof(invalid),cudaMemcpyHostToDevice)==cudaSuccess);
  require(mgbfs_archive_pack_permutation_u8(n,stride,states,1,output,ring,nullptr)==0);
  require(cudaDeviceSynchronize()==cudaSuccess);
  require(cudaMemcpy(&state,ring,sizeof(state),cudaMemcpyDeviceToHost)==cudaSuccess);
  require(state.fatal==18);
  cudaFree(ring);cudaFree(output);cudaFree(states);
  std::puts("ARCHIVE_PACK_PASS");
}
