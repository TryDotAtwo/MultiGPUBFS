#include "generic_parent_action.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CUDA_CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);}}while(0)
__global__ void bounds_gate(uint32_t* counts,uint32_t* failures,uint32_t* result){
 if(blockIdx.x||threadIdx.x)return;__shared__ uint32_t error;error=0;
 IncomingAllAction<uint8_t,false,true> shared{25,32,16,1,0,2,1,3,nullptr,nullptr,nullptr,nullptr,counts,counts,nullptr};
 if(!shared.valid(16,&error)||error)++*failures;
 for(uint32_t i=0;i<4;++i){const uint32_t bad[4]={0u,32u,96u,0xffffffffu};uint32_t child=bad[i];
  error=0;if(shared.valid(child,&error)||error!=128u)++*failures;
 }
 IncomingAllAction<int64_t,false,false> flat{25,32,16,1,0,2,1,3,nullptr,nullptr,nullptr,nullptr,counts,counts,nullptr};
 error=0;if(!flat.valid(0,&error)||error)++*failures;
 for(uint32_t i=0;i<2;++i){const uint32_t bad[2]={32u,0xffffffffu};uint32_t child=bad[i];error=0;if(flat.valid(child,&error)||error!=128u)++*failures;}
 ActionT<uint8_t> graph{};
 ParentOriginAction<uint8_t> parent{shared,graph,nullptr,nullptr,nullptr,0,0,0};
 for(uint32_t i=0;i<4;++i){const uint32_t bad[4]={0u,32u,96u,0xffffffffu};uint32_t child=bad[i];error=0;if(parent.valid(child,&error)||error!=128u)++*failures;}
 error=0;counts[1]=17;if(shared.valid(16,&error)||error!=32u)++*failures;
 *result=*failures;
}
int main(){int devices=0;CUDA_CHECK(cudaGetDeviceCount(&devices));if(devices<2)return 5;
 for(int device=0;device<2;++device){CUDA_CHECK(cudaSetDevice(device));uint32_t *counts,*failures,*result;
  CUDA_CHECK(cudaMalloc(&counts,6*sizeof(uint32_t)));CUDA_CHECK(cudaMalloc(&failures,sizeof(uint32_t)));CUDA_CHECK(cudaMalloc(&result,sizeof(uint32_t)));
  uint32_t host[6]={8,8,8,8,8,8};CUDA_CHECK(cudaMemcpy(counts,host,sizeof(host),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemset(failures,0,4));
  bounds_gate<<<1,1>>>(counts,failures,result);CUDA_CHECK(cudaDeviceSynchronize());uint32_t failed=0;CUDA_CHECK(cudaMemcpy(&failed,result,4,cudaMemcpyDeviceToHost));if(failed)return 3;
  CUDA_CHECK(cudaFree(counts));CUDA_CHECK(cudaFree(failures));CUDA_CHECK(cudaFree(result));printf("INCOMING_ORIGIN_BOUNDS_PASS device=%d\n",device);
 }
}
