#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" int mgbfs_generic_max_queue_count(const uint32_t*,uint32_t,uint32_t*,void*);
#define CHECK(x) do{auto err=(x);if(err!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(err));return 1;}}while(0)
int main(){
 int devices=0;CHECK(cudaGetDeviceCount(&devices));if(devices<2)return 2;
 for(int dev=0;dev<2;dev++){
  CHECK(cudaSetDevice(dev));
  for(uint32_t n:{1u,7u,8u,255u,256u,257u,1024u,524288u}){
   std::vector<uint32_t> host(n);uint32_t x=0x9e3779b9u;
   for(auto& v:host){x^=x<<13;x^=x>>17;x^=x<<5;v=x&0x1ffffu;}
   host[n/2]=0xf1234567u;
   auto expected=*std::max_element(host.begin(),host.end());
   uint32_t *d=nullptr,*maximum=nullptr;CHECK(cudaMalloc(&d,n*sizeof(uint32_t)));CHECK(cudaMalloc(&maximum,sizeof(uint32_t)));
   CHECK(cudaMemcpy(d,host.data(),n*sizeof(uint32_t),cudaMemcpyHostToDevice));CHECK(cudaMemset(maximum,0,sizeof(uint32_t)));
   auto rc=mgbfs_generic_max_queue_count(d,n,maximum,nullptr);if(rc){std::fprintf(stderr,"launch error %d\n",rc);return 3;}
   uint32_t actual=0;CHECK(cudaMemcpy(&actual,maximum,sizeof(actual),cudaMemcpyDeviceToHost));
   if(actual!=expected){std::fprintf(stderr,"GPU %d n=%u expected=%u actual=%u\n",dev,n,expected,actual);return 4;}
   CHECK(cudaFree(d));CHECK(cudaFree(maximum));
   std::printf("PASS gpu=%d queue_count=%u maximum=%u\n",dev,n,actual);
  }
 }
 return 0;
}
