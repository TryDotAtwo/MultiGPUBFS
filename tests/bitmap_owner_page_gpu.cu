#include "../cuda/bitmap_owner_page.cuh"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do{auto err=(x);if(err!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(err));exit(1);}}while(0)
__global__ void map_kernel(const uint64_t* input,uint64_t* output,uint32_t n,uint32_t world,uint32_t page,uint64_t boundary){
 uint32_t i=threadIdx.x+blockIdx.x*blockDim.x;if(i<n)output[i]=mgbfs_bitmap_permute_owner_id(input[i],world,page,boundary);
}
static uint64_t mix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
static uint64_t ref(uint64_t id,uint32_t world,uint32_t page,uint64_t boundary){
 if(!page||id>=boundary)return id;
 uint64_t p=id/page,stripe=p/world,slot=p%world,owner=(slot+mix(stripe)%world)%world;
 return (stripe*page+id%page)*world+owner;
}
int main(){
 int count=0;CHECK(cudaGetDeviceCount(&count));if(count<2)return 77;
 for(int dev=0;dev<2;dev++){
  CHECK(cudaSetDevice(dev));
  for(uint32_t world:{1u,3u,5u,7u,8u,128u})for(uint32_t page:{0u,1u,2u,8u,32u,64u,256u})
  for(uint64_t total:std::vector<uint64_t>{1ull,129ull,4095ull,4294967296ull,9223372036854775945ull,UINT64_MAX}){
   uint64_t stripe=uint64_t(page)*world,boundary=page?total-total%stripe:0;
   std::vector<uint64_t> v={0,total/2,total-1};if(boundary){v.push_back(boundary-1);if(boundary<total)v.push_back(boundary);}
   uint64_t *in,*out;CHECK(cudaMalloc(&in,v.size()*8));CHECK(cudaMalloc(&out,v.size()*8));CHECK(cudaMemcpy(in,v.data(),v.size()*8,cudaMemcpyHostToDevice));
   map_kernel<<<1,32>>>(in,out,v.size(),world,page,boundary);CHECK(cudaDeviceSynchronize());
   std::vector<uint64_t> result(v.size());CHECK(cudaMemcpy(result.data(),out,v.size()*8,cudaMemcpyDeviceToHost));
   for(size_t i=0;i<v.size();i++)if(result[i]!=ref(v[i],world,page,boundary)||result[i]>=total){fprintf(stderr,"FAIL gpu=%d world=%u page=%u total=%llu id=%llu got=%llu\n",dev,world,page,(unsigned long long)total,(unsigned long long)v[i],(unsigned long long)result[i]);return 1;}
   CHECK(cudaFree(in));CHECK(cudaFree(out));
  }
  printf("PASS GPU %d page-map, partial tails and UINT64_MAX\n",dev);
 }
}
