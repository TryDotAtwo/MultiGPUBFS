#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include "generic_action.cuh"
__global__ void buckets(const uint64_t* hashes,uint32_t* bins,uint32_t* old_bins,uint32_t count){
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x){
  bins[i]=uint32_t(mgbfs_shared_table_bucket(hashes[i],65536)>>12);
  old_bins[i]=uint32_t(__umul64hi(hashes[i],uint64_t(65536))>>12);
 }
}
uint64_t random_word(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
int main(){
 constexpr uint32_t count=8192;uint64_t* input;uint32_t *out,*old;
 if(cudaMalloc(&input,count*8)||cudaMalloc(&out,count*4)||cudaMalloc(&old,count*4))return 2;
 for(uint32_t world: {1u,2u,8u,128u})for(uint32_t shards:{1u,4u,16u,4096u})for(uint32_t rank:{0u,world-1}){
  std::vector<uint64_t> hashes(count);std::vector<uint32_t> bins(count),legacy(count);uint32_t h[16]={},before[16]={};
  const uint64_t rank_begin=(uint64_t(rank)<<32)/world,rank_end=(uint64_t(rank+1)<<32)/world;
  const uint32_t shard=shards-1;const uint64_t shard_begin=(uint64_t(shard)<<32)/shards,shard_end=(uint64_t(shard+1)<<32)/shards;
  for(uint32_t i=0;i<count;i++){uint64_t hi=rank_begin+random_word(i+123)%(rank_end-rank_begin),lo=shard_begin+random_word(i+999)%(shard_end-shard_begin);hashes[i]=(hi<<32)|lo;}
  if(cudaMemcpy(input,hashes.data(),count*8,cudaMemcpyHostToDevice))return 3;
  buckets<<<32,256>>>(input,out,old,count);if(cudaDeviceSynchronize())return 4;
  if(cudaMemcpy(bins.data(),out,count*4,cudaMemcpyDeviceToHost)||cudaMemcpy(legacy.data(),old,count*4,cudaMemcpyDeviceToHost))return 5;
  for(uint32_t i=0;i<count;i++){if(bins[i]>=16)return 6;h[bins[i]]++;before[legacy[i]]++;}
  uint32_t occupied=0,maximum=0,legacy_occupied=0;for(uint32_t i=0;i<16;i++){occupied+=h[i]!=0;legacy_occupied+=before[i]!=0;if(h[i]>maximum)maximum=h[i];}
  if(occupied!=16||maximum>count/10)return 7;
  if(world>=8&&legacy_occupied>(16+world-1)/world)return 8;
  printf("world=%u shards=%u rank=%u occupied=%u maximum=%u legacy_occupied=%u\n",world,shards,rank,occupied,maximum,legacy_occupied);
 }
 cudaFree(input);cudaFree(out);cudaFree(old);puts("VERIFIED_CONDITIONAL_SHARED_BUCKET_DISTRIBUTION");return 0;
}
