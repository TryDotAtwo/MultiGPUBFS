#include "../cuda/permutation_bitmap.cuh"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
struct Flat {const int64_t* states;uint32_t stride;__device__ int64_t value(uint32_t row,uint32_t e)const{return states[uint64_t(e)*stride+row];}};
__global__ void rank_test(Flat f,BitmapDeviceDomain d,uint64_t* ranks,uint32_t* errors,uint32_t n){
 uint32_t i=threadIdx.x+blockIdx.x*blockDim.x;if(i>=n)return;uint64_t r=0;bool ok=bitmap_rank(d,f,i,&r);ranks[i]=r;errors[i]=ok?0:1;
}
int main(){
 int devices=0;CK(cudaGetDeviceCount(&devices));if(devices<2)return 1;
 const std::vector<int64_t> symbols={-7,8,10,12,90,91};
 std::vector<uint32_t> metadata(4*7);
 // Two independent two-symbol components and two fixed singletons.
 const uint32_t positions[][2]={{0,1},{2,3},{4,4},{5,5}};
 const uint32_t sizes[]={2,2,1,1},offsets[]={0,2,4,5};
 for(uint32_t i=0;i<4;i++){
  uint32_t b=7*i;metadata[b]=sizes[i];metadata[b+1]=sizes[i];metadata[b+3]=offsets[i];
  metadata[b+4]=metadata.size();for(uint32_t j=0;j<sizes[i];j++)metadata.push_back(1);
  metadata[b+2]=metadata.size();for(uint32_t j=0;j<sizes[i];j++)metadata.push_back(positions[i][j]);
  metadata[b+5]=sizes[i]==2?2:1;metadata[b+6]=0;
 }
 std::vector<std::vector<int64_t>> rows={
  {-7,8,10,12,90,91},{-7,8,12,10,90,91},{8,-7,10,12,90,91},{8,-7,12,10,90,91},
  {90,8,10,12,-7,91},{-7,-7,10,12,90,91}};
 std::vector<int64_t> soa(rows.size()*6);for(size_t r=0;r<rows.size();r++)for(size_t e=0;e<6;e++)soa[e*rows.size()+r]=rows[r][e];
 for(int dev=0;dev<2;dev++){
  CK(cudaSetDevice(dev));int64_t *dstate,*dsymbol;uint32_t *dmeta,*derr;uint64_t* drank;
  CK(cudaMalloc(&dstate,soa.size()*8));CK(cudaMalloc(&dsymbol,symbols.size()*8));CK(cudaMalloc(&dmeta,metadata.size()*4));CK(cudaMalloc(&derr,rows.size()*4));CK(cudaMalloc(&drank,rows.size()*8));
  CK(cudaMemcpy(dstate,soa.data(),soa.size()*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(dsymbol,symbols.data(),symbols.size()*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(dmeta,metadata.data(),metadata.size()*4,cudaMemcpyHostToDevice));
  BitmapDeviceDomain d{dsymbol,dmeta,0x80000004u,6,4};rank_test<<<1,32>>>(Flat{dstate,(uint32_t)rows.size()},d,drank,derr,rows.size());CK(cudaDeviceSynchronize());
  std::vector<uint64_t> ranks(rows.size());std::vector<uint32_t> errors(rows.size());CK(cudaMemcpy(ranks.data(),drank,ranks.size()*8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(errors.data(),derr,errors.size()*4,cudaMemcpyDeviceToHost));
  for(uint32_t i=0;i<4;i++){assert(errors[i]==0);assert(ranks[i]==i);}assert(errors[4]==1&&errors[5]==1);
  CK(cudaFree(dstate));CK(cudaFree(dsymbol));CK(cudaFree(dmeta));CK(cudaFree(derr));CK(cudaFree(drank));printf("PASS orbit-domain rank gpu %d\n",dev);
 }
}
