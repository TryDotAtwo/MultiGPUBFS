#include "../cuda/permutation_bitmap.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d CUDA %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
struct Flat {
 const int64_t* states;uint32_t stride;
 __device__ int64_t value(uint32_t row,uint32_t element)const{return states[uint64_t(element)*stride+row];}
};
__global__ void rank_test(Flat f,BitmapDeviceDomain d,uint64_t* ranks,uint32_t* errors,uint32_t count){
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 uint64_t rank=0;bool ok=bitmap_rank(d,f,i,&rank);
 ranks[i]=rank;errors[i]=ok?0:1;
}
__global__ void claim_test(const uint64_t* ranks,uint32_t count,uint32_t world,uint32_t owner,uint64_t* words,uint32_t* wins){
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 uint64_t r=ranks[i];if(r%world!=owner)return;
 uint64_t local=r/world,bit=1ull<<(local%64);
 auto old=atomicOr(reinterpret_cast<unsigned long long*>(&words[local/64]),(unsigned long long)bit);
 if(!(old&bit))atomicAdd(&wins[r],1u);
}
static std::vector<int64_t> unrank_unique(int n,uint64_t r){
 std::vector<int64_t> values(n),out;for(int i=0;i<n;i++)values[i]=i;
 uint64_t f=1;for(int i=2;i<n;i++)f*=i;
 for(int remain=n;remain;remain--){
  uint64_t j=remain==1?0:r/f;r=remain==1?0:r%f;out.push_back(values.at(j));values.erase(values.begin()+j);
  if(remain>1)f/=uint64_t(remain-1);
 }return out;
}
static void run(int device,std::vector<int64_t> start,bool exhaustive){
 CK(cudaSetDevice(device));std::sort(start.begin(),start.end());uint32_t n=start.size();
 std::vector<std::vector<int64_t>> states;std::vector<uint64_t> expect;
 if(exhaustive){auto s=start;do{expect.push_back(states.size());states.push_back(s);}while(std::next_permutation(s.begin(),s.end()));}
 else{
  uint64_t factorial=1;for(uint64_t i=2;i<=uint64_t(n);i++)factorial*=i;
  for(uint64_t r:std::vector<uint64_t>{0,1,4294967313ull,factorial/2,factorial-1}){states.push_back(unrank_unique(n,r));expect.push_back(r);}
 }
 std::vector<int64_t> symbols;std::vector<uint32_t> counts;
 for(auto x:start){if(symbols.empty()||symbols.back()!=x){symbols.push_back(x);counts.push_back(1);}else counts.back()++;}
 const uint32_t m=states.size();std::vector<int64_t> soa(uint64_t(n)*m);
 for(uint32_t i=0;i<m;i++)for(uint32_t e=0;e<n;e++)soa[uint64_t(e)*m+i]=states[i][e];
 int64_t *ds,*dv;uint32_t *dc,*de;uint64_t *dr;
 CK(cudaMalloc(&ds,soa.size()*8));CK(cudaMalloc(&dv,symbols.size()*8));CK(cudaMalloc(&dc,counts.size()*4));CK(cudaMalloc(&de,m*4));CK(cudaMalloc(&dr,m*8));
 CK(cudaMemcpy(ds,soa.data(),soa.size()*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(dv,symbols.data(),symbols.size()*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(dc,counts.data(),counts.size()*4,cudaMemcpyHostToDevice));
 uint64_t total=1;if(exhaustive)total=m;else for(uint64_t i=2;i<=uint64_t(n);i++)total*=i;
 BitmapDeviceDomain d{dv,dc,(uint32_t)symbols.size(),n,total};
 rank_test<<<(m+255)/256,256>>>(Flat{ds,m},d,dr,de,m);CK(cudaDeviceSynchronize());
 std::vector<uint64_t> got(m);std::vector<uint32_t> errors(m);
 CK(cudaMemcpy(got.data(),dr,m*8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(errors.data(),de,m*4,cudaMemcpyDeviceToHost));
 for(uint32_t i=0;i<m;i++){if(errors[i]||got[i]!=expect[i]){fprintf(stderr,"rank mismatch n=%u i=%u got=%llu expected=%llu err=%u\n",n,i,(unsigned long long)got[i],(unsigned long long)expect[i],errors[i]);exit(1);}}
 if(exhaustive){
  uint64_t* bits;uint32_t* wins;size_t bytes=((uint64_t(m)+63)/64)*8;
  CK(cudaMalloc(&bits,bytes));CK(cudaMemset(bits,0,bytes));CK(cudaMalloc(&wins,m*4));CK(cudaMemset(wins,0,m*4));
  cudaStream_t a,b;CK(cudaStreamCreateWithFlags(&a,cudaStreamNonBlocking));CK(cudaStreamCreateWithFlags(&b,cudaStreamNonBlocking));
  for(int x=0;x<5;x++){claim_test<<<(m+255)/256,256,0,a>>>(dr,m,1,0,bits,wins);claim_test<<<(m+255)/256,256,0,b>>>(dr,m,1,0,bits,wins);}
  CK(cudaDeviceSynchronize());std::vector<uint32_t> host(m);CK(cudaMemcpy(host.data(),wins,m*4,cudaMemcpyDeviceToHost));
  for(uint32_t i=0;i<m;i++)if(host[i]!=1){fprintf(stderr,"claim mismatch %u %u\n",i,host[i]);exit(1);}
  CK(cudaStreamDestroy(a));CK(cudaStreamDestroy(b));CK(cudaFree(bits));CK(cudaFree(wins));
 }
 // An input containing a duplicate in a unique domain must be rejected.
 if(symbols.size()==n&&n>1){
  int64_t duplicate=soa[m];CK(cudaMemcpy(ds,&duplicate,8,cudaMemcpyHostToDevice));
  rank_test<<<(m+255)/256,256>>>(Flat{ds,m},d,dr,de,m);CK(cudaDeviceSynchronize());uint32_t bad=0;CK(cudaMemcpy(&bad,de,4,cudaMemcpyDeviceToHost));assert(bad==1);
 }
 // An inconsistent unique descriptor must be rejected, even when the
 // symbols themselves happen to look like a valid permutation.
 if(symbols.size()==n){
  int64_t restore=soa[0];CK(cudaMemcpy(ds,&restore,8,cudaMemcpyHostToDevice));
  uint32_t bad_count=2;CK(cudaMemcpy(dc,&bad_count,4,cudaMemcpyHostToDevice));
  rank_test<<<1,1>>>(Flat{ds,m},d,dr,de,1);CK(cudaDeviceSynchronize());
  uint32_t bad=0;CK(cudaMemcpy(&bad,de,4,cudaMemcpyDeviceToHost));assert(bad==1);
  bad_count=1;CK(cudaMemcpy(dc,&bad_count,4,cudaMemcpyHostToDevice));
 }
 // Mutate first symbol outside domain and require a clean rejected state.
 int64_t invalid=INT64_MIN;CK(cudaMemcpy(ds,&invalid,8,cudaMemcpyHostToDevice));
 rank_test<<<(m+255)/256,256>>>(Flat{ds,m},d,dr,de,m);CK(cudaDeviceSynchronize());uint32_t bad=0;CK(cudaMemcpy(&bad,de,4,cudaMemcpyDeviceToHost));assert(bad==1);
 CK(cudaFree(ds));CK(cudaFree(dv));CK(cudaFree(dc));CK(cudaFree(de));CK(cudaFree(dr));
 printf("PASS GPU %d domain %llu states %u\n",device,(unsigned long long)total,m);
}
int main(){
 int devices=0;CK(cudaGetDeviceCount(&devices));if(devices<2){fprintf(stderr,"need two GPUs\n");return 1;}
 for(int d=0;d<2;d++){
  if(std::getenv("MGBFS_TEST_SMALL")){
   run(d,{0,1,2,3},true);
   run(d,{-7,-7,0,99,99},true);
   run(d,{-200,-7,13,999},true);
   continue;
  }
  run(d,{0,1,2,3,4,5,6,7},true);
  run(d,{0,1,2,3,4,5,6,7,8},true);
  run(d,{-7,-7,0,99,99},true);
  run(d,{-200,-7,13,999},true);
  run(d,{0,1,2,3,4,5,6,7,8,9,10,11,12},false);
  run(d,{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19},false);
 }
 return 0;
}
