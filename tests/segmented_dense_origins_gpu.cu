// Device validation of compact ordinals, including empty middle sources.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
extern "C" int mgbfs_segmented_dense_origins(const uint32_t*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t*,void*);
#define CUDA(x) do { auto rc=(x); if(rc!=cudaSuccess){fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(rc));exit(1);} } while(0)
static void check(const std::vector<uint32_t>& counts,uint32_t q,uint32_t* d_offsets,uint32_t* d_out,cudaStream_t stream){
 std::vector<uint32_t> offsets(counts.size()+1),want;
 for(size_t i=0;i<counts.size();++i){
  offsets[i+1]=offsets[i]+counts[i];
  for(uint32_t j=0;j<counts[i];++j)want.push_back(uint32_t(i)*q+j);
 }
 CUDA(cudaMemcpyAsync(d_offsets,offsets.data(),offsets.size()*sizeof(uint32_t),cudaMemcpyHostToDevice,stream));
 if(want.empty()){CUDA(cudaStreamSynchronize(stream));return;}
 for(uint32_t begin=0;begin<want.size();begin+=2){
  uint32_t n=uint32_t(want.size())-begin;if(n>2)n=2;
  int rc=mgbfs_segmented_dense_origins(d_offsets,uint32_t(counts.size()),q,begin,n,d_out,stream);
  if(rc!=0){fprintf(stderr,"origin kernel error %d\n",rc);exit(2);}
  uint32_t got[2]={0xffffffffu,0xffffffffu};
  CUDA(cudaMemcpyAsync(got,d_out,n*sizeof(uint32_t),cudaMemcpyDeviceToHost,stream));
  CUDA(cudaStreamSynchronize(stream));
  for(uint32_t j=0;j<n;++j)if(got[j]!=want[begin+j]){
   fprintf(stderr,"mismatch ordinal=%u got=%u expected=%u\n",begin+j,got[j],want[begin+j]);exit(3);
  }
 }
}
int main(){
 int count=0;CUDA(cudaGetDeviceCount(&count));if(count<2){fprintf(stderr,"need two GPUs\n");return 4;}
 int limit=3125;if(const char* value=std::getenv("MGBFS_DENSE_MAP_LIMIT")){limit=std::atoi(value);if(limit<0||limit>3125)return 5;}
 for(int device=0;device<2;++device){
  CUDA(cudaSetDevice(device));uint32_t *offsets,*out;cudaStream_t stream;
  CUDA(cudaMalloc(&offsets,9*sizeof(uint32_t)));CUDA(cudaMalloc(&out,2*sizeof(uint32_t)));CUDA(cudaStreamCreate(&stream));
  check({0,3,0,1,0,0,2,0},16,offsets,out,stream);
  for(uint32_t code=0;code<uint32_t(limit);++code){
   uint32_t n=code;std::vector<uint32_t> v(5);
   for(auto &x:v){x=n%5;n/=5;}
   check(v,16,offsets,out,stream);
  }
  CUDA(cudaStreamDestroy(stream));CUDA(cudaFree(offsets));CUDA(cudaFree(out));
  printf("PASS gpu=%d vector-count=%d tile=2\n",device,limit+1);
 }
 return 0;
}
