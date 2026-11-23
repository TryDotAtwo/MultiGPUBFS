#include "../cuda/device_batch_partition.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { \
  std::fprintf(stderr,"%s:%d CUDA: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1); } } while(0)

__global__ void partition_test(mgbfs_device_batch::Input* input,
    mgbfs_device_batch::Payload* output,uint64_t n,uint64_t cap,uint32_t world,
    uint64_t* positions,uint64_t* hist,uint64_t* offsets,uint32_t* error) {
  mgbfs_device_batch::partition(input,output,n,cap,world,positions,hist,offsets,error);
}

static void run(uint64_t n,uint32_t world,int blocks,int mode,bool invalid,bool short_output) {
  using namespace mgbfs_device_batch;
  std::vector<Input> source(n);
  for(uint64_t i=0;i<n;++i) {
    uint32_t dst=mode==0?0:mode==1?uint32_t((i*17+(i>>4))%world):uint32_t((i*i+3*i+7)%world);
    source[i]={i, i^0x9192939495969798ull, dst, 0};
  }
  if(invalid && n) source[n/2].destination=world;
  uint64_t capacity=short_output && n?n-1:n;
  Input* in=nullptr;Payload* out=nullptr;
  uint64_t *positions=nullptr,*hist=nullptr,*offsets=nullptr;
  uint32_t* error=nullptr;
  CHECK(cudaMalloc(&in,std::max<uint64_t>(n,1)*sizeof(Input)));
  CHECK(cudaMalloc(&out,(n+2)*sizeof(Payload)));
  CHECK(cudaMalloc(&positions,std::max<uint64_t>(n,1)*sizeof(uint64_t)));
  CHECK(cudaMalloc(&hist,uint64_t(blocks)*world*sizeof(uint64_t)));
  CHECK(cudaMalloc(&offsets,(world+1)*sizeof(uint64_t)));
  CHECK(cudaMalloc(&error,sizeof(uint32_t)));
  if(n)CHECK(cudaMemcpy(in,source.data(),n*sizeof(Input),cudaMemcpyHostToDevice));
  CHECK(cudaMemset(out,0xa5,(n+2)*sizeof(Payload)));
  void* args[]={&in,&out,&n,&capacity,&world,&positions,&hist,&offsets,&error};
  CHECK(cudaLaunchCooperativeKernel((void*)partition_test,dim3(blocks),dim3(128),args,0,nullptr));
  CHECK(cudaDeviceSynchronize());
  uint32_t result=0;
  CHECK(cudaMemcpy(&result,error,sizeof(result),cudaMemcpyDeviceToHost));
  std::vector<Payload> observed(n+2);
  CHECK(cudaMemcpy(observed.data(),out,observed.size()*sizeof(Payload),cudaMemcpyDeviceToHost));
  auto untouched=[](const Payload& p){return p.key==0xa5a5a5a5a5a5a5a5ull && p.packed==0xa5a5a5a5a5a5a5a5ull;};
  if(!untouched(observed[n])||!untouched(observed[n+1])) { std::fprintf(stderr,"OUTPUT_OVERRUN\n");std::exit(1); }
  if(invalid || short_output) {
    if(!result) {std::fprintf(stderr,"MISSING_FAILURE n=%llu\n",(unsigned long long)n);std::exit(1);}
    for(const auto& p:observed) if(!untouched(p)) {std::fprintf(stderr,"PARTIAL_SCATTER\n");std::exit(1);}
  } else {
    if(result) {std::fprintf(stderr,"UNEXPECTED_ERROR %u\n",result);std::exit(1);}
    std::vector<uint64_t> start(world+1);
    CHECK(cudaMemcpy(start.data(),offsets,start.size()*sizeof(uint64_t),cudaMemcpyDeviceToHost));
    if(start[0]||start[world]!=n){std::fprintf(stderr,"BAD_OFFSETS\n");std::exit(1);}
    for(uint32_t d=0;d<world;++d) {
      std::vector<uint64_t> got,expected;
      if(start[d]>start[d+1]||start[d+1]>n){std::fprintf(stderr,"BAD_RANGE\n");std::exit(1);}
      for(uint64_t j=start[d];j<start[d+1];++j){
        if(observed[j].packed!=(observed[j].key^0x9192939495969798ull)) {
          std::fprintf(stderr,"PAYLOAD_CORRUPT\n");std::exit(1);
        }
        got.push_back(observed[j].key);
      }
      for(const auto& x:source)if(x.destination==d)expected.push_back(x.key);
      std::sort(got.begin(),got.end());std::sort(expected.begin(),expected.end());
      if(got!=expected) {std::fprintf(stderr,"MULTISET_MISMATCH d=%u n=%llu\n",d,(unsigned long long)n);std::exit(1);}
    }
  }
  CHECK(cudaFree(error));CHECK(cudaFree(offsets));CHECK(cudaFree(hist));CHECK(cudaFree(positions));
  CHECK(cudaFree(out));CHECK(cudaFree(in));
}

int main(){
  int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
  int cooperative=0;CHECK(cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,0));
  if(!cooperative)return 77;
  int sm=0,active=0;CHECK(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,0));
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,partition_test,128,0));
  const int blocks=std::min(sm*active,32);
  if(blocks<2){std::fprintf(stderr,"Need a multi-CTA cooperative grid\n");return 77;}
  for(uint64_t n: {0ull,1ull,31ull,32ull,33ull,1023ull,8191ull,131073ull})
    for(uint32_t world: {1u,2u,4u,8u})
      for(int mode=0;mode<3;++mode) {
        run(n,world,1,mode,false,false);
        run(n,world,blocks,mode,false,false);
      }
  run(33,4,blocks,1,true,false);
  run(8191,4,blocks,1,false,true);
  std::printf("PASS partition single/multi CTA; blocks=%d sm=%d\n",blocks,sm);
}
