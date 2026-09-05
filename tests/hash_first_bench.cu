// Synthetic canonical matrix arithmetic, not a graph BFS benchmark.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../cuda/hash_first_generate.h"
static void require(bool value){if(!value){std::fprintf(stderr,"HASH_BENCH_FAILURE\n");std::exit(1);}}
template<class T> struct Device {
 T* p=nullptr;
 explicit Device(const std::vector<T>& v){require(cudaMalloc(&p,v.size()*sizeof(T))==cudaSuccess);require(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)==cudaSuccess);}
 ~Device(){cudaFree(p);}
};
static void bench(unsigned n,unsigned parents_n,unsigned moves){
 const unsigned width=n*n,stride=(width+15)&~15u,cap=parents_n*moves;
 std::vector<uint8_t> parents(parents_n*stride,0),generators(moves*width);
 std::vector<uint32_t> coefficients(width*4),offsets={0,1,17,4294967290u};
 uint32_t rng=20260828;
 auto next=[&](){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;};
 for(unsigned p=0;p<parents_n;++p)for(unsigned j=0;j<width;++j)parents[p*stride+j]=next()%2;
 for(auto& x:generators)x=next()%2;
 for(auto& x:coefficients)x=next()%4294967291u;
 Device<uint8_t> p(parents),g(generators);
 Device<uint32_t> w(coefficients),b(offsets),count({parents_n}),out({0}),fatal({0});
 Device<uint32_t> hashes(std::vector<uint32_t>(cap*4));
 Device<MgbfsRegenerateOrigin> origins{std::vector<MgbfsRegenerateOrigin>(cap)};
 auto launch=[&](bool tc){auto f=tc?mgbfs_generate_hash_only_tc:mgbfs_generate_hash_only;
  require(f(n,moves,2,stride,parents_n,cap,0,0,p.p,g.p,w.p,b.p,count.p,hashes.p,origins.p,out.p,fatal.p,nullptr)==0);};
 // Full hash+origin comparison before timing. Independent CPU vectors are
 // exercised by the companion leaf gate; this only checks backend agreement.
 std::vector<uint32_t> reference(cap*4),actual(cap*4);
 std::vector<MgbfsRegenerateOrigin> refs(cap),got(cap);
 launch(false);require(cudaDeviceSynchronize()==cudaSuccess);
 require(cudaMemcpy(reference.data(),hashes.p,cap*16,cudaMemcpyDeviceToHost)==cudaSuccess);
 require(cudaMemcpy(refs.data(),origins.p,cap*16,cudaMemcpyDeviceToHost)==cudaSuccess);
 launch(true);require(cudaDeviceSynchronize()==cudaSuccess);
 require(cudaMemcpy(actual.data(),hashes.p,cap*16,cudaMemcpyDeviceToHost)==cudaSuccess);
 require(cudaMemcpy(got.data(),origins.p,cap*16,cudaMemcpyDeviceToHost)==cudaSuccess);
 require(reference==actual&&std::memcmp(refs.data(),got.data(),cap*16)==0);
 uint32_t emitted=0,error=0;
 require(cudaMemcpy(&emitted,out.p,4,cudaMemcpyDeviceToHost)==cudaSuccess);
 require(cudaMemcpy(&error,fatal.p,4,cudaMemcpyDeviceToHost)==cudaSuccess);
 require(emitted==cap&&error==0);
 cudaEvent_t start,end;require(cudaEventCreate(&start)==cudaSuccess);require(cudaEventCreate(&end)==cudaSuccess);
 for(bool tc:{false,true})for(int i=0;i<3;++i)launch(tc);
 require(cudaDeviceSynchronize()==cudaSuccess);
 for(int rep=0;rep<5;++rep)for(int order=0;order<2;++order){
  const bool tc=(order+rep)%2;
  const auto wall_start=std::chrono::steady_clock::now();
  require(cudaEventRecord(start)==cudaSuccess);
  for(int i=0;i<10;++i)launch(tc);
  require(cudaEventRecord(end)==cudaSuccess);require(cudaEventSynchronize(end)==cudaSuccess);
  const double wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wall_start).count()/10;
  float elapsed=0;require(cudaEventElapsedTime(&elapsed,start,end)==cudaSuccess);
  const double ms=elapsed/10.0;
  const unsigned long long bytes=parents.size()+generators.size()+coefficients.size()*4+16+12+uint64_t(cap)*32;
  std::printf("{\"status\":\"PASS\",\"scope\":\"hash_only_generation_leaf\",\"backend\":\"%s\",\"n\":%u,\"parents\":%u,\"moves\":%u,\"repeat\":%d,\"gpu_ms\":%.9f,\"wall_ms\":%.9f,\"candidates_per_second\":%.3f,\"explicit_device_payload_bytes\":%llu}\n",tc?"INT_MMA_SM75":"SCALAR",n,parents_n,moves,rep,ms,wall_ms,cap*1000.0/ms,bytes);
  std::fflush(stdout);
 }
 cudaEventDestroy(end);cudaEventDestroy(start);
}
int main(){
 require(cudaSetDevice(0)==cudaSuccess);
 for(unsigned n:{4u,12u,16u})for(unsigned moves:{6u,24u})for(unsigned parents:{16384u,65536u})bench(n,parents,moves);
 std::puts("HASH_FIRST_BENCH_PASS");
}
