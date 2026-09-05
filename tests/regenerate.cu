// Selected requests must regenerate from parents, never from stored children.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
struct Origin { uint32_t source; uint16_t move, reserved; uint64_t parent; };
static_assert(sizeof(Origin)==16);
extern "C" int mgbfs_regenerate_selected(uint32_t n,uint32_t moves,uint32_t modulus,
 uint32_t stride,uint32_t capacity,uint32_t source_rank,uint64_t parent_begin,
 uint32_t parent_count,const uint8_t* parents,const uint8_t* generators,
 const Origin* requests,const uint32_t* count,uint8_t* output,uint32_t* fatal,void* stream);
static void require(bool x){if(!x)std::abort();}
template<class T> T* upload(const std::vector<T>& v){T* p=nullptr;require(cudaMalloc(&p,v.size()*sizeof(T))==cudaSuccess);require(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)==cudaSuccess);return p;}
int main(){
  // Two 2x2 parents, two generators; request order differs from parent order.
  std::vector<uint8_t> parents(32),generators={1,1,0,1, 1,0,1,1};
  parents[0]=parents[3]=1; parents[16]=parents[19]=1; parents[17]=2;
  auto p=upload(parents),g=upload(generators);
  std::vector<Origin> req={{1,1,0,101},{1,0,0,100},{1,0,0,101}};
  auto r=upload(req); auto count=upload<uint32_t>({3}),fatal=upload<uint32_t>({0});
  auto out=upload<uint8_t>(std::vector<uint8_t>(48,0xcc));
  require(mgbfs_regenerate_selected(2,2,5,16,3,1,100,2,p,g,r,count,out,fatal,nullptr)==0);
  require(cudaDeviceSynchronize()==cudaSuccess);
  std::vector<uint8_t> actual(48); uint32_t error=99;
  require(cudaMemcpy(actual.data(),out,48,cudaMemcpyDeviceToHost)==cudaSuccess);
  require(cudaMemcpy(&error,fatal,4,cudaMemcpyDeviceToHost)==cudaSuccess);require(error==0);
  const uint8_t expected[12]={1,2,1,3, 1,1,0,1, 1,3,0,1};
  for(unsigned i=0;i<3;++i)for(unsigned j=0;j<16;++j)require(actual[i*16+j]==(j<4?expected[i*4+j]:0));
  // One stale request rejects the entire batch before any destination write.
  req[2].parent=99;require(cudaMemcpy(r,req.data(),48,cudaMemcpyHostToDevice)==cudaSuccess);
  require(cudaMemset(out,0xcc,48)==cudaSuccess);
  require(mgbfs_regenerate_selected(2,2,5,16,3,1,100,2,p,g,r,count,out,fatal,nullptr)==0);
  require(cudaDeviceSynchronize()==cudaSuccess);
  require(cudaMemcpy(actual.data(),out,48,cudaMemcpyDeviceToHost)==cudaSuccess);
  require(cudaMemcpy(&error,fatal,4,cudaMemcpyDeviceToHost)==cudaSuccess);require(error!=0);
  for(auto x:actual)require(x==0xcc);
  cudaFree(out);cudaFree(fatal);cudaFree(count);cudaFree(r);cudaFree(g);cudaFree(p);
  std::puts("REGENERATE_PASS");
}
