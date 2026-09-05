#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../cuda/mgbfs_cuda.h"
#include "../cuda/regenerate.h"
extern "C" int mgbfs_materialize_sort_origins(void*,uint32_t,const MgbfsRegenerateOrigin*,
 const uint64_t*,const uint32_t*,MgbfsRegenerateOrigin*,uint64_t*,uint32_t*,void*);
static void require(bool x){if(!x)std::abort();}
template<class T> T* upload(const std::vector<T>& v){T* p=nullptr;require(cudaMalloc(&p,v.size()*sizeof(T))==cudaSuccess);require(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)==cudaSuccess);return p;}
int main(){
 void* plan=nullptr;char error[512]{};
 require(mgbfs_materialize_create(16,8,8,&plan,error,sizeof(error))==0);
 auto in=upload<MgbfsRegenerateOrigin>({{1,2,0,102},{1,1,0,100},{1,0,0,100},{1,3,0,101}});
 auto targets=upload<uint64_t>({21,22,23,24});auto count=upload<uint32_t>({4});auto fatal=upload<uint32_t>({0});
 auto out=upload<MgbfsRegenerateOrigin>(std::vector<MgbfsRegenerateOrigin>(8));
 auto dest=upload<uint64_t>(std::vector<uint64_t>(8,99));
 require(mgbfs_materialize_sort_origins(plan,1,in,targets,count,out,dest,fatal,nullptr)==0);
 require(cudaDeviceSynchronize()==cudaSuccess);
 MgbfsRegenerateOrigin actual[4]{};uint64_t refs[4]{};uint32_t status=99;
 require(cudaMemcpy(actual,out,sizeof(actual),cudaMemcpyDeviceToHost)==cudaSuccess);
 require(cudaMemcpy(refs,dest,sizeof(refs),cudaMemcpyDeviceToHost)==cudaSuccess);
 require(cudaMemcpy(&status,fatal,4,cudaMemcpyDeviceToHost)==cudaSuccess);require(status==0);
 const uint64_t parents[4]={100,100,101,102},expected[4]={22,23,24,21};
 const uint16_t moves[4]={1,0,3,2};
 for(unsigned i=0;i<4;++i)require(actual[i].source==1&&actual[i].parent==parents[i]&&actual[i].move==moves[i]&&refs[i]==expected[i]);
 // Source jobs must never mix ranks, and a malformed job cannot write output.
 require(cudaMemset(out,0xcc,8*16)==cudaSuccess);
 require(mgbfs_materialize_sort_origins(plan,0,in,targets,count,out,dest,fatal,nullptr)==0);
 require(cudaDeviceSynchronize()==cudaSuccess);
 require(cudaMemcpy(&status,fatal,4,cudaMemcpyDeviceToHost)==cudaSuccess);require(status!=0);
 require(cudaMemcpy(actual,out,sizeof(actual),cudaMemcpyDeviceToHost)==cudaSuccess);
 for(unsigned i=0;i<sizeof(actual);++i)require(reinterpret_cast<unsigned char*>(actual)[i]==0xcc);
 cudaFree(dest);cudaFree(out);cudaFree(fatal);cudaFree(count);cudaFree(targets);cudaFree(in);mgbfs_materialize_destroy(plan);
 std::puts("SORT_ORIGINS_PASS");
}
