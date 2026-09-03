#include "mgbfs_cuda.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK(value) do { if(!(value)){std::fprintf(stderr,"CHECK failed at %s:%d: %s\n",__FILE__,__LINE__,#value);std::abort();} } while(0)

struct alignas(16) Key { uint32_t word[4]; };
Key key(uint32_t value){return {{value,0,0,0}};}
template<class T>T* device(const T* values,size_t count){T* p{};CHECK(cudaMalloc(&p,sizeof(T)*count)==cudaSuccess);if(values)CHECK(cudaMemcpy(p,values,sizeof(T)*count,cudaMemcpyHostToDevice)==cudaSuccess);return p;}
int main(){
  MgbfsMacroSettleBytes bytes{};CHECK(mgbfs_macro_settle_query(8,4,4,&bytes)==0);CHECK(bytes.indices>=32&&bytes.flags>=8&&bytes.scratch);
  void* plan{};char error[256]{};CHECK(mgbfs_macro_settle_create(8,4,4,&plan,error,sizeof(error))==0);
  Key f[8]={key(1),key(2),key(2),key(4),key(7)};uint64_t refs[8]={10,20,21,40,70};uint32_t count=5;
  Key h[16]={key(0),key(1),key(0),key(0),key(3),key(0),key(0),key(0),key(4),key(5),key(0),key(0),key(0),key(0),key(0),key(0)};
  uint32_t hc[4]={2,1,2,0};MgbfsMacroSettleState initial{};
  auto df=device(f,8);auto dh=device(h,16);auto dr=device(refs,8);auto dc=device(&count,1);auto dhc=device(hc,4);auto ds=device(&initial,1);
  auto out=device<Key>(nullptr,8);auto outrefs=device<uint64_t>(nullptr,8);auto outcount=device<uint32_t>(nullptr,1);
  CHECK(mgbfs_macro_settle_run(plan,df,dr,dc,dh,dhc,out,outrefs,outcount,ds,1,nullptr)==0);CHECK(cudaDeviceSynchronize()==cudaSuccess);
  Key actual[2]{};uint64_t actual_refs[2]{};uint32_t n{};MgbfsMacroSettleState state{};
  CHECK(cudaMemcpy(actual,out,sizeof(actual),cudaMemcpyDeviceToHost)==cudaSuccess);CHECK(cudaMemcpy(actual_refs,outrefs,sizeof(actual_refs),cudaMemcpyDeviceToHost)==cudaSuccess);
  CHECK(cudaMemcpy(&n,outcount,4,cudaMemcpyDeviceToHost)==cudaSuccess);CHECK(cudaMemcpy(&state,ds,sizeof(state),cudaMemcpyDeviceToHost)==cudaSuccess);
  CHECK(n==2&&actual[0].word[0]==2&&actual[1].word[0]==7&&actual_refs[0]==20&&actual_refs[1]==70&&state.count==2&&state.fatal==0&&state.last_epoch==1);
  mgbfs_macro_settle_destroy(plan);cudaFree(outcount);cudaFree(outrefs);cudaFree(out);cudaFree(ds);cudaFree(dhc);cudaFree(dc);cudaFree(dr);cudaFree(dh);cudaFree(df);
}
