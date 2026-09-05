#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../cuda/mgbfs_cuda.h"
#include "../cuda/state_commit.h"
extern "C" int mgbfs_state_apply_responses(void*,const uint8_t*,const uint64_t*,
 const uint32_t*,const uint32_t*,uint8_t*,MgbfsStateRingControl*,MgbfsOwnerControl*,MgbfsStateExtent*,void*);
static void require(bool x){if(!x)std::abort();}
template<class T> T* upload(const std::vector<T>& v){T* p=nullptr;require(cudaMalloc(&p,v.size()*sizeof(T))==cudaSuccess);require(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)==cudaSuccess);return p;}
int main(){
 void* plan=nullptr;char error[512]{};
 require(mgbfs_materialize_create(32,4,16,&plan,error,sizeof(error))==0);
 std::vector<uint8_t> source(64,0x22);for(unsigned i=32;i<64;++i)source[i]=0x11;
 auto input=upload(source);auto targets=upload<uint64_t>({22,21});auto count=upload<uint32_t>({2});
 auto fatal=upload<uint32_t>({0});auto output=upload<uint8_t>(std::vector<uint8_t>(512,0xcc));
 MgbfsStateRingControl ring{};ring.head=16;ring.tail=23;ring.capacity=16;ring.descriptor_capacity=8;ring.descriptor_tail=1;
 MgbfsOwnerControl owner{};owner.stage=2;owner.survivors=2;
 MgbfsStateExtent extent{};extent.sequence=21;extent.begin=5;extent.count=2;extent.granted_rows=2;
 auto r=upload<MgbfsStateRingControl>({ring});auto o=upload<MgbfsOwnerControl>({owner});auto e=upload<MgbfsStateExtent>({extent});
 require(mgbfs_state_apply_responses(plan,input,targets,count,fatal,output,r,o,e,nullptr)==0);
 require(cudaDeviceSynchronize()==cudaSuccess);
 std::vector<uint8_t> actual(512);MgbfsStateExtent result{};
 require(cudaMemcpy(actual.data(),output,512,cudaMemcpyDeviceToHost)==cudaSuccess);
 for(unsigned i=0;i<512;++i)require(actual[i]==(i>=160&&i<192?0x11:(i>=192&&i<224?0x22:0xcc)));
 require(cudaMemcpy(&result,e,sizeof(result),cudaMemcpyDeviceToHost)==cudaSuccess);require(result.ready==1);
 // Duplicate, missing, foreign target and remote fatal may not publish states.
 for(unsigned scenario=0;scenario<4;++scenario){
   uint64_t refs[2]={21,scenario==0?21ULL:(scenario==2?23ULL:22ULL)};
   uint32_t rows=scenario==1?1:2,poison=scenario==3?2:0;
   require(cudaMemcpy(targets,refs,16,cudaMemcpyHostToDevice)==cudaSuccess);
   require(cudaMemcpy(count,&rows,4,cudaMemcpyHostToDevice)==cudaSuccess);
   require(cudaMemcpy(fatal,&poison,4,cudaMemcpyHostToDevice)==cudaSuccess);
   require(cudaMemcpy(r,&ring,sizeof(ring),cudaMemcpyHostToDevice)==cudaSuccess);
   require(cudaMemcpy(o,&owner,sizeof(owner),cudaMemcpyHostToDevice)==cudaSuccess);
   require(cudaMemcpy(e,&extent,sizeof(extent),cudaMemcpyHostToDevice)==cudaSuccess);
   require(cudaMemset(output,0xcc,512)==cudaSuccess);
   require(mgbfs_state_apply_responses(plan,input,targets,count,fatal,output,r,o,e,nullptr)==0);
   require(cudaDeviceSynchronize()==cudaSuccess);
   MgbfsOwnerControl status{};
   require(cudaMemcpy(&status,o,sizeof(status),cudaMemcpyDeviceToHost)==cudaSuccess);require(status.error==18);
   require(cudaMemcpy(&result,e,sizeof(result),cudaMemcpyDeviceToHost)==cudaSuccess);require(result.ready==0);
   require(cudaMemcpy(actual.data(),output,512,cudaMemcpyDeviceToHost)==cudaSuccess);
   for(auto x:actual)require(x==0xcc);
 }
 cudaFree(e);cudaFree(o);cudaFree(r);cudaFree(output);cudaFree(fatal);cudaFree(count);cudaFree(targets);cudaFree(input);
 mgbfs_materialize_destroy(plan);std::puts("APPLY_RESPONSES_PASS");
}
