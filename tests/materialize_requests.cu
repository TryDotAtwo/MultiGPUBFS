#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../cuda/state_commit.h"
#include "../cuda/regenerate.h"
static void require(bool x){if(!x)std::abort();}
template<class T> T* upload(const std::vector<T>& v){T* p=nullptr;require(cudaMalloc(&p,v.size()*sizeof(T))==cudaSuccess);require(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)==cudaSuccess);return p;}
int main(){
 auto origins=upload<MgbfsRegenerateOrigin>({{1,0,0,100},{0,1,0,101},{1,1,0,102},{0,0,0,103}});
 auto refs=upload<uint64_t>({2,0,3,1});auto selected=upload<uint32_t>({1,3});
 MgbfsStateRingControl ring{};ring.head=16;ring.tail=23;ring.capacity=16;ring.descriptor_capacity=8;ring.descriptor_tail=1;
 MgbfsOwnerControl owner{};owner.stage=2;owner.survivors=2;
 MgbfsStateExtent extent{};extent.sequence=21;extent.begin=5;extent.count=2;extent.granted_rows=2;
 auto r=upload<MgbfsStateRingControl>({ring});auto o=upload<MgbfsOwnerControl>({owner});auto e=upload<MgbfsStateExtent>({extent});
 auto out=upload<MgbfsRegenerateOrigin>(std::vector<MgbfsRegenerateOrigin>(2));
 auto targets=upload<uint64_t>({99,99});auto count=upload<uint32_t>({99});
 require(mgbfs_state_build_requests(origins,4,refs,4,selected,2,out,targets,count,r,o,e,nullptr)==0);
 require(cudaDeviceSynchronize()==cudaSuccess);
 MgbfsRegenerateOrigin actual[2]{};uint64_t dest[2]{};uint32_t rows=99;
 require(cudaMemcpy(actual,out,32,cudaMemcpyDeviceToHost)==cudaSuccess);
 require(cudaMemcpy(dest,targets,16,cudaMemcpyDeviceToHost)==cudaSuccess);
 require(cudaMemcpy(&rows,count,4,cudaMemcpyDeviceToHost)==cudaSuccess);require(rows==2);
 require(actual[0].source==1&&actual[0].move==0&&actual[0].parent==100&&actual[0].reserved==0);
 require(actual[1].source==0&&actual[1].move==1&&actual[1].parent==101&&actual[1].reserved==0);
 require(dest[0]==21&&dest[1]==22); // Absolute StateRef, not wrapped physical row.
 require(cudaMemcpy(&extent,e,sizeof(extent),cudaMemcpyDeviceToHost)==cudaSuccess);
 require(extent.ready==0&&extent.sequence==21&&extent.count==2); // Commit is final, states not ready.
 // Any invalid selection rejects all requests and never publishes StateReady.
 uint32_t bad[2]={1,4};require(cudaMemcpy(selected,bad,8,cudaMemcpyHostToDevice)==cudaSuccess);
 require(cudaMemset(out,0xcc,32)==cudaSuccess);require(cudaMemset(targets,0xcc,16)==cudaSuccess);
 require(mgbfs_state_build_requests(origins,4,refs,4,selected,2,out,targets,count,r,o,e,nullptr)==0);
 require(cudaDeviceSynchronize()==cudaSuccess);
 require(cudaMemcpy(&owner,o,sizeof(owner),cudaMemcpyDeviceToHost)==cudaSuccess);require(owner.error==15);
 require(cudaMemcpy(&rows,count,4,cudaMemcpyDeviceToHost)==cudaSuccess);require(rows==0);
 require(cudaMemcpy(actual,out,32,cudaMemcpyDeviceToHost)==cudaSuccess);
 for(unsigned i=0;i<32;++i)require(reinterpret_cast<const unsigned char*>(actual)[i]==0xcc);
 require(cudaMemcpy(dest,targets,16,cudaMemcpyDeviceToHost)==cudaSuccess);
 require(dest[0]==0xccccccccccccccccULL&&dest[1]==0xccccccccccccccccULL);
 cudaFree(count);cudaFree(targets);cudaFree(out);cudaFree(e);cudaFree(o);cudaFree(r);cudaFree(selected);cudaFree(refs);cudaFree(origins);
 std::puts("MATERIALIZE_REQUESTS_PASS");
}
