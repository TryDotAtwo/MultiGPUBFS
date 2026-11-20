// Physical tests of grouped bitmap acceptance. No full-layer state buffering.
#include "generic_route.h"
#include "segmented_history.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

extern "C" int mgbfs_bitmap_accept_grouped(uint32_t,uint32_t,uint32_t,const void*,const void*,const GenericRouteRecord*,const GenericRouteRecord*,const uint32_t*,const uint32_t*,uint32_t,uint32_t,uint32_t,uint64_t,uint64_t,uint64_t*,uint32_t,uint32_t,uint32_t,uint32_t,const uint32_t*,uint32_t,MgbfsSegment,uint32_t,uint64_t*,uint32_t*,uint32_t*,uint32_t*,uint64_t*,uint64_t,void*);
#define C(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);exit(1);}}while(0)
template<class T> T* alloc(size_t n=1){T* p=nullptr;C(cudaMalloc((void**)&p,n*sizeof(T)));return p;}
template<class T> T get(T* p){T x{};C(cudaMemcpy(&x,p,sizeof(T),cudaMemcpyDeviceToHost));return x;}
template<class T> void put(T* p,T x){C(cudaMemcpy(p,&x,sizeof(T),cudaMemcpyHostToDevice));}
constexpr uint32_t Q=193, CAP=7, TOTAL=4096;
static void run(bool limit){
 std::vector<uint32_t> ids(Q);
 for(uint32_t i=0;i<Q;i++){uint32_t x=(i*37)%73;ids[i]=x<5?uint32_t((uint32_t[]){0,63,64,127,128}[x]):x+128;}
 std::set<uint32_t> expected(ids.begin(),ids.end()),accepted;
 std::vector<GenericRouteRecord> meta(Q);
 for(uint32_t i=0;i<Q;i++){meta[i]={};meta[i].hash=ids[i];meta[i].parent=ids[i];}
 auto* dm=alloc<GenericRouteRecord>(Q);C(cudaMemcpy(dm,meta.data(),Q*sizeof(meta[0]),cudaMemcpyHostToDevice));
 auto* counts=alloc<uint32_t>();put(counts,Q);
 auto* bits=alloc<uint64_t>(TOTAL/64);C(cudaMemset(bits,0,TOTAL/8));
 auto* global=alloc<uint64_t>();put(global,uint64_t(0));
 auto* sel=alloc<uint32_t>(Q);auto* spill=alloc<uint32_t>(Q);auto* ns=alloc<uint32_t>();auto* err=alloc<uint32_t>();
 uint32_t selected=Q;bool initial=true,cap_was_tested=!limit;unsigned rounds=0;uint64_t global_limit=limit?3:Q;
 while(selected){
  CHECK(++rounds<100);
  MgbfsSegment seg{};seg.capacity=CAP;seg.banks=3;seg.states=alloc<uint8_t>(2*CAP*3);C(cudaMemset(seg.states,0xff,2*CAP*3));
  auto* n=alloc<uint64_t>();put(n,uint64_t(0));
  for(;;){
   put(ns,uint32_t(0));put(err,uint32_t(0));
   C(static_cast<cudaError_t>(mgbfs_bitmap_accept_grouped(1,1,2,nullptr,nullptr,dm,dm,counts,counts,0,1,0,TOTAL,TOTAL,bits,0,1,Q,Q,initial?nullptr:sel,selected,seg,0,n,spill,ns,err,global,global_limit,nullptr)));
   C(cudaDeviceSynchronize());uint32_t status=get(err);if(!cap_was_tested)CHECK(status==16);else CHECK(status==0);
   uint64_t nrows=get(n);uint32_t nspill=get(ns);CHECK(nrows<=CAP && nspill<=selected);
   std::vector<uint8_t> rows(2*CAP*3);C(cudaMemcpy(rows.data(),seg.states,rows.size(),cudaMemcpyDeviceToHost));
   std::set<uint32_t> current;
   for(uint64_t j=0;j<nrows;j++){uint32_t id=rows[j]+(uint32_t(rows[CAP*3+j])<<8);CHECK(expected.count(id));CHECK(current.insert(id).second);}
   for(auto x:current)CHECK(!accepted.count(x));
   std::vector<uint64_t> words(TOTAL/64),required(TOTAL/64);C(cudaMemcpy(words.data(),bits,TOTAL/8,cudaMemcpyDeviceToHost));
   for(auto id:accepted)required[id/64]|=1ull<<(id%64);
   for(auto id:current)required[id/64]|=1ull<<(id%64);
   CHECK(words==required);CHECK(get(global)==accepted.size()+current.size());
   if(!cap_was_tested){
    CHECK(nrows==3 && nspill>0);cap_was_tested=true;global_limit=Q;
    C(cudaMemcpy(sel,spill,nspill*4,cudaMemcpyDeviceToDevice));selected=nspill;initial=false;
    continue; // same bank after temporary cap pressure
   }
   accepted.insert(current.begin(),current.end());
   if(nspill){CHECK(nrows==CAP);C(cudaMemcpy(sel,spill,nspill*4,cudaMemcpyDeviceToDevice));}
   selected=nspill;initial=false;break;
  }
  C(cudaFree(n));C(cudaFree(seg.states));
 }
 CHECK(accepted==expected);CHECK(cap_was_tested);
 C(cudaFree(dm));C(cudaFree(counts));C(cudaFree(bits));C(cudaFree(global));C(cudaFree(sel));C(cudaFree(spill));C(cudaFree(ns));C(cudaFree(err));
 printf("PASS bitmap grouped spill/retry bits cap=%s rounds=%u unique=%zu\n",limit?"yes":"no",rounds,accepted.size());fflush(stdout);
}
int main(){int count=0;C(cudaGetDeviceCount(&count));CHECK(count>=2);
 for(int d=0;d<2;d++){C(cudaSetDevice(d));run(false);run(true);}
 puts("WARP_BITMAP_TWO_GPU_PASS");return 0;
}
