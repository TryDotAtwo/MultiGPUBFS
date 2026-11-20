// Same-shard tiles: a full bank still suppresses duplicates before spilling
// a genuinely new child. Reuse the original inbox until spill is consumed.
#include "generic_route.h"
#include "segmented_history.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
extern "C" int mgbfs_bitmap_accept_grouped(uint32_t,uint32_t,uint32_t,const void*,const void*,const GenericRouteRecord*,const GenericRouteRecord*,const uint32_t*,const uint32_t*,uint32_t,uint32_t,uint32_t,uint64_t,uint64_t,uint64_t*,uint32_t,uint32_t,uint32_t,uint32_t,const uint32_t*,uint32_t,MgbfsSegment,uint32_t,uint64_t*,uint32_t*,uint32_t*,uint32_t*,uint64_t*,uint64_t,void*);
extern "C" int mgbfs_segmented_accept_routes_limited_grouped_u8(uint32_t,const uint8_t*,const uint8_t*,const GenericRouteRecord*,const GenericRouteRecord*,const uint32_t*,const uint32_t*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,const uint32_t*,uint32_t,const MgbfsSegment*,uint32_t,MgbfsSegment,uint32_t,uint64_t*,uint64_t*,uint32_t*,uint32_t*,uint32_t*,uint64_t*,uint64_t,void*);
#define C(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);exit(1);}}while(0)
template<class T>T* alloc(size_t n=1){T* p=nullptr;C(cudaMalloc((void**)&p,n*sizeof(T)));return p;}
template<class T>void put(T* p,T x){C(cudaMemcpy(p,&x,sizeof(x),cudaMemcpyHostToDevice));}
template<class T>T get(T* p){T x;C(cudaMemcpy(&x,p,sizeof(x),cudaMemcpyDeviceToHost));return x;}
MgbfsSegment segment(bool bitmap){
 MgbfsSegment s{};s.capacity=1;s.banks=3;s.states=alloc<uint8_t>(6);C(cudaMemset(s.states,255,6));
 if(!bitmap){s.slots_count=16;s.slots=alloc<uint64_t>(16);C(cudaMemset(s.slots,255,128));s.position_item_bytes=4;s.positions=(uint64_t*)alloc<uint32_t>(3);}
 return s;
}
void release(MgbfsSegment s){C(cudaFree(s.states));if(s.slots)C(cudaFree(s.slots));if(s.positions)C(cudaFree(s.positions));}
void run(bool bitmap){
 GenericRouteRecord h[3]{};h[0].hash=h[1].hash=0;h[2].hash=bitmap?1:0;h[0].parent=h[1].parent=0;h[2].parent=1;
 auto* meta=alloc<GenericRouteRecord>(3);C(cudaMemcpy(meta,h,sizeof(h),cudaMemcpyHostToDevice));
 auto* count=alloc<uint32_t>();put(count,3u);auto* selection=alloc<uint32_t>(3);auto* spill=alloc<uint32_t>(3);
 auto* ns=alloc<uint32_t>();auto* err=alloc<uint32_t>();auto* accepted=alloc<uint64_t>();auto* bits=alloc<uint64_t>();put(bits,uint64_t(0));
 auto* sealed=alloc<MgbfsSegment>();MgbfsSegment a=segment(bitmap),b=segment(bitmap);
 auto execute=[&](MgbfsSegment seg,uint32_t n,uint32_t nsealed){
  put(ns,0u);put(err,0u);
  int result=bitmap?
   mgbfs_bitmap_accept_grouped(1,1,2,nullptr,nullptr,meta,meta,count,count,0,1,0,2,2,bits,0,1,3,3,selection,n,seg,0,accepted,spill,ns,err,nullptr,0,nullptr):
   mgbfs_segmented_accept_routes_limited_grouped_u8(2,nullptr,nullptr,meta,meta,count,count,0,1,0,1,3,3,1,selection,n,sealed,nsealed,seg,0,accepted,nullptr,spill,ns,err,nullptr,0,nullptr);
  C((cudaError_t)result);C(cudaDeviceSynchronize());CHECK(get(err)==0);
 };
 put(accepted,uint64_t(0));put(selection,0u);execute(a,1,0);CHECK(get(accepted)==1&&get(ns)==0);
 uint32_t two[2]={1,2};C(cudaMemcpy(selection,two,sizeof(two),cudaMemcpyHostToDevice));execute(a,2,0);
 CHECK(get(accepted)==1&&get(ns)==1&&get(spill)==2);
 C(cudaMemcpy(selection,spill,4,cudaMemcpyDeviceToDevice));put(accepted,uint64_t(0));C(cudaMemcpy(sealed,&a,sizeof(a),cudaMemcpyHostToDevice));
 execute(b,1,bitmap?0:1);CHECK(get(accepted)==1&&get(ns)==0);
 put(selection,2u);execute(b,1,bitmap?0:1);CHECK(get(accepted)==1&&get(ns)==0);
 if(bitmap)CHECK(get(bits)==3);uint8_t va=255,vb=255;C(cudaMemcpy(&va,a.states,1,cudaMemcpyDeviceToHost));C(cudaMemcpy(&vb,b.states,1,cudaMemcpyDeviceToHost));CHECK(va==0&&vb==1);
 release(a);release(b);C(cudaFree(sealed));C(cudaFree(meta));C(cudaFree(count));C(cudaFree(selection));C(cudaFree(spill));C(cudaFree(ns));C(cudaFree(err));C(cudaFree(accepted));C(cudaFree(bits));
 printf("PASS grouped two tiles duplicate full bank backend=%s\n",bitmap?"bitmap":"hash");fflush(stdout);
}
int main(){int n=0;C(cudaGetDeviceCount(&n));CHECK(n>=2);for(int d=0;d<2;d++){C(cudaSetDevice(d));run(false);run(true);}puts("WARP_TWO_TILES_TWO_GPU_PASS");}
