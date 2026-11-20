// Physical GPU regression of the production rank-wide 64-bit reservation.
// Initial counts are synthetic; this does not materialize 2^32 live states.
#include "generic_route.h"
#include "segmented_history.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

extern "C" int mgbfs_segmented_accept_routes_limited_u8(uint32_t,const uint8_t*,const uint8_t*,const GenericRouteRecord*,const GenericRouteRecord*,const uint32_t*,const uint32_t*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,const uint32_t*,uint32_t,const MgbfsSegment*,uint32_t,MgbfsSegment,uint32_t,uint64_t*,uint64_t*,uint32_t*,uint32_t*,uint32_t*,uint64_t*,uint64_t,void*);
extern "C" int mgbfs_bitmap_accept(uint32_t,uint32_t,uint32_t,const void*,const void*,const GenericRouteRecord*,const GenericRouteRecord*,const uint32_t*,const uint32_t*,uint32_t,uint32_t,uint32_t,uint64_t,uint64_t,uint64_t*,uint32_t,uint32_t,uint32_t,uint32_t,const uint32_t*,uint32_t,MgbfsSegment,uint32_t,uint64_t*,uint32_t*,uint32_t*,uint32_t*,uint64_t*,uint64_t,void*);
#define C(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s at %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
static void check(bool ok,const char* what){if(!ok){fprintf(stderr,"FAIL %s\n",what);exit(1);}}
template<class T> static T* alloc(size_t count=1){T* p=nullptr;C(cudaMalloc((void**)&p,sizeof(T)*count));return p;}
template<class T> static void put(T* p,const T& x){C(cudaMemcpy(p,&x,sizeof(T),cudaMemcpyHostToDevice));}
template<class T> static T get(T* p){T x;C(cudaMemcpy(&x,p,sizeof(T),cudaMemcpyDeviceToHost));return x;}
constexpr uint32_t SHARDS=8, Q=1024;
constexpr uint64_t BASE=uint64_t(1)<<32;
struct Lane {
 MgbfsSegment seg{}; uint64_t* accepted{}; uint32_t* spill{}; uint32_t* nspill{}; uint32_t* err{}; cudaStream_t stream{};
};
enum Case {Cross,Limit,Spill,Duplicates};
static const char* case_name(Case k){switch(k){case Cross:return "cross";case Limit:return "limit";case Spill:return "spill";default:return "duplicates";}}
static void run(bool bitmap,uint32_t width,Case mode){
 const uint64_t cap=(mode==Spill||mode==Duplicates)?1:Q;
 std::vector<GenericRouteRecord> h(SHARDS*Q);
 for(uint32_t s=0;s<SHARDS;s++)for(uint32_t i=0;i<Q;i++){
  const uint32_t v=mode==Duplicates?42:i;
  auto& x=h[s*Q+i];x={};x.hash=uint64_t(s)*Q+v;x.parent=v;
 }
 GenericRouteRecord* meta=alloc<GenericRouteRecord>(h.size()); C(cudaMemcpy(meta,h.data(),h.size()*sizeof(h[0]),cudaMemcpyHostToDevice));
 uint32_t* counts=alloc<uint32_t>(SHARDS);std::vector<uint32_t> hc(SHARDS,Q);C(cudaMemcpy(counts,hc.data(),SHARDS*4,cudaMemcpyHostToDevice));
 uint64_t* bits=alloc<uint64_t>((SHARDS*Q)/64);C(cudaMemset(bits,0,(SHARDS*Q)/8));
 uint64_t initial=mode==Limit?BASE-2048:(mode==Spill?BASE-4:BASE-128);
 uint64_t limit=mode==Limit?BASE+2048:BASE+uint64_t(SHARDS)*Q+1;
 uint64_t* global=alloc<uint64_t>();put(global,initial);
 std::vector<Lane> a(SHARDS);
 for(auto& l:a){
  l.seg.capacity=cap;l.seg.banks=3;l.seg.position_item_bytes=width;l.seg.states=alloc<uint8_t>(size_t(2*cap*3));
  C(cudaMemset(l.seg.states,0xff,size_t(2*cap*3)));
  if(!bitmap){l.seg.slots_count=4096;l.seg.slots=alloc<uint64_t>(4096);C(cudaMemset(l.seg.slots,0xff,4096*8));l.seg.positions=reinterpret_cast<uint64_t*>(alloc<uint8_t>(size_t(cap*3*width)));}
  l.accepted=alloc<uint64_t>();l.spill=alloc<uint32_t>(Q);l.nspill=alloc<uint32_t>();l.err=alloc<uint32_t>();
  C(cudaMemset(l.accepted,0,8));C(cudaMemset(l.nspill,0,4));C(cudaMemset(l.err,0,4));C(cudaStreamCreateWithFlags(&l.stream,cudaStreamNonBlocking));
 }
 for(uint32_t s=0;s<SHARDS;s++){
  auto& l=a[s];int status=bitmap?
   mgbfs_bitmap_accept(1,1,2,nullptr,nullptr,meta,meta,counts,counts,0,1,0,SHARDS*Q,SHARDS*Q,bits,s,SHARDS,Q,Q,nullptr,Q,l.seg,0,l.accepted,l.spill,l.nspill,l.err,global,limit,l.stream):
   mgbfs_segmented_accept_routes_limited_u8(2,nullptr,nullptr,meta,meta,counts,counts,0,1,s,SHARDS,Q,Q,1,nullptr,Q,nullptr,0,l.seg,0,l.accepted,nullptr,l.spill,l.nspill,l.err,global,limit,l.stream);
  C(static_cast<cudaError_t>(status));
 }
 for(auto& l:a)C(cudaStreamSynchronize(l.stream));
 const uint64_t total=get(global);uint64_t accepted=0;uint32_t errors=0;uint64_t spills=0;
 std::vector<uint64_t> expected_bits(SHARDS*Q/64,0), actual_bits(expected_bits.size());
 for(uint32_t s=0;s<SHARDS;s++){
  auto& l=a[s];uint64_t n=get(l.accepted);uint32_t ns=get(l.nspill),err=get(l.err);accepted+=n;spills+=ns;errors|=err;
  check(n<=cap,"lane count exceeds capacity");if(mode==Cross)check(n==Q&&ns==0&&err==0,"cross lane");
  if(mode==Spill)check(n==1&&ns==Q-1&&err==0,"spill lane");
  if(mode==Duplicates)check(n==1&&ns==0&&err==0,"duplicate full lane");
  std::vector<uint8_t> state(size_t(2*cap*3));C(cudaMemcpy(state.data(),l.seg.states,state.size(),cudaMemcpyDeviceToHost));
  std::set<uint32_t> seen;
  for(uint64_t i=0;i<n;i++){
   uint32_t v=uint32_t(state[i])+(uint32_t(state[cap*3+i])<<8);
   check(v<Q,"invalid materialized child");check(seen.insert(v).second,"duplicate child accepted");
   const uint64_t bit=uint64_t(s)*Q+v;expected_bits[bit/64]|=uint64_t(1)<<(bit%64);
   if(mode==Duplicates)check(v==42,"duplicate materialization");
  }
 }
 check(total==initial+accepted,"shared count != initial + accepted");
 if(mode==Cross)check(accepted==uint64_t(SHARDS)*Q&&total>BASE&&errors==0,"counter crossing");
 if(mode==Limit)check(accepted==4096&&total==limit&&(errors&MGBFS_SEGMENTED_CAPACITY_LIMIT),"hard limit at 64 bits");
 if(mode==Spill)check(accepted==SHARDS&&spills==uint64_t(SHARDS)*(Q-1)&&total>BASE,"spill rollback");
 if(mode==Duplicates)check(accepted==SHARDS&&spills==0&&errors==0,"duplicate at full capacity");
 if(bitmap){C(cudaMemcpy(actual_bits.data(),bits,actual_bits.size()*8,cudaMemcpyDeviceToHost));check(actual_bits==expected_bits,"bitmap committed bits disagree with actual accepted states");}
 printf("PASS backend=%s positions=%u case=%s initial=%llu accepted=%llu final=%llu spills=%llu errors=%u\n",bitmap?"bitmap":"hash",width,case_name(mode),(unsigned long long)initial,(unsigned long long)accepted,(unsigned long long)total,(unsigned long long)spills,errors);fflush(stdout);
 for(auto& l:a){C(cudaStreamDestroy(l.stream));C(cudaFree(l.seg.states));if(!bitmap){C(cudaFree(l.seg.slots));C(cudaFree(l.seg.positions));}C(cudaFree(l.accepted));C(cudaFree(l.spill));C(cudaFree(l.nspill));C(cudaFree(l.err));}
 C(cudaFree(global));C(cudaFree(bits));C(cudaFree(counts));C(cudaFree(meta));
}
int main(){
 int devices=0;C(cudaGetDeviceCount(&devices));check(devices>0,"no GPU");C(cudaSetDevice(0));
 for(int iteration=0;iteration<5;iteration++)for(Case c:{Cross,Limit,Spill,Duplicates}){run(false,4,c);run(false,8,c);run(true,0,c);}
 puts("SHARED_U64_PHYSICAL_GPU_PASS");return 0;
}
