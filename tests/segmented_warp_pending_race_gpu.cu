// Deterministic two-shard GPU schedule against the actual grouped accept kernel.
// Build twice: fixed and -DMGBFS_TEST_OLD_DEFER=1. The latter must reproduce
// duplicate X rows; the former must preserve Y until the wave drains.
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
struct RaceControl { unsigned held,y_claim,x1_scanned,y_deferred,released,x2_claim,timeout; };
__device__ RaceControl* test_control;
__device__ uint64_t* test_other_slots;
__device__ uint64_t* test_target_slots;
__device__ void test_wait(unsigned* p){
 auto start=clock64();while(!atomicAdd(p,0u)){if(clock64()-start>40000000000ull){atomicExch(&test_control->timeout,1u);break;}__nanosleep(64);}
}
__device__ void test_hook(int stage,uint64_t* slots,uint32_t child){
 auto* c=test_control;if(!c)return;
 if(slots==test_target_slots){
  if(stage==1&&child==0)test_wait(&c->held);
  if(stage==1&&child==32)test_wait(&c->y_claim);
  if(stage==1&&child==64){test_wait(&c->y_deferred);test_wait(&c->released);}
  if(stage==2&&child==32){atomicExch(&c->x1_scanned,1u);test_wait(&c->x2_claim);}
  if(stage==3&&child==0){atomicExch(&c->y_claim,1u);test_wait(&c->x1_scanned);}
  if(stage==3&&child==64)atomicExch(&c->x2_claim,1u);
  if(stage==6&&child==0)atomicExch(&c->y_deferred,1u);
 }else if(slots==test_other_slots){
  if(stage==4){atomicExch(&c->held,1u);test_wait(&c->y_deferred);}
  if(stage==5)atomicExch(&c->released,1u);
 }
}
#define MGBFS_SEGMENTED_TEST_HOOK(stage,segment,child) test_hook(stage,(segment).slots,child)
#include "../cuda/segmented_history.cu"
struct RaceInput {
 uint32_t elements,count;bool target;
 __device__ uint32_t child(uint32_t i)const{return i;}
 __device__ uint32_t origin_limit()const{return count;}
 __device__ bool valid(uint32_t i,uint32_t*)const{return target?(i==0||i==32||i==64):(i<5);}
 __device__ int64_t value(uint32_t i,uint32_t)const{return target?(i==0?11:22):int64_t(100+i);}
 __device__ uint64_t hash(uint32_t,uint64_t,uint32_t)const{return target?0x1234567800000000ull:0x1234567880000000ull;}
};
#define CK(x) do { auto e=(x); if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(2);} } while(0)
template<class T> T* alloc(size_t n){T* p=nullptr;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T> T get(T* p){T v;CK(cudaMemcpy(&v,p,sizeof(T),cudaMemcpyDeviceToHost));return v;}
static uint64_t test_mix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ull;x^=x>>27;x*=0x94d049bb133111ebull;return x^(x>>31);}
int main(){
 CK(cudaSetDevice(0));cudaStream_t s0,s1;CK(cudaStreamCreate(&s0));CK(cudaStreamCreate(&s1));
 auto* control=alloc<RaceControl>(1);CK(cudaMemset(control,0,sizeof(RaceControl)));
 auto* other=alloc<uint64_t>(8);auto* target=alloc<uint64_t>(8);CK(cudaMemset(other,255,64));CK(cudaMemset(target,255,64));
 CK(cudaMemcpyToSymbol(test_control,&control,sizeof(control)));CK(cudaMemcpyToSymbol(test_other_slots,&other,sizeof(other)));CK(cudaMemcpyToSymbol(test_target_slots,&target,sizeof(target)));
 auto* os=alloc<uint8_t>(1);auto* ts=alloc<uint8_t>(8);auto* oc=alloc<uint64_t>(1);auto* tc=alloc<uint64_t>(1);auto* total=alloc<uint64_t>(1);
 const uint64_t other_hash=0x1234567880000000ull,other_slot=test_mix(other_hash^0x9e3779b97f4a7c15ull)&7,token=other_hash&PREFIX;
 const uint8_t old_state=99;CK(cudaMemcpy(os,&old_state,1,cudaMemcpyHostToDevice));CK(cudaMemcpy(other+other_slot,&token,8,cudaMemcpyHostToDevice));
 std::printf("ROUTING target_owner=0 other_owner=0 target_shard=0 other_shard=1 prefilled_slot=%llu\n",(unsigned long long)other_slot);
 uint64_t one=1,zero=0,sixteen=16;CK(cudaMemcpy(oc,&one,8,cudaMemcpyHostToDevice));CK(cudaMemcpy(tc,&zero,8,cudaMemcpyHostToDevice));CK(cudaMemcpy(total,&sixteen,8,cudaMemcpyHostToDevice));
 auto* ospill=alloc<uint32_t>(96);auto* tspill=alloc<uint32_t>(96);auto* osc=alloc<uint32_t>(1);auto* tsc=alloc<uint32_t>(1);auto* oe=alloc<uint32_t>(1);auto* te=alloc<uint32_t>(1);
 CK(cudaMemset(osc,0,4));CK(cudaMemset(tsc,0,4));CK(cudaMemset(oe,0,4));CK(cudaMemset(te,0,4));
 MgbfsSegment o{other,os,nullptr,8,1,1,0};MgbfsSegment t{target,ts,nullptr,8,8,1,0};
 RaceInput oi{1,96,false},ti{1,96,true};
 accept_grouped<uint8_t><<<1,256,0,s0>>>(oi,nullptr,0,o,0,oc,nullptr,oe,nullptr,5,ospill,osc,total,21);
 accept_grouped<uint8_t><<<1,256,0,s1>>>(ti,nullptr,0,t,0,tc,nullptr,te,nullptr,96,tspill,tsc,total,21);
 CK(cudaGetLastError());CK(cudaStreamSynchronize(s0));CK(cudaStreamSynchronize(s1));
 auto c=get(control);auto count=get(tc);auto spilled=get(tsc);auto error=get(te);auto ranks=get(total);uint8_t rows[8]{};CK(cudaMemcpy(rows,ts,8,cudaMemcpyDeviceToHost));
 std::printf("TRACE held=%u y_claim=%u scanned=%u deferred=%u released=%u x2_claim=%u timeout=%u cap_retry=%u other_spill=%u target_spill=%u accepted=%llu rank_count=%llu rows=%u,%u\n",c.held,c.y_claim,c.x1_scanned,c.y_deferred,c.released,c.x2_claim,c.timeout,(error&16u)!=0,get(osc),spilled,(unsigned long long)count,(unsigned long long)ranks,rows[0],rows[1]);
 if(c.timeout||!c.held||!c.y_claim||!c.x1_scanned||!c.y_deferred||!c.released||!c.x2_claim||!(error&16u)||get(osc)!=5||spilled!=1)return 3;
#ifdef MGBFS_TEST_OLD_DEFER
 if(count!=2||rows[0]!=22||rows[1]!=22)return 4;
 std::puts("OLD_DUPLICATE_REPRODUCED");return 0;
#else
 if(count!=1||rows[0]!=22)return 5;
 // Hooks only control the initial interleaving. Cleanup runs after every
 // reader on the shard, then Y is replayed into the same bank.
 RaceControl* none=nullptr;CK(cudaMemcpyToSymbol(test_control,&none,sizeof(none)));
 retire_deferred<<<1,256,0,s0>>>(oi,o,ospill,osc,96,oe);
 retire_deferred<<<1,256,0,s1>>>(ti,t,tspill,tsc,96,te);
 CK(cudaGetLastError());CK(cudaStreamSynchronize(s0));CK(cudaStreamSynchronize(s1));
 uint32_t origin=get(tspill);if(origin!=0)return 6;
 CK(cudaMemset(tsc,0,4));CK(cudaMemset(te,0,4));auto* selection=alloc<uint32_t>(1);CK(cudaMemcpy(selection,&origin,4,cudaMemcpyHostToDevice));
 accept_grouped<uint8_t><<<1,256,0,s1>>>(ti,nullptr,0,t,0,tc,nullptr,te,selection,1,tspill,tsc,total,21);
 retire_deferred<<<1,256,0,s1>>>(ti,t,tspill,tsc,1,te);
 CK(cudaGetLastError());CK(cudaStreamSynchronize(s1));
 count=get(tc);CK(cudaMemcpy(rows,ts,8,cudaMemcpyDeviceToHost));
 std::printf("REPLAY accepted=%llu rows=%u,%u error=%u remaining_spill=%u\n",(unsigned long long)count,rows[0],rows[1],get(te),get(tsc));
 if(count!=2||rows[0]!=22||rows[1]!=11||get(te)!=0||get(tsc)!=0)return 7;
 std::puts("FIXED_EXACT_REPLAY_PASS");return 0;
#endif
}
