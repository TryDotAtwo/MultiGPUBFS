// Exact collision, cross-tile lookup, spill, and publication for batch hash.
#include "generic_route.h"
#include "segmented_history.cuh"
#include "segmented_hash_batch.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>
extern "C" int mgbfs_hash_batch_scratch_bytes(uint32_t,uint32_t,uint32_t,size_t*);
extern "C" int mgbfs_hash_batch_accept(uint32_t,uint32_t,uint32_t,const void*,const void*,
 const GenericRouteRecord*,const GenericRouteRecord*,const uint32_t*,const uint32_t*,
 uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,const uint32_t*,uint32_t,uint32_t,
 const MgbfsBatchHashSegment*,uint32_t,uint32_t,uint32_t,MgbfsBatchHashRun*,uint32_t*,uint32_t,
 uint64_t*,uint32_t*,uint32_t*,uint32_t*,uint64_t*,uint64_t,void*,size_t,void*);
extern "C" int mgbfs_hash_batch_seed(uint32_t,uint32_t,const void*,MgbfsBatchHashSegment,uint64_t,MgbfsBatchHashRun*,uint32_t*,uint64_t*,void*);
extern "C" int mgbfs_hash_batch_retire_bank(MgbfsBatchHashRun*,uint32_t*,uint32_t,const MgbfsBatchHashSegment*,uint32_t,uint32_t,uint32_t*,void*);
extern "C" int mgbfs_hash_batch_compact_scratch_bytes(uint32_t,size_t*);
extern "C" int mgbfs_hash_batch_compact_tail(uint32_t,uint32_t,const MgbfsBatchHashSegment*,uint32_t,
 MgbfsBatchHashRun*,uint32_t*,uint32_t,void*,size_t,uint32_t*,uint32_t*,void*);
#define C(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);exit(1);}}while(0)
template<class T> T* alloc(size_t n=1){T*p=nullptr;C(cudaMalloc((void**)&p,n*sizeof(T)));return p;}
template<class T> T get(T*p){T v{};C(cudaMemcpy(&v,p,sizeof(T),cudaMemcpyDeviceToHost));return v;}
template<class T> void put(T*p,T v){C(cudaMemcpy(p,&v,sizeof(T),cudaMemcpyHostToDevice));}
using State=std::pair<uint8_t,uint8_t>;
static GenericRouteRecord record(State v){GenericRouteRecord r{};r.hash=0;r.parent=uint64_t(v.first)|(uint64_t(v.second)<<8);return r;}
static void run(){
 constexpr uint32_t Q=8;
 std::vector<State> first={{5,2},{0,1},{5,2},{3,4},{7,8},{0,1},{9,1},{3,4}};
 std::vector<GenericRouteRecord> host(Q);for(unsigned i=0;i<Q;i++)host[i]=record(first[i]);
 auto* meta=alloc<GenericRouteRecord>(Q);C(cudaMemcpy(meta,host.data(),Q*sizeof(host[0]),cudaMemcpyHostToDevice));
 auto* counts=alloc<uint32_t>();put(counts,Q);
 MgbfsBatchHashSegment hs[3]{};
 for(unsigned i=0;i<3;i++){hs[i].capacity=i==0?3:(i==1?2:1);hs[i].banks=1;hs[i].states=alloc<uint8_t>(hs[i].capacity*2);hs[i].keys=alloc<uint64_t>(hs[i].capacity);hs[i].rows=alloc<uint32_t>(hs[i].capacity);}
 auto* segments=alloc<MgbfsBatchHashSegment>(3);C(cudaMemcpy(segments,hs,sizeof(hs),cudaMemcpyHostToDevice));
 auto* runs=alloc<MgbfsBatchHashRun>(8);auto* nruns=alloc<uint32_t>();put(nruns,uint32_t(0));
 auto* accepted=alloc<uint64_t>(3);C(cudaMemset(accepted,0,3*sizeof(uint64_t)));
 auto* global=alloc<uint64_t>();put(global,uint64_t(0));
 auto* selected=alloc<uint32_t>(Q);auto* spill=alloc<uint32_t>(Q);auto* ns=alloc<uint32_t>();auto* err=alloc<uint32_t>();
 size_t bytes=0;C(cudaError_t(mgbfs_hash_batch_scratch_bytes(Q,1,1,&bytes)));void* scratch=nullptr;C(cudaMalloc(&scratch,bytes));
 auto call=[&](uint32_t active,const uint32_t* selection,uint32_t n,uint32_t capacity){
  put(ns,uint32_t(0));put(err,uint32_t(0));
  C(cudaError_t(mgbfs_hash_batch_accept(1,1,2,nullptr,nullptr,meta,meta,counts,counts,
   0,1,0,1,Q,Q,selection,n,0,segments,3,active,0,runs,nruns,capacity,accepted+active,
   spill,ns,err,global,100,scratch,bytes,nullptr)));
  C(cudaDeviceSynchronize());
 };
 call(0,nullptr,Q,8);CHECK(get(err)==0);CHECK(get(accepted)==3);CHECK(get(nruns)==1);CHECK(get(ns)==2);CHECK(get(global)==3);
 C(cudaMemcpy(selected,spill,2*sizeof(uint32_t),cudaMemcpyDeviceToDevice));
 call(1,selected,2,8);CHECK(get(err)==0);CHECK(get(accepted+1)==2);CHECK(get(nruns)==2);CHECK(get(ns)==0);CHECK(get(global)==5);
 host[0]=record({0,1});host[1]=record({10,2});host[2]=record({9,1});for(unsigned i=3;i<Q;i++)host[i]=record({0,1});
 C(cudaMemcpy(meta,host.data(),Q*sizeof(host[0]),cudaMemcpyHostToDevice));put(counts,uint32_t(3));
 call(2,nullptr,Q,8);CHECK(get(err)==0);CHECK(get(accepted+2)==1);CHECK(get(nruns)==3);CHECK(get(ns)==0);CHECK(get(global)==6);
 std::set<State> got;
 for(unsigned seg=0;seg<3;seg++){
  std::vector<uint8_t> states(hs[seg].capacity*2);C(cudaMemcpy(states.data(),hs[seg].states,states.size(),cudaMemcpyDeviceToHost));
  for(unsigned row=0;row<hs[seg].capacity;row++)CHECK(got.insert({states[row],states[hs[seg].capacity+row]}).second);
 }
 CHECK(got==std::set<State>({{0,1},{3,4},{5,2},{7,8},{9,1},{10,2}}));
 call(2,nullptr,Q,8);CHECK(get(err)==0);CHECK(get(accepted+2)==1);CHECK(get(nruns)==3);CHECK(get(ns)==0);CHECK(get(global)==6);
 MgbfsBatchHashRun observed[3]{};C(cudaMemcpy(observed,runs,sizeof(observed),cudaMemcpyDeviceToHost));
 for(unsigned i=0;i<3;i++){CHECK(observed[i].segment==i&&observed[i].bank==0&&observed[i].begin==0&&observed[i].length==hs[i].capacity);CHECK(observed[i].min_hash==0&&observed[i].max_hash==0);}
 for(auto& seg:hs){C(cudaFree(seg.states));C(cudaFree(seg.keys));C(cudaFree(seg.rows));}
 C(cudaFree(meta));C(cudaFree(counts));C(cudaFree(segments));C(cudaFree(runs));C(cudaFree(nruns));C(cudaFree(accepted));C(cudaFree(global));C(cudaFree(selected));C(cudaFree(spill));C(cudaFree(ns));C(cudaFree(err));C(cudaFree(scratch));
 puts("PASS forced equal hashes, exact compare, cross-tile history, segment spill, duplicate-only full bank");
}

static void compact_runs(){
 constexpr uint32_t Q=4;
 auto* meta=alloc<GenericRouteRecord>(Q);auto* counts=alloc<uint32_t>();put(counts,uint32_t(2));
 MgbfsBatchHashSegment h{};h.capacity=8;h.banks=1;h.states=alloc<uint8_t>(16);h.keys=alloc<uint64_t>(8);h.rows=alloc<uint32_t>(8);
 auto* segments=alloc<MgbfsBatchHashSegment>();C(cudaMemcpy(segments,&h,sizeof(h),cudaMemcpyHostToDevice));
 auto* runs=alloc<MgbfsBatchHashRun>(8);auto* nruns=alloc<uint32_t>();put(nruns,uint32_t(0));
 auto* accepted=alloc<uint64_t>();put(accepted,uint64_t(0));auto* global=alloc<uint64_t>();put(global,uint64_t(0));
 auto* spill=alloc<uint32_t>(Q);auto* ns=alloc<uint32_t>();auto* err=alloc<uint32_t>();auto* result=alloc<uint32_t>();
 size_t scratch_bytes=0;C(cudaError_t(mgbfs_hash_batch_scratch_bytes(Q,1,1,&scratch_bytes)));void* scratch=nullptr;C(cudaMalloc(&scratch,scratch_bytes));
 auto accept=[&](State x,State y){
  GenericRouteRecord rows[Q]={record(x),record(y),record(x),record(y)};
  C(cudaMemcpy(meta,rows,sizeof(rows),cudaMemcpyHostToDevice));put(ns,uint32_t(0));put(err,uint32_t(0));
  C(cudaError_t(mgbfs_hash_batch_accept(1,1,2,nullptr,nullptr,meta,meta,counts,counts,0,1,0,1,Q,Q,nullptr,Q,0,
   segments,1,0,0,runs,nruns,8,accepted,spill,ns,err,global,100,scratch,scratch_bytes,nullptr)));
  C(cudaDeviceSynchronize());CHECK(get(err)==0&&get(ns)==0);
 };
 accept({5,2},{1,1});accept({2,0},{4,4});CHECK(get(accepted)==4&&get(nruns)==2);
 std::vector<uint8_t> before(16),after(16);C(cudaMemcpy(before.data(),h.states,16,cudaMemcpyDeviceToHost));
 size_t mb=0;C(cudaError_t(mgbfs_hash_batch_compact_scratch_bytes(2,&mb)));void* ms=nullptr;C(cudaMalloc(&ms,mb));
 put(result,uint32_t(99));put(err,uint32_t(0));
 C(cudaError_t(mgbfs_hash_batch_compact_tail(1,2,segments,1,runs,nruns,2,ms,mb,result,err,nullptr)));
 C(cudaDeviceSynchronize());CHECK(get(result)==2&&get(err)==0&&get(nruns)==2);
 accept({1,1},{2,0});CHECK(get(nruns)==2&&get(accepted)==4);
 C(cudaFree(ms));C(cudaError_t(mgbfs_hash_batch_compact_scratch_bytes(4,&mb)));C(cudaMalloc(&ms,mb));
 put(result,uint32_t(99));put(err,uint32_t(0));
 C(cudaError_t(mgbfs_hash_batch_compact_tail(1,2,segments,1,runs,nruns,4,ms,mb,result,err,nullptr)));
 C(cudaDeviceSynchronize());CHECK(get(result)==1&&get(err)==0&&get(nruns)==1);
 MgbfsBatchHashRun run{};C(cudaMemcpy(&run,runs,sizeof(run),cudaMemcpyDeviceToHost));CHECK(run.begin==0&&run.length==4&&run.min_hash==0&&run.max_hash==0);
 uint32_t rows[4];C(cudaMemcpy(rows,h.rows,sizeof(rows),cudaMemcpyDeviceToHost));CHECK(rows[0]==0&&rows[1]==2&&rows[2]==3&&rows[3]==1);
 C(cudaMemcpy(after.data(),h.states,16,cudaMemcpyDeviceToHost));CHECK(before==after);
 accept({1,1},{2,0});CHECK(get(nruns)==1&&get(accepted)==4);
 C(cudaFree(ms));C(cudaFree(scratch));C(cudaFree(result));C(cudaFree(err));C(cudaFree(ns));C(cudaFree(spill));C(cudaFree(global));C(cudaFree(accepted));C(cudaFree(nruns));C(cudaFree(runs));C(cudaFree(segments));C(cudaFree(h.states));C(cudaFree(h.keys));C(cudaFree(h.rows));C(cudaFree(counts));C(cudaFree(meta));
 puts("PASS deferred/merged runs remain searchable; no state movement");
}

static void bounded_raw_tiles(){
 constexpr uint32_t Q=6;uint8_t ids[Q]={1,1,1,2,1,3};
 GenericRouteRecord host[Q]{};for(unsigned i=0;i<Q;i++)host[i]=record({ids[i],0});
 auto* meta=alloc<GenericRouteRecord>(Q);C(cudaMemcpy(meta,host,sizeof(host),cudaMemcpyHostToDevice));
 auto* counts=alloc<uint32_t>();put(counts,Q);
 MgbfsBatchHashSegment hs[2]{};
 for(unsigned i=0;i<2;i++){hs[i].capacity=i?1:2;hs[i].banks=1;hs[i].states=alloc<uint8_t>(hs[i].capacity*2);hs[i].keys=alloc<uint64_t>(hs[i].capacity);hs[i].rows=alloc<uint32_t>(hs[i].capacity);}
 auto* segments=alloc<MgbfsBatchHashSegment>(2);C(cudaMemcpy(segments,hs,sizeof(hs),cudaMemcpyHostToDevice));
 auto* runs=alloc<MgbfsBatchHashRun>(8);auto* nruns=alloc<uint32_t>();put(nruns,uint32_t(0));
 auto* accepted=alloc<uint64_t>(2);C(cudaMemset(accepted,0,16));auto* global=alloc<uint64_t>();put(global,uint64_t(0));
 auto* spill=alloc<uint32_t>(2);auto* selection=alloc<uint32_t>(2);auto* ns=alloc<uint32_t>();auto* err=alloc<uint32_t>();
 size_t bytes=0;C(cudaError_t(mgbfs_hash_batch_scratch_bytes(2,1,1,&bytes)));void* scratch=nullptr;C(cudaMalloc(&scratch,bytes));
 auto call=[&](const uint32_t* sel,uint32_t begin,uint32_t active,uint32_t n){
  put(ns,uint32_t(0));put(err,uint32_t(0));
  C(cudaError_t(mgbfs_hash_batch_accept(1,1,2,nullptr,nullptr,meta,meta,counts,counts,0,1,0,1,Q,Q,sel,n,begin,segments,2,active,0,runs,nruns,8,accepted+active,spill,ns,err,global,3,scratch,bytes,nullptr)));
  C(cudaDeviceSynchronize());CHECK(get(err)==0);
 };
 int invalid=mgbfs_hash_batch_accept(1,1,2,nullptr,nullptr,meta,meta,counts,counts,0,1,0,1,Q,Q,nullptr,2,5,segments,2,0,0,runs,nruns,8,accepted,spill,ns,err,global,3,scratch,bytes,nullptr);
 CHECK(invalid==int(cudaErrorInvalidValue));
 call(nullptr,0,0,2);CHECK(get(accepted)==1&&get(ns)==0&&get(global)==1);
 call(nullptr,2,0,2);CHECK(get(accepted)==2&&get(ns)==0&&get(global)==2);
 call(nullptr,4,0,2);CHECK(get(accepted)==2&&get(ns)==1&&get(global)==2);
 CHECK(get(spill)==5);C(cudaMemcpy(selection,spill,4,cudaMemcpyDeviceToDevice));call(selection,0,1,1);
 CHECK(get(accepted+1)==1&&get(ns)==0&&get(global)==3&&get(nruns)==3);
 uint8_t rows_a[4]{},rows_b[2]{};C(cudaMemcpy(rows_a,hs[0].states,4,cudaMemcpyDeviceToHost));C(cudaMemcpy(rows_b,hs[1].states,2,cudaMemcpyDeviceToHost));
 CHECK(rows_a[0]==1&&rows_a[1]==2&&rows_b[0]==3);
 for(auto& seg:hs){C(cudaFree(seg.states));C(cudaFree(seg.keys));C(cudaFree(seg.rows));}
 C(cudaFree(meta));C(cudaFree(counts));C(cudaFree(segments));C(cudaFree(runs));C(cudaFree(nruns));C(cudaFree(accepted));C(cudaFree(global));C(cudaFree(spill));C(cudaFree(selection));C(cudaFree(ns));C(cudaFree(err));C(cudaFree(scratch));
 puts("PASS bounded raw tiles; duplicate+new use last bank row; live queue and spill replay");
}


static void seed_and_retire(){
 MgbfsBatchHashSegment hs[2]{};
 for(auto& seg:hs){seg.capacity=2;seg.banks=3;seg.states=alloc<uint8_t>(12);seg.keys=alloc<uint64_t>(6);seg.rows=alloc<uint32_t>(6);}
 auto* segments=alloc<MgbfsBatchHashSegment>(2);C(cudaMemcpy(segments,hs,sizeof(hs),cudaMemcpyHostToDevice));
 auto* input=alloc<uint8_t>(2);uint8_t start[2]={4,7};C(cudaMemcpy(input,start,2,cudaMemcpyHostToDevice));
 auto* runs=alloc<MgbfsBatchHashRun>(8);auto* count=alloc<uint32_t>();auto* accepted=alloc<uint64_t>();
 C(cudaError_t(mgbfs_hash_batch_seed(1,2,input,hs[0],123,runs,count,accepted,nullptr)));C(cudaDeviceSynchronize());
 CHECK(get(count)==1&&get(accepted)==1&&get(hs[0].keys)==123&&get(hs[0].rows)==0);
 uint8_t states[12];C(cudaMemcpy(states,hs[0].states,12,cudaMemcpyDeviceToHost));CHECK(states[0]==4&&states[6]==7);
 MgbfsBatchHashRun seeded{};C(cudaMemcpy(&seeded,runs,sizeof(seeded),cudaMemcpyDeviceToHost));
 CHECK(seeded.segment==0&&seeded.bank==0&&seeded.length==1&&seeded.min_hash==123&&seeded.max_hash==123);
 MgbfsBatchHashRun host[4]={{0,0,0,1,1,1},{0,1,0,1,2,2},{1,0,0,1,3,3},{1,2,0,1,4,4}};
 C(cudaMemcpy(runs,host,sizeof(host),cudaMemcpyHostToDevice));put(count,uint32_t(4));auto* error=alloc<uint32_t>();put(error,uint32_t(0));
 C(cudaError_t(mgbfs_hash_batch_retire_bank(runs,count,8,segments,2,0,error,nullptr)));C(cudaDeviceSynchronize());
 CHECK(get(count)==2&&get(error)==0);MgbfsBatchHashRun retained[2];C(cudaMemcpy(retained,runs,sizeof(retained),cudaMemcpyDeviceToHost));
 CHECK(retained[0].segment==0&&retained[0].bank==1&&retained[1].segment==1&&retained[1].bank==2);
 for(auto& seg:hs){C(cudaFree(seg.states));C(cudaFree(seg.keys));C(cudaFree(seg.rows));}
 C(cudaFree(segments));C(cudaFree(input));C(cudaFree(runs));C(cudaFree(count));C(cudaFree(accepted));C(cudaFree(error));
 puts("PASS exact seed index and bank-local run retirement");
}
int main(){int count=0;C(cudaGetDeviceCount(&count));CHECK(count>=2);for(int d=0;d<2;d++){C(cudaSetDevice(d));run();compact_runs();bounded_raw_tiles();seed_and_retire();}puts("BATCH_HASH_TWO_GPU_PASS");}
