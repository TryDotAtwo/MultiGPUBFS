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

extern "C" int mgbfs_bitmap_accept_batch(uint32_t,uint32_t,uint32_t,const void*,const void*,const GenericRouteRecord*,const GenericRouteRecord*,const uint32_t*,const uint32_t*,uint32_t,uint32_t,uint32_t,uint64_t,uint64_t,uint64_t*,uint32_t,uint32_t,uint32_t,uint32_t,const uint32_t*,uint32_t,uint32_t,MgbfsSegment,uint32_t,uint64_t*,uint32_t*,uint32_t*,uint32_t*,uint64_t*,uint64_t,void*,size_t,void*);
extern "C" int mgbfs_bitmap_accept_batch_scratch_bytes(uint32_t,size_t*);
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
 size_t scratch_bytes=0;C(static_cast<cudaError_t>(mgbfs_bitmap_accept_batch_scratch_bytes(Q,&scratch_bytes)));void* scratch=nullptr;C(cudaMalloc(&scratch,scratch_bytes));
 uint32_t selected=Q;bool initial=true,cap_was_tested=!limit;unsigned rounds=0;uint64_t global_limit=limit?3:Q;
 while(selected){
  CHECK(++rounds<100);
  MgbfsSegment seg{};seg.capacity=CAP;seg.banks=3;seg.states=alloc<uint8_t>(2*CAP*3);C(cudaMemset(seg.states,0xff,2*CAP*3));
  auto* n=alloc<uint64_t>();put(n,uint64_t(0));
  for(;;){
   put(ns,uint32_t(0));put(err,uint32_t(0));
   C(static_cast<cudaError_t>(mgbfs_bitmap_accept_batch(1,1,2,nullptr,nullptr,dm,dm,counts,counts,0,1,0,TOTAL,TOTAL,bits,0,1,Q,Q,initial?nullptr:sel,selected,0,seg,0,n,spill,ns,err,global,global_limit,scratch,scratch_bytes,nullptr)));
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
 C(cudaFree(scratch));C(cudaFree(dm));C(cudaFree(counts));C(cudaFree(bits));C(cudaFree(global));C(cudaFree(sel));C(cudaFree(spill));C(cudaFree(ns));C(cudaFree(err));
 printf("PASS bitmap batch spill/retry bits cap=%s rounds=%u unique=%zu\n",limit?"yes":"no",rounds,accepted.size());fflush(stdout);
}

static void world2_word_boundaries(){
 constexpr uint32_t q=4,world=2;
 std::vector<GenericRouteRecord> lm(8),rm(8);
 auto record=[](uint64_t id){GenericRouteRecord r{};r.hash=id;r.parent=id;return r;};
 rm[0]=record(1);rm[1]=record(127);rm[2]=record(129);
 lm[4]=record(127);lm[5]=record(127);
 auto* dl=alloc<GenericRouteRecord>(8);auto* dr=alloc<GenericRouteRecord>(8);
 C(cudaMemcpy(dl,lm.data(),8*sizeof(lm[0]),cudaMemcpyHostToDevice));
 C(cudaMemcpy(dr,rm.data(),8*sizeof(rm[0]),cudaMemcpyHostToDevice));
 uint32_t lcount[2]={0,2},rcount[2]={3,0};
 auto* lc=alloc<uint32_t>(2);auto* rc=alloc<uint32_t>(2);
 C(cudaMemcpy(lc,lcount,8,cudaMemcpyHostToDevice));C(cudaMemcpy(rc,rcount,8,cudaMemcpyHostToDevice));
 auto* bits=alloc<uint64_t>(2);C(cudaMemset(bits,0,16));
 auto* global=alloc<uint64_t>();put(global,uint64_t(0));
 auto* spill=alloc<uint32_t>(8);auto* sel=alloc<uint32_t>(8);auto* ns=alloc<uint32_t>();auto* err=alloc<uint32_t>();auto* accepted=alloc<uint64_t>();
 size_t bytes=0;C(static_cast<cudaError_t>(mgbfs_bitmap_accept_batch_scratch_bytes(8,&bytes)));void* scratch=nullptr;C(cudaMalloc(&scratch,bytes));
 MgbfsSegment seg{};seg.banks=2;seg.capacity=1;seg.states=alloc<uint8_t>(2);C(cudaMemset(seg.states,0xff,2));
 put(accepted,uint64_t(0));put(ns,uint32_t(0));put(err,uint32_t(0));
 C(static_cast<cudaError_t>(mgbfs_bitmap_accept_batch(1,1,1,nullptr,nullptr,dl,dr,lc,rc,1,world,1,256,128,bits,0,1,q,q,nullptr,8,0,seg,0,accepted,spill,ns,err,global,1,scratch,bytes,nullptr)));
 C(cudaDeviceSynchronize());CHECK(get(err)==0);CHECK(get(accepted)==1);CHECK(get(global)==1);CHECK(get(ns)==2);
 uint64_t words[2]={};C(cudaMemcpy(words,bits,16,cudaMemcpyDeviceToHost));CHECK(words[0]==1&&words[1]==0);
 uint8_t row=0;C(cudaMemcpy(&row,seg.states,1,cudaMemcpyDeviceToHost));CHECK(row==1);
 C(cudaMemcpy(sel,spill,8,cudaMemcpyDeviceToDevice));C(cudaFree(seg.states));
 seg.capacity=2;seg.states=alloc<uint8_t>(4);C(cudaMemset(seg.states,0xff,4));
 put(accepted,uint64_t(0));put(ns,uint32_t(0));put(err,uint32_t(0));
 C(static_cast<cudaError_t>(mgbfs_bitmap_accept_batch(1,1,1,nullptr,nullptr,dl,dr,lc,rc,1,world,1,256,128,bits,0,1,q,q,sel,2,0,seg,1,accepted,spill,ns,err,global,3,scratch,bytes,nullptr)));
 C(cudaDeviceSynchronize());CHECK(get(err)==0);CHECK(get(accepted)==2);CHECK(get(global)==3);CHECK(get(ns)==0);
 C(cudaMemcpy(words,bits,16,cudaMemcpyDeviceToHost));CHECK(words[0]==((1ull<<63)|1)&&words[1]==1);
 uint8_t rows[4];C(cudaMemcpy(rows,seg.states,4,cudaMemcpyDeviceToHost));CHECK(rows[2]==127&&rows[3]==129);
 put(ns,uint32_t(0));put(err,uint32_t(0));
 C(static_cast<cudaError_t>(mgbfs_bitmap_accept_batch(1,1,1,nullptr,nullptr,dl,dr,lc,rc,1,world,1,256,128,bits,0,1,q,q,nullptr,8,0,seg,1,accepted,spill,ns,err,global,3,scratch,bytes,nullptr)));
 C(cudaDeviceSynchronize());CHECK(get(err)==0);CHECK(get(accepted)==2);CHECK(get(global)==3);CHECK(get(ns)==0);
 C(cudaFree(seg.states));C(cudaFree(dl));C(cudaFree(dr));C(cudaFree(lc));C(cudaFree(rc));C(cudaFree(bits));C(cudaFree(global));C(cudaFree(spill));C(cudaFree(sel));C(cudaFree(ns));C(cudaFree(err));C(cudaFree(accepted));C(cudaFree(scratch));
 puts("PASS logical two-rank word 0/63/64 partial grant, replay and duplicate-only full bank");
}

static void independent_shards_parallel(){
 constexpr unsigned SHARDS=2;
 std::vector<GenericRouteRecord> host(SHARDS);
 host[0].hash=0;host[0].parent=0;
 host[1].hash=192;host[1].parent=192;
 auto* meta=alloc<GenericRouteRecord>(SHARDS);C(cudaMemcpy(meta,host.data(),sizeof(host[0])*SHARDS,cudaMemcpyHostToDevice));
 uint32_t hcounts[SHARDS]={1,1};auto* counts=alloc<uint32_t>(SHARDS);C(cudaMemcpy(counts,hcounts,sizeof(hcounts),cudaMemcpyHostToDevice));
 auto* bits=alloc<uint64_t>(4);auto* global=alloc<uint64_t>();
 MgbfsSegment seg[2]{};uint64_t* accepted[2];uint32_t* spill[2];uint32_t* ns[2];uint32_t* err[2];void* scratch[2];cudaStream_t stream[2];
 size_t bytes=0;C(cudaError_t(mgbfs_bitmap_accept_batch_scratch_bytes(1,&bytes)));
 for(unsigned k=0;k<2;k++){seg[k].capacity=1;seg[k].banks=2;seg[k].states=alloc<uint8_t>(2);accepted[k]=alloc<uint64_t>();spill[k]=alloc<uint32_t>();ns[k]=alloc<uint32_t>();err[k]=alloc<uint32_t>();C(cudaMalloc(&scratch[k],bytes));C(cudaStreamCreateWithFlags(&stream[k],cudaStreamNonBlocking));}
 for(unsigned repeat=0;repeat<30;repeat++){
  C(cudaMemset(bits,0,32));put(global,uint64_t(0));
  for(unsigned k=0;k<2;k++){put(accepted[k],uint64_t(0));put(ns[k],uint32_t(0));put(err[k],uint32_t(0));}
  for(unsigned k=0;k<2;k++)C(cudaError_t(mgbfs_bitmap_accept_batch(1,1,1,nullptr,nullptr,meta,meta,counts,counts,0,1,0,256,256,bits,k,SHARDS,1,1,nullptr,1,0,seg[k],0,accepted[k],spill[k],ns[k],err[k],global,2,scratch[k],bytes,stream[k])));
  for(unsigned k=0;k<2;k++)C(cudaStreamSynchronize(stream[k]));
  uint64_t words[4];C(cudaMemcpy(words,bits,sizeof(words),cudaMemcpyDeviceToHost));
  CHECK(words[0]==1&&words[1]==0&&words[2]==0&&words[3]==1&&get(global)==2);
  for(unsigned k=0;k<2;k++){CHECK(get(accepted[k])==1&&get(ns[k])==0&&get(err[k])==0);uint8_t row;C(cudaMemcpy(&row,seg[k].states,1,cudaMemcpyDeviceToHost));CHECK(row==(k?192:0));}
 }
 for(unsigned k=0;k<2;k++){C(cudaStreamDestroy(stream[k]));C(cudaFree(scratch[k]));C(cudaFree(seg[k].states));C(cudaFree(accepted[k]));C(cudaFree(spill[k]));C(cudaFree(ns[k]));C(cudaFree(err[k]));}
 C(cudaFree(meta));C(cudaFree(counts));C(cudaFree(bits));C(cudaFree(global));
 puts("PASS 30 overlapping transactions on disjoint word-owned shards");
}
static void same_word_stream_event(){
 GenericRouteRecord h[2]{};h[0].hash=0;h[0].parent=0;h[1].hash=63;h[1].parent=63;
 auto* m0=alloc<GenericRouteRecord>();auto* m1=alloc<GenericRouteRecord>();
 C(cudaMemcpy(m0,&h[0],sizeof(h[0]),cudaMemcpyHostToDevice));C(cudaMemcpy(m1,&h[1],sizeof(h[1]),cudaMemcpyHostToDevice));
 auto* count=alloc<uint32_t>();put(count,uint32_t(1));auto* bits=alloc<uint64_t>();put(bits,uint64_t(0));
 auto* global=alloc<uint64_t>();put(global,uint64_t(0));auto* accepted=alloc<uint64_t>();put(accepted,uint64_t(0));
 MgbfsSegment seg{};seg.capacity=2;seg.banks=2;seg.states=alloc<uint8_t>(4);
 auto* spill0=alloc<uint32_t>();auto* spill1=alloc<uint32_t>();auto* n0=alloc<uint32_t>();auto* n1=alloc<uint32_t>();auto* e0=alloc<uint32_t>();auto* e1=alloc<uint32_t>();
 put(n0,uint32_t(0));put(n1,uint32_t(0));put(e0,uint32_t(0));put(e1,uint32_t(0));
 size_t bytes=0;C(cudaError_t(mgbfs_bitmap_accept_batch_scratch_bytes(1,&bytes)));void *a=nullptr,*b=nullptr;C(cudaMalloc(&a,bytes));C(cudaMalloc(&b,bytes));
 cudaStream_t s0,s1;cudaEvent_t done;C(cudaStreamCreateWithFlags(&s0,cudaStreamNonBlocking));C(cudaStreamCreateWithFlags(&s1,cudaStreamNonBlocking));C(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));
 C(cudaError_t(mgbfs_bitmap_accept_batch(1,1,1,nullptr,nullptr,m0,m0,count,count,0,1,0,64,64,bits,0,1,1,1,nullptr,1,0,seg,0,accepted,spill0,n0,e0,global,2,a,bytes,s0)));
 C(cudaEventRecord(done,s0));C(cudaStreamWaitEvent(s1,done,0));
 C(cudaError_t(mgbfs_bitmap_accept_batch(1,1,1,nullptr,nullptr,m1,m1,count,count,0,1,0,64,64,bits,0,1,1,1,nullptr,1,0,seg,0,accepted,spill1,n1,e1,global,2,b,bytes,s1)));
 C(cudaStreamSynchronize(s1));CHECK(get(bits)==((1ull<<63)|1));CHECK(get(accepted)==2&&get(global)==2);CHECK(get(e0)==0&&get(e1)==0&&get(n0)==0&&get(n1)==0);
 uint8_t rows[4];C(cudaMemcpy(rows,seg.states,4,cudaMemcpyDeviceToHost));CHECK(rows[0]==0&&rows[1]==63);
 C(cudaEventDestroy(done));C(cudaStreamDestroy(s0));C(cudaStreamDestroy(s1));C(cudaFree(a));C(cudaFree(b));C(cudaFree(seg.states));C(cudaFree(m0));C(cudaFree(m1));C(cudaFree(count));C(cudaFree(bits));C(cudaFree(global));C(cudaFree(accepted));C(cudaFree(spill0));C(cudaFree(spill1));C(cudaFree(n0));C(cudaFree(n1));C(cudaFree(e0));C(cudaFree(e1));
 puts("PASS same-word transactions ordered across two streams by event");
}

static void bounded_raw_tiles(){
 constexpr uint32_t Q=6;uint8_t ids[Q]={1,1,1,2,1,3};
 GenericRouteRecord h[Q]{};for(unsigned i=0;i<Q;i++){h[i].hash=ids[i];h[i].parent=ids[i];}
 auto* meta=alloc<GenericRouteRecord>(Q);C(cudaMemcpy(meta,h,sizeof(h),cudaMemcpyHostToDevice));
 auto* counts=alloc<uint32_t>();put(counts,Q);
 auto* bits=alloc<uint64_t>(4);C(cudaMemset(bits,0,32));
 auto* global=alloc<uint64_t>();put(global,uint64_t(0));
 auto* accepted=alloc<uint64_t>();put(accepted,uint64_t(0));
 auto* spill=alloc<uint32_t>(2);auto* selection=alloc<uint32_t>(2);
 auto* ns=alloc<uint32_t>();auto* err=alloc<uint32_t>();
 MgbfsSegment a{};a.banks=2;a.capacity=2;a.states=alloc<uint8_t>(4);
 MgbfsSegment b{};b.banks=2;b.capacity=1;b.states=alloc<uint8_t>(2);
 size_t bytes=0;C(cudaError_t(mgbfs_bitmap_accept_batch_scratch_bytes(2,&bytes)));
 void* scratch=nullptr;C(cudaMalloc(&scratch,bytes));
 auto call=[&](const uint32_t* sel,uint32_t begin,MgbfsSegment s,uint32_t n){
  put(ns,uint32_t(0));put(err,uint32_t(0));
  C(cudaError_t(mgbfs_bitmap_accept_batch(1,1,1,nullptr,nullptr,meta,meta,counts,counts,0,1,0,256,256,bits,0,1,Q,Q,sel,n,begin,s,0,accepted,spill,ns,err,global,3,scratch,bytes,nullptr)));
  C(cudaDeviceSynchronize());CHECK(get(err)==0);
 };
 int invalid=mgbfs_bitmap_accept_batch(1,1,1,nullptr,nullptr,meta,meta,counts,counts,0,1,0,256,256,bits,0,1,Q,Q,nullptr,2,5,a,0,accepted,spill,ns,err,global,3,scratch,bytes,nullptr);
 CHECK(invalid==int(cudaErrorInvalidValue));
 call(nullptr,0,a,2);CHECK(get(accepted)==1&&get(ns)==0&&get(global)==1);
 call(nullptr,2,a,2);CHECK(get(accepted)==2&&get(ns)==0&&get(global)==2);
 call(nullptr,4,a,2);CHECK(get(accepted)==2&&get(ns)==1&&get(global)==2);
 CHECK(get(spill)==5);C(cudaMemcpy(selection,spill,4,cudaMemcpyDeviceToDevice));
 put(accepted,uint64_t(0));call(selection,0,b,1);
 CHECK(get(accepted)==1&&get(ns)==0&&get(global)==3&&get(bits)==14);
 uint8_t rows_a[4]{},rows_b[2]{};C(cudaMemcpy(rows_a,a.states,4,cudaMemcpyDeviceToHost));C(cudaMemcpy(rows_b,b.states,2,cudaMemcpyDeviceToHost));
 CHECK(rows_a[0]==1&&rows_a[1]==2&&rows_b[0]==3);
 C(cudaFree(a.states));C(cudaFree(b.states));C(cudaFree(meta));C(cudaFree(counts));C(cudaFree(bits));C(cudaFree(global));C(cudaFree(accepted));C(cudaFree(spill));C(cudaFree(selection));C(cudaFree(ns));C(cudaFree(err));C(cudaFree(scratch));
 puts("PASS bounded raw tiles; duplicate+new use last bank row; live queue and spill replay");
}

int main(){int count=0;C(cudaGetDeviceCount(&count));CHECK(count>=2);
 for(int d=0;d<2;d++){C(cudaSetDevice(d));run(false);run(true);world2_word_boundaries();bounded_raw_tiles();independent_shards_parallel();same_word_stream_event();}
 puts("BATCH_BITMAP_TWO_GPU_PASS");return 0;
}
