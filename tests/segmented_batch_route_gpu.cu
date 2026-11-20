// Match grouped CUDA route output against the original producer, including
// exact metadata, queue counts, tails, mapped owners, and bounded overflow.
#include "generic_route.h"
#include "permutation_bitmap.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define C(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);exit(1);}}while(0)
extern "C" int mgbfs_segmented_route_storage(uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,const void*,uint32_t,uint64_t,const uint32_t*,const int64_t*,const uint32_t*,uint64_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint64_t,const uint32_t*,const uint64_t*,GenericRouteRecord*,uint32_t*,uint32_t*,void*);
extern "C" int mgbfs_segmented_route_storage_grouped(uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,const void*,uint32_t,uint64_t,const uint32_t*,const int64_t*,const uint32_t*,uint64_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint64_t,const uint32_t*,const uint64_t*,GenericRouteRecord*,uint32_t*,uint32_t*,void*);
extern "C" int mgbfs_bitmap_route(uint32_t,BitmapDeviceDomain,uint32_t,uint32_t,const void*,uint32_t,uint64_t,const uint32_t*,uint32_t,uint32_t,uint32_t,uint32_t,uint64_t,const uint32_t*,GenericRouteRecord*,uint32_t*,uint32_t*,void*);
extern "C" int mgbfs_bitmap_route_grouped(uint32_t,BitmapDeviceDomain,uint32_t,uint32_t,const void*,uint32_t,uint64_t,const uint32_t*,uint32_t,uint32_t,uint32_t,uint32_t,uint64_t,const uint32_t*,GenericRouteRecord*,uint32_t*,uint32_t*,void*);
extern "C" int mgbfs_segmented_batch_route_scratch_bytes(uint32_t,uint32_t,size_t*);
extern "C" int mgbfs_segmented_batch_route(uint32_t,uint32_t,BitmapDeviceDomain,
 uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,const void*,uint32_t,uint64_t,
 const uint32_t*,const int64_t*,const uint32_t*,uint64_t,uint32_t,uint32_t,uint32_t,
 uint32_t,uint32_t,uint64_t,const uint32_t*,const uint64_t*,GenericRouteRecord*,
 uint32_t*,uint32_t*,void*,size_t,void*);
template<class T> static T* alloc(size_t n){T* x=nullptr;C(cudaMalloc((void**)&x,std::max<size_t>(n,1)*sizeof(T)));return x;}
template<class T> static void upload(T* p,const std::vector<T>& h){if(!h.empty())C(cudaMemcpy(p,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));}
using Bytes=std::array<unsigned char,sizeof(GenericRouteRecord)>;
struct Result {std::vector<uint32_t> counts;std::vector<std::vector<Bytes>> queues;uint32_t error;};
static Result call(bool bitmap,int mode,uint32_t bytes,uint32_t world,uint32_t shards,uint32_t capacity,uint32_t count,uint32_t begin=0,bool word_shard=false){
 constexpr uint32_t N=6,G=4;
 std::vector<uint32_t> perms(G*N),map(world),freq(N,1);
 std::vector<int64_t> values(N),parents64(N*count);std::vector<uint8_t> parents8(N*count);
 for(uint32_t i=0;i<N;i++)values[i]=i;
 for(uint32_t k=0;k<G;k++){for(uint32_t i=0;i<N;i++)perms[k*N+i]=i;std::swap(perms[k*N+k],perms[k*N+k+1]);}
 for(uint32_t i=0;i<world;i++)map[i]=world-1-i;
 std::vector<int64_t> p={0,1,2,3,4,5};
 for(uint32_t i=0;i<begin;i++)if(!std::next_permutation(p.begin(),p.end()))p={0,1,2,3,4,5};
 for(uint32_t i=0;i<count;i++){for(uint32_t e=0;e<N;e++){parents64[e*count+i]=p[e];parents8[e*count+i]=uint8_t(p[e]);}if(!std::next_permutation(p.begin(),p.end()))p={0,1,2,3,4,5};}
 auto* dp=alloc<uint32_t>(perms.size());upload(dp,perms);auto* dm=alloc<uint32_t>(map.size());upload(dm,map);
 auto* dv=alloc<int64_t>(values.size());upload(dv,values);auto* df=alloc<uint32_t>(freq.size());upload(df,freq);
 void* parent=nullptr;if(bytes==1){auto* x=alloc<uint8_t>(parents8.size());upload(x,parents8);parent=x;}else{auto* x=alloc<int64_t>(parents64.size());upload(x,parents64);parent=x;}
 const uint32_t nq=world*shards;auto* counts=alloc<uint32_t>(nq);auto* error=alloc<uint32_t>(1);C(cudaMemset(counts,0,nq*4));C(cudaMemset(error,0,4));
 // Red zones around the output detect a write past the queue allocation.
 auto* guarded=alloc<GenericRouteRecord>(uint64_t(nq)*capacity+2);C(cudaMemset(guarded,0xa5,(uint64_t(nq)*capacity+2)*sizeof(GenericRouteRecord)));
 auto* records=guarded+1;
 int status=0;
 BitmapDeviceDomain domain{dv,df,N,N,720};
 if(mode==2){
  size_t scratch_bytes=0;C(static_cast<cudaError_t>(mgbfs_segmented_batch_route_scratch_bytes(count*G,nq,&scratch_bytes)));
  void* scratch=nullptr;C(cudaMalloc(&scratch,std::max<size_t>(scratch_bytes,1)));
  status=mgbfs_segmented_batch_route(uint32_t(bitmap),uint32_t(word_shard),domain,bytes,0,N,N,1,G,parent,count,count,dp,nullptr,nullptr,123,64,world,0,shards,capacity,begin,dm,nullptr,records,counts,error,scratch,scratch_bytes,nullptr);
  C(static_cast<cudaError_t>(status));C(cudaDeviceSynchronize());C(cudaFree(scratch));
 }else if(bitmap){auto fn=mode==1?mgbfs_bitmap_route_grouped:mgbfs_bitmap_route;
  status=fn(bytes,domain,N,G,parent,count,count,dp,world,0,shards,capacity,begin,dm,records,counts,error,nullptr);
 }else{auto fn=mode==1?mgbfs_segmented_route_storage_grouped:mgbfs_segmented_route_storage;
  status=fn(bytes,0,N,N,1,G,parent,count,count,dp,nullptr,nullptr,123,64,world,0,shards,capacity,begin,dm,nullptr,records,counts,error,nullptr);
 }
 C(static_cast<cudaError_t>(status));C(cudaDeviceSynchronize());
 Result out;out.counts.resize(nq);out.queues.resize(nq);C(cudaMemcpy(out.counts.data(),counts,nq*4,cudaMemcpyDeviceToHost));C(cudaMemcpy(&out.error,error,4,cudaMemcpyDeviceToHost));
 std::vector<GenericRouteRecord> all(uint64_t(nq)*capacity+2);C(cudaMemcpy(all.data(),guarded,all.size()*sizeof(all[0]),cudaMemcpyDeviceToHost));
 Bytes guard{};guard.fill(0xa5);CHECK(std::memcmp(&all.front(),guard.data(),guard.size())==0);CHECK(std::memcmp(&all.back(),guard.data(),guard.size())==0);
 if(mode==2 && (out.error&1u))for(size_t i=1;i+1<all.size();i++)CHECK(std::memcmp(&all[i],guard.data(),guard.size())==0);
 if(mode==2&&bitmap&&word_shard&&out.error==0){
  auto mix=[](uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);};
  for(uint32_t q=0;q<nq;q++)for(uint32_t i=0;i<out.counts[q];i++){
   uint64_t id=all[1+uint64_t(q)*capacity+i].hash,local=id/world;
   uint32_t owner=uint32_t(id%world),rank=world-1-owner;
   uint32_t shard=uint32_t((uint64_t(uint32_t(mix(local>>6)))*shards)>>32);
   CHECK(q==rank*shards+shard);
  }
 }
 for(uint32_t q=0;q<nq;q++){uint32_t take=std::min(capacity,out.counts[q]);for(uint32_t i=0;i<take;i++){Bytes x;std::memcpy(x.data(),&all[1+uint64_t(q)*capacity+i],x.size());out.queues[q].push_back(x);}std::sort(out.queues[q].begin(),out.queues[q].end());}
 C(cudaFree(dp));C(cudaFree(dm));C(cudaFree(dv));C(cudaFree(df));C(cudaFree(parent));C(cudaFree(counts));C(cudaFree(error));C(cudaFree(guarded));return out;
}

static void retry(bool bitmap,uint32_t width,uint32_t world,uint32_t shards,int mode){
 const uint32_t cap=4, n=33; Result reference=call(bitmap,0,width,world,shards,4096,n);
 std::vector<std::vector<Bytes>> received(world*shards);
 std::vector<std::pair<uint32_t,uint32_t>> work={{0,n}}; unsigned retries=0;
 while(!work.empty()){
  auto [begin,count]=work.back();work.pop_back();
  Result r=call(bitmap,mode,width,world,shards,cap,count,begin);
  if(r.error&1u){CHECK(count>1);uint32_t left=count/2;work.push_back({begin,left});work.push_back({begin+left,count-left});retries++;continue;}
  CHECK(r.error==0);
  for(uint32_t q=0;q<world*shards;q++)received[q].insert(received[q].end(),r.queues[q].begin(),r.queues[q].end());
 }
 CHECK(retries>0);for(auto& q:received)std::sort(q.begin(),q.end());CHECK(received==reference.queues);
}
int main(){int devices=0;C(cudaGetDeviceCount(&devices));CHECK(devices>=2);
 for(int device=0;device<2;device++){C(cudaSetDevice(device));for(bool bitmap:{false,true})for(uint32_t width:{1u,8u})for(uint32_t world:{1u,2u,8u})for(uint32_t shards:{1u,4u})for(uint32_t count:{0u,1u,31u,32u,33u,513u})for(uint32_t capacity:{1u,4096u}){
  Result a=call(bitmap,0,width,world,shards,capacity,count),b=call(bitmap,1,width,world,shards,capacity,count),c=call(bitmap,2,width,world,shards,capacity,count);
  CHECK(a.counts==b.counts&&a.counts==c.counts);uint64_t total=0;for(auto n:a.counts)total+=n;CHECK(total==uint64_t(count)*4);
  bool over=false;for(auto n:a.counts)over|=n>capacity;CHECK((a.error&1u)==uint32_t(over));CHECK((b.error&1u)==uint32_t(over));CHECK((c.error&1u)==uint32_t(over));
  CHECK((a.error&~1u)==0&&(b.error&~1u)==0&&(c.error&~1u)==0);if(!over)CHECK(a.queues==b.queues&&a.queues==c.queues);
 }
 for(bool bitmap:{false,true})for(uint32_t width:{1u,8u})for(uint32_t world:{1u,2u})for(int mode:{1,2})retry(bitmap,width,world,world==1?1:4,mode);
 for(uint32_t width:{1u,8u})for(uint32_t world:{1u,2u,8u})call(true,2,width,world,4,4096,513,0,true);
 printf("PASS batch route/count/overflow/retry/word-shard device=%d\n",device);fflush(stdout);
 }
 puts("BATCH_ROUTE_TWO_GPU_PASS");return 0;
}
