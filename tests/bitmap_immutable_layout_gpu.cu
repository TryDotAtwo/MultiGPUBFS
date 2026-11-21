// Two streams route independent jobs concurrently, with job-owned layouts.
// Compare every (packed state, exact mapped id) and every destination queue
// against a CPU permutation rank/map implementation on two physical GPUs.
#include "../cuda/generic_route.h"
#include "../cuda/permutation_bitmap.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>
#define CK(x) do{cudaError_t e=(cudaError_t)(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"ASSERT %s line %d\n",#x,__LINE__);exit(1);}}while(0)
extern "C" int mgbfs_bitmap_route_v2(uint32_t,BitmapDeviceDomain,uint32_t,uint32_t,const void*,uint32_t,uint64_t,const uint32_t*,uint32_t,uint32_t,uint32_t,uint32_t,uint64_t,const uint32_t*,GenericRouteRecord*,uint32_t*,uint32_t*,void*,uint32_t,uint64_t);
extern "C" int mgbfs_bitmap_route_grouped_v2(uint32_t,BitmapDeviceDomain,uint32_t,uint32_t,const void*,uint32_t,uint64_t,const uint32_t*,uint32_t,uint32_t,uint32_t,uint32_t,uint64_t,const uint32_t*,GenericRouteRecord*,uint32_t*,uint32_t*,void*,uint32_t,uint64_t);
extern "C" int mgbfs_segmented_batch_route_scratch_bytes(uint32_t,uint32_t,size_t*);
extern "C" int mgbfs_segmented_batch_route_v2(uint32_t,uint32_t,BitmapDeviceDomain,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,const void*,uint32_t,uint64_t,const uint32_t*,const int64_t*,const uint32_t*,uint64_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint64_t,const uint32_t*,const uint64_t*,GenericRouteRecord*,uint32_t*,uint32_t*,void*,void*,size_t,void*,uint32_t,uint64_t);
template<class T> T* alloc(size_t n){T* p=nullptr;CK(cudaMalloc((void**)&p,n*sizeof(T)));return p;}
template<class T> T* upload(const std::vector<T>& v){auto p=alloc<T>(v.size());CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
static uint64_t mix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
static uint64_t map_id(uint64_t id,uint32_t world,uint32_t page,uint64_t boundary){
 if(!page||id>=boundary)return id;uint64_t p=id/page,stripe=p/world,slot=p%world;
 return (stripe*page+id%page)*world+(slot+mix(stripe)%world)%world;
}
static uint64_t rank_perm(const std::array<uint8_t,7>& p){
 uint64_t rank=0;for(int i=0;i<7;i++){int count=0;for(int j=i+1;j<7;j++)count+=p[j]<p[i];rank=rank*(7-i)+count;}return rank;
}
using Key=std::pair<uint64_t,uint64_t>;
struct Job{cudaStream_t stream;GenericRouteRecord* records;uint32_t* counts;uint32_t* error;void* scratch;size_t scratch_bytes;uint32_t page;uint64_t boundary;std::vector<std::vector<Key>> want;};
int main(){
 constexpr uint32_t N=7,G=6,C=1024,W=8,S=4,Q=2048,BUCKETS=W*S;constexpr uint64_t TOTAL=5040;
 std::array<uint8_t,N> p={0,1,2,3,4,5,6};std::vector<std::array<uint8_t,N>> originals;
 std::vector<uint8_t> parents(N*C);for(uint32_t i=0;i<C;i++){originals.push_back(p);for(uint32_t e=0;e<N;e++)parents[e*C+i]=p[e];std::next_permutation(p.begin(),p.end());}
 std::vector<uint32_t> perms(G*N),mapping(W),freq(N,1);std::vector<int64_t> symbols(N);
 for(uint32_t i=0;i<N;i++)symbols[i]=i;for(uint32_t g=0;g<G;g++){for(uint32_t e=0;e<N;e++)perms[g*N+e]=e;std::swap(perms[g*N+g],perms[g*N+g+1]);}
 for(uint32_t i=0;i<W;i++)mapping[i]=W-1-i;
 int devices=0;CK(cudaGetDeviceCount(&devices));REQUIRE(devices>=2);
 for(int device=0;device<2;device++){
  CK(cudaSetDevice(device));auto dp=upload(parents);auto dg=upload(perms);auto dm=upload(mapping);auto ds=upload(symbols);auto df=upload(freq);BitmapDeviceDomain domain{ds,df,N,N,TOTAL};
  for(int mode=0;mode<3;mode++){
   std::array<Job,2> jobs{};
   for(int j=0;j<2;j++){
    Job& z=jobs[j];z.page=j?64:0;z.boundary=z.page?TOTAL-TOTAL%(z.page*W):0;z.want.resize(BUCKETS);
    CK(cudaStreamCreateWithFlags(&z.stream,cudaStreamNonBlocking));z.records=alloc<GenericRouteRecord>(uint64_t(BUCKETS)*Q);z.counts=alloc<uint32_t>(BUCKETS);z.error=alloc<uint32_t>(1);
    if(mode==2){CK(mgbfs_segmented_batch_route_scratch_bytes(C*G,BUCKETS,&z.scratch_bytes));CK(cudaMalloc(&z.scratch,z.scratch_bytes));}
    for(auto parent:originals)for(uint32_t g=0;g<G;g++){
     auto child=parent;std::swap(child[g],child[g+1]);uint64_t id=map_id(rank_perm(child),W,z.page,z.boundary),packed=0;
     for(uint32_t e=0;e<N;e++)packed|=uint64_t(child[e])<<(e*8);
     uint32_t shard=(uint64_t(uint32_t(mix(mode==2?(id/W)>>6:id)))*S)>>32;
     z.want[mapping[id%W]*S+shard].push_back({id,packed});
    }
    for(auto& q:z.want){REQUIRE(q.size()<=Q);std::sort(q.begin(),q.end());}
   }
   for(int pass=0;pass<4;pass++){
    for(auto& z:jobs){
     CK(cudaMemsetAsync(z.counts,0,BUCKETS*4,z.stream));CK(cudaMemsetAsync(z.error,0,4,z.stream));
     int rc=0;
     if(mode==2)rc=mgbfs_segmented_batch_route_v2(1,1,domain,1,0,N,N,1,G,dp,C,C,dg,nullptr,nullptr,0,64,W,0,S,Q,0,dm,nullptr,z.records,z.counts,z.error,nullptr,z.scratch,z.scratch_bytes,z.stream,z.page,z.boundary);
     else{auto fn=mode==1?mgbfs_bitmap_route_grouped_v2:mgbfs_bitmap_route_v2;rc=fn(1,domain,N,G,dp,C,C,dg,W,0,S,Q,0,dm,z.records,z.counts,z.error,z.stream,z.page,z.boundary);}
     CK(rc);
    }
    for(auto& z:jobs){
     CK(cudaStreamSynchronize(z.stream));std::vector<uint32_t> counts(BUCKETS);uint32_t err=0;CK(cudaMemcpy(counts.data(),z.counts,BUCKETS*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&err,z.error,4,cudaMemcpyDeviceToHost));REQUIRE(err==0);
     std::vector<GenericRouteRecord> records(uint64_t(BUCKETS)*Q);CK(cudaMemcpy(records.data(),z.records,records.size()*sizeof(records[0]),cudaMemcpyDeviceToHost));
     for(uint32_t q=0;q<BUCKETS;q++){
      REQUIRE(counts[q]==z.want[q].size());std::vector<Key> got;for(uint32_t i=0;i<counts[q];i++){auto r=records[uint64_t(q)*Q+i];got.push_back({r.hash,r.parent});}
      std::sort(got.begin(),got.end());REQUIRE(got==z.want[q]);
     }
    }
   }
   auto& z=jobs[1];auto fn=mgbfs_bitmap_route_v2;
   REQUIRE(fn(1,domain,N,G,dp,C,C,dg,W,0,S,Q,0,dm,z.records,z.counts,z.error,z.stream,3,z.boundary)==int(cudaErrorInvalidValue));
   REQUIRE(fn(1,domain,N,G,dp,C,C,dg,W,0,S,Q,0,dm,z.records,z.counts,z.error,z.stream,64,0)==int(cudaErrorInvalidValue));
   for(auto& a:jobs){if(a.scratch)CK(cudaFree(a.scratch));CK(cudaFree(a.records));CK(cudaFree(a.counts));CK(cudaFree(a.error));CK(cudaStreamDestroy(a.stream));}
   printf("PASS gpu=%d mode=%d concurrent page=0/64 exact queues x4\n",device,mode);fflush(stdout);
  }
  CK(cudaFree(dp));CK(cudaFree(dg));CK(cudaFree(dm));CK(cudaFree(ds));CK(cudaFree(df));
 }
 puts("BITMAP_IMMUTABLE_TWO_GPU_PASS");return 0;
}
