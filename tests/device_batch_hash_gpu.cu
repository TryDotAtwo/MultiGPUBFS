#include "device_batch_hash.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>
#define C(x) do{auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s:%d %s %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e));std::exit(1);}}while(0)
#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"%s:%d CHECK %s\n",__FILE__,__LINE__,#x);std::exit(1);}}while(0)
using namespace mgbfs_device_batch;
template<class T>T* managed(size_t n){T* p=nullptr;C(cudaMallocManaged(&p,n*sizeof(T)));return p;}
__global__ void run_hash(const Payload* input,uint64_t n,HashBatch a){hash_accept(input,n,a);}
template<class T> void check_type(bool tight) {
 constexpr unsigned extents_n=32,shards=2,capacity=512;
 auto* ext=managed<mgbfs_device_hash::Extent>(extents_n);
 auto* owners=managed<uint32_t>(extents_n);
 auto* heads=managed<uint64_t>(shards);auto* next=managed<uint64_t>(extents_n);
 auto* cursor=managed<uint64_t>(shards);
 std::fill(heads,heads+shards,UINT64_MAX);std::fill(next,next+extents_n,UINT64_MAX);
 std::fill(cursor,cursor+shards,0);
 for(unsigned e=0;e<extents_n;++e) {
  auto* slots=managed<uint64_t>(8);std::fill(slots,slots+8,UINT64_MAX);
  auto* rows=managed<T>(4*3);std::fill(rows,rows+12,T(0));
  auto* pos=managed<uint64_t>(3);std::fill(pos,pos+3,0);
  auto* counts=managed<uint64_t>(3);std::fill(counts,counts+3,0);
  ext[e]={MgbfsSegment{slots,rows,pos,8,1,3,8},counts};owners[e]=UINT32_MAX;
 }
 MgbfsPack4Dictionary dictionary{};dictionary.elements=4;dictionary.distinct=4;
 for(unsigned i=0;i<4;++i)dictionary.symbols[i]=sizeof(T)==1?int64_t(i):int64_t(i)*10000000000LL-10000000000LL;
 auto* input=managed<Payload>(capacity);auto* grouped=managed<Payload>(capacity);
 auto* offsets=managed<uint64_t>(shards+1);auto* counts=managed<uint64_t>(shards);
 auto* positions=managed<uint64_t>(capacity);auto* rank_count=managed<uint64_t>(1);
 auto* error=managed<uint32_t>(1);auto* mask=managed<uint64_t>(1);
 *rank_count=0;*error=0;*mask=0;
 HashBatch batch{ext,owners,extents_n,heads,next,cursor,dictionary,rank_count,tight?2ull:24ull,
                 shards,1,uint32_t(sizeof(T)),grouped,positions,counts,offsets,mask,error,nullptr,0,nullptr,0};
 std::array<unsigned,4> perm{0,1,2,3};std::vector<Payload> all;
 do {
  uint64_t word=0;for(unsigned k=0;k<4;++k)word|=uint64_t(perm[k])<<(4*k);
  const uint64_t hash=(all.size()%2)?0x80000000ull:0ull;
  all.push_back({hash,word});
 }while(std::next_permutation(perm.begin(),perm.end()));
 auto launch=[&](const std::vector<Payload>& in){
  std::copy(in.begin(),in.end(),input);uint64_t n=in.size();
  void* args[]={&input,&n,&batch};
  C(cudaLaunchCooperativeKernel((void*)run_hash,dim3(8),dim3(128),args,0,nullptr));
  C(cudaDeviceSynchronize());
 };
 std::vector<Payload> first=tight?std::vector<Payload>{all[0],all[1],all[0],all[1]}:all;
 if(!tight)for(unsigned i=0;i<11;++i)first.insert(first.end(),all.begin(),all.end());
 launch(first);CHECK(*error==0);CHECK(*rank_count==(tight?2:24));CHECK(*mask==3);
 std::map<uint64_t,unsigned> got;
 for(unsigned e=0;e<extents_n;++e)if(owners[e]!=UINT32_MAX){
  CHECK(owners[e]<2);CHECK(ext[e].counts[1]<=1);
  if(ext[e].counts[1]){
   uint64_t word=0;auto* values=static_cast<T*>(ext[e].segment.states);
   for(unsigned k=0;k<4;++k){
    const auto value=int64_t(values[k*3+1]);unsigned ord=0;
    while(ord<4&&dictionary.symbols[ord]!=value)++ord;
    CHECK(ord<4);word|=uint64_t(ord)<<(4*k);
   }
   ++got[word];
  }
 }
 CHECK(got.size()==(tight?2:24));for(auto x:got)CHECK(x.second==1);
 // Every live extent occurs exactly once in its owner's list; no unrelated
 // extent or cycle may be hidden from the next collision/duplicate lookup.
 std::vector<unsigned> linked(extents_n,0);
 for(unsigned shard=0;shard<shards;++shard) {
  for(uint64_t e=heads[shard];e!=UINT64_MAX;e=next[e]) {
   CHECK(e<extents_n);CHECK(owners[e]==shard);CHECK(++linked[e]==1);
  }
 }
 for(unsigned e=0;e<extents_n;++e)CHECK(linked[e]==unsigned(owners[e]!=UINT32_MAX));
 launch(first);CHECK(*error==0);CHECK(*rank_count==(tight?2:24));
 if(tight){launch(std::vector<Payload>{all[2]});CHECK(*error==HASH_RESOURCE);CHECK(*rank_count==2);}
 std::printf("PASS %s batch exact collision, two shards, small extents, duplicate%s\n",
  sizeof(T)==1?"u8":"wide",tight?" at cap + resource":" repeated across tiles");
}
int main(){
 int n=0;C(cudaGetDeviceCount(&n));if(!n)return 77;
 C(cudaSetDevice(0));C(cudaFree(nullptr));
 check_type<uint8_t>(false);check_type<int64_t>(false);
 check_type<uint8_t>(true);check_type<int64_t>(true);
 std::puts("PASS batched resident hash owner primitive; no distributed BFS claim");
}
