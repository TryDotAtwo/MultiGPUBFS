#include "../cuda/bitmap_pack4_route.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <tuple>
#include <vector>
#define CK(x) do{cudaError_t ec=(x);if(ec!=cudaSuccess){std::cerr<<"CUDA "<<__LINE__<<" "<<cudaGetErrorString(ec)<<"\n";std::exit(1);}}while(0)
#define REQUIRE(x) do{if(!(x)){std::cerr<<"FAIL "<<__LINE__<<" "#x"\n";std::exit(1);}}while(0)
static uint64_t mix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
static uint64_t map_page(uint64_t id,uint32_t world,uint32_t page,uint64_t boundary){
 if(!page||id>=boundary)return id;
 uint64_t p=id/page,stripe=p/world,slot=p%world,owner=(slot+mix(stripe)%world)%world,local=stripe*page+id%page;
 return local*world+owner;
}
template<class T>T* alloc(size_t n){T* p=nullptr;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T>void upload(T* p,const std::vector<T>& v){CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
using Item=std::tuple<uint32_t,uint64_t,std::vector<int64_t>>;
static void run(int gpu,bool grouped,bool byte_wire,bool repeated,bool reverse,bool paged) {
 CK(cudaSetDevice(gpu));
 const uint32_t n=5,gens=4,world=2,shards=2,queues=world*shards,cap=512;
 std::vector<int64_t> root=byte_wire?(repeated?std::vector<int64_t>{2,2,7,11,25}:std::vector<int64_t>{2,3,7,11,25}):
  (repeated?std::vector<int64_t>{-50,-50,0,12,900}:std::vector<int64_t>{-50,-1,0,12,900});
 std::vector<std::vector<int64_t>> states;auto x=root;
 do{states.push_back(x);}while(std::next_permutation(x.begin(),x.end()));
 std::vector<int64_t> symbols;std::vector<uint32_t> symbol_counts;
 for(auto v:root){if(symbols.empty()||symbols.back()!=v){symbols.push_back(v);symbol_counts.push_back(1);}else symbol_counts.back()++;}
 MgbfsPack4Dictionary dict{};dict.elements=n;dict.distinct=symbols.size();for(uint32_t i=0;i<symbols.size();i++)dict.symbols[i]=symbols[i];
 const uint32_t begin=states.size()*gpu/2,end=states.size()*(gpu+1)/2,count=end-begin;
 std::vector<uint64_t> parents(count);
 for(uint32_t p=0;p<count;p++)for(uint32_t e=0;e<n;e++)parents[p]|=uint64_t(std::lower_bound(symbols.begin(),symbols.end(),states[begin+p][e])-symbols.begin())<<(4*e);
 std::vector<uint32_t> perms(gens*n);for(uint32_t g=0;g<gens;g++){for(uint32_t e=0;e<n;e++)perms[g*n+e]=e;std::swap(perms[g*n+g],perms[g*n+g+1]);}
 std::vector<uint32_t> rank_map=reverse?std::vector<uint32_t>{1,0}:std::vector<uint32_t>{0,1};
 const uint32_t page=paged?8:0;const uint64_t boundary=page?states.size()-states.size()%(page*world):0;
 std::vector<Item> expected;std::vector<uint32_t> expected_counts(queues);
 for(uint32_t p=0;p<count;p++)for(uint32_t g=0;g<gens;g++){
  auto child=states[begin+p];std::swap(child[g],child[g+1]);
  auto it=std::lower_bound(states.begin(),states.end(),child);REQUIRE(it!=states.end()&&*it==child);
  uint64_t id=map_page(it-states.begin(),world,page,boundary);
  uint32_t target=rank_map[id%world],sh=(uint64_t(uint32_t(mix(id)))*shards)>>32,queue=target*shards+sh;
  expected.emplace_back(queue,id,child);expected_counts[queue]++;
 }
 auto *dp=alloc<uint64_t>(parents.size());auto *dperms=alloc<uint32_t>(perms.size()),*dmap=alloc<uint32_t>(world);
 auto *dsym=alloc<int64_t>(symbols.size());auto *dcnt=alloc<uint32_t>(symbol_counts.size());
 upload(dp,parents);upload(dperms,perms);upload(dmap,rank_map);upload(dsym,symbols);upload(dcnt,symbol_counts);
 auto *records=alloc<GenericRouteRecord>(queues*cap);auto *payload=alloc<int64_t>(queues*n*cap);
 auto *counts=alloc<uint32_t>(queues),*error=alloc<uint32_t>(1);
 BitmapDeviceDomain domain{dsym,dcnt,uint32_t(symbols.size()),n,uint64_t(states.size())};
 MgbfsPack4RouteArgs a{};a.dictionary=dict;a.domain=domain;a.parents=dp;a.permutations=dperms;
 a.count=count;a.generators=gens;a.world=world;a.source=gpu;a.shards=shards;a.capacity=cap;
 a.state_bytes=byte_wire?1:8;a.grouped=grouped;a.map=dmap;a.records=records;a.payload=payload;
 a.counts=counts;a.error=error;a.owner_page=page;a.owner_boundary=boundary;
 auto launch=[&](){
  CK(cudaMemset(counts,0,queues*4));CK(cudaMemset(error,0,4));
  // The host copies whole padded queues for inspection; initialize the unused
  // lanes to a conspicuous sentinel, then compare only the counted records.
  CK(cudaMemset(records,0xa5,queues*cap*sizeof(GenericRouteRecord)));
  CK(cudaMemset(payload,0xa5,queues*n*cap*sizeof(int64_t)));
  CK((cudaError_t)mgbfs_pack4_route(a,nullptr));CK(cudaDeviceSynchronize());
 };
 launch();
 std::vector<uint32_t> actual_counts(queues);uint32_t err=0;
 CK(cudaMemcpy(actual_counts.data(),counts,queues*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&err,error,4,cudaMemcpyDeviceToHost));
 REQUIRE(!err&&actual_counts==expected_counts);
 std::vector<GenericRouteRecord> hm(queues*cap);std::vector<int64_t> hp(queues*n*cap);
 CK(cudaMemcpy(hm.data(),records,hm.size()*sizeof(GenericRouteRecord),cudaMemcpyDeviceToHost));
 if(!byte_wire)CK(cudaMemcpy(hp.data(),payload,hp.size()*8,cudaMemcpyDeviceToHost));
 std::vector<Item> got;
 for(uint32_t q=0;q<queues;q++)for(uint32_t row=0;row<actual_counts[q];row++){
  const auto& rec=hm[q*cap+row];std::vector<int64_t> child(n);
  for(uint32_t e=0;e<n;e++)child[e]=byte_wire?int64_t((rec.parent>>(8*e))&255u):hp[q*n*cap+e*cap+row];
  got.emplace_back(q,rec.hash,child);
 }
 std::sort(expected.begin(),expected.end());std::sort(got.begin(),got.end());REQUIRE(got==expected);
 // A rejected overflow wave is not sent; counts still report the exact demand.
 a.capacity=1;launch();CK(cudaMemcpy(actual_counts.data(),counts,queues*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&err,error,4,cudaMemcpyDeviceToHost));
 REQUIRE((err&1u)!=0&&actual_counts==expected_counts);
 // A subsequent full-capacity retry starts from the same parent input.
 a.capacity=cap;launch();CK(cudaMemcpy(&err,error,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(actual_counts.data(),counts,queues*4,cudaMemcpyDeviceToHost));
 REQUIRE(!err&&actual_counts==expected_counts);
 CK(cudaMemcpy(hm.data(),records,hm.size()*sizeof(GenericRouteRecord),cudaMemcpyDeviceToHost));
 if(!byte_wire)CK(cudaMemcpy(hp.data(),payload,hp.size()*8,cudaMemcpyDeviceToHost));
 got.clear();
 for(uint32_t q=0;q<queues;q++)for(uint32_t row=0;row<actual_counts[q];row++){
  const auto& rec=hm[q*cap+row];std::vector<int64_t> child(n);
  for(uint32_t e=0;e<n;e++)child[e]=byte_wire?int64_t((rec.parent>>(8*e))&255u):hp[q*n*cap+e*cap+row];
  got.emplace_back(q,rec.hash,child);
 }
 std::sort(got.begin(),got.end());REQUIRE(got==expected);
 CK(cudaFree(dp));CK(cudaFree(dperms));CK(cudaFree(dmap));CK(cudaFree(dsym));CK(cudaFree(dcnt));CK(cudaFree(records));CK(cudaFree(payload));CK(cudaFree(counts));CK(cudaFree(error));
}

static void run_sixteen(int gpu) {
 CK(cudaSetDevice(gpu));
 constexpr uint32_t n=16,gens=15,world=2,shards=2,cap=32,queues=world*shards;
 std::vector<int64_t> symbols(n);std::vector<uint32_t> counts_host(n,1),perms(gens*n),map={0,1};
 MgbfsPack4Dictionary dict{};dict.elements=n;dict.distinct=n;uint64_t packed=0,total=1;
 for(uint32_t e=0;e<n;e++){symbols[e]=dict.symbols[e]=e;packed|=uint64_t(e)<<(4*e);total*=e+1;}
 for(uint32_t g=0;g<gens;g++){for(uint32_t e=0;e<n;e++)perms[g*n+e]=e;std::swap(perms[g*n+g],perms[g*n+g+1]);}
 auto *dp=alloc<uint64_t>(1);auto *dperms=alloc<uint32_t>(perms.size()),*dmap=alloc<uint32_t>(world);
 auto *dsym=alloc<int64_t>(n);auto *dcnt=alloc<uint32_t>(n),*counts=alloc<uint32_t>(queues),*err=alloc<uint32_t>(1);
 auto *records=alloc<GenericRouteRecord>(queues*cap);CK(cudaMemcpy(dp,&packed,8,cudaMemcpyHostToDevice));
 upload(dperms,perms);upload(dmap,map);upload(dsym,symbols);upload(dcnt,counts_host);
 MgbfsPack4RouteArgs a{};a.dictionary=dict;a.domain=BitmapDeviceDomain{dsym,dcnt,n,n,total};
 a.parents=dp;a.permutations=dperms;a.count=1;a.generators=gens;a.world=world;a.source=gpu;
 a.shards=shards;a.capacity=cap;a.state_bytes=1;a.grouped=1;a.map=dmap;a.records=records;a.counts=counts;a.error=err;
 CK(cudaMemset(counts,0,queues*4));CK(cudaMemset(err,0,4));CK(cudaMemset(records,0xa5,queues*cap*sizeof(GenericRouteRecord)));
 CK((cudaError_t)mgbfs_pack4_route(a,nullptr));CK(cudaDeviceSynchronize());
 uint32_t error=0;std::vector<uint32_t> hc(queues);std::vector<GenericRouteRecord> hm(queues*cap);
 CK(cudaMemcpy(&error,err,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hc.data(),counts,queues*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hm.data(),records,hm.size()*sizeof(GenericRouteRecord),cudaMemcpyDeviceToHost));REQUIRE(!error);
 std::vector<std::vector<int64_t>> observed;
 for(uint32_t q=0;q<queues;q++)for(uint32_t row=0;row<hc[q];row++){
  const auto& rec=hm[q*cap+row];uint64_t hi=uint64_t(rec.source)|(uint64_t(rec.generator)<<32);
  std::vector<int64_t> child(n);
  for(uint32_t e=0;e<n;e++)child[e]=int64_t(((e<8?rec.parent:hi)>>(8*(e%8)))&255u);
  observed.push_back(child);
 }
 std::vector<std::vector<int64_t>> expected;for(uint32_t g=0;g<gens;g++){auto child=symbols;std::swap(child[g],child[g+1]);expected.push_back(child);}
 std::sort(observed.begin(),observed.end());std::sort(expected.begin(),expected.end());REQUIRE(observed==expected);
 // Reject an invalid generator before it can emit the corresponding child.
 uint32_t bad=n;CK(cudaMemcpy(dperms,&bad,4,cudaMemcpyHostToDevice));CK(cudaMemset(counts,0,queues*4));CK(cudaMemset(err,0,4));
 CK((cudaError_t)mgbfs_pack4_route(a,nullptr));CK(cudaDeviceSynchronize());CK(cudaMemcpy(&error,err,4,cudaMemcpyDeviceToHost));REQUIRE(error&8u);
 // Byte-wire cannot silently truncate a signed dictionary value.
 a.dictionary.symbols[0]=-1;REQUIRE(mgbfs_pack4_route(a,nullptr)==int(cudaErrorInvalidValue));
 CK(cudaFree(dp));CK(cudaFree(dperms));CK(cudaFree(dmap));CK(cudaFree(dsym));CK(cudaFree(dcnt));CK(cudaFree(counts));CK(cudaFree(err));CK(cudaFree(records));
 std::cout<<"ROUTE_SIXTEEN_PASS gpu="<<gpu<<" children=15 last_nibble=checked\n";
}

int main(){
 int devices=0;CK(cudaGetDeviceCount(&devices));REQUIRE(devices>=2);uint32_t cases=0;
 for(int gpu=0;gpu<2;gpu++)for(bool grouped:{false,true})for(bool byte:{false,true})
  for(bool repeated:{false,true})for(bool reverse:{false,true})for(bool paged:{false,true}){
   run(gpu,grouped,byte,repeated,reverse,paged);cases++;
  }
 for(int gpu=0;gpu<2;gpu++)run_sixteen(gpu);
 std::cout<<"PACK4_ROUTE_PASS gpu=2 configurations="<<cases<<" overflow_and_retry=each exact_candidate_multisets=each\n";
}
