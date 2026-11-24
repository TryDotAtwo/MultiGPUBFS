#include "../cuda/bitmap_pack4_route.cuh"
#include "../cuda/bitmap_pack4_owner.cuh"
#include <cuda_runtime.h>
#include <nccl.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){std::cerr<<"CUDA "<<__LINE__<<" "<<cudaGetErrorString(e)<<"\n";std::exit(1);}}while(0)
#define NK(x) do{ncclResult_t e=(x);if(e!=ncclSuccess){std::cerr<<"NCCL "<<__LINE__<<" "<<ncclGetErrorString(e)<<"\n";std::exit(1);}}while(0)
#define REQUIRE(x) do{if(!(x)){std::cerr<<"FAIL "<<__LINE__<<" "#x"\n";std::exit(1);}}while(0)
template<class T>T* alloc(size_t n){T* p=nullptr;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T>void upload(T* p,const std::vector<T>& v){if(!v.empty())CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
struct Rank {
 int dev=0;cudaStream_t stream=nullptr;ncclComm_t comm=nullptr;
 uint64_t *parents=nullptr,*bits=nullptr,*rank_count=nullptr;
 uint32_t *perms=nullptr,*mapping=nullptr,*domain_counts=nullptr,*counts=nullptr,*recv_counts=nullptr,*error=nullptr;
 int64_t *symbols=nullptr,*payload=nullptr,*recv_payload=nullptr;
 GenericRouteRecord *records=nullptr,*recv_records=nullptr;
 std::vector<uint64_t> frontier;std::vector<uint32_t> host_counts,host_recv_counts;
};
static uint64_t pack(const MgbfsPack4Dictionary& d,const std::vector<int64_t>& row){uint64_t x=0;REQUIRE(mgbfs_pack4_encode(d,row.data(),&x));return x;}
static std::vector<int64_t> unpack(const MgbfsPack4Dictionary& d,uint64_t row){std::vector<int64_t> out(d.elements);REQUIRE(mgbfs_pack4_decode(d,row,out.data()));return out;}
static void run(bool byte_wire,bool repeated,bool reverse,bool grouped,int cap_mode) {
 constexpr uint32_t world=2,n=5,gens=4,shards=2,q=128,queues=world*shards;
 std::vector<int64_t> start=byte_wire?(repeated?std::vector<int64_t>{2,2,7,11,25}:std::vector<int64_t>{2,3,7,11,25}):
  (repeated?std::vector<int64_t>{-50,-50,0,12,900}:std::vector<int64_t>{-50,-1,0,12,900});
 std::vector<std::vector<int64_t>> domain_states;auto cur=start;
 do{domain_states.push_back(cur);}while(std::next_permutation(cur.begin(),cur.end()));
 REQUIRE(domain_states.size()<=128);
 std::vector<int64_t> symbols;std::vector<uint32_t> symbol_counts;
 for(auto v:start){if(symbols.empty()||symbols.back()!=v){symbols.push_back(v);symbol_counts.push_back(1);}else symbol_counts.back()++;}
 MgbfsPack4Dictionary dict{};dict.elements=n;dict.distinct=symbols.size();for(uint32_t i=0;i<symbols.size();i++)dict.symbols[i]=symbols[i];
 std::vector<uint32_t> perms(gens*n),mapping=reverse?std::vector<uint32_t>{1,0}:std::vector<uint32_t>{0,1};
 for(uint32_t g=0;g<gens;g++){for(uint32_t e=0;e<n;e++)perms[g*n+e]=e;std::swap(perms[g*n+g],perms[g*n+g+1]);}
 // Independent oracle: maximum number of states in one layer on a rank.
 uint64_t peak_per_rank=0;
 {
  std::set<std::vector<int64_t>> seen{start},frontier{start};
  while(!frontier.empty()){
   std::array<uint64_t,2> by_rank={0,0};
   for(const auto& state:frontier){auto it=std::lower_bound(domain_states.begin(),domain_states.end(),state);REQUIRE(it!=domain_states.end()&&*it==state);by_rank[mapping[(it-domain_states.begin())%world]]++;}
   peak_per_rank=std::max(peak_per_rank,std::max(by_rank[0],by_rank[1]));
   std::set<std::vector<int64_t>> next;
   for(const auto& parent:frontier)for(uint32_t g=0;g<gens;g++){auto child=parent;std::swap(child[g],child[g+1]);if(!seen.count(child))next.insert(child);}
   seen.insert(next.begin(),next.end());frontier=next;
  }
 }
 REQUIRE(peak_per_rank>1);const uint64_t hard_cap=cap_mode==2?peak_per_rank-1:peak_per_rank;
 std::array<Rank,2> r;
 for(uint32_t rank=0;rank<world;rank++){
  auto& t=r[rank];t.dev=rank;CK(cudaSetDevice(rank));CK(cudaStreamCreateWithFlags(&t.stream,cudaStreamNonBlocking));
  t.parents=alloc<uint64_t>(domain_states.size());t.bits=alloc<uint64_t>(1);t.rank_count=alloc<uint64_t>(1);CK(cudaMemset(t.bits,0,8));
  t.perms=alloc<uint32_t>(perms.size());t.mapping=alloc<uint32_t>(world);t.domain_counts=alloc<uint32_t>(symbol_counts.size());
  t.symbols=alloc<int64_t>(symbols.size());t.counts=alloc<uint32_t>(queues);t.recv_counts=alloc<uint32_t>(queues);t.error=alloc<uint32_t>(1);
  t.records=alloc<GenericRouteRecord>(queues*q);t.recv_records=alloc<GenericRouteRecord>(queues*q);
  t.payload=alloc<int64_t>(queues*n*q);t.recv_payload=alloc<int64_t>(queues*n*q);
  upload(t.perms,perms);upload(t.mapping,mapping);upload(t.domain_counts,symbol_counts);upload(t.symbols,symbols);
  t.host_counts.resize(queues);t.host_recv_counts.resize(queues);
 }
 int devices[2]={0,1};ncclComm_t comm[2];NK(ncclCommInitAll(comm,2,devices));for(int i=0;i<2;i++)r[i].comm=comm[i];
 const uint32_t root_rank=mapping[0];CK(cudaSetDevice(root_rank));
 auto *seed_input=alloc<int64_t>(n);auto *seed_row=alloc<uint64_t>(1),*seed_count=alloc<uint64_t>(1);auto *seed_error=alloc<uint32_t>(1);
 upload(seed_input,start);CK(cudaMemset(seed_count,0,8));CK(cudaMemset(seed_error,0,4));
 MgbfsPack4SeedArgs seed{};seed.dictionary=dict;seed.input=seed_input;seed.row=seed_row;seed.bits=r[root_rank].bits;
 seed.accepted=seed_count;seed.error=seed_error;seed.local=0;seed.owner_len=(domain_states.size()+1)/2;
 CK((cudaError_t)mgbfs_pack4_seed(seed,r[root_rank].stream));CK(cudaStreamSynchronize(r[root_rank].stream));
 uint64_t root_word=0,root_count=0;uint32_t root_error=0;
 CK(cudaMemcpy(&root_word,seed_row,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&root_count,seed_count,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&root_error,seed_error,4,cudaMemcpyDeviceToHost));
 REQUIRE(!root_error&&root_count==1&&root_word==pack(dict,start));r[root_rank].frontier.push_back(root_word);
 CK(cudaFree(seed_input));CK(cudaFree(seed_row));CK(cudaFree(seed_count));CK(cudaFree(seed_error));
 std::set<std::vector<int64_t>> cpu_seen{start},cpu_frontier{start};
 uint64_t spills_total=0,retries=0,remote_records=0;uint32_t depth=0,verified=1;bool incomplete=false;
 while(!cpu_frontier.empty()){
  REQUIRE(depth<30);
  std::set<std::vector<int64_t>> expected;
  for(const auto& parent:cpu_frontier)for(uint32_t g=0;g<gens;g++){
   auto child=parent;std::swap(child[g],child[g+1]);if(!cpu_seen.count(child))expected.insert(child);
  }
  std::array<uint64_t,2> expected_by_rank={0,0};
  for(const auto& state:expected){auto it=std::lower_bound(domain_states.begin(),domain_states.end(),state);REQUIRE(it!=domain_states.end()&&*it==state);expected_by_rank[mapping[(it-domain_states.begin())%world]]++;}
  // The first undersized route is intentionally discarded after overflow.
  for(uint32_t rank=0;rank<world;rank++){
   auto& t=r[rank];CK(cudaSetDevice(rank));upload(t.parents,t.frontier);
   BitmapDeviceDomain domain{t.symbols,t.domain_counts,uint32_t(symbols.size()),n,uint64_t(domain_states.size())};
   MgbfsPack4RouteArgs a{};a.dictionary=dict;a.domain=domain;a.parents=t.parents;a.permutations=t.perms;
   a.count=t.frontier.size();a.generators=gens;a.world=world;a.source=rank;a.shards=shards;a.state_bytes=byte_wire?1:8;
   a.grouped=grouped;a.map=t.mapping;a.records=t.records;a.payload=t.payload;a.counts=t.counts;a.error=t.error;
   for(uint32_t capacity:std::array<uint32_t,2>{1,q}){
    a.capacity=capacity;CK(cudaMemset(t.counts,0,queues*4));CK(cudaMemset(t.error,0,4));
    CK(cudaMemset(t.records,0xa5,queues*q*sizeof(GenericRouteRecord)));CK(cudaMemset(t.payload,0xa5,queues*n*q*8));
    CK((cudaError_t)mgbfs_pack4_route(a,t.stream));CK(cudaStreamSynchronize(t.stream));
    uint32_t error=0;CK(cudaMemcpy(&error,t.error,4,cudaMemcpyDeviceToHost));
    if(capacity==1){REQUIRE((error&~1u)==0);if(error&1u)retries++;}
    else REQUIRE(error==0);
   }
   CK(cudaMemcpy(t.host_counts.data(),t.counts,queues*4,cudaMemcpyDeviceToHost));
   CK(cudaMemset(t.recv_counts,0,queues*4));CK(cudaMemset(t.recv_records,0xa5,queues*q*sizeof(GenericRouteRecord)));
   CK(cudaMemset(t.recv_payload,0xa5,queues*n*q*8));
  }
  NK(ncclGroupStart());
  for(uint32_t src=0;src<world;src++){
   uint32_t dst=1-src;auto& a=r[src];auto& b=r[dst];
   CK(cudaSetDevice(src));
   NK(ncclSend(a.counts+dst*shards,shards,ncclUint32,dst,a.comm,a.stream));
   NK(ncclSend(a.records+dst*shards*q,shards*q*sizeof(GenericRouteRecord),ncclUint8,dst,a.comm,a.stream));
   if(!byte_wire)NK(ncclSend(a.payload+dst*shards*n*q,shards*n*q,ncclInt64,dst,a.comm,a.stream));
   CK(cudaSetDevice(dst));
   NK(ncclRecv(b.recv_counts+src*shards,shards,ncclUint32,src,b.comm,b.stream));
   NK(ncclRecv(b.recv_records+src*shards*q,shards*q*sizeof(GenericRouteRecord),ncclUint8,src,b.comm,b.stream));
   if(!byte_wire)NK(ncclRecv(b.recv_payload+src*shards*n*q,shards*n*q,ncclInt64,src,b.comm,b.stream));
  }
  NK(ncclGroupEnd());
  for(uint32_t rank=0;rank<world;rank++){
   auto& t=r[rank];CK(cudaSetDevice(rank));CK(cudaStreamSynchronize(t.stream));
   CK(cudaMemcpy(t.host_recv_counts.data(),t.recv_counts,queues*4,cudaMemcpyDeviceToHost));
   uint32_t peer=1-rank;
   for(uint32_t sh=0;sh<shards;sh++){
    REQUIRE(t.host_recv_counts[peer*shards+sh]==r[peer].host_counts[rank*shards+sh]);
    remote_records+=t.host_recv_counts[peer*shards+sh];
   }
  }
  std::array<std::vector<uint64_t>,2> next;
  std::multiset<std::vector<int64_t>> actual;
  bool resource=false;
  for(uint32_t rank=0;rank<world&&!resource;rank++){
   auto& t=r[rank];CK(cudaSetDevice(rank));const uint32_t peer=1-rank,owner=reverse?peer:rank;
   CK(cudaMemset(t.rank_count,0,8));
   for(uint32_t sh=0;sh<shards&&!resource;sh++){
    const uint32_t visible=std::max(t.host_counts[rank*shards+sh],t.host_recv_counts[peer*shards+sh]);
    uint32_t selected=world*visible;if(!selected)continue;
    auto *selection=alloc<uint32_t>(world*q),*spill=alloc<uint32_t>(world*q),*spill_count=alloc<uint32_t>(1),*error=alloc<uint32_t>(1);
    auto *rows=alloc<uint64_t>(2),*accepted=alloc<uint64_t>(1);bool replay=false;uint32_t stalled=0;
    for(uint32_t pass=0;selected;pass++){
     REQUIRE(pass<512);CK(cudaMemset(spill_count,0,4));CK(cudaMemset(error,0,4));CK(cudaMemset(accepted,0,8));CK(cudaMemset(rows,0xff,16));
     MgbfsPack4OwnerArgs a{};a.dictionary=dict;a.elements=n;a.state_bytes=byte_wire?1:8;a.packed=byte_wire;
     a.local=t.payload;a.remote=t.recv_payload;a.local_meta=t.records;a.remote_meta=t.recv_records;a.local_counts=t.counts;a.remote_counts=t.recv_counts;
     a.rank=rank;a.world=world;a.owner=owner;a.total=domain_states.size();a.owner_len=(domain_states.size()+world-1-owner)/world;a.bits=t.bits;
     a.shard=sh;a.shards=shards;a.q=q;a.visible=visible;a.selection=replay?selection:nullptr;a.selected=selected;
     a.capacity=1;a.banks=2;a.rows=rows;a.accepted=accepted;a.spill=spill;a.spill_count=spill_count;a.error=error;
     if(cap_mode){a.global_count=t.rank_count;a.global_limit=hard_cap;}
     CK((cudaError_t)mgbfs_pack4_owner_accept(a,t.stream));CK(cudaStreamSynchronize(t.stream));
     uint32_t err=0,num_spill=0;uint64_t count=0,word=0;
     CK(cudaMemcpy(&err,error,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&num_spill,spill_count,4,cudaMemcpyDeviceToHost));
     CK(cudaMemcpy(&count,accepted,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&word,rows,8,cudaMemcpyDeviceToHost));
     REQUIRE(!(err&~16u)&&count<=1&&num_spill<=selected);spills_total+=num_spill;
     uint64_t rank_count=0;if(cap_mode)CK(cudaMemcpy(&rank_count,t.rank_count,8,cudaMemcpyDeviceToHost));
     if((err&16u)&&num_spill&&rank_count>=hard_cap)stalled++;else stalled=0;
     if(count){
      auto row=unpack(dict,word);auto it=std::lower_bound(domain_states.begin(),domain_states.end(),row);
      REQUIRE(it!=domain_states.end()&&*it==row);
      const uint64_t id=it-domain_states.begin();REQUIRE(mapping[id%world]==rank);
      actual.insert(row);next[rank].push_back(word);
     }
     selected=num_spill;replay=true;
     if(stalled>=2){resource=true;break;}
     if(selected)CK(cudaMemcpy(selection,spill,selected*4,cudaMemcpyDeviceToDevice));
    }
    CK(cudaFree(selection));CK(cudaFree(spill));CK(cudaFree(spill_count));CK(cudaFree(error));CK(cudaFree(rows));CK(cudaFree(accepted));
   }
  }
  if(resource){
   REQUIRE(cap_mode==2&&(expected_by_rank[0]>hard_cap||expected_by_rank[1]>hard_cap));
   std::set<std::vector<int64_t>> partial(actual.begin(),actual.end());REQUIRE(partial.size()==actual.size());
   for(const auto& state:actual)REQUIRE(expected.count(state)==1);
   incomplete=true;
   std::cout<<"PACK4_NCCL_RESOURCE_PREFIX last_depth="<<depth<<" first_unfinished="<<depth+1
            <<" cap="<<hard_cap<<" oracle_rank_sizes="<<expected_by_rank[0]<<","<<expected_by_rank[1]<<"\n";
   break;
  }
  if(cap_mode)REQUIRE(expected_by_rank[0]<=hard_cap&&expected_by_rank[1]<=hard_cap);
  std::multiset<std::vector<int64_t>> expected_multi(expected.begin(),expected.end());
  if(actual!=expected_multi){std::cerr<<"LAYER_MISMATCH depth="<<depth+1<<" got="<<actual.size()<<" expected="<<expected.size()<<"\n";std::exit(1);}
  cpu_seen.insert(expected.begin(),expected.end());cpu_frontier=expected;
  r[0].frontier=next[0];r[1].frontier=next[1];depth++;
  if(!expected.empty())verified++;
  // Confirm the visited bitmap agrees with the independently enumerated IDs.
  std::array<uint64_t,2> wanted={0,0};
  for(const auto& state:cpu_seen){auto it=std::lower_bound(domain_states.begin(),domain_states.end(),state);REQUIRE(it!=domain_states.end()&&*it==state);uint64_t id=it-domain_states.begin();wanted[mapping[id%world]]|=1ull<<(id/world);}
  for(uint32_t rank=0;rank<world;rank++){CK(cudaSetDevice(rank));uint64_t bits=0;CK(cudaMemcpy(&bits,r[rank].bits,8,cudaMemcpyDeviceToHost));REQUIRE(bits==wanted[rank]);}
 }
 if(cap_mode==2)REQUIRE(incomplete&&cpu_seen.size()<domain_states.size());
 else REQUIRE(!incomplete&&cpu_seen.size()==domain_states.size());
 REQUIRE(spills_total>0&&retries>0&&remote_records>0);
 for(uint32_t rank=0;rank<world;rank++){
  auto& t=r[rank];CK(cudaSetDevice(rank));NK(ncclCommDestroy(t.comm));CK(cudaFree(t.parents));CK(cudaFree(t.bits));CK(cudaFree(t.rank_count));CK(cudaFree(t.perms));CK(cudaFree(t.mapping));CK(cudaFree(t.domain_counts));CK(cudaFree(t.counts));CK(cudaFree(t.recv_counts));CK(cudaFree(t.error));CK(cudaFree(t.symbols));CK(cudaFree(t.payload));CK(cudaFree(t.recv_payload));CK(cudaFree(t.records));CK(cudaFree(t.recv_records));CK(cudaStreamDestroy(t.stream));
 }
 std::cout<<"PACK4_NCCL_CASE_PASS byte="<<byte_wire<<" repeated="<<repeated<<" reverse="<<reverse<<" grouped="<<grouped
          <<" cap_mode="<<cap_mode<<" states="<<cpu_seen.size()<<" verified_layers="<<verified<<" forced_spills="<<spills_total<<" discarded_overflow_waves="<<retries<<" remote_records="<<remote_records<<"\n";
}
int main(){int n=0;CK(cudaGetDeviceCount(&n));REQUIRE(n>=2);
 run(false,false,false,false,0);run(true,false,true,true,0);run(false,true,true,true,0);run(true,true,false,false,0);
 run(false,false,false,false,1);run(false,false,false,false,2);
 run(true,true,false,false,1);run(true,true,false,false,2);
 std::cout<<"PACK4_NCCL_BFS_FIXTURE_PASS physical_gpus=2 configurations=8\n";
}
