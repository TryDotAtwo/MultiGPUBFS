// Integration fixture: exercises the exported resident CUDA library entry, not
// the experimental scheduler. Capturing all states is explicit and test-only.
#include "device_pack4_bfs.cuh"
#ifdef MGBFS_LIBRARY_TEST_BATCH
#include "device_batch_bitmap_bfs.cuh"
extern "C" int mgbfs_device_batch_bitmap_geometry_v1(uint32_t*,uint32_t*);
extern "C" int mgbfs_device_batch_bitmap_init_v1(void*,uint64_t,uint64_t,void*);
extern "C" int mgbfs_device_batch_bitmap_launch_v1(const void*,const void*,const mgbfs_resident_batch::Args*,uint64_t,mgbfs_resident::Report*,void*);
#endif
#include <nccl.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <queue>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>
using namespace mgbfs_resident;
#ifndef MGBFS_LIBRARY_TEST_WORLD
#define MGBFS_LIBRARY_TEST_WORLD 2
#endif
constexpr unsigned WORLD=MGBFS_LIBRARY_TEST_WORLD;
extern "C" int mgbfs_device_pack4_bfs_smoke_v1();
extern "C" int mgbfs_device_pack4_bfs_init_v1(void*,uint64_t,const Args*,void*);
extern "C" int mgbfs_device_pack4_bfs_launch_v1(const void*,const void*,const Args*,uint64_t,Report*,void*);
#ifdef MGBFS_LIBRARY_TEST_TRANSPORT
extern "C" int mgbfs_nccl_create(uint32_t,uint32_t,uint32_t,const void*,void**,char*,size_t);
extern "C" int mgbfs_nccl_resident_prepare_v1(void*,uint64_t,char*,size_t);
extern "C" int mgbfs_nccl_resident_activate_v1(void*,char*,size_t);
extern "C" int mgbfs_nccl_resident_init_v1(void*,const Args*,void*);
extern "C" int mgbfs_nccl_resident_smoke_v1(void*,const Args*,uint32_t*,void*);
extern "C" int mgbfs_nccl_resident_launch_v1(void*,const Args*,Report*,void*);
extern "C" int mgbfs_nccl_resident_retire_v1(void*);
extern "C" int mgbfs_nccl_all_reduce_max_pair_u32(void*,const uint32_t*,uint32_t*,void*);
extern "C" int mgbfs_nccl_session_park(void*);
extern "C" void mgbfs_nccl_destroy(void*);
#endif
#define C(x) do{auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s:%d %s: %s\\n",__FILE__,__LINE__,#x,cudaGetErrorString(e));return 2;}}while(0)
#define N(x) do{auto e=(x);if(e!=ncclSuccess){std::fprintf(stderr,"%s:%d %s: %s\\n",__FILE__,__LINE__,#x,ncclGetErrorString(e));return 3;}}while(0)
#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"%s:%d CHECK %s\\n",__FILE__,__LINE__,#x);return 4;}}while(0)
template<class T> static cudaError_t alloc(T** p,size_t n){return cudaMalloc(reinterpret_cast<void**>(p),n*sizeof(T));}
struct Graph {
  std::string name;unsigned n,root;
  std::vector<std::vector<unsigned>> e;
  std::vector<uint64_t> words;
  std::vector<uint32_t> permutations,counts;
  std::vector<int64_t> symbols;
  MgbfsPack4Dictionary dictionary{};
};
static Graph make_permutation_graph(const char* name,const std::vector<int64_t>& initial,
                                    const std::vector<std::vector<uint32_t>>& generators) {
  Graph g{};g.name=name;
  auto sorted=initial;std::sort(sorted.begin(),sorted.end());
  for(int64_t x:sorted) {
    if(g.symbols.empty()||g.symbols.back()!=x){g.symbols.push_back(x);g.counts.push_back(1);}
    else g.counts.back()++;
  }
  g.dictionary.elements=unsigned(initial.size());g.dictionary.distinct=unsigned(g.symbols.size());
  for(unsigned i=0;i<g.symbols.size();i++)g.dictionary.symbols[i]=g.symbols[i];
  std::map<uint64_t,unsigned> index;
  do {
    uint64_t word=0;if(!mgbfs_pack4_encode(g.dictionary,sorted.data(),&word))std::abort();
    index[word]=unsigned(g.words.size());g.words.push_back(word);
  } while(std::next_permutation(sorted.begin(),sorted.end()));
  g.n=unsigned(g.words.size());
  uint64_t root=0;if(!mgbfs_pack4_encode(g.dictionary,initial.data(),&root))std::abort();
  g.root=index.at(root);g.e.resize(g.n);
  for(auto& gen:generators)for(auto i:gen)g.permutations.push_back(i);
  for(unsigned i=0;i<g.n;i++)for(const auto& gen:generators) {
    uint64_t child=0;
    for(unsigned j=0;j<gen.size();j++)child|=((g.words[i]>>(4*gen[j]))&15ull)<<(4*j);
    g.e[i].push_back(index.at(child));
  }
  return g;
}
static std::vector<std::vector<uint32_t>> adjacent(unsigned n) {
  std::vector<std::vector<uint32_t>> gens;
  for(unsigned i=0;i+1<n;i++) {
    std::vector<uint32_t> p(n);for(unsigned j=0;j<n;j++)p[j]=j;
    std::swap(p[i],p[i+1]);gens.push_back(p);
  }
  return gens;
}
static Graph graph(unsigned which) {
  if(which==0)return make_permutation_graph("s5_arbitrary_root",{4,0,3,1,2},adjacent(5));
  if(which==1)return make_permutation_graph("repeated_001122",{1,0,2,0,1,2},adjacent(6));
  if(which==2)return make_permutation_graph("signed_directed_rotation",{-7,25,0,13},{{1,2,3,0}});
  if(which==3)return make_permutation_graph("s6_adjacent",{0,1,2,3,4,5},adjacent(6));
  if(which==4)return make_permutation_graph("duplicate_when_full",{0,1,2},{{1,0,2},{1,0,2},{0,1,2}});
  return make_permutation_graph("s8_adjacent",{0,1,2,3,4,5,6,7},adjacent(8));
}
// Independent reference uses a standard FIFO BFS and does not use the GPU protocol.
static std::vector<int> oracle(const Graph& g) {
  std::vector<int> distance(g.n,-1);std::queue<unsigned> q;
  distance[g.root]=0;q.push(g.root);
  while(!q.empty()) {
    unsigned v=q.front();q.pop();
    for(unsigned w:g.e[v])if(distance[w]<0){distance[w]=distance[v]+1;q.push(w);}
  }
  return distance;
}
static std::string encode(const ncclUniqueId& id) {
  const char* h="0123456789abcdef";std::string s;
  for(unsigned char c:id.internal){s.push_back(h[c>>4]);s.push_back(h[c&15]);}return s;
}
static bool decode(const char* s,ncclUniqueId& id) {
  if(std::strlen(s)!=2*sizeof(id.internal))return false;
  auto d=[](char c)->int {if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;return -1;};
  for(size_t i=0;i<sizeof(id.internal);i++){int a=d(s[2*i]),b=d(s[2*i+1]);if(a<0||b<0)return false;id.internal[i]=char(a*16+b);}
  return true;
}
static uint64_t host_mix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
static int rank_main(unsigned rank,const ncclUniqueId& id) {
 int visible=0;auto status=cudaGetDeviceCount(&visible);
 if(status==cudaErrorNoDevice||(status==cudaSuccess&&visible<int(WORLD)))return 77;C(status);
 bool swapped=std::getenv("MGBFS_DEVICE_TEST_SWAP")!=nullptr;
 int device=swapped?int(WORLD-1-rank):int(rank);C(cudaSetDevice(device));
 CHECK(!mgbfs_device_pack4_bfs_smoke_v1());
 for(unsigned r=0;r<WORLD;++r)if(r!=rank){
  int peer=swapped?int(WORLD-1-r):int(r),access=0,atom=0;
  C(cudaDeviceCanAccessPeer(&access,device,peer));
  C(cudaDeviceGetP2PAttribute(&atom,cudaDevP2PAttrNativeAtomicSupported,device,peer));
  if(!access||!atom)return 77;
  auto enabled=cudaDeviceEnablePeerAccess(peer,0);
  if(enabled!=cudaSuccess&&enabled!=cudaErrorPeerAccessAlreadyEnabled)return 2;
  if(enabled==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();
 }
#ifdef MGBFS_LIBRARY_TEST_TRANSPORT
 void* production_comm=nullptr;char transport_error[512]{};
 CHECK(!mgbfs_nccl_create(rank,WORLD,device,&id,&production_comm,transport_error,sizeof(transport_error)));
#else
 ncclComm_t comm{};N(ncclCommInitRank(&comm,WORLD,id,rank));
 ncclCommProperties_t props=NCCL_COMM_PROPERTIES_INITIALIZER;N(ncclCommQueryProperties(comm,&props));
 if(!props.deviceApiSupport||props.nLsaTeams!=1)return 77;
#endif
 cudaStream_t stream{};C(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 const char* lanes_text=std::getenv("MGBFS_DEVICE_TEST_LANES");
 const unsigned lanes=lanes_text?unsigned(std::strtoul(lanes_text,nullptr,10)):mgbfs_resident::lanes;
 CHECK(lanes>=1&&lanes<=max_lanes);
 const char* records_text=std::getenv("MGBFS_DEVICE_TEST_PACKET_RECORDS");
 const unsigned packet_records=records_text?unsigned(std::strtoul(records_text,nullptr,10)):1;
 CHECK(packet_records>=1&&packet_records<=max_packet_records);
 MgbfsDeviceMailboxLayoutV1 maximum{};
 CHECK(!mgbfs_device_mailbox_layout_v1(WORLD,lanes,3,sizeof(Payload)*packet_records,UINT64_MAX,sizeof(maximum),&maximum));
 size_t window_bytes=sizeof(Control)+maximum.total_bytes;
#ifdef MGBFS_LIBRARY_TEST_BATCH
 constexpr uint64_t batch_send=33,batch_receive=7;
 window_bytes=std::max(window_bytes,sizeof(mgbfs_device_batch::Window)+batch_send*sizeof(mgbfs_device_batch::Payload));
 uint32_t batch_blocks=0,batch_threads=0;
 CHECK(!mgbfs_device_batch_bitmap_geometry_v1(&batch_blocks,&batch_threads));
 CHECK(batch_blocks>1 && batch_threads==128);
#endif
#ifdef MGBFS_LIBRARY_TEST_TRANSPORT
 CHECK(!mgbfs_nccl_resident_prepare_v1(production_comm,window_bytes,transport_error,sizeof(transport_error)));
 CHECK(!mgbfs_nccl_resident_activate_v1(production_comm,transport_error,sizeof(transport_error)));
#else
 void* base=nullptr;N(ncclMemAlloc(&base,window_bytes));
 ncclWindow_t window{};N(ncclCommWindowRegister(comm,base,window_bytes,&window,NCCL_WIN_COLL_SYMMETRIC));
 ncclDevComm dev{};ncclDevCommRequirements req=NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
 req.lsaBarrierCount=1;N(ncclDevCommCreate(comm,&req,&dev));
 CHECK(dev.lsaRank==int(rank)&&dev.lsaSize==int(WORLD));
#endif
 bool failed=false;unsigned case_no=0;
 // The old graph builder and FIFO oracle are independent of the library.
 for(unsigned graph_index=0;graph_index<5&&!failed;++graph_index){
  Graph g=graph(graph_index);auto dist=oracle(g);unsigned max_depth=0;
  for(int d:dist)if(d>=0)max_depth=std::max(max_depth,unsigned(d));
  std::vector<uint32_t> map(WORLD);for(unsigned r=0;r<WORLD;++r)map[r]=r;
  std::swap(map[0],map[g.root%WORLD]);
  unsigned owner=0;while(owner<WORLD&&map[owner]!=rank)++owner;CHECK(owner<WORLD);
  unsigned needed=1;
  for(unsigned d=0;d<=max_depth;++d)for(unsigned r=0;r<WORLD;++r){
   unsigned n=0;for(unsigned i=0;i<g.n;++i)n+=dist[i]==int(d)&&map[i%WORLD]==r;
   needed=std::max(needed,n);
  }
#ifdef MGBFS_LIBRARY_TEST_BATCH
  for(unsigned queue:{1u})for(unsigned variant=0;variant<4;++variant){
#else
  for(unsigned queue:{1u,3u})for(unsigned variant=0;variant<4;++variant){
#endif
   if(failed)break;
   if(variant==2&&needed==1)continue;
   // Force metadata exhaustion independently of per-rank storage; every rank
   // must report the same reason and keep only completed layers.
   if(variant==3&&(graph_index!=0||queue!=1))continue;
   unsigned cap=(variant==0||variant==3)?g.n:variant==1?needed:needed-1;
   int first_bad=-1;
   for(unsigned d=0;d<=max_depth;++d)for(unsigned r=0;r<WORLD;++r){
    unsigned n=0;for(unsigned i=0;i<g.n;++i)n+=dist[i]==int(d)&&map[i%WORLD]==r;
    if(n>cap&&(first_bad<0||int(d)<first_bad))first_bad=int(d);
   }
   if(variant==3)first_bad=3;
   unsigned expected_depth=first_bad<0?max_depth:unsigned(first_bad-1);
   unsigned expected_status=first_bad<0?mgbfs_device_epoch::COMPLETE:mgbfs_device_epoch::INCOMPLETE;
   MgbfsDeviceMailboxLayoutV1 layout{};
   CHECK(!mgbfs_device_mailbox_layout_v1(WORLD,lanes,queue,sizeof(Payload)*packet_records,maximum.total_bytes,sizeof(layout),&layout));
   uint32_t *perms=nullptr,*counts=nullptr,*gpu_map=nullptr,*leases=nullptr,*locks=nullptr,*terminal_word=nullptr;
   int64_t* symbols=nullptr;uint64_t *visited=nullptr,*rank_count=nullptr,*layers=nullptr,*rank_counts=nullptr,*capture_count=nullptr;
   Capture* capture=nullptr;Report* report=nullptr;mgbfs_device_pack4::Extent* extents=nullptr;uint64_t *rows[2]{},*used[2]{};
   C(alloc(&perms,g.permutations.size()));C(alloc(&counts,g.counts.size()));C(alloc(&symbols,g.symbols.size()));C(alloc(&gpu_map,WORLD));
   C(alloc(&leases,2));C(alloc(&locks,2));C(alloc(&terminal_word,1));C(alloc(&rank_count,1));
   size_t bit_words=std::max<size_t>(1,(g.n+63)/64);
   C(alloc(&visited,bit_words));C(alloc(&layers,g.n+1));C(alloc(&rank_counts,WORLD));
   C(alloc(&capture_count,1));C(alloc(&capture,g.n+1));C(alloc(&report,1));C(alloc(&extents,2));
   mgbfs_device_pack4::Extent host_extents[2]{};
   for(unsigned i=0;i<2;++i){C(alloc(&rows[i],size_t(cap)*3));C(alloc(&used[i],3));C(cudaMemsetAsync(used[i],0,24,stream));host_extents[i]={rows[i],used[i],cap,3,0};}
   uint32_t host_leases[2]={UINT32_MAX,UINT32_MAX};
   C(cudaMemsetAsync(visited,0,bit_words*8,stream));
   if(rank==0){
    uint64_t local=g.root/WORLD,mask=uint64_t(1)<<(local%64),one=1;
    host_leases[0]=unsigned((uint64_t(uint32_t(host_mix(g.root)))*2)>>32);
    C(cudaMemcpyAsync(rows[0],&g.words[g.root],8,cudaMemcpyHostToDevice,stream));
    C(cudaMemcpyAsync(used[0],&one,8,cudaMemcpyHostToDevice,stream));
    C(cudaMemcpyAsync(visited+local/64,&mask,8,cudaMemcpyHostToDevice,stream));
    C(cudaStreamSynchronize(stream));
   }
   C(cudaMemcpyAsync(extents,host_extents,sizeof(host_extents),cudaMemcpyHostToDevice,stream));
   C(cudaMemcpyAsync(leases,host_leases,sizeof(host_leases),cudaMemcpyHostToDevice,stream));
   C(cudaMemcpyAsync(perms,g.permutations.data(),g.permutations.size()*4,cudaMemcpyHostToDevice,stream));
   C(cudaMemcpyAsync(counts,g.counts.data(),g.counts.size()*4,cudaMemcpyHostToDevice,stream));
   C(cudaMemcpyAsync(symbols,g.symbols.data(),g.symbols.size()*8,cudaMemcpyHostToDevice,stream));
   C(cudaMemcpyAsync(gpu_map,map.data(),map.size()*4,cudaMemcpyHostToDevice,stream));
   C(cudaMemsetAsync(terminal_word,0,4,stream));C(cudaMemsetAsync(report,0,sizeof(Report),stream));
   Args a{};a.abi_version=1;a.struct_bytes=sizeof(a);a.world=WORLD;a.rank=rank;a.owner=owner;a.shards=2;
   a.generators=g.permutations.size()/g.dictionary.elements;a.banks=3;a.owner_len=g.n/WORLD+(owner<g.n%WORLD);
   a.job_generation=1000+(case_no++);a.rank_limit=cap;
   a.domain={symbols,counts,g.dictionary.distinct,g.dictionary.elements,g.n};a.dictionary=g.dictionary;
   a.mailbox=layout;a.permutations=perms;a.physical_of_logical=gpu_map;a.extents=extents;a.extent_owner=leases;a.extent_count=2;
   a.shard_locks=locks;a.visited=visited;a.rank_count=rank_count;a.layer_sizes=layers;a.layer_capacity=variant==3?3:g.n+1;
   a.rank_counts=rank_counts;a.capture=capture;a.capture_capacity=g.n+1;a.capture_count=capture_count;a.terminal=terminal_word;
#ifdef MGBFS_LIBRARY_TEST_TRANSPORT
   CHECK(!mgbfs_nccl_resident_init_v1(production_comm,&a,stream));
#else
   CHECK(!mgbfs_device_pack4_bfs_init_v1(base,window_bytes,&a,stream));
#endif
#ifdef MGBFS_LIBRARY_TEST_BATCH
   using namespace mgbfs_device_batch;
   Input* batch_raw=nullptr;
   mgbfs_device_batch::Payload* batch_incoming=nullptr;
   uint64_t *batch_positions=nullptr,*batch_hist=nullptr,*batch_offsets=nullptr,
      *batch_shards=nullptr,*batch_starts=nullptr,*batch_prefix=nullptr;
   BitmapSpan* batch_spans=nullptr;uint32_t* batch_error=nullptr;
   C(alloc(&batch_raw,batch_send));C(alloc(&batch_incoming,batch_receive));
   C(alloc(&batch_positions,std::max(batch_send,batch_receive)));C(alloc(&batch_hist,batch_blocks*WORLD));
   C(alloc(&batch_offsets,WORLD+1));C(alloc(&batch_shards,a.shards));C(alloc(&batch_starts,a.shards+1));
   C(alloc(&batch_spans,a.extent_count));C(alloc(&batch_prefix,a.extent_count+1));C(alloc(&batch_error,1));
   mgbfs_resident_batch::Args batch{a,batch_send,batch_receive,batch_raw,batch_positions,
     batch_hist,batch_offsets,batch_incoming,batch_shards,batch_starts,batch_spans,batch_prefix,batch_error};
   CHECK(!mgbfs_device_batch_bitmap_init_v1(base,window_bytes,batch_send,stream));
#endif
   uint32_t* barrier=nullptr;C(alloc(&barrier,2));C(cudaMemsetAsync(barrier,0,8,stream));
#ifdef MGBFS_LIBRARY_TEST_TRANSPORT
   CHECK(!mgbfs_nccl_all_reduce_max_pair_u32(production_comm,barrier,barrier,stream));
#else
   N(ncclAllReduce(barrier,barrier,1,ncclUint32,ncclMax,comm,stream));
#endif
   C(cudaStreamSynchronize(stream));
#ifdef MGBFS_LIBRARY_TEST_TRANSPORT
   if(graph_index==0&&queue==1&&variant==0) {
     uint32_t* smoke=nullptr;C(alloc(&smoke,1));C(cudaMemsetAsync(smoke,255,4,stream));
     CHECK(!mgbfs_nccl_resident_smoke_v1(production_comm,&a,smoke,stream));
     uint32_t smoke_status=99;C(cudaMemcpyAsync(&smoke_status,smoke,4,cudaMemcpyDeviceToHost,stream));
     // The allreduce is a fixture-only all-rank acknowledgement before the
     // next initialization resets the shared window.
     CHECK(!mgbfs_nccl_all_reduce_max_pair_u32(production_comm,barrier,barrier,stream));
     C(cudaStreamSynchronize(stream));CHECK(smoke_status==0);C(cudaFree(smoke));
     std::printf("PASS resident-lsa-data-credit-smoke rank=%u world=%u\n",rank,WORLD);
     CHECK(!mgbfs_nccl_resident_init_v1(production_comm,&a,stream));
     CHECK(!mgbfs_nccl_all_reduce_max_pair_u32(production_comm,barrier,barrier,stream));
     C(cudaStreamSynchronize(stream));
   }
   CHECK(!mgbfs_nccl_resident_launch_v1(production_comm,&a,report,stream));
#else
#ifdef MGBFS_LIBRARY_TEST_BATCH
   CHECK(!mgbfs_device_batch_bitmap_launch_v1(&window,&dev,&batch,window_bytes,report,stream));
#else
   CHECK(!mgbfs_device_pack4_bfs_launch_v1(&window,&dev,&a,window_bytes,report,stream));
#endif
#endif
   C(cudaStreamSynchronize(stream));
   Report out{};C(cudaMemcpy(&out,report,sizeof(out),cudaMemcpyDeviceToHost));CHECK(out.capture_count<=g.n+1);
   std::vector<Capture> actual(out.capture_count);
   if(!actual.empty())C(cudaMemcpy(actual.data(),capture,actual.size()*sizeof(Capture),cudaMemcpyDeviceToHost));
   std::vector<std::pair<unsigned,uint64_t>> got,want;if(rank==0)got.emplace_back(0,g.words[g.root]);
   for(auto r:actual)if(r.depth<=out.depth)got.emplace_back(unsigned(r.depth),r.packed);
   for(unsigned i=0;i<g.n;++i)if(dist[i]>=0&&unsigned(dist[i])<=expected_depth&&map[i%WORLD]==rank)want.emplace_back(unsigned(dist[i]),g.words[i]);
   std::sort(got.begin(),got.end());std::sort(want.begin(),want.end());
   std::vector<uint64_t> sizes(expected_depth+1),ranks(WORLD);
   C(cudaMemcpy(sizes.data(),layers,sizes.size()*8,cudaMemcpyDeviceToHost));C(cudaMemcpy(ranks.data(),rank_counts,WORLD*8,cudaMemcpyDeviceToHost));
   const unsigned expected_reason=variant==3?reason_log_capacity:
       expected_status==mgbfs_device_epoch::INCOMPLETE?reason_capacity:reason_none;
   bool ok=out.status==expected_status&&out.reason==expected_reason&&out.depth==expected_depth&&got==want;
   if(out.reason!=expected_reason)std::fprintf(stderr,"BAD_REASON rank=%u actual=%u expected=%u\n",rank,out.reason,expected_reason);
   for(unsigned d=0;d<=expected_depth;++d){uint64_t n=0;for(int x:dist)n+=x==int(d);ok&=sizes[d]==n;}
   for(unsigned r=0;r<WORLD;++r){uint64_t n=0;for(unsigned i=0;i<g.n;++i)n+=dist[i]==int(expected_depth)&&map[i%WORLD]==r;ok&=ranks[r]==n;}
   if(graph_index==4&&rank==map[g.e[g.root][0]%WORLD]&&cap==1)ok&=out.duplicates_full>0;
   std::printf("%s library graph=%s rank=%u q=%u cap=%u depth=%llu status=%u exact=%zu duplicate_full=%llu\n",
     ok?"PASS":"FAIL",g.name.c_str(),rank,queue,cap,(unsigned long long)out.depth,out.status,got.size(),(unsigned long long)out.duplicates_full);
   if(!ok)std::fprintf(stderr,"EXPECTED rank=%u depth=%u status=%u got=%zu wanted=%zu\n",rank,expected_depth,expected_status,got.size(),want.size());
   int fail=ok?0:1;C(cudaMemcpyAsync(barrier,&fail,4,cudaMemcpyHostToDevice,stream));
#ifdef MGBFS_LIBRARY_TEST_TRANSPORT
   CHECK(!mgbfs_nccl_all_reduce_max_pair_u32(production_comm,barrier,barrier,stream));
#else
   N(ncclAllReduce(barrier,barrier,1,ncclUint32,ncclMax,comm,stream));
#endif
   C(cudaMemcpyAsync(&fail,barrier,4,cudaMemcpyDeviceToHost,stream));C(cudaStreamSynchronize(stream));failed=fail!=0;
#ifdef MGBFS_LIBRARY_TEST_BATCH
   C(cudaFree(batch_error));C(cudaFree(batch_prefix));C(cudaFree(batch_spans));C(cudaFree(batch_starts));
   C(cudaFree(batch_shards));C(cudaFree(batch_offsets));C(cudaFree(batch_hist));C(cudaFree(batch_positions));
   C(cudaFree(batch_incoming));C(cudaFree(batch_raw));
#endif
   C(cudaFree(barrier));C(cudaFree(perms));C(cudaFree(counts));C(cudaFree(symbols));C(cudaFree(gpu_map));C(cudaFree(leases));C(cudaFree(locks));C(cudaFree(terminal_word));
   C(cudaFree(rank_count));C(cudaFree(visited));C(cudaFree(layers));C(cudaFree(rank_counts));C(cudaFree(capture_count));C(cudaFree(capture));C(cudaFree(report));C(cudaFree(extents));
   for(unsigned i=0;i<2;++i){C(cudaFree(rows[i]));C(cudaFree(used[i]));}
#ifdef MGBFS_LIBRARY_TEST_TRANSPORT
   // The barrier after the result proves all test ranks have stopped reading
   // the old window before it is deregistered. Then park and reuse the same
   // NCCL communicator for a new resident window and subsequent jobs.
   if(!failed&&graph_index==0&&queue==1&&variant==0) {
     CHECK(!mgbfs_nccl_resident_retire_v1(production_comm));
     CHECK(!mgbfs_nccl_session_park(production_comm));
     CHECK(!mgbfs_nccl_resident_prepare_v1(production_comm,window_bytes,transport_error,sizeof(transport_error)));
     CHECK(!mgbfs_nccl_resident_activate_v1(production_comm,transport_error,sizeof(transport_error)));
     std::printf("PASS resident-window-retire-and-reopen rank=%u\n",rank);
   }
#endif
   if(failed)break;
  }
 }
#ifdef MGBFS_LIBRARY_TEST_TRANSPORT
 if(!failed) {
   CHECK(!mgbfs_nccl_resident_retire_v1(production_comm));
   CHECK(!mgbfs_nccl_session_park(production_comm));
   mgbfs_nccl_destroy(production_comm);
 }
#else
 N(ncclDevCommDestroy(comm,&dev));N(ncclCommWindowDeregister(comm,window));N(ncclMemFree(base));N(ncclCommDestroy(comm));
#endif
 C(cudaStreamDestroy(stream));
 return failed?4:0;
}
int main(int argc,char** argv){
 setvbuf(stdout,nullptr,_IONBF,0);
 if(argc==4&&!std::strcmp(argv[1],"--rank")){int rank=std::atoi(argv[2]);ncclUniqueId id{};if(rank<0||rank>=int(WORLD)||!decode(argv[3],id))return 6;return rank_main(rank,id);}
 if(argc!=1)return 6;ncclUniqueId id{};N(ncclGetUniqueId(&id));auto value=encode(id);std::array<pid_t,WORLD> pids{};
 for(unsigned i=0;i<WORLD;++i){pids[i]=fork();if(!pids[i]){auto rank=std::to_string(i);execl(argv[0],argv[0],"--rank",rank.c_str(),value.c_str(),nullptr);_exit(127);}if(pids[i]<0)return 7;}
 bool ok=true;unsigned skipped=0;
 for(unsigned i=0;i<WORLD;++i){int s=0;if(waitpid(pids[i],&s,0)<0)ok=false;int code=WIFEXITED(s)?WEXITSTATUS(s):-1;
  std::printf("CHILD rank=%u exit=%d\n",i,code);if(code==77)++skipped;else if(code!=0)ok=false;}
 if(skipped==WORLD)return 77;return ok&&skipped==0?0:4;
}
