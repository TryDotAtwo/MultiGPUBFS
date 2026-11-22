// Physical owner-primitive test; does not claim end-to-end resident BFS.
// Independent CPU multiset oracle checks all materialized rows, including
// forced equal full hashes, concurrent shards, tiny extents, and cap replay.
#include "device_segmented_hash_accept.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <type_traits>
using namespace mgbfs_device_hash;
#define C(x) do { auto e_=(x); if(e_!=cudaSuccess){std::fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e_));std::exit(1);} }while(0)
#define CHECK(x) do { if(!(x)){std::fprintf(stderr,"%s:%d CHECK %s\n",__FILE__,__LINE__,#x);std::exit(1);} }while(0)
template<class T> T* managed(size_t n) {
 T* p=nullptr;C(cudaMallocManaged(&p,n*sizeof(T)));return p;
}
template<class T> struct Input {
 T v[4];uint64_t hash;uint32_t shard;
 __device__ int64_t value(uint32_t e) const { return int64_t(v[e]); }
};
template<class T> __global__ void wave(Owner* owners,Input<T>* input,unsigned n,Outcome* out,unsigned* gate) {
 const unsigned shard=blockIdx.x;
 if(shard>=2||threadIdx.x)return;
 atomicAdd(gate,1u);
 unsigned long long start=clock64();
 while(atomicAdd(gate,0u)<2u && clock64()-start<3000000000ULL){}
 if(atomicAdd(gate,0u)!=2u)return;
 out[shard*n+shard]=accept<T>(owners[shard],input[shard].hash,input[shard]);
 atomicAdd(gate,1u);
 start=clock64();
 while(atomicAdd(gate,0u)<4u && clock64()-start<3000000000ULL){}
 if(atomicAdd(gate,0u)!=4u)return;
 for(unsigned i=2;i<n;++i)
  if(input[i].shard==shard)out[shard*n+i]=accept<T>(owners[shard],input[i].hash,input[i]);
}
template<class T> __global__ void once(Owner o,Input<T> input,Outcome* out) {
 if(!blockIdx.x&&!threadIdx.x)*out=accept<T>(o,input.hash,input);
}
template<class T> struct Fixture {
 static constexpr unsigned n=8, ns=6;
 bool wide;
 Extent* extents=managed<Extent>(ns);
 uint32_t* leases=managed<uint32_t>(ns);
 uint64_t* rank_count=managed<uint64_t>(1);
 uint64_t* heads=managed<uint64_t>(2);
 uint64_t* next=managed<uint64_t>(ns);
 uint64_t* cursors=managed<uint64_t>(2);
 Owner* owners=managed<Owner>(2);
 Input<T>* input=managed<Input<T>>(n);
 Outcome* out=managed<Outcome>(2*n);
 unsigned* gate=managed<unsigned>(1);
 Fixture(bool value,bool indexed=false):wide(value) {
  std::fill(heads,heads+2,UINT64_MAX);
  std::fill(next,next+ns,UINT64_MAX);
  std::fill(cursors,cursors+2,0);
  for(unsigned i=0;i<ns;++i) {
   auto* slots=managed<uint64_t>(8);
   auto* rows=managed<T>(12);
   auto* positions=managed<uint64_t>(3);
   auto* counts=managed<uint64_t>(3);
   std::fill(slots,slots+8,UINT64_MAX);
   std::fill(rows,rows+12,T(0));
   std::fill(positions,positions+3,0);
   std::fill(counts,counts+3,0);
   MgbfsSegment segment{slots,rows,positions,8,1,3,(i%2)?4u:8u};
   extents[i]={segment,counts};
   leases[i]=UINT32_MAX;
  }
  *rank_count=0;
  for(unsigned s=0;s<2;++s) {
   Owner o{};o.extents=extents;o.extent_count=ns;o.extent_owner=leases;
   o.rank_count=rank_count;o.rank_limit=2;o.elements=4;o.shards=2;o.shard=s;o.bank=1;
   if(indexed){o.shard_head=heads;o.extent_next=next;o.free_cursor=cursors;}
   owners[s]=o;
  }
  auto set=[&](unsigned i,unsigned s,int a,int b) {
   input[i].v[0]=T(wide?-10000000007LL:a);
   input[i].v[1]=T(wide?40000000001LL:b);
   input[i].v[2]=T(a);input[i].v[3]=T(b);
   // Exact full-hash collision within a shard; the second shard has another hash.
   input[i].hash=s?0x100000000ULL:0;input[i].shard=s;
  };
  set(0,0,1,2);set(1,1,10,11);
  input[2]=input[0];input[3]=input[1];
  set(4,0,3,4);set(5,1,12,13);set(6,0,5,6);set(7,1,14,15);
 }
 void launch() {
  for(unsigned i=0;i<2*n;++i)out[i]={Status::invalid,UINT64_MAX,0};
  *gate=0;wave<T><<<2,1>>>(owners,input,n,out,gate);C(cudaGetLastError());C(cudaDeviceSynchronize());CHECK(*gate==4);
 }
 std::map<std::array<T,4>,unsigned> rows() {
  std::map<std::array<T,4>,unsigned> found;
  uint64_t count=0;unsigned active[2]={};
  for(unsigned i=0;i<ns;++i) {
   auto& e=extents[i];
   if(leases[i]==UINT32_MAX){CHECK(e.counts[0]==0&&e.counts[1]==0&&e.counts[2]==0);continue;}
   CHECK(leases[i]<2);CHECK(e.counts[0]==0&&e.counts[1]<=1&&e.counts[2]==0);
   for(uint64_t r=0;r<e.counts[1];++r) {
    std::array<T,4> row{};auto* states=static_cast<T*>(e.segment.states);
    for(unsigned k=0;k<4;++k)row[k]=states[k*3+1+r];
    ++found[row];++count;++active[leases[i]];
   }
  }
  CHECK(count==*rank_count);CHECK(active[0]>0&&active[1]>0);
  for(const auto& item:found)CHECK(item.second==1);
  return found;
 }
 void check() {
  launch();CHECK(*rank_count==2);auto first=rows();CHECK(first.size()==2);
  CHECK(out[2].status==Status::duplicate&&out[n+3].status==Status::duplicate);
  unsigned cap=0;for(unsigned i=4;i<n;++i)cap+=out[input[i].shard*n+i].status==Status::cap_full;
  CHECK(cap==4);
  unsigned free=0;for(unsigned i=0;i<ns;++i)free+=leases[i]==UINT32_MAX;
  CHECK(free==4); // temporary empty leases were returned after rejected cap.
  owners[0].rank_limit=owners[1].rank_limit=6;launch();CHECK(*rank_count==6);
  std::map<std::array<T,4>,unsigned> expected;
  for(unsigned i: {0u,1u,4u,5u,6u,7u}) {
   std::array<T,4> row{};std::copy(input[i].v,input[i].v+4,row.begin());++expected[row];
  }
  CHECK(rows()==expected);
  for(unsigned i=0;i<n;++i) {
   auto status=out[input[i].shard*n+i].status;
   CHECK(status==Status::accepted||status==Status::duplicate);
  }
  // Use the real library retirement API on a drained bank, then verify the
  // tombstone still allows exact probing and row reuse by the new owner primitive.
  unsigned chosen=0;while(chosen<ns&&leases[chosen]!=0)++chosen;CHECK(chosen<ns);
  auto* error=managed<uint32_t>(1);*error=0;
  CHECK(mgbfs_segmented_retire(extents[chosen].segment,1,1,error,nullptr)==int(cudaSuccess));
  C(cudaDeviceSynchronize());CHECK(*error==0);
  extents[chosen].counts[1]=0;*rank_count=0;
  auto* result=managed<Outcome>(1);
  Input<T> fresh{};fresh.v[0]=T(wide?-10000000007LL:31);fresh.v[1]=T(wide?40000000001LL:32);fresh.v[2]=T(31);fresh.v[3]=T(32);fresh.hash=0;fresh.shard=0;
  once<T><<<1,1>>>(owners[0],fresh,result);C(cudaGetLastError());C(cudaDeviceSynchronize());
  CHECK(result->status==Status::accepted&&result->extent==chosen&&result->row==0&&*rank_count==1);
  once<T><<<1,1>>>(owners[0],fresh,result);C(cudaGetLastError());C(cudaDeviceSynchronize());
  CHECK(result->status==Status::duplicate&&*rank_count==1);
  std::printf("PASS %s exact collisions, 2 active shards, cap/replay, six segments, retire/tombstone\n",wide?"i64":"u8");
 }
};

template<class T> void chain_and_full_table() {
 auto* slots=managed<uint64_t>(4);std::fill(slots,slots+4,UINT64_MAX);
 auto* rows=managed<T>(4*3*3);auto* positions=managed<uint64_t>(3*3);
 auto* counts=managed<uint64_t>(3);std::fill(counts,counts+3,0);
 auto* ext=managed<Extent>(1);
 ext[0]={MgbfsSegment{slots,rows,positions,4,3,3,8},counts};
 Owner o{};o.extents=ext;o.extent_count=1;o.elements=4;o.shards=1;o.bank=1;
 auto* out=managed<Outcome>(1);
 auto* error=managed<uint32_t>(1);*error=0;
 Input<T> in{};in.hash=0;
 auto insert=[&](int value) {
  for(int i=0;i<4;++i)in.v[i]=T(value+i);
  once<T><<<1,1>>>(o,in,out);C(cudaGetLastError());C(cudaDeviceSynchronize());return out->status;
 };
 CHECK(insert(1)==Status::accepted);
 CHECK(insert(11)==Status::accepted);
 CHECK(insert(21)==Status::accepted);
 CHECK(insert(11)==Status::duplicate); // full bank still resolves collision
 CHECK(insert(31)==Status::extent_full);
 // Existing host backend can rebuild this exact index, forced hash_bits=0.
 if constexpr(std::is_same_v<T,uint8_t>)
  CHECK(mgbfs_segmented_rehash_u8(4,ext[0].segment,0,3,0,123,0,error,nullptr)==int(cudaSuccess));
 else
  CHECK(mgbfs_segmented_rehash_i64(4,ext[0].segment,0,3,0,123,0,error,nullptr)==int(cudaSuccess));
 C(cudaDeviceSynchronize());CHECK(*error==0);CHECK(insert(21)==Status::duplicate);
 // Deliberately retire only the first row to exercise a tombstone in the
 // middle of a probe chain. Production retires the entire drained bank.
 CHECK(mgbfs_segmented_retire(ext[0].segment,1,1,error,nullptr)==int(cudaSuccess));
 C(cudaDeviceSynchronize());CHECK(*error==0);
 o.bank=2;CHECK(insert(11)==Status::duplicate);
 CHECK(insert(31)==Status::accepted);CHECK(insert(11)==Status::duplicate);
 std::puts("PASS same-segment full-hash collision, rehash, tombstone and full-bank duplicate");

 // Row space remains, but a deliberately undersized table is full. This is
 // a retriable index-space condition and must not consume the candidate/cap.
 auto* one_slot=managed<uint64_t>(1);*one_slot=UINT64_MAX;
 auto* one_rows=managed<T>(8);auto* one_count=managed<uint64_t>(1);*one_count=0;
 auto* rank_count=managed<uint64_t>(1);*rank_count=0;
 auto* tiny=managed<Extent>(1);
 tiny[0]={MgbfsSegment{one_slot,one_rows,nullptr,1,2,1,0},one_count};
 o.extents=tiny;o.bank=0;o.rank_count=rank_count;o.rank_limit=2;
 CHECK(insert(1)==Status::accepted);CHECK(insert(11)==Status::table_full);
 CHECK(insert(1)==Status::duplicate);CHECK(*one_count==1&&*rank_count==1);
 std::puts("PASS full index leaves row and rank capacity unchanged");
}
void virtual_last_row() {
 CUdevice dev;int supported=0;CUresult r=cuInit(0);
 if(r!=CUDA_SUCCESS||cuDeviceGet(&dev,0)!=CUDA_SUCCESS||
    cuDeviceGetAttribute(&supported,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,dev)!=CUDA_SUCCESS||!supported) {
  std::puts("SKIP virtual hash boundary: VMM unavailable");return;
 }
 CUmemAllocationProp prop{};prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
 prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=dev;
 size_t page=0;CHECK(cuMemGetAllocationGranularity(&page,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM)==CUDA_SUCCESS);
 const uint64_t cap=uint64_t(1)<<32;
 if(cap%page){std::puts("SKIP virtual hash boundary: page alignment");return;}
 CUdeviceptr addr=0;CHECK(cuMemAddressReserve(&addr,cap,page,0,0)==CUDA_SUCCESS);
 CUmemGenericAllocationHandle handle;CHECK(cuMemCreate(&handle,page,&prop,0)==CUDA_SUCCESS);
 CHECK(cuMemMap(addr+cap-page,page,0,handle,0)==CUDA_SUCCESS);
 CUmemAccessDesc access{};access.location=prop.location;access.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
 CHECK(cuMemSetAccess(addr+cap-page,page,&access,1)==CUDA_SUCCESS);
 auto* slots=managed<uint64_t>(8);std::fill(slots,slots+8,UINT64_MAX);
 auto* counts=managed<uint64_t>(1);*counts=cap-1;
 auto* ext=managed<Extent>(1);
 ext[0]={MgbfsSegment{slots,reinterpret_cast<void*>(addr),nullptr,8,cap,1,0},counts};
 Owner o{};o.extents=ext;o.extent_count=1;o.elements=1;o.shards=1;
 auto* out=managed<Outcome>(1);Input<uint8_t> input{};input.v[0]=42;
 once<uint8_t><<<1,1>>>(o,input,out);C(cudaGetLastError());C(cudaDeviceSynchronize());
 CHECK(out->status==Status::accepted&&out->row==UINT32_MAX&&*counts==cap);
 uint8_t got=0;C(cudaMemcpy(&got,reinterpret_cast<void*>(addr+cap-1),1,cudaMemcpyDeviceToHost));CHECK(got==42);
 once<uint8_t><<<1,1>>>(o,input,out);C(cudaGetLastError());C(cudaDeviceSynchronize());CHECK(out->status==Status::duplicate);
 input.v[0]=43;once<uint8_t><<<1,1>>>(o,input,out);C(cudaGetLastError());C(cudaDeviceSynchronize());
 CHECK(out->status==Status::extent_full&&*counts==cap);
 CHECK(cuMemUnmap(addr+cap-page,page)==CUDA_SUCCESS);
 CHECK(cuMemRelease(handle)==CUDA_SUCCESS);CHECK(cuMemAddressFree(addr,cap)==CUDA_SUCCESS);
 std::printf("PASS virtual hash row 2^32-1, full and duplicate (physical page %zu bytes only)\n",page);
}
int main() {
 int n=0;auto e=cudaGetDeviceCount(&n);if(e==cudaErrorNoDevice||(e==cudaSuccess&&!n))return 77;
 C(e);C(cudaSetDevice(0));C(cudaFree(nullptr));
 Fixture<uint8_t> u8(false);u8.check();
 Fixture<int64_t> wide(true);wide.check();
 Fixture<uint8_t> indexed_u8(false,true);indexed_u8.check();
 Fixture<int64_t> indexed_wide(true,true);indexed_wide.check();
 std::puts("PASS indexed owner lists with exact collisions and cap replay");
 chain_and_full_table<uint8_t>();chain_and_full_table<int64_t>();
 virtual_last_row();
 if(n>=2) { C(cudaSetDevice(1));Fixture<uint8_t> second(false);second.check();std::puts("PASS same primitive on second physical GPU; no cross-GPU scheduler tested"); }
 std::puts("PASS resident segmented hash primitive (not yet a BFS scheduler)");
}
