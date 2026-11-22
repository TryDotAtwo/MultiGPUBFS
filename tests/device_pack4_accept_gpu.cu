// Physical CUDA unit for the resident-owner primitive, not a BFS scheduler.
// CPU builds an independent lexicographic multiset oracle; GPU stores packed
// rows with the existing domain rank and compares the complete result.
#include "device_pack4_accept.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace mgbfs_device_pack4;
#define C(x) do { auto status_=(x); if(status_!=cudaSuccess){ \
 std::fprintf(stderr,"%s at %d: %s\n",#x,__LINE__,cudaGetErrorString(status_));std::exit(1);} } while(0)
#define CHECK(x) do { if(!(x)){std::fprintf(stderr,"CHECK %s at %d\n",#x,__LINE__);std::exit(1);} } while(0)
template<class T> T* managed(size_t n) {
 T* p=nullptr; C(cudaMallocManaged(&p,n*sizeof(T))); return p;
}
uint64_t mix_host(uint64_t x) {
 x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);
}
struct Candidate {uint64_t id,packed;};
__global__ void run(Owner* owners,Candidate* candidates,unsigned n,Outcome* outcomes,unsigned* gate) {
 unsigned shard=blockIdx.x;
 if(shard>=2 || threadIdx.x)return;
 atomicAdd(gate,1u);
 unsigned long long start=clock64();
 while(atomicAdd(gate,0u)<2u && clock64()-start<3000000000ULL) {}
 if(atomicAdd(gate,0u)!=2u)return;
 for(unsigned i=0;i<n;++i)
  if((uint64_t(uint32_t(mix64(candidates[i].id)))*2)>>32 == shard)
   outcomes[uint64_t(shard)*n+i]=accept(owners[shard],candidates[i].id,candidates[i].packed);
}
__global__ void one(Owner owner,Candidate candidate,Outcome* out) {
 if(!blockIdx.x&&!threadIdx.x)*out=accept(owner,candidate.id,candidate.packed);
}
struct Fixture {
 uint32_t* counts;
 int64_t* symbols;
 uint64_t* bits;
 uint64_t* rank_count;
 Owner* owners;
 Candidate* candidates;
 Outcome* outcomes;
 unsigned* gate;
 std::vector<uint64_t> words;
 unsigned n;
 Fixture() {
  std::vector<int64_t> row{-7,2,2,91}, sorted_symbols{-7,2,91};
  symbols=managed<int64_t>(3); counts=managed<uint32_t>(3);
  for(unsigned i=0;i<3;++i) symbols[i]=sorted_symbols[i];
  counts[0]=1;counts[1]=2;counts[2]=1;
  do {
   uint64_t packed=0;
   for(unsigned i=0;i<row.size();++i) {
    auto it=std::find(sorted_symbols.begin(),sorted_symbols.end(),row[i]);
    CHECK(it!=sorted_symbols.end());packed|=uint64_t(it-sorted_symbols.begin())<<(4*i);
   }
   words.push_back(packed);
  } while(std::next_permutation(row.begin(),row.end()));
  CHECK(words.size()==12);
  n=unsigned(words.size()*2);candidates=managed<Candidate>(n);
  for(unsigned i=0;i<n;++i)candidates[i]={i%words.size(),words[i%words.size()]};
  outcomes=managed<Outcome>(2*n);gate=managed<unsigned>(1);bits=managed<uint64_t>(1);rank_count=managed<uint64_t>(1);
  owners=managed<Owner>(2);
  for(unsigned shard=0;shard<2;++shard) {
   auto* extents=managed<Extent>(3);
   const uint64_t caps[3]={1,2,16};
   for(unsigned i=0;i<3;++i) {
    auto* rows=managed<uint64_t>(caps[i]*2);
    auto* used=managed<uint64_t>(2);
    extents[i]={rows,used,caps[i],2,0};
   }
   Owner o{};
   o.domain={symbols,counts,3,4,uint64_t(words.size())};
   o.dictionary.elements=4;o.dictionary.distinct=3;
   for(unsigned i=0;i<3;++i)o.dictionary.symbols[i]=symbols[i];
   o.extents=extents;o.extent_count=3;o.bits=bits;o.owner_len=words.size();
   o.world=1;o.owner=0;o.shards=2;o.shard=shard;o.bank=1;
   owners[shard]=o;
  }
  reset();
 }
 void reset() {
  *bits=0;*rank_count=0;
  for(unsigned shard=0;shard<2;++shard) {
   owners[shard].rank_count=nullptr;owners[shard].rank_limit=0;
   owners[shard].extent_count=3;
   for(unsigned i=0;i<3;++i) {
    const Extent& e=owners[shard].extents[i];
    e.counts[0]=0;e.counts[1]=0;
    std::fill(e.rows,e.rows+e.capacity*e.banks,0xccccccccccccccccULL);
   }
  }
 }
 void launch() {
  for(unsigned i=0;i<2*n;++i)outcomes[i]={Status::invalid,UINT64_MAX,0};
  *gate=0;run<<<2,32>>>(owners,candidates,n,outcomes,gate);C(cudaGetLastError());C(cudaDeviceSynchronize());CHECK(*gate==2);
  for(unsigned i=0;i<n;++i) {
   unsigned shard=unsigned((uint64_t(uint32_t(mix_host(candidates[i].id)))*2)>>32);
   Status s=outcomes[shard*n+i].status;
   CHECK(s==Status::accepted||s==Status::duplicate||s==Status::extent_full||s==Status::cap_full);
  }
 }
 uint64_t validate(bool limited) {
  std::vector<unsigned> seen(words.size(),0);
  uint64_t total=0;
  for(unsigned shard=0;shard<2;++shard) for(unsigned i=0;i<3;++i) {
   const Extent& e=owners[shard].extents[i];
   CHECK(e.counts[0]==0 && e.counts[1]<=e.capacity);
   for(uint64_t j=0;j<e.capacity;++j)CHECK(e.rows[j]==0xccccccccccccccccULL);
   for(uint64_t j=0;j<e.counts[1];++j) {
    auto it=std::find(words.begin(),words.end(),e.rows[e.capacity+j]);
    CHECK(it!=words.end());unsigned id=unsigned(it-words.begin());
    CHECK(unsigned((uint64_t(uint32_t(mix_host(id)))*2)>>32)==shard);
    CHECK(++seen[id]==1);++total;
   }
  }
  for(unsigned id=0;id<words.size();++id)CHECK(seen[id]==unsigned((*bits>>id)&1));
  if(limited)CHECK(*rank_count==total);
  return total;
 }
};
int main() {
 int count=0;auto e=cudaGetDeviceCount(&count);
 if(e==cudaErrorNoDevice || (e==cudaSuccess&&!count))return 77;
 C(e);C(cudaSetDevice(0));C(cudaFree(nullptr));
 Fixture f;
 // Two full one-row segments: untouched candidates remain retryable.
 f.owners[0].extent_count=f.owners[1].extent_count=1;
 f.launch();CHECK(f.validate(false)==2);
 unsigned spills=0;
 for(unsigned i=0;i<f.n;++i){unsigned s=unsigned((uint64_t(uint32_t(mix_host(f.candidates[i].id)))*2)>>32);spills+=f.outcomes[s*f.n+i].status==Status::extent_full;}
 CHECK(spills>0);
 f.owners[0].extent_count=f.owners[1].extent_count=3;
 f.launch();CHECK(f.validate(false)==f.words.size());
 // Rank-wide cap shared by concurrently active shards; retry after raising it.
 f.reset();for(unsigned s=0;s<2;++s){f.owners[s].rank_count=f.rank_count;f.owners[s].rank_limit=4;}
 f.launch();CHECK(f.validate(true)==4);
 unsigned capfull=0,duplicates=0;
 for(unsigned i=0;i<f.n;++i){unsigned s=unsigned((uint64_t(uint32_t(mix_host(f.candidates[i].id)))*2)>>32);auto st=f.outcomes[s*f.n+i].status;capfull+=st==Status::cap_full;duplicates+=st==Status::duplicate;}
 CHECK(capfull>0 && duplicates>0);
 f.owners[0].rank_limit=f.owners[1].rank_limit=f.words.size();
 f.launch();CHECK(f.validate(true)==f.words.size());
 // Invalid claimed ID and invalid packed symbols never modify bitmap or rows.
 auto* result=managed<Outcome>(1);const uint64_t before=*f.bits, before_count=*f.rank_count;
 Candidate bad{1,f.words[0]};one<<<1,1>>>(f.owners[0],bad,result);C(cudaGetLastError());C(cudaDeviceSynchronize());
 CHECK(result->status==Status::invalid && *f.bits==before && *f.rank_count==before_count);
 bad={0,15};one<<<1,1>>>(f.owners[0],bad,result);C(cudaGetLastError());C(cudaDeviceSynchronize());
 CHECK(result->status==Status::invalid && *f.bits==before && *f.rank_count==before_count);
 std::puts("PASS resident packed bitmap owner: exact repeated-symbol states, two shards, segments, cap and replay");
}
