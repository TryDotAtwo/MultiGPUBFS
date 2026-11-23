#include "../cuda/device_batch_bitmap.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>
using namespace mgbfs_device_batch;
#define CU(x) do{auto cuda_status_=(x);if(cuda_status_!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(cuda_status_));std::exit(1);}}while(0)
__global__ void test_accept(const Payload* incoming,uint64_t n,BitmapBatch batch) {
  bitmap_accept(incoming,n,batch);
}
template<class T> T* allocate(size_t n){T* p=nullptr;CU(cudaMalloc(&p,n*sizeof(T)));return p;}
static void check(bool v,const char* what){if(!v){std::fprintf(stderr,"FAIL %s\n",what);std::exit(1);}}

int main(){
  int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
  int cooperative=0;CU(cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,0));if(!cooperative)return 77;
  int sm=0,active=0;CU(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,0));
  CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,test_accept,128,0));
  int blocks=std::min(sm*active,32);if(blocks<2)return 77;
  constexpr uint32_t shards=4,banks=3;constexpr uint64_t extents=24,capacity=16,domain=1024,limit=129;
  using Extent=mgbfs_device_pack4::Extent;
  std::vector<Extent> host_extents(extents);
  std::vector<uint64_t*> all_rows,all_counts;
  for(auto& e:host_extents) {
    e.rows=allocate<uint64_t>(banks*capacity);e.counts=allocate<uint64_t>(banks);
    e.capacity=capacity;e.banks=banks;e.reserved=0;
    CU(cudaMemset(e.rows,0,banks*capacity*8));CU(cudaMemset(e.counts,0,banks*8));
    all_rows.push_back(e.rows);all_counts.push_back(e.counts);
  }
  Extent* dev_extents=allocate<Extent>(extents);
  uint32_t *owners=allocate<uint32_t>(extents),*error=allocate<uint32_t>(1);
  uint64_t *bits=allocate<uint64_t>((domain+63)/64),*rank_count=allocate<uint64_t>(1);
  uint64_t *positions=allocate<uint64_t>(8192),*counts=allocate<uint64_t>(shards);
  uint64_t *starts=allocate<uint64_t>(shards+1);
  BitmapSpan* spans=allocate<BitmapSpan>(extents);
  Payload* input=allocate<Payload>(8192);
  CU(cudaMemcpy(dev_extents,host_extents.data(),extents*sizeof(Extent),cudaMemcpyHostToDevice));
  CU(cudaMemset(owners,0xff,extents*4));CU(cudaMemset(bits,0,(domain+63)/64*8));
  CU(cudaMemset(rank_count,0,8));
  BitmapBatch batch{dev_extents,owners,extents,bits,domain,domain,rank_count,limit,
                    1,0,shards,1,positions,counts,starts,spans,error};
  auto wave=[&](const std::vector<uint64_t>& ids) {
    std::vector<Payload> payloads;
    for(auto id:ids)payloads.push_back({id,id^0x4192736455463728ull});
    if(!payloads.empty())CU(cudaMemcpy(input,payloads.data(),payloads.size()*sizeof(Payload),cudaMemcpyHostToDevice));
    uint64_t n=payloads.size();void* args[]={&input,&n,&batch};
    CU(cudaLaunchCooperativeKernel((void*)test_accept,dim3(blocks),dim3(128),args,0,nullptr));
    CU(cudaDeviceSynchronize());uint32_t e=0;CU(cudaMemcpy(&e,error,4,cudaMemcpyDeviceToHost));return e;
  };
  std::vector<uint64_t> ids(8192);
  for(uint64_t i=0;i<ids.size();++i)ids[i]=(i*37+(i>>5))%limit;
  check(wave(ids)==0,"new and duplicate candidates");
  uint64_t accepted=0;CU(cudaMemcpy(&accepted,rank_count,8,cudaMemcpyDeviceToHost));
  check(accepted==limit,"rank count");
  std::vector<uint32_t> tags(extents);CU(cudaMemcpy(tags.data(),owners,extents*4,cudaMemcpyDeviceToHost));
  std::set<uint32_t> active_shards;
  std::vector<uint64_t> stored;
  for(size_t e=0;e<extents;++e) {
    uint64_t count[banks];CU(cudaMemcpy(count,all_counts[e],sizeof(count),cudaMemcpyDeviceToHost));
    check(count[1]<=capacity,"extent bound");
    if(count[1])active_shards.insert(tags[e]);
    std::vector<uint64_t> rows(count[1]);
    if(!rows.empty())CU(cudaMemcpy(rows.data(),all_rows[e]+capacity,rows.size()*8,cudaMemcpyDeviceToHost));
    stored.insert(stored.end(),rows.begin(),rows.end());
  }
  check(active_shards.size()>=2,"two active shards");
  std::sort(stored.begin(),stored.end());
  std::vector<uint64_t> expected;
  for(uint64_t i=0;i<limit;++i)expected.push_back(i^0x4192736455463728ull);
  std::sort(expected.begin(),expected.end());
  check(stored==expected,"exact unique states");
  check(wave(ids)==0,"duplicate-only full cap");
  CU(cudaMemcpy(&accepted,rank_count,8,cudaMemcpyDeviceToHost));check(accepted==limit,"duplicate changed count");
  check(wave({limit})==BITMAP_RESOURCE,"new state beyond cap");
  std::printf("PASS bitmap batched CTA dedup blocks=%d exact=%llu active_shards=%zu\n",
      blocks,(unsigned long long)limit,active_shards.size());
  auto reset=[&] {
    CU(cudaMemset(owners,0xff,extents*4));CU(cudaMemset(bits,0,(domain+63)/64*8));
    CU(cudaMemset(rank_count,0,8));
    for(auto p:all_counts)CU(cudaMemset(p,0,banks*8));
  };
  reset();
  batch.world=2;batch.owner=1;batch.owner_len=domain/2;batch.rank_limit=2;
  check(wave({1,127,127,1,127})==0,"owner word boundaries");
  check(wave({127,1,127})==0,"duplicates at full rank cap");
  check(wave({129})==BITMAP_RESOURCE,"adjacent word beyond cap");
  reset();
  batch.world=1;batch.owner=0;batch.owner_len=domain;batch.shards=1;
  batch.extent_count=1;batch.rank_limit=0;
  std::vector<uint64_t> full;
  for(uint64_t i=0;i<capacity;++i)full.push_back(i);
  check(wave(full)==0,"fill only extent");
  check(wave(full)==0,"duplicate-only full extent");
  check(wave({capacity})==BITMAP_RESOURCE,"new state beyond physical pool");
  reset();
  batch.extent_count=extents;batch.shards=shards;batch.rank_limit=0;
  check(wave({domain})==BITMAP_FATAL,"invalid exact ID");
  check(wave({})==0,"empty tile");
  CU(cudaFree(input));CU(cudaFree(spans));CU(cudaFree(starts));CU(cudaFree(counts));
  CU(cudaFree(positions));CU(cudaFree(rank_count));CU(cudaFree(bits));CU(cudaFree(error));CU(cudaFree(owners));
  CU(cudaFree(dev_extents));
  for(auto p:all_rows)CU(cudaFree(p));for(auto p:all_counts)CU(cudaFree(p));
}
