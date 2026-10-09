#include "shard_ab_pipeline.h"
#include "state_commit.h"
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <climits>
#include <cstddef>
#include <stdexcept>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <cub/device/device_scan.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda/atomic>
static_assert(offsetof(MgbfsStateRingControl,fatal)==48,"Rust ring fatal ABI");
namespace {
struct alignas(16) Key{uint32_t w[4];};
constexpr uint32_t EMPTY=UINT32_MAX,LOCK=UINT32_MAX-1;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
bool power2(uint32_t n){return n&&!(n&(n-1));}
uint32_t bits(uint32_t n){uint32_t b=0;while(n>1){++b;n>>=1;}return b;}
uint32_t span_for(uint32_t n){uint32_t s=1;while(uint64_t(s)<uint64_t(n)*2)s*=2;return s;}
__device__ bool equal(Key a,Key b){return a.w[0]==b.w[0]&&a.w[1]==b.w[1]&&a.w[2]==b.w[2]&&a.w[3]==b.w[3];}

__device__ uint32_t hash(Key x){uint32_t h=0x9e3779b9u;for(int i=0;i<4;++i){h^=x.w[i]+0x9e3779b9u+(h<<6)+(h>>2);h^=h>>16;h*=0x85ebca6bu;}return h^(h>>16);}
__device__ uint32_t shard_of(Key k,uint32_t shift){return shift==32?0:k.w[3]>>shift;}
struct alignas(16) MergeRecord{Key key;uint64_t ref;};
constexpr uint64_t NEW_BIT=uint64_t(1)<<63,HISTORY_REF=NEW_BIT-1;
struct SortedWork{
 MergeRecord *b=nullptr;
 uint32_t *positions=nullptr,*tile_counts=nullptr,*tile_offsets=nullptr;
 void* temp=nullptr;size_t temp_bytes=0;uint32_t bound=0,tiles=0;
 ~SortedWork(){cudaFree(b);cudaFree(positions);cudaFree(tile_counts);cudaFree(tile_offsets);cudaFree(temp);}
};
size_t sorted_temp(uint32_t tiles){size_t bytes=0;ck(cub::DeviceScan::ExclusiveSum(nullptr,bytes,(uint32_t*)nullptr,(uint32_t*)nullptr,int(tiles)));return bytes;}
bool sorted_mode(){const char* mode=std::getenv("MGBFS_SHARD_AB_DEDUP");if(!mode||std::string(mode)=="HASH")return false;if(std::string(mode)=="SORT_MERGE")return true;throw std::runtime_error("SHARD_AB_DEDUP");}
struct Slot{cudaStream_t stream=nullptr;std::unique_ptr<SortedWork> sorted;};
struct ResponseLeaseMeta { MgbfsOwnerControl owner; MgbfsStateExtent extent; uint32_t count,start; };
bool future_only_mode(){const char*v=std::getenv("MGBFS_SORT_HISTORY_LOOKUP");return v&&std::string(v)=="1";}
bool async_response_mode(){const char*v=std::getenv("MGBFS_SHARD_AB_ASYNC_MATERIALIZE");return v&&std::string(v)=="1";}
uint64_t response_lease_bytes(uint32_t maximum,uint32_t stage){return sizeof(ResponseLeaseMeta)+uint64_t(2*maximum+1)*4;}
struct Pipeline{
 bool fused_response_meta=false;bool async_response=false,response_pending=false;cudaStream_t response_stream=nullptr;cudaEvent_t response_ready=nullptr,response_done=nullptr;
 ResponseLeaseMeta* response_meta=nullptr;uint32_t *response_counts=nullptr,*response_offsets=nullptr,*response_rows=nullptr;

 bool rotated_history=false;bool trace_history=false;bool direct_input=false;bool future_only=false; bool indexed=false,sorted=false,pending_materialization=false,requests_ready=false;const uint32_t* pending_begin=nullptr;uint32_t pending_input_capacity=0;Key* sorted_history=nullptr;uint32_t* history_offsets=nullptr;uint32_t* run_counts=nullptr; MgbfsStateRingControl* live_ring=nullptr; MgbfsOwnerControl* live_owner=nullptr; MgbfsStateExtent* live_extent=nullptr; uint8_t* live_states=nullptr; uint32_t *live_layer=nullptr,*live_next_count=nullptr; MgbfsStateExtent* live_next=nullptr; uint32_t live_capacity=0;
 uint32_t *state_indices=nullptr,*batch_counts=nullptr,*batch_refs=nullptr,*batch_key_rows=nullptr,*batch_offsets=nullptr,*batch_start=nullptr;
 uint32_t maximum,capacity,stride,k,stage_capacity,active=0,owner=0,world=1,shift=32;
 uint32_t table_span=0;uint64_t pushes=0;bool open=false;cudaStream_t producer;
 Key *keys=nullptr,*stage_keys=nullptr,*union_keys=nullptr;uint8_t *states=nullptr,*stage_states=nullptr,*union_states=nullptr;
 uint32_t *counts=nullptr,*stage_counts=nullptr,*ranges=nullptr,*output_count=nullptr,*table=nullptr;
 uint32_t *history_previous=nullptr,*history_current=nullptr,*history_storage=nullptr;uint32_t previous_span=1,current_span=1,history_capacity=0;
 const Key *previous=nullptr,*current=nullptr,*last_current=nullptr;uint32_t pn=0,cn=0,last_cn=0;bool history_valid=false;uint32_t* fatal=nullptr;
 std::vector<Slot>slots;std::vector<cudaEvent_t>done;cudaEvent_t copied=nullptr;
 ~Pipeline(){cudaStreamSynchronize(producer);
  if(response_stream)cudaStreamSynchronize(response_stream);
  if(response_stream)cudaStreamDestroy(response_stream);if(response_ready)cudaEventDestroy(response_ready);if(response_done)cudaEventDestroy(response_done);cudaFree(response_meta);
for(auto&x:slots)if(x.stream)cudaStreamSynchronize(x.stream);
  for(auto&x:slots)if(x.stream)cudaStreamDestroy(x.stream);for(auto e:done)if(e)cudaEventDestroy(e);if(copied)cudaEventDestroy(copied);
  cudaFree(state_indices);cudaFree(batch_counts);cudaFree(batch_refs);cudaFree(batch_key_rows);cudaFree(batch_offsets);cudaFree(batch_start);
   cudaFree(sorted_history);cudaFree(history_offsets);cudaFree(run_counts);if(history_storage)cudaFree(history_storage);else{cudaFree(history_previous);cudaFree(history_current);}cudaFree(table);cudaFree(stage_counts);cudaFree(stage_keys);cudaFree(stage_states);
  cudaFree(union_keys);cudaFree(union_states);cudaFree(output_count);cudaFree(ranges);cudaFree(counts);cudaFree(keys);cudaFree(states);}
};
int finish_response_lease(Pipeline*p){
 if(!p->response_pending)return 0;
 ck(cudaStreamWaitEvent(p->producer,p->response_done,0));
 if(mgbfs_state_finish_next_extent(p->live_ring,&p->response_meta->owner,&p->response_meta->extent,p->live_next_count,p->live_next,2,p->producer))return 2;
 ck(cudaMemcpyAsync(p->live_owner,&p->response_meta->owner,sizeof(MgbfsOwnerControl),cudaMemcpyDeviceToDevice,p->producer));
 ck(cudaMemcpyAsync(p->live_extent,&p->response_meta->extent,sizeof(MgbfsStateExtent),cudaMemcpyDeviceToDevice,p->producer));
 p->response_pending=false;return 0;
}
__global__ void assign_selected_state_slots(const uint32_t* counts,const uint32_t* rows,const uint32_t* offsets,const uint32_t* start,uint32_t active,uint32_t stage,uint32_t cap,uint32_t* indices,MgbfsStateRingControl* ring,const MgbfsOwnerControl* owner){
 if(ring->fatal||owner->error)return;
 uint32_t shard=blockIdx.x%active,worker=blockIdx.x/active,workers=(gridDim.x+active-1-shard)/active;
 for(uint32_t i=worker*blockDim.x+threadIdx.x;i<counts[shard];i+=workers*blockDim.x){
  uint32_t row=rows[uint64_t(shard)*stage+i];if(row>=cap){atomicCAS(&ring->fatal,0u,114u);continue;}
  indices[uint64_t(shard)*cap+row]=*start+offsets[shard]+i;
 }
}
__global__ void snapshot_response_lease(const uint32_t* counts,const uint32_t* offsets,const uint32_t* start,const uint32_t* actual_count,const MgbfsOwnerControl* owner,const MgbfsStateExtent* extent,uint32_t active,ResponseLeaseMeta* meta,uint32_t* out_counts,uint32_t* out_offsets){
 uint32_t t=threadIdx.x;
 if(!t){meta->owner=*owner;meta->extent=*extent;meta->count=*actual_count;meta->start=*start;out_offsets[active]=offsets[active];}
 if(t<active){out_counts[t]=counts[t];out_offsets[t]=offsets[t];}
}
// All blocks finish before response_ready is recorded. Key slots are assigned
// on the producer; only the independent payload lease crosses to the consumer.
// Count validation remains a preceding kernel so malformed counts cannot write.
__global__ void assign_and_snapshot_response(const uint32_t* counts,const uint32_t* rows,const uint32_t* offsets,const uint32_t* start,uint32_t active,uint32_t stage,uint32_t cap,uint32_t* indices,MgbfsStateRingControl* ring,const MgbfsOwnerControl* owner,const MgbfsStateExtent* extent,const uint32_t* actual_count,ResponseLeaseMeta* meta,uint32_t* out_counts,uint32_t* out_offsets){
 if(blockIdx.x==0){uint32_t t=threadIdx.x;if(!t){meta->owner=*owner;meta->extent=*extent;meta->count=*actual_count;meta->start=*start;out_offsets[active]=offsets[active];}for(uint32_t i=t;i<active;i+=blockDim.x){out_counts[i]=counts[i];out_offsets[i]=offsets[i];}}
 if(ring->fatal||owner->error)return;
 uint32_t shard=blockIdx.x%active,worker=blockIdx.x/active,workers=(gridDim.x+active-1-shard)/active;
 for(uint32_t i=worker*blockDim.x+threadIdx.x;i<counts[shard];i+=workers*blockDim.x){uint32_t row=rows[uint64_t(shard)*stage+i];if(row>=cap){atomicCAS(&ring->fatal,0u,114u);continue;}indices[uint64_t(shard)*cap+row]=*start+offsets[shard]+i;}
}
__global__ void clear_future(uint32_t* table,uint32_t active,uint32_t capacity,uint32_t span){
 for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<uint64_t(active)*span;i+=uint64_t(gridDim.x)*blockDim.x)
  table[(i/span)*uint64_t(2*capacity)+i%span]=EMPTY;
}
__global__ void history_build(const Key* keys,uint32_t n,uint32_t* table,uint32_t span,uint32_t* fatal){
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){
  uint32_t slot=hash(keys[i])&(span-1);bool done=false;
  for(uint32_t probe=0;probe<span;++probe){uint32_t old=atomicCAS(table+slot,EMPTY,i);
   if(old==EMPTY||equal(keys[old],keys[i])){done=true;break;}slot=(slot+1)&(span-1);}
  if(!done)atomicCAS(fatal,0u,115u);
 }
}
__device__ bool contains(const Key* keys,const uint32_t* table,uint32_t span,uint32_t n,Key key){
 if(!n)return false;uint32_t slot=hash(key)&(span-1);
 for(uint32_t probe=0;probe<span;++probe){uint32_t row=table[slot];if(row==EMPTY)return false;if(equal(keys[row],key))return true;slot=(slot+1)&(span-1);}return false;
}
constexpr uint32_t CURRENT_HISTORY_BIT=1u<<31;
__device__ Key history_key(const Key* previous,const Key* current,uint32_t code){return code&CURRENT_HISTORY_BIT?current[code&~CURRENT_HISTORY_BIT]:previous[code];}
__global__ void clear_sharded_history(uint32_t* table,uint32_t active,uint32_t cap,uint32_t span){
 for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<uint64_t(active)*span;i+=uint64_t(gridDim.x)*blockDim.x)table[(i/span)*uint64_t(4)*cap+i%span]=EMPTY;
}
__global__ void build_sharded_history(const Key* previous,uint32_t pn,const Key* current,uint32_t cn,uint32_t active,uint32_t shift,uint32_t cap,uint32_t span,uint32_t* table,uint32_t* fatal){
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<pn+cn;i+=gridDim.x*blockDim.x){
  uint32_t code=i<pn?i:((i-pn)|CURRENT_HISTORY_BIT);Key key=history_key(previous,current,code);
  uint32_t shard=shard_of(key,shift)&(active-1);uint32_t* slots=table+uint64_t(shard)*4*cap;uint32_t at=hash(key)&(span-1);bool done=false;
  for(uint32_t probe=0;probe<span;++probe){uint32_t old=atomicCAS(slots+at,EMPTY,code);if(old==EMPTY||equal(history_key(previous,current,old),key)){done=true;break;}at=(at+1)&(span-1);}
  if(!done)atomicCAS(fatal,0u,115u);
 }
}
__device__ bool contains_sharded_history(const Key* previous,const Key* current,const uint32_t* table,uint32_t span,uint32_t cap,uint32_t shard,uint32_t n,Key key){
 if(!n)return false;const uint32_t* slots=table+uint64_t(shard)*4*cap;uint32_t at=hash(key)&(span-1);
 for(uint32_t probe=0;probe<span;++probe){uint32_t row=slots[at];if(row==EMPTY)return false;if(equal(history_key(previous,current,row),key))return true;at=(at+1)&(span-1);}return false;
}
__device__ bool less(Key a,Key b){for(int i=3;i>=0;--i)if(a.w[i]!=b.w[i])return a.w[i]<b.w[i];return false;}
struct Run{
 const Key* keys=nullptr;const uint32_t* refs=nullptr;const MergeRecord* records=nullptr;
 uint32_t n=0;uint64_t tag=0;bool implicit_ref=false;
 __device__ MergeRecord at(uint32_t i)const{return records?records[i]:MergeRecord{keys[i],refs?(tag==0&&refs[i]==EMPTY?HISTORY_REF:tag|refs[i]):(tag|(implicit_ref?uint64_t(i):0))};}
};
__device__ uint32_t partition(Run a,Run b,uint32_t diagonal){
 uint32_t lo=diagonal>b.n?diagonal-b.n:0,hi=min(diagonal,a.n);
 while(lo<hi){uint32_t i=lo+(hi-lo)/2,j=diagonal-i;if(j&&i<a.n&&!less(b.at(j-1).key,a.at(i).key))lo=i+1;else hi=i;}return lo;
}
// Each block reads contiguous tiles; the only global binary searches locate
// tile boundaries, never one search for every incoming state.
__device__ void merge_tile(Run a,Run b,MergeRecord* output,MergeRecord* sa,MergeRecord* sb,uint32_t* bounds,uint32_t tile_index){
 uint32_t start=tile_index*256,total=a.n+b.n;
 if(start>=total)return;
 uint32_t end=min(start+256,total);
 if(!threadIdx.x){bounds[0]=partition(a,b,start);bounds[1]=start-bounds[0];uint32_t a1=partition(a,b,end);bounds[2]=a1-bounds[0];bounds[3]=end-a1-bounds[1];}
 __syncthreads();uint32_t ac=bounds[2],bc=bounds[3];
 for(uint32_t i=threadIdx.x;i<ac;i+=blockDim.x)sa[i]=a.at(bounds[0]+i);
 for(uint32_t i=threadIdx.x;i<bc;i+=blockDim.x)sb[i]=b.at(bounds[1]+i);
 __syncthreads();Run ar{},br{};ar.records=sa;ar.n=ac;br.records=sb;br.n=bc;
 for(uint32_t d=threadIdx.x;d<end-start;d+=blockDim.x){uint32_t i=partition(ar,br,d),j=d-i;output[start+d]=(i<ac&&(j==bc||!less(sb[j].key,sa[i].key)))?sa[i]:sb[j];}
}
__global__ void history_directory_pair(const Key* previous,uint32_t pn,const Key* current,uint32_t cn,uint32_t owner,uint32_t shards,uint32_t shift,uint32_t cap,uint32_t* offsets,uint32_t* fatal){
 uint32_t t=threadIdx.x;
 if(t<=shards){uint32_t target=owner*shards+t;
  uint32_t lo=0,hi=pn;while(lo<hi){uint32_t m=lo+(hi-lo)/2;if(shard_of(previous[m],shift)<target)lo=m+1;else hi=m;}offsets[t]=lo;
  lo=0;hi=cn;while(lo<hi){uint32_t m=lo+(hi-lo)/2;if(shard_of(current[m],shift)<target)lo=m+1;else hi=m;}offsets[shards+1+t]=lo;
 }
 __syncthreads();if(t<shards&&uint64_t(offsets[t+1]-offsets[t])+offsets[shards+1+t+1]-offsets[shards+1+t]>uint64_t(2)*cap)atomicCAS(fatal,0u,112u);
}
__global__ void initialize_unified(const Key* previous,const Key* current,const uint32_t* offsets,uint32_t active,uint32_t cap,uint32_t tiles,
 Key* keys,uint32_t* state_indices,uint32_t* run_counts,const uint32_t* fatal){
 if(*fatal)return;uint32_t shard=blockIdx.x/tiles,tile=blockIdx.x%tiles,start=tile*256;
 Run a{},b{};a.keys=previous+offsets[shard];a.n=offsets[shard+1]-offsets[shard];
 b.keys=current+offsets[active+1+shard];b.n=offsets[active+1+shard+1]-offsets[active+1+shard];
 if(!tile&&!threadIdx.x)run_counts[shard]=a.n+b.n;
 if(start>=a.n+b.n)return;uint32_t end=min(start+256,a.n+b.n);
 __shared__ MergeRecord sa[256],sb[256];__shared__ uint32_t bounds[4];
 if(!threadIdx.x){bounds[0]=partition(a,b,start);bounds[1]=start-bounds[0];uint32_t a1=partition(a,b,end);bounds[2]=a1-bounds[0];bounds[3]=end-a1-bounds[1];}
 __syncthreads();for(uint32_t i=threadIdx.x;i<bounds[2];i+=blockDim.x)sa[i]=a.at(bounds[0]+i);for(uint32_t i=threadIdx.x;i<bounds[3];i+=blockDim.x)sb[i]=b.at(bounds[1]+i);__syncthreads();
 Run ar{},br{};ar.records=sa;ar.n=bounds[2];br.records=sb;br.n=bounds[3];
 for(uint32_t d=threadIdx.x;d<end-start;d+=blockDim.x){uint32_t i=partition(ar,br,d),j=d-i;uint64_t dst=uint64_t(shard)*3*cap+start+d;
  keys[dst]=(i<ar.n&&(j==br.n||!less(sb[j].key,sa[i].key)))?sa[i].key:sb[j].key;state_indices[dst]=EMPTY;
 }
}
__global__ void merge_incoming(const Key* unified,const uint32_t* state_indices,const uint32_t* run_counts,uint32_t shard,uint32_t cap,
 const Key* incoming,uint32_t stage,const uint32_t* stage_counts,uint32_t buffer,MergeRecord* out,const uint32_t* fatal,const uint32_t* input_begin,const uint32_t* input_ranges){
 if(*fatal)return;Run a{},b{};a.keys=unified+uint64_t(shard)*3*cap;a.refs=state_indices+uint64_t(shard)*3*cap;a.n=run_counts[shard];
 b.keys=incoming+(input_begin?uint64_t(*input_begin)+input_ranges[shard]:uint64_t(shard*2+buffer)*stage);b.implicit_ref=true;b.tag=NEW_BIT;b.n=stage_counts[shard*2+buffer];
 __shared__ MergeRecord sa[256],sb[256];__shared__ uint32_t bounds[4];merge_tile(a,b,out,sa,sb,bounds,blockIdx.x);
}
__device__ bool sorted_history_has(const Key* keys,uint32_t lo,uint32_t hi,Key key){
 uint32_t end=hi;while(lo<hi){uint32_t mid=lo+(hi-lo)/2;if(less(keys[mid],key))lo=mid+1;else hi=mid;}return lo<end&&equal(keys[lo],key);
}
__global__ void mark_merge(const MergeRecord* merged,const uint32_t* offsets,const uint32_t* counts,const uint32_t* input_counts,uint32_t shard,uint32_t buffer,uint32_t* positions,uint32_t* tile_counts,const uint32_t* fatal,const Key* previous,const Key* current,uint32_t active,bool future_only){
 uint32_t i=blockIdx.x*256+threadIdx.x,total=counts[shard]+input_counts[shard*2+buffer];bool keep=false;
 if(!*fatal&&i<total){MergeRecord r=merged[i];bool fresh=r.ref&NEW_BIT;keep=!fresh||(i==0||!equal(r.key,merged[i-1].key));if(keep&&fresh&&future_only)keep=!sorted_history_has(previous,offsets[shard],offsets[shard+1],r.key)&&!sorted_history_has(current,offsets[active+1+shard],offsets[active+2+shard],r.key);}
 using Scan=cub::BlockScan<uint32_t,256>;__shared__ typename Scan::TempStorage scratch;uint32_t local,total_kept;
 Scan(scratch).ExclusiveSum(uint32_t(keep),local,total_kept);positions[i]=keep?local:EMPTY;if(!threadIdx.x)tile_counts[blockIdx.x]=total_kept;
}
__global__ void guard_sorted_count(const uint32_t* tile_counts,const uint32_t* tile_offsets,uint32_t tiles,uint32_t cap,uint32_t shard,
 const uint32_t* history_offsets,uint32_t active,uint32_t* run_counts,uint32_t* counts,uint32_t* fatal,bool future_only){
 if(*fatal)return;uint32_t total=tile_offsets[tiles-1]+tile_counts[tiles-1];
 uint32_t history=history_offsets[shard+1]-history_offsets[shard]+history_offsets[active+1+shard+1]-history_offsets[active+1+shard];
 if(future_only)history=0;
 if(total<history||total>3*cap||total-history>cap){atomicCAS(fatal,0u,112u);return;}run_counts[shard]=total;counts[shard]=total-history;
}
__global__ void copy_sorted_future(const MergeRecord* merged,const uint32_t* positions,const uint32_t* tile_offsets,uint32_t bound,uint32_t shard,uint32_t cap,uint32_t stage,
 Key* keys,uint32_t* state_indices,uint32_t* batch_counts,uint32_t* batch_refs,uint32_t* batch_key_rows,uint32_t* fatal){
 if(*fatal)return;for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<bound;i+=gridDim.x*blockDim.x){uint32_t local=positions[i];if(local==EMPTY)continue;uint32_t row=tile_offsets[i/256]+local;
 if(row>=3*cap){atomicCAS(fatal,0u,112u);return;}MergeRecord r=merged[i];uint64_t dest=uint64_t(shard)*3*cap+row;keys[dest]=r.key;
 if(r.ref&NEW_BIT){uint32_t selected=atomicAdd(batch_counts+shard,1u);if(selected>=stage){atomicCAS(fatal,0u,112u);return;}uint64_t at=uint64_t(shard)*stage+selected;batch_refs[at]=uint32_t(r.ref&~NEW_BIT);batch_key_rows[at]=row;state_indices[dest]=EMPTY;}else state_indices[dest]=r.ref==HISTORY_REF?EMPTY:uint32_t(r.ref);
 }
}
__global__ void prepare_input(const Key* keys,const uint32_t* begin,const uint32_t* rows,const uint32_t* source_rows,
 uint32_t input_cap,uint32_t owner,uint32_t world,uint32_t shards,uint32_t shift,uint32_t stage_cap,uint32_t buffer,
 uint32_t* ranges,uint32_t* stage_counts,uint32_t* fatal,uint32_t* batch_counts,uint32_t maximum){
 uint32_t t=threadIdx.x,n=*rows,b=*begin;
  if(batch_counts&&t<maximum)batch_counts[t]=0;
 if(n>input_cap||*source_rows>input_cap||b>*source_rows||n>*source_rows-b){if(t==0)atomicCAS(fatal,0u,110u);return;}
 if(t<=shards){uint32_t lo=0,hi=n,target=owner*shards+t;while(lo<hi){uint32_t m=lo+(hi-lo)/2;if(shard_of(keys[b+m],shift)<target)lo=m+1;else hi=m;}ranges[t]=lo;}
 __syncthreads();
 if(t<shards){uint32_t add=ranges[t+1]-ranges[t];stage_counts[t*2+buffer]=add;if(add>stage_cap)atomicCAS(fatal,0u,112u);}
}
__global__ void stage_copy(const Key* input,const uint8_t* states,const uint32_t* begin,const uint32_t* rows,
 uint32_t input_cap,uint32_t owner,uint32_t world,uint32_t shards,uint32_t shift,uint32_t buffer,uint32_t stage_cap,
 uint32_t chunks,const uint32_t* ranges,Key* keys,uint8_t* destination,uint32_t* fatal){
 for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<uint64_t(input_cap)*chunks;i+=uint64_t(gridDim.x)*blockDim.x){
  if(*fatal)return;uint32_t row=i/chunks,part=i%chunks;if(row>=*rows)continue;uint32_t source=*begin+row;Key key=input[source];
  uint32_t actual=world==1?0:key.w[3]>>(32-__ffs(world)+1);
  if(part==0 && (actual!=owner||(row&&shard_of(key,shift)<shard_of(input[source-1],shift)))){atomicCAS(fatal,0u,111u);continue;}
  if(part==0)for(int w=0;w<4;++w)if(key.w[w]>=4294967291u){atomicCAS(fatal,0u,102u);return;}
  uint32_t shard=shard_of(key,shift)&(shards-1),local=row-ranges[shard];if(local>=stage_cap){atomicCAS(fatal,0u,112u);continue;}
  uint64_t at=uint64_t(shard*2+buffer)*stage_cap+local;
  if(destination)reinterpret_cast<uint4*>(destination)[at*chunks+part]=reinterpret_cast<const uint4*>(states)[uint64_t(source)*chunks+part];if(part==0&&keys)keys[at]=key;
 }
}
// Each shard is owned by one stream. Its table survives all input portions of
// this layer. A/B input is recycled only after that stream records completion.
__global__ void insert_new(const Key* input,const uint8_t* input_states,const uint32_t* stage_counts,
 uint32_t stage_cap,uint32_t buffer,uint32_t active,uint32_t first,uint32_t step,uint32_t blocks_per_shard,
 uint32_t cap,uint32_t table_span,uint32_t chunks,Key* keys,uint8_t* states,uint32_t* counts,uint32_t* tables,
  bool indexed,bool rotated_history,uint32_t* batch_counts,uint32_t* batch_refs,uint32_t* batch_key_rows,
 const Key* previous,const uint32_t* hp,uint32_t ps,uint32_t pn,const Key* current,const uint32_t* hc,uint32_t cs,uint32_t cn,uint32_t* fatal,const uint32_t* input_begin,const uint32_t* input_ranges){
 uint32_t group=blockIdx.x/blocks_per_shard,shard=first+group*step;if(shard>=active)return;
 uint32_t lane_block=blockIdx.x%blocks_per_shard,n=stage_counts[shard*2+buffer];uint64_t source_base=input_begin?uint64_t(*input_begin)+input_ranges[shard]:uint64_t(shard*2+buffer)*stage_cap,out_base=uint64_t(shard)*cap;
 uint32_t span=table_span;uint32_t* table=tables+uint64_t(shard)*(2*cap);
 for(uint32_t i=lane_block*blockDim.x+threadIdx.x;i<n;i+=blocks_per_shard*blockDim.x){
  if(*fatal)return;Key key=input[source_base+i];
  if(indexed?(rotated_history?(contains_sharded_history(previous,nullptr,hp,ps,cap/2,shard,pn,key)||contains_sharded_history(current,nullptr,hc,cs,cap/2,shard,cn,key)):contains_sharded_history(previous,current,hp,ps,cap,shard,pn+cn,key)):(contains(previous,hp,ps,pn,key)||contains(current,hc,cs,cn,key)))continue;
  uint32_t slot=hash(key)&(span-1);bool done=false;
  for(uint32_t probe=0;probe<span;){
   // The table row publishes a mutable key. A legacy relaxed atomic plus
    // an ordinary payload load does not express the required reader acquire.
    cuda::atomic_ref<uint32_t,cuda::thread_scope_device> entry(table[slot]);
    // Read-mostly probes need an acquire load, not a read-modify-write.
    // Only an empty slot competes for ownership. A failed CAS retries the
    // acquire load before reading any winner's immutable payload.
    uint32_t old=entry.load(cuda::memory_order_acquire);
    if(old==EMPTY){uint32_t expected=EMPTY;
     if(!entry.compare_exchange_strong(expected,LOCK,cuda::memory_order_relaxed,cuda::memory_order_relaxed))continue;uint32_t row=atomicAdd(counts+shard,1u);
    if(row>=cap){atomicCAS(fatal,0u,112u);entry.store(EMPTY,cuda::memory_order_release);done=true;break;}
    keys[out_base+row]=key;
    if(indexed){uint32_t selected=atomicAdd(batch_counts+shard,1u);
      if(selected>=stage_cap){atomicCAS(fatal,0u,112u);entry.store(EMPTY,cuda::memory_order_release);done=true;break;}
      batch_refs[uint64_t(shard)*stage_cap+selected]=i;batch_key_rows[uint64_t(shard)*stage_cap+selected]=row;
     }else for(uint32_t c=0;c<chunks;++c)reinterpret_cast<uint4*>(states)[(out_base+row)*chunks+c]=reinterpret_cast<const uint4*>(input_states)[(source_base+i)*chunks+c];
    entry.store(row,cuda::memory_order_release);done=true;break;
   }
   if(old==LOCK){if(*fatal)return;continue;}
   if(equal(keys[out_base+old],key)){done=true;break;}
   slot=(slot+1)&(span-1);++probe;
  }
  if(!done)atomicCAS(fatal,0u,103u);
 }
}

__global__ void shard_offsets(const uint32_t* counts,uint32_t active,uint32_t cap,
 uint32_t* ranges,uint32_t* total,uint32_t* fatal){
 if(*fatal)return;uint64_t sum=0;
 for(uint32_t shard=0;shard<active;++shard){ranges[shard]=uint32_t(sum);
  if(counts[shard]>cap){atomicCAS(fatal,0u,114u);return;}sum+=counts[shard];
  if(sum>UINT32_MAX){atomicCAS(fatal,0u,114u);return;}}
 ranges[active]=uint32_t(sum);*total=uint32_t(sum);
}
__global__ void begin_publish(const uint32_t* count,uint32_t cap,MgbfsStateRingControl* ring,MgbfsOwnerControl* owner){
 if(ring->fatal||owner->error)return;if(*count>cap){atomicCAS(&ring->fatal,0u,16u);atomicCAS(&owner->error,0u,16u);return;}*owner={};owner->stage=1;owner->survivors=*count;}

__global__ void copy_published(const Key* keys,const uint8_t* states,const uint32_t* counts,
 const uint32_t* ranges,uint32_t active,uint32_t cap,uint32_t chunks,Key* out_keys,uint8_t* out_states,
 const MgbfsStateRingControl* ring,const MgbfsOwnerControl* owner,const MgbfsStateExtent* extent){
 if(ring->fatal||owner->error)return;uint32_t shard=blockIdx.x%active,worker=blockIdx.x/active;
 uint32_t workers=(gridDim.x+active-1-shard)/active;
 for(uint64_t i=uint64_t(worker)*blockDim.x+threadIdx.x;i<uint64_t(counts[shard])*chunks;i+=uint64_t(workers)*blockDim.x){
  uint64_t row=i/chunks,source=uint64_t(shard)*cap+row,dest=uint64_t(ranges[shard])+row;uint32_t part=i%chunks;
  reinterpret_cast<uint4*>(out_states)[(extent->begin+dest)*chunks+part]=reinterpret_cast<const uint4*>(states)[source*chunks+part];
  if(!part)out_keys[dest]=keys[source];}
}
// All ring mutations stay on the producer stream. Worker streams publish only
// hash/index selections. A/B input remains leased until materialization completes.
__global__ void copy_batch_states(const uint8_t* source,const uint32_t* counts,const uint32_t* refs,const uint32_t* key_rows,
 const uint32_t* offsets,const uint32_t* start,const uint32_t* source_begin,const uint32_t* source_ranges,uint32_t active,uint32_t stage,uint32_t cap,uint32_t buffer,uint32_t chunks,
 uint32_t* state_indices,uint8_t* output,const MgbfsStateRingControl* ring,const MgbfsOwnerControl* owner,const MgbfsStateExtent* extent){
 if(ring->fatal||owner->error)return;uint32_t shard=blockIdx.x%active,worker=blockIdx.x/active;
 uint32_t workers=(gridDim.x+active-1-shard)/active;
 for(uint64_t i=uint64_t(worker)*blockDim.x+threadIdx.x;i<uint64_t(counts[shard])*chunks;i+=uint64_t(workers)*blockDim.x){
  uint32_t row=i/chunks,part=i%chunks;uint64_t selected=uint64_t(shard)*stage+row;
  // The producer joins shard workers before this gather and before recycling the input lease.
  uint64_t src=uint64_t(*source_begin)+source_ranges[shard]+refs[selected];uint32_t dest=offsets[shard]+row;
  reinterpret_cast<uint4*>(output)[(extent->begin+dest)*chunks+part]=reinterpret_cast<const uint4*>(source)[src*chunks+part];
  if(!part)state_indices[uint64_t(shard)*cap+key_rows[selected]]=*start+dest;
 }
}
// Origin records are 16-byte source/move/absolute-parent identifiers.
// Destination order is the contiguous reservation, independent of source order.
__global__ void gather_selected_origins(const uint4* origins,const uint32_t* counts,const uint32_t* refs,
 const uint32_t* offsets,const uint32_t* begin,const uint32_t* ranges,uint32_t active,uint32_t stage,uint32_t capacity,
 uint4* output,uint32_t* output_count,MgbfsStateRingControl* ring,MgbfsOwnerControl* owner){
 if(ring->fatal||owner->error)return;if(owner->survivors>capacity){if(!blockIdx.x&&!threadIdx.x){atomicCAS(&ring->fatal,0u,114u);atomicCAS(&owner->error,0u,114u);}return;}
 if(!blockIdx.x&&!threadIdx.x)*output_count=owner->survivors;
 uint32_t shard=blockIdx.x%active,worker=blockIdx.x/active,workers=(gridDim.x+active-1-shard)/active;
 for(uint32_t row=worker*blockDim.x+threadIdx.x;row<counts[shard];row+=workers*blockDim.x){
  uint32_t dest=offsets[shard]+row;uint64_t src=uint64_t(*begin)+ranges[shard]+refs[uint64_t(shard)*stage+row];
  if(dest>=capacity){atomicCAS(&ring->fatal,0u,114u);atomicCAS(&owner->error,0u,114u);continue;}output[dest]=origins[src];
 }
}
__global__ void gather_selected_rows(const uint32_t* counts,const uint32_t* refs,const uint32_t* offsets,const uint32_t* begin,const uint32_t* ranges,uint32_t active,uint32_t stage,uint32_t capacity,uint32_t* output,uint32_t* output_count,MgbfsStateRingControl* ring,MgbfsOwnerControl* owner){
 if(ring->fatal||owner->error)return;if(owner->survivors>capacity){if(!blockIdx.x&&!threadIdx.x){atomicCAS(&ring->fatal,0u,114u);atomicCAS(&owner->error,0u,114u);}return;}if(!blockIdx.x&&!threadIdx.x)*output_count=owner->survivors;
 uint32_t shard=blockIdx.x%active,worker=blockIdx.x/active,workers=(gridDim.x+active-1-shard)/active;
 for(uint32_t row=worker*blockDim.x+threadIdx.x;row<counts[shard];row+=workers*blockDim.x){uint32_t dest=offsets[shard]+row;if(dest>=capacity){atomicCAS(&ring->fatal,0u,114u);atomicCAS(&owner->error,0u,114u);continue;}output[dest]=*begin+ranges[shard]+refs[uint64_t(shard)*stage+row];}
}
__global__ void validate_selected_response(const uint32_t* count,MgbfsStateRingControl* ring,MgbfsOwnerControl* owner){
 if(!ring->fatal&&!owner->error&&*count!=owner->survivors){atomicCAS(&ring->fatal,0u,114u);atomicCAS(&owner->error,0u,114u);}
}
__global__ void apply_selected_response(const uint4* source,const uint32_t* counts,const uint32_t* key_rows,
 const uint32_t* offsets,const uint32_t* start,uint32_t active,uint32_t stage,uint32_t cap,uint32_t chunks,
 uint32_t* indices,uint4* output,const MgbfsStateRingControl* ring,const MgbfsOwnerControl* owner,const MgbfsStateExtent* extent){
 if(ring->fatal||owner->error)return;uint32_t shard=blockIdx.x%active,worker=blockIdx.x/active,workers=(gridDim.x+active-1-shard)/active;
 for(uint64_t i=uint64_t(worker)*blockDim.x+threadIdx.x;i<uint64_t(counts[shard])*chunks;i+=uint64_t(workers)*blockDim.x){
  uint32_t row=i/chunks,part=i%chunks,dest=offsets[shard]+row;
  output[(extent->begin+dest)*chunks+part]=source[uint64_t(dest)*chunks+part];
  if(!part&&indices)indices[uint64_t(shard)*cap+key_rows[uint64_t(shard)*stage+row]]=*start+dest;
 }
}
__global__ void publish_indexed_keys(const Key* keys,const uint32_t* state_indices,const uint32_t* counts,
 uint32_t active,uint32_t cap,bool sorted,const uint32_t* offsets,Key* output,const uint32_t* layer,MgbfsStateRingControl* ring,MgbfsOwnerControl* owner){
 if(ring->fatal||owner->error)return;uint32_t shard=blockIdx.x%active,worker=blockIdx.x/active;
 uint32_t workers=(gridDim.x+active-1-shard)/active;
 for(uint32_t row=worker*blockDim.x+threadIdx.x;row<counts[shard];row+=workers*blockDim.x){
  uint64_t i=uint64_t(shard)*cap+row;uint32_t dest=state_indices[i];
  if(dest>=*layer){atomicCAS(&ring->fatal,0u,114u);atomicCAS(&owner->error,0u,114u);continue;}output[sorted?offsets[shard]+row:dest]=keys[i];
 }
}
__global__ void mark_final_future(const uint32_t* indices,const uint32_t* run_counts,uint32_t shard,uint32_t cap,uint32_t* positions,uint32_t* tile_counts,const uint32_t* fatal){
 uint32_t i=blockIdx.x*256+threadIdx.x;bool keep=!*fatal&&i<run_counts[shard]&&indices[uint64_t(shard)*3*cap+i]!=EMPTY;
 using Scan=cub::BlockScan<uint32_t,256>;__shared__ typename Scan::TempStorage scratch;uint32_t local,total;
 Scan(scratch).ExclusiveSum(uint32_t(keep),local,total);positions[i]=keep?local:EMPTY;if(!threadIdx.x)tile_counts[blockIdx.x]=total;
}
__global__ void copy_final_sorted(const Key* keys,const uint32_t* indices,const uint32_t* positions,const uint32_t* tile_offsets,uint32_t bound,uint32_t shard,uint32_t cap,const uint32_t* offsets,
 Key* output,const uint32_t* layer,MgbfsStateRingControl* ring,MgbfsOwnerControl* owner){
 if(ring->fatal||owner->error)return;for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<bound;i+=gridDim.x*blockDim.x){if(positions[i]==EMPTY)continue;
  uint64_t source=uint64_t(shard)*3*cap+i;uint32_t dest=offsets[shard]+tile_offsets[i/256]+positions[i];
  if(dest>=offsets[shard+1]||dest>=*layer||indices[source]>=*layer){atomicCAS(&ring->fatal,0u,114u);atomicCAS(&owner->error,0u,114u);continue;}output[dest]=keys[source];
 }
}
__global__ void validate_indexed_count(const uint32_t* count,const uint32_t* layer,MgbfsStateRingControl* ring,MgbfsOwnerControl* owner){
 if(!ring->fatal&&!owner->error&&*count!=*layer){atomicCAS(&ring->fatal,0u,114u);atomicCAS(&owner->error,0u,114u);}
}
__global__ void ready_published(const MgbfsStateRingControl* ring,MgbfsOwnerControl* owner,MgbfsStateExtent* extent){if(!ring->fatal&&!owner->error){extent->ready=1;owner->stage=2;}}
}
extern "C" int mgbfs_shard_ab_pipeline_query(uint32_t shards,uint32_t cap,uint32_t stride,uint32_t slots,uint64_t* bytes){
 if(bytes)*bytes=0;if(!bytes||!power2(shards)||shards>256||!power2(cap)||cap>INT_MAX/4||!stride||stride%16||!slots||slots>32||uint64_t(shards)*2*cap>INT_MAX)return 1;
 uint32_t stage=cap<256?256:(cap<131072?cap:131072);uint64_t sc=uint64_t(shards)*cap;
 *bytes=(sc+uint64_t(shards)*2*stage)*(16+stride)+24*sc+uint64_t(shards*3+shards+1+1)*4;return 0;}
static int pipeline_create(uint32_t shards,uint32_t cap,uint32_t stride,uint32_t k,void* stream,void** out,bool indexed){
 if(!out)return 1;*out=nullptr;uint64_t bytes;if(mgbfs_shard_ab_pipeline_query(shards,cap,stride,k,&bytes))return 1;
 try{auto p=std::make_unique<Pipeline>();p->indexed=indexed;p->direct_input=indexed&&std::getenv("MGBFS_SHARD_AB_DIRECT_INPUT")&&std::string(std::getenv("MGBFS_SHARD_AB_DIRECT_INPUT"))=="1";p->sorted=indexed&&sorted_mode();p->future_only=p->sorted&&std::getenv("MGBFS_SORT_HISTORY_LOOKUP")&&std::string(std::getenv("MGBFS_SORT_HISTORY_LOOKUP"))=="1";p->rotated_history=indexed&&!p->sorted&&cap>=2&&std::getenv("MGBFS_SHARD_AB_REUSE_HISTORY")&&std::string(std::getenv("MGBFS_SHARD_AB_REUSE_HISTORY"))=="1";p->trace_history=std::getenv("MGBFS_SHARD_AB_REUSE_HISTORY_TRACE")!=nullptr;p->maximum=shards;p->capacity=cap;p->stride=stride;p->k=k;p->stage_capacity=cap<256?256:(cap<131072?cap:131072);p->producer=static_cast<cudaStream_t>(stream);
  p->async_response=indexed&&async_response_mode();p->fused_response_meta=p->async_response&&std::getenv("MGBFS_SHARD_AB_FUSED_RESPONSE_META")&&std::string(std::getenv("MGBFS_SHARD_AB_FUSED_RESPONSE_META"))=="1";
  if(p->async_response){
   ck(cudaMalloc(&p->response_meta,response_lease_bytes(shards,p->stage_capacity)));
   p->response_counts=reinterpret_cast<uint32_t*>(p->response_meta+1);p->response_offsets=p->response_counts+shards;p->response_rows=nullptr;
   ck(cudaStreamCreateWithFlags(&p->response_stream,cudaStreamNonBlocking));ck(cudaEventCreateWithFlags(&p->response_ready,cudaEventDisableTiming));ck(cudaEventCreateWithFlags(&p->response_done,cudaEventDisableTiming));
  }
  uint64_t sc=uint64_t(shards)*cap,st=uint64_t(shards)*2*p->stage_capacity;p->history_capacity=uint32_t(sc*2);
  ck(cudaMalloc(&p->keys,sc*(p->sorted?3:1)*16));if(!indexed)ck(cudaMalloc(&p->states,sc*stride));
   if(indexed){ck(cudaMalloc(&p->state_indices,sc*(p->sorted?3:1)*4));ck(cudaMalloc(&p->batch_counts,shards*4));ck(cudaMalloc(&p->batch_refs,uint64_t(shards)*p->stage_capacity*4));ck(cudaMalloc(&p->batch_key_rows,uint64_t(shards)*p->stage_capacity*4));ck(cudaMalloc(&p->batch_offsets,(shards+1)*4));ck(cudaMalloc(&p->batch_start,4));}if(!p->direct_input)ck(cudaMalloc(&p->stage_keys,st*16));if(!indexed)ck(cudaMalloc(&p->stage_states,st*stride));
   if(p->sorted){ck(cudaMalloc(&p->history_offsets,2*(shards+1)*4));ck(cudaMalloc(&p->run_counts,shards*4));}
   else{ck(cudaMalloc(&p->table,sc*2*4));ck(cudaMalloc(&p->history_previous,sc*(indexed?4:2)*4));if(!indexed)ck(cudaMalloc(&p->history_current,sc*2*4));if(p->rotated_history){p->history_storage=p->history_previous;p->history_current=p->history_previous+sc*2;}}
  ck(cudaMalloc(&p->counts,shards*4));ck(cudaMalloc(&p->stage_counts,shards*2*4));ck(cudaMalloc(&p->ranges,(shards+1)*4));ck(cudaMalloc(&p->output_count,4));
  ck(cudaEventCreateWithFlags(&p->copied,cudaEventDisableTiming));p->done.resize(k*2,nullptr);
  for(auto&e:p->done){ck(cudaEventCreateWithFlags(&e,cudaEventDisableTiming));ck(cudaEventRecord(e,p->producer));}
  p->slots.resize(k);for(auto&x:p->slots){ck(cudaStreamCreateWithFlags(&x.stream,cudaStreamNonBlocking));
    if(p->sorted){x.sorted=std::make_unique<SortedWork>();auto&w=*x.sorted;w.bound=(p->future_only?cap:3*cap)+p->stage_capacity;w.tiles=(w.bound+255)/256;w.temp_bytes=sorted_temp(w.tiles);
     ck(cudaMalloc(&w.b,uint64_t(w.bound)*sizeof(MergeRecord)));ck(cudaMalloc(&w.positions,uint64_t(w.tiles)*256*4));ck(cudaMalloc(&w.tile_counts,uint64_t(w.tiles)*4));ck(cudaMalloc(&w.tile_offsets,uint64_t(w.tiles)*4));
     ck(cudaMalloc(&w.temp,w.temp_bytes));}}
*out=p.release();return 0;
 }catch(...){return 2;}}
extern "C" int mgbfs_shard_ab_pipeline_create(uint32_t s,uint32_t c,uint32_t w,uint32_t k,void* stream,void** out){return pipeline_create(s,c,w,k,stream,out,false);}
extern "C" int mgbfs_shard_ab_indexed_query(uint32_t s,uint32_t c,uint32_t w,uint32_t k,uint64_t* bytes){
 int result=mgbfs_shard_ab_pipeline_query(s,c,w,k,bytes);if(result)return result;
 uint32_t stage=c<256?256:(c<131072?c:131072);
 *bytes-=uint64_t(s)*c*w+uint64_t(s)*2*stage*w;*bytes+=uint64_t(s)*c*4+uint64_t(s)*stage*8+uint64_t(s*2+2)*4;
  if(std::getenv("MGBFS_SHARD_AB_DIRECT_INPUT")&&std::string(std::getenv("MGBFS_SHARD_AB_DIRECT_INPUT"))=="1")*bytes-=uint64_t(s)*2*stage*16;
  try{if(sorted_mode()){uint64_t sc=uint64_t(s)*c,bound=uint64_t(c)*(future_only_mode()?1:3)+stage,tiles=(bound+255)/256;*bytes-=24*sc;*bytes+=40*sc+uint64_t(3*s+2)*4+uint64_t(k)*(bound*32+tiles*256*4+tiles*8+sorted_temp(uint32_t(tiles)));}if(async_response_mode())*bytes+=response_lease_bytes(s,stage);}catch(...){return 2;}return 0;
}
extern "C" int mgbfs_shard_ab_indexed_create(uint32_t s,uint32_t c,uint32_t w,uint32_t k,void* stream,void** out){return pipeline_create(s,c,w,k,stream,out,true);}
extern "C" int mgbfs_shard_ab_indexed_bind(void* raw,void* ring,void* owner,void* extent,uint8_t* states,uint32_t* layer,uint32_t capacity,uint32_t* next_count,void* next){
 auto*p=static_cast<Pipeline*>(raw);if(!p||!p->indexed||p->open||!ring||!owner||!extent||!states||!layer||!capacity||!next_count||!next)return 1;
 p->live_ring=static_cast<MgbfsStateRingControl*>(ring);p->live_owner=static_cast<MgbfsOwnerControl*>(owner);p->live_extent=static_cast<MgbfsStateExtent*>(extent);
 p->live_states=states;p->live_layer=layer;p->live_capacity=capacity;p->live_next_count=next_count;p->live_next=static_cast<MgbfsStateExtent*>(next);return 0;
}
extern "C" int mgbfs_shard_ab_pipeline_begin(void* raw,uint32_t shards,uint32_t owner,uint32_t world,const void* previous,uint32_t pn,
 const void* current,uint32_t cn,uint32_t bound,uint32_t* fatal){
 auto*p=static_cast<Pipeline*>(raw);if(!p||p->open||(p->indexed&&!p->live_ring)||!power2(shards)||shards>p->maximum||!power2(world)||world>128||owner>=world||!power2(bound)||bound>p->capacity||!fatal||(pn&&!previous)||(cn&&!current)||uint64_t(pn)>uint64_t(p->maximum)*p->capacity||uint64_t(cn)>uint64_t(p->maximum)*p->capacity)return 1;
 try{bool rotate_geometry=p->active==shards&&p->owner==owner&&p->world==world;p->active=shards;p->owner=owner;p->world=world;p->shift=32-bits(world*shards);p->previous=static_cast<const Key*>(previous);p->current=static_cast<const Key*>(current);p->pn=pn;p->cn=cn;p->fatal=fatal;p->pushes=0;
  auto s=p->producer;ck(cudaMemsetAsync(p->counts,0,p->maximum*4,s));ck(cudaMemsetAsync(p->output_count,0,4,s));p->table_span=2*bound;
   if(p->sorted){
    history_directory_pair<<<1,512,0,s>>>(p->previous,pn,p->current,cn,owner,shards,p->shift,p->capacity,p->history_offsets,fatal);
    uint32_t history_bound=uint32_t(std::min(uint64_t(pn)+cn,uint64_t(2)*p->capacity));uint32_t tiles=(history_bound+255)/256;if(!tiles)tiles=1;
    if(p->future_only)ck(cudaMemsetAsync(p->run_counts,0,shards*4,s));else initialize_unified<<<shards*tiles,128,0,s>>>(p->previous,p->current,p->history_offsets,shards,p->capacity,tiles,p->keys,p->state_indices,p->run_counts,fatal);
   }else{
  unsigned clear_grid=unsigned((uint64_t(shards)*p->table_span+255)/256);if(clear_grid>4096)clear_grid=4096;
  clear_future<<<clear_grid,256,0,s>>>(p->table,shards,p->capacity,p->table_span);
   if(p->indexed){
   if(p->rotated_history){
    bool rotate=rotate_geometry&&p->history_valid&&p->previous==p->last_current&&pn==p->last_cn;
    if(p->trace_history)std::fprintf(stderr,"MGBFS_HISTORY_REUSE owner=%u active=%u reused=%u previous=%u current=%u\n",owner,shards,unsigned(rotate),pn,cn);
    if(rotate){std::swap(p->history_previous,p->history_current);p->previous_span=p->current_span;}
    else{
     p->previous_span=span_for(uint32_t(std::min(uint64_t(pn),uint64_t(p->capacity))));
     uint64_t total=uint64_t(shards)*p->previous_span;unsigned grid=unsigned(std::min(uint64_t(4096),(total+255)/256));
     clear_sharded_history<<<grid,256,0,s>>>(p->history_previous,shards,p->capacity/2,p->previous_span);
     if(pn)build_sharded_history<<<std::min(4096u,(pn+255)/256),256,0,s>>>(p->previous,pn,nullptr,0,shards,p->shift,p->capacity/2,p->previous_span,p->history_previous,fatal);
    }
    p->current_span=span_for(uint32_t(std::min(uint64_t(cn),uint64_t(p->capacity))));
    uint64_t total=uint64_t(shards)*p->current_span;unsigned grid=unsigned(std::min(uint64_t(4096),(total+255)/256));
    clear_sharded_history<<<grid,256,0,s>>>(p->history_current,shards,p->capacity/2,p->current_span);
    if(cn)build_sharded_history<<<std::min(4096u,(cn+255)/256),256,0,s>>>(p->current,cn,nullptr,0,shards,p->shift,p->capacity/2,p->current_span,p->history_current,fatal);
   }else{
   p->previous_span=span_for(uint32_t(std::min(uint64_t(pn)+cn,uint64_t(2)*p->capacity)));
   uint64_t total=uint64_t(shards)*p->previous_span;unsigned grid=unsigned(std::min(uint64_t(4096),(total+255)/256));
   clear_sharded_history<<<grid,256,0,s>>>(p->history_previous,shards,p->capacity,p->previous_span);
   if(pn+cn)build_sharded_history<<<std::min(4096u,(pn+cn+255)/256),256,0,s>>>(p->previous,pn,p->current,cn,shards,p->shift,p->capacity,p->previous_span,p->history_previous,fatal);
   }
  }else{
  bool rotate=p->history_valid&&p->previous==p->last_current&&pn==p->last_cn;
  if(rotate){std::swap(p->history_previous,p->history_current);p->previous_span=p->current_span;}
  else{p->previous_span=span_for(pn);ck(cudaMemsetAsync(p->history_previous,255,uint64_t(p->previous_span)*4,s));if(pn)history_build<<<(pn+255)/256,256,0,s>>>(p->previous,pn,p->history_previous,p->previous_span,fatal);}
  p->current_span=span_for(cn);ck(cudaMemsetAsync(p->history_current,255,uint64_t(p->current_span)*4,s));if(cn)history_build<<<(cn+255)/256,256,0,s>>>(p->current,cn,p->history_current,p->current_span,fatal);
  }
   }
  p->last_current=p->current;p->last_cn=cn;p->history_valid=true;p->open=true;ck(cudaGetLastError());return 0;
 }catch(...){return 2;}}
static int push_impl(void* raw,const void* keys,const uint8_t* states,const uint32_t* begin,const uint32_t* rows,const uint32_t* source_rows,uint32_t input_capacity,bool defer_materialization){
 auto*p=static_cast<Pipeline*>(raw);if(!p||!p->open||p->pending_materialization||!keys||(!states&&!defer_materialization)||!begin||!rows||!source_rows||!input_capacity||input_capacity>INT_MAX||(defer_materialization&&!p->indexed))return 1;
 try{auto s=p->producer;uint32_t buffer=uint32_t(p->pushes++&1);
  // Indexed pushes join every worker before the producer recycles input.
  if(!p->indexed)for(uint32_t slot=0;slot<p->k&&slot<p->active;++slot)ck(cudaStreamWaitEvent(s,p->done[slot*2+buffer],0));

  prepare_input<<<1,512,0,s>>>(static_cast<const Key*>(keys),begin,rows,source_rows,input_capacity,p->owner,p->world,p->active,p->shift,p->stage_capacity,buffer,p->ranges,p->stage_counts,p->fatal,p->indexed?p->batch_counts:nullptr,p->maximum);
  uint32_t chunks=p->stride/16;unsigned grid=unsigned((uint64_t(input_capacity)*chunks+255)/256);if(grid>4096)grid=4096;
  uint32_t stage_chunks=p->indexed?1:chunks;unsigned stage_grid=unsigned(std::min(uint64_t(4096),(uint64_t(input_capacity)*stage_chunks+255)/256));
  stage_copy<<<stage_grid,256,0,s>>>(static_cast<const Key*>(keys),states,begin,rows,input_capacity,p->owner,p->world,p->active,p->shift,buffer,p->stage_capacity,stage_chunks,p->ranges,p->direct_input?nullptr:p->stage_keys,p->stage_states,p->fatal);
  ck(cudaEventRecord(p->copied,s));uint32_t blocks_per_shard=256/p->active;if(blocks_per_shard<4)blocks_per_shard=4;
  for(uint32_t slot=0;slot<p->k&&slot<p->active;++slot){auto stream=p->slots[slot].stream;ck(cudaStreamWaitEvent(stream,p->copied,0));
   if(slot<p->active){uint32_t nshard=(p->active-1-slot)/p->k+1;
     if(p->sorted){auto&w=*p->slots[slot].sorted;uint32_t sort_bound=1;while(sort_bound<input_capacity&&sort_bound<p->stage_capacity)sort_bound*=2;
     uint32_t old_bound=(p->future_only?0:uint32_t(std::min(uint64_t(p->pn)+p->cn,uint64_t(2)*p->capacity)))+p->table_span/2;
     uint32_t work_tiles=(old_bound+sort_bound+255)/256;
     for(uint32_t shard=slot;shard<p->active;shard+=p->k){
      size_t bytes=w.temp_bytes;
      merge_incoming<<<work_tiles,128,0,stream>>>(p->keys,p->state_indices,p->run_counts,shard,p->capacity,p->direct_input?static_cast<const Key*>(keys):p->stage_keys,p->stage_capacity,p->stage_counts,buffer,w.b,p->fatal,p->direct_input?begin:nullptr,p->direct_input?p->ranges:nullptr);
      mark_merge<<<work_tiles,256,0,stream>>>(w.b,p->history_offsets,p->run_counts,p->stage_counts,shard,buffer,w.positions,w.tile_counts,p->fatal,p->previous,p->current,p->active,p->future_only);
      bytes=w.temp_bytes;ck(cub::DeviceScan::ExclusiveSum(w.temp,bytes,w.tile_counts,w.tile_offsets,int(work_tiles),stream));
      guard_sorted_count<<<1,1,0,stream>>>(w.tile_counts,w.tile_offsets,work_tiles,p->capacity,shard,p->history_offsets,p->active,p->run_counts,p->counts,p->fatal,p->future_only);
      copy_sorted_future<<<std::min(4096u,work_tiles),256,0,stream>>>(w.b,w.positions,w.tile_offsets,work_tiles*256,shard,p->capacity,p->stage_capacity,p->keys,p->state_indices,p->batch_counts,p->batch_refs,p->batch_key_rows,p->fatal);
     }
    }else insert_new<<<nshard*blocks_per_shard,256,0,stream>>>(p->direct_input?static_cast<const Key*>(keys):p->stage_keys,p->stage_states,p->stage_counts,p->stage_capacity,buffer,p->active,slot,p->k,blocks_per_shard,p->capacity,p->table_span,chunks,p->keys,p->states,p->counts,p->table,p->indexed,p->rotated_history,p->batch_counts,p->batch_refs,p->batch_key_rows,p->previous,p->history_previous,p->previous_span,p->pn,p->current,p->history_current,p->current_span,p->cn,p->fatal,p->direct_input?begin:nullptr,p->direct_input?p->ranges:nullptr);}
   ck(cudaEventRecord(p->done[slot*2+buffer],stream));}
   if(p->indexed){
    for(uint32_t slot=0;slot<p->k&&slot<p->active;++slot)ck(cudaStreamWaitEvent(s,p->done[slot*2+buffer],0));
    if(finish_response_lease(p))return 2;
    if(mgbfs_state_reserve_indexed_batch(p->live_ring,p->live_owner,p->live_extent,p->counts,p->batch_counts,p->active,p->capacity,p->stage_capacity,p->batch_offsets,p->batch_start,p->live_layer,p->live_capacity,s))return 2;
    if(defer_materialization){p->pending_materialization=true;p->pending_begin=begin;p->pending_input_capacity=input_capacity;return 0;}
    copy_batch_states<<<grid<p->active?p->active:grid,256,0,s>>>(states,p->batch_counts,p->batch_refs,p->batch_key_rows,p->batch_offsets,p->batch_start,begin,p->ranges,p->active,p->stage_capacity,p->sorted?3*p->capacity:p->capacity,buffer,p->stride/16,p->state_indices,p->live_states,p->live_ring,p->live_owner,p->live_extent);
    if(mgbfs_state_finish_next_extent(p->live_ring,p->live_owner,p->live_extent,p->live_next_count,p->live_next,2,s))return 2;
   }
   ck(cudaGetLastError());return 0;
  }catch(...){return 2;}}
// The selection lease owns batch refs, offsets and staging until materialize.
// This split permits routing origins instead of full states; it does not permit
// recycling input or starting another selection before this lease is resolved.
extern "C" int mgbfs_shard_ab_pipeline_push(void* raw,const void* keys,const uint8_t* states,const uint32_t* begin,const uint32_t* rows,const uint32_t* source_rows,uint32_t input_capacity){
 auto*p=static_cast<Pipeline*>(raw);
 if(!p||!p->indexed)return push_impl(raw,keys,states,begin,rows,source_rows,input_capacity,false);
 int status=push_impl(raw,keys,nullptr,begin,rows,source_rows,input_capacity,true);
 return status?status:mgbfs_shard_ab_pipeline_materialize(raw,states,begin,input_capacity);
}
extern "C" int mgbfs_shard_ab_pipeline_select(void* raw,const void* keys,const uint32_t* begin,const uint32_t* rows,const uint32_t* source_rows,uint32_t input_capacity){
 return push_impl(raw,keys,nullptr,begin,rows,source_rows,input_capacity,true);
}
extern "C" int mgbfs_shard_ab_pipeline_selection(void* raw,const uint32_t** counts,const uint32_t** refs,const uint32_t** key_rows,const uint32_t** offsets,const uint32_t** start){
 auto*p=static_cast<Pipeline*>(raw);if(!p||!p->pending_materialization||!counts||!refs||!key_rows||!offsets||!start)return 1;
 *counts=p->batch_counts;*refs=p->batch_refs;*key_rows=p->batch_key_rows;*offsets=p->batch_offsets;*start=p->batch_start;return 0;
}
extern "C" int mgbfs_shard_ab_pipeline_materialize(void* raw,const uint8_t* states,const uint32_t* begin,uint32_t input_capacity){
 auto*p=static_cast<Pipeline*>(raw);if(!p||!p->open||!p->pending_materialization||!states||begin!=p->pending_begin||input_capacity!=p->pending_input_capacity)return 1;
 try{auto s=p->producer;uint32_t buffer=uint32_t((p->pushes-1)&1);uint32_t grid=uint32_t(std::min(uint64_t(4096),(uint64_t(input_capacity)*(p->stride/16)+255)/256));
    copy_batch_states<<<grid<p->active?p->active:grid,256,0,s>>>(states,p->batch_counts,p->batch_refs,p->batch_key_rows,p->batch_offsets,p->batch_start,begin,p->ranges,p->active,p->stage_capacity,p->sorted?3*p->capacity:p->capacity,buffer,p->stride/16,p->state_indices,p->live_states,p->live_ring,p->live_owner,p->live_extent);
    if(mgbfs_state_finish_next_extent(p->live_ring,p->live_owner,p->live_extent,p->live_next_count,p->live_next,2,s))return 2;
  ck(cudaGetLastError());p->pending_materialization=false;return 0;
 }catch(...){return 2;}
}
extern "C" int mgbfs_shard_ab_pipeline_requests(void* raw,const void* origins,void* requests,uint32_t* count,uint32_t capacity){
 auto*p=static_cast<Pipeline*>(raw);if(!p||!p->pending_materialization||p->requests_ready||!origins||!requests||!count||!capacity)return 1;
 try{gather_selected_origins<<<std::max(p->active,256u),256,0,p->producer>>>(static_cast<const uint4*>(origins),p->batch_counts,p->batch_refs,p->batch_offsets,p->pending_begin,p->ranges,p->active,p->stage_capacity,capacity,static_cast<uint4*>(requests),count,p->live_ring,p->live_owner);ck(cudaGetLastError());p->requests_ready=true;return 0;}catch(...){return 2;}
}
extern "C" int mgbfs_shard_ab_pipeline_row_requests(void* raw,uint32_t* requests,uint32_t* count,uint32_t capacity){
 auto*p=static_cast<Pipeline*>(raw);if(!p||!p->pending_materialization||p->requests_ready||!requests||!count||!capacity)return 1;
 try{gather_selected_rows<<<std::max(p->active,256u),256,0,p->producer>>>(p->batch_counts,p->batch_refs,p->batch_offsets,p->pending_begin,p->ranges,p->active,p->stage_capacity,capacity,requests,count,p->live_ring,p->live_owner);ck(cudaGetLastError());p->requests_ready=true;return 0;}catch(...){return 2;}
}
extern "C" int mgbfs_shard_ab_pipeline_apply_responses(void* raw,const uint8_t* responses,const uint32_t* count){
 auto*p=static_cast<Pipeline*>(raw);if(!p||!p->pending_materialization||!p->requests_ready||!responses||!count)return 1;
 try{auto s=p->producer;validate_selected_response<<<1,1,0,s>>>(count,p->live_ring,p->live_owner);
 if(p->async_response){
  if(p->response_pending)return 1;
  if(p->fused_response_meta){
   assign_and_snapshot_response<<<std::max(p->active,256u),256,0,s>>>(p->batch_counts,p->batch_key_rows,p->batch_offsets,p->batch_start,p->active,p->stage_capacity,p->sorted?3*p->capacity:p->capacity,p->state_indices,p->live_ring,p->live_owner,p->live_extent,count,p->response_meta,p->response_counts,p->response_offsets);
  }else{
  assign_selected_state_slots<<<std::max(p->active,256u),256,0,s>>>(p->batch_counts,p->batch_key_rows,p->batch_offsets,p->batch_start,p->active,p->stage_capacity,p->sorted?3*p->capacity:p->capacity,p->state_indices,p->live_ring,p->live_owner);
  snapshot_response_lease<<<1,256,0,s>>>(p->batch_counts,p->batch_offsets,p->batch_start,count,p->live_owner,p->live_extent,p->active,p->response_meta,p->response_counts,p->response_offsets);
  }
  ck(cudaEventRecord(p->response_ready,s));ck(cudaStreamWaitEvent(p->response_stream,p->response_ready,0));
  apply_selected_response<<<std::max(p->active,256u),256,0,p->response_stream>>>(reinterpret_cast<const uint4*>(responses),p->response_counts,nullptr,p->response_offsets,&p->response_meta->start,p->active,p->stage_capacity,p->sorted?3*p->capacity:p->capacity,p->stride/16,nullptr,reinterpret_cast<uint4*>(p->live_states),p->live_ring,&p->response_meta->owner,&p->response_meta->extent);
  ck(cudaEventRecord(p->response_done,p->response_stream));ck(cudaGetLastError());p->response_pending=true;p->pending_materialization=false;p->requests_ready=false;return 0;
 }

 apply_selected_response<<<std::max(p->active,256u),256,0,s>>>(reinterpret_cast<const uint4*>(responses),p->batch_counts,p->batch_key_rows,p->batch_offsets,p->batch_start,p->active,p->stage_capacity,p->sorted?3*p->capacity:p->capacity,p->stride/16,p->state_indices,reinterpret_cast<uint4*>(p->live_states),p->live_ring,p->live_owner,p->live_extent);
 if(mgbfs_state_finish_next_extent(p->live_ring,p->live_owner,p->live_extent,p->live_next_count,p->live_next,2,s))return 2;
 ck(cudaGetLastError());p->pending_materialization=false;p->requests_ready=false;return 0;}catch(...){return 2;}
}
extern "C" int mgbfs_shard_ab_pipeline_finalize(void* raw,void** keys,uint8_t** states,const uint32_t** count){
 auto*p=static_cast<Pipeline*>(raw);if(!p||!p->open||p->pending_materialization||!keys||!states||!count)return 1;
 try{auto s=p->producer;if(finish_response_lease(p))return 2;if(!p->indexed)for(auto e:p->done)ck(cudaStreamWaitEvent(s,e,0));ck(cudaMemsetAsync(p->output_count,0,4,s));
  shard_offsets<<<1,1,0,s>>>(p->counts,p->active,p->capacity,p->ranges,p->output_count,p->fatal);
  ck(cudaGetLastError());*keys=p->keys;*states=p->states;*count=p->output_count;p->open=false;return 0;
 }catch(...){return 2;}}
extern "C" int mgbfs_shard_ab_pipeline_publish(void* raw,void* raw_ring,void* raw_owner,void* raw_extent,uint8_t* out_states,
 void* out_keys,uint32_t* layer_count,uint32_t layer_capacity,uint32_t* next_count,void* next_extents,uint32_t* route_count){
 auto*p=static_cast<Pipeline*>(raw);if(!p||p->open||!raw_ring||!raw_owner||!raw_extent||!out_states||!out_keys||!layer_count||!layer_capacity||!next_count||!next_extents||!route_count)return 1;
 try{auto s=p->producer;auto*r=static_cast<MgbfsStateRingControl*>(raw_ring);auto*o=static_cast<MgbfsOwnerControl*>(raw_owner);auto*e=static_cast<MgbfsStateExtent*>(raw_extent);
  if(p->indexed){
   if(p->sorted){ck(cudaEventRecord(p->copied,s));
    uint32_t bound=(p->future_only?0:uint32_t(std::min(uint64_t(p->pn)+p->cn,uint64_t(2)*p->capacity)))+p->table_span/2;uint32_t tiles=(bound+255)/256;
    for(uint32_t slot=0;slot<p->k&&slot<p->active;++slot){auto stream=p->slots[slot].stream;ck(cudaStreamWaitEvent(stream,p->copied,0));auto&w=*p->slots[slot].sorted;
     for(uint32_t shard=slot;shard<p->active;shard+=p->k){
      mark_final_future<<<tiles,256,0,stream>>>(p->state_indices,p->run_counts,shard,p->capacity,w.positions,w.tile_counts,p->fatal);
      size_t bytes=w.temp_bytes;ck(cub::DeviceScan::ExclusiveSum(w.temp,bytes,w.tile_counts,w.tile_offsets,int(tiles),stream));
      copy_final_sorted<<<std::min(4096u,tiles),256,0,stream>>>(p->keys,p->state_indices,w.positions,w.tile_offsets,tiles*256,shard,p->capacity,p->ranges,static_cast<Key*>(out_keys),layer_count,r,o);
     }
     ck(cudaEventRecord(p->done[slot*2],stream));ck(cudaStreamWaitEvent(s,p->done[slot*2],0));
    }
   }else publish_indexed_keys<<<4096,256,0,s>>>(p->keys,p->state_indices,p->counts,p->active,p->capacity,false,p->ranges,static_cast<Key*>(out_keys),layer_count,r,o);
   ck(cudaMemcpyAsync(route_count,p->output_count,4,cudaMemcpyDeviceToDevice,s));ck(cudaGetLastError());return 0;}
  // prepare has reserved and globally admitted the destination already.
  copy_published<<<4096,256,0,s>>>(p->keys,p->states,p->counts,p->ranges,p->active,p->capacity,p->stride/16,static_cast<Key*>(out_keys),out_states,r,o,e);
  ready_published<<<1,1,0,s>>>(r,o,e);if(mgbfs_state_publish_next_extent(r,o,e,next_count,static_cast<MgbfsStateExtent*>(next_extents),2,s))return 2;
  ck(cudaMemcpyAsync(route_count,p->output_count,4,cudaMemcpyDeviceToDevice,s));ck(cudaGetLastError());return 0;
 }catch(...){return 2;}}
extern "C" int mgbfs_shard_ab_pipeline_destroy(void* raw){delete static_cast<Pipeline*>(raw);return 0;}

extern "C" int mgbfs_shard_ab_pipeline_views(void* raw,const uint32_t** counts,const uint32_t** offsets){
 auto*p=static_cast<Pipeline*>(raw);if(!p||p->open||!counts||!offsets)return 1;
 *counts=p->counts;*offsets=p->ranges;return 0;
}

extern "C" int mgbfs_shard_ab_pipeline_prepare_publish(void* raw,void* raw_ring,void* raw_owner,
 void* raw_extent,uint32_t* layer_count,uint32_t layer_capacity){
 auto*p=static_cast<Pipeline*>(raw);if(!p||p->open||!raw_ring||!raw_owner||!raw_extent||!layer_count||!layer_capacity)return 1;
 auto*r=static_cast<MgbfsStateRingControl*>(raw_ring);auto*o=static_cast<MgbfsOwnerControl*>(raw_owner);auto*e=static_cast<MgbfsStateExtent*>(raw_extent);
 if(p->indexed){validate_indexed_count<<<1,1,0,p->producer>>>(p->output_count,layer_count,r,o);return cudaGetLastError()==cudaSuccess?0:2;}
 begin_publish<<<1,1,0,p->producer>>>(p->output_count,layer_capacity,r,o);
 if(mgbfs_state_reserve_layer(r,o,e,layer_count,layer_capacity,p->producer))return 2;
 return cudaGetLastError()==cudaSuccess?0:2;
}

namespace {
__global__ void compare_hashes(const Key* expected,const Key* actual,uint32_t n,uint32_t* fatal){
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)
  if(!equal(expected[i],actual[i])){if(atomicCAS(fatal,0u,130u)==0)printf("MGBFS_STATE_HASH_MISMATCH row=%u expected=%u,%u,%u,%u actual=%u,%u,%u,%u\n",i,expected[i].w[0],expected[i].w[1],expected[i].w[2],expected[i].w[3],actual[i].w[0],actual[i].w[1],actual[i].w[2],actual[i].w[3]);}
}
}
extern "C" int mgbfs_debug_hashes_equal(const void* expected,const void* actual,uint32_t n,uint32_t* fatal,void* stream){
 if(!expected||!actual||!fatal)return 1;if(!n)return 0;compare_hashes<<<(n+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const Key*>(expected),static_cast<const Key*>(actual),n,fatal);return cudaGetLastError()==cudaSuccess?0:2;
}

extern "C" int mgbfs_shard_ab_pipeline_drain(void* raw){auto*p=static_cast<Pipeline*>(raw);if(!p||p->pending_materialization)return 1;try{return finish_response_lease(p);}catch(...){return 2;}}
