// Bounded bitmap owner transaction, prototype for one exclusive shard lane.
// Incoming records and the workspace stay live until materialize and commit
// complete. The caller must serialize all transactions of the same shard.
#include "generic_action.cuh"
#include "segmented_incoming.cuh"
#include "segmented_history.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {
constexpr size_t kAlign=256;
struct Grant {uint64_t base;uint32_t total,granted;};
struct Workspace {
 size_t keys_in,keys_out,origins_in,origins_out,word_first,representative;
 size_t deltas,snapshots,counts,prefix,grant,temp,temp_bytes,total;
};
bool region(size_t& cursor,size_t n,size_t* off){
 if(cursor>std::numeric_limits<size_t>::max()-(kAlign-1))return false;
 cursor=(cursor+kAlign-1)&~(kAlign-1);*off=cursor;
 if(n>std::numeric_limits<size_t>::max()-cursor)return false;
 cursor+=n;return true;
}
cudaError_t layout(uint32_t n,Workspace* w){
 if(!w||n>=0x7fffffffU)return cudaErrorInvalidValue;*w={};if(!n)return cudaSuccess;
 size_t sort=0,scan=0;
 cudaError_t rc=cub::DeviceRadixSort::SortPairs(nullptr,sort,
  static_cast<const uint64_t*>(nullptr),static_cast<uint64_t*>(nullptr),
  static_cast<const uint32_t*>(nullptr),static_cast<uint32_t*>(nullptr),int(n),0,64);
 if(rc!=cudaSuccess)return rc;
 rc=cub::DeviceScan::ExclusiveSum(nullptr,scan,static_cast<const uint32_t*>(nullptr),
  static_cast<uint32_t*>(nullptr),int(n));
 if(rc!=cudaSuccess)return rc;
 size_t c=0,temp=sort>scan?sort:scan;
 if(!region(c,size_t(n)*8,&w->keys_in)||!region(c,size_t(n)*8,&w->keys_out)||
    !region(c,size_t(n)*4,&w->origins_in)||!region(c,size_t(n)*4,&w->origins_out)||
    !region(c,size_t(n)*4,&w->word_first)||!region(c,size_t(n)*4,&w->representative)||
    !region(c,size_t(n)*8,&w->deltas)||!region(c,size_t(n)*8,&w->snapshots)||
    !region(c,size_t(n)*4,&w->counts)||!region(c,size_t(n)*4,&w->prefix)||
    !region(c,sizeof(Grant),&w->grant)||!region(c,temp,&w->temp))return cudaErrorInvalidValue;
 w->temp_bytes=temp;w->total=c;return cudaSuccess;
}
__device__ inline bool fatal(uint32_t x){return (x&~uint32_t(MGBFS_SEGMENTED_CAP_RETRY))!=0;}
template<class Input>
__global__ void prepare(Input in,const uint32_t* selection,uint32_t raw_begin,uint32_t n,uint32_t world,uint32_t owner,
 uint64_t total,uint64_t owner_len,uint32_t shard,uint32_t shards,uint64_t* keys,uint32_t* origins,uint32_t* error){
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
  uint32_t child=selection?selection[i]:in.child(raw_begin+i);origins[i]=child;
  if(!in.valid(child,error)){keys[i]=UINT64_MAX;continue;}
  uint64_t id=in.hash(child,0,64),local=id/world;
  if(id>=total||id%world!=owner||local>=owner_len||
    uint32_t((uint64_t(uint32_t(mix64(local>>6)))*shards)>>32)!=shard){
   atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));keys[i]=UINT64_MAX;continue;
  }
  keys[i]=local;
 }
}
__global__ void group_words(const uint64_t* keys,const uint32_t* origins,uint32_t n,const uint64_t* visited,
 uint32_t* word_first,uint32_t* representative,uint64_t* deltas,uint64_t* snapshots,uint32_t* counts,const uint32_t* error){
 if(fatal(*error))return;
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
  uint64_t id=keys[i];if(id==UINT64_MAX)continue;uint64_t word=id>>6;
  if(i&&keys[i-1]!=UINT64_MAX&&(keys[i-1]>>6)==word)continue;
  uint64_t mask=0;uint32_t j=i;
  while(j<n&&keys[j]!=UINT64_MAX&&(keys[j]>>6)==word){
   uint32_t start=j,rep=origins[j];uint64_t one=keys[j];
   do{word_first[j]=i;if(origins[j]<rep)rep=origins[j];j++;}while(j<n&&keys[j]==one);
   representative[start]=rep;mask|=1ull<<(one&63);
  }
  uint64_t old=visited[word],delta=mask&~old;
  snapshots[i]=old;deltas[i]=delta;counts[i]=uint32_t(__popcll(delta));
 }
}
__global__ void reserve_rows(const uint32_t* counts,const uint32_t* prefix,uint32_t n,
 uint64_t* accepted,uint64_t capacity,uint64_t* global_count,uint64_t limit,
 Grant* g,uint32_t* spill_count,uint32_t* error){
 if(blockIdx.x||threadIdx.x||fatal(*error))return;
 uint64_t base=*accepted;if(base>capacity){atomicOr(error,uint32_t(MGBFS_SEGMENTED_OVERFLOW));return;}
 uint32_t total=prefix[n-1]+counts[n-1];
 uint64_t avail=capacity-base;uint32_t want=avail<total?uint32_t(avail):total,grant=want;
 if(global_count&&want){
  cuda::atomic_ref<uint64_t,cuda::thread_scope_device> count(*global_count);
  uint64_t old=count.load(cuda::memory_order_relaxed);
  for(;;){
   if(old>=limit){grant=0;break;}
   uint64_t left=limit-old;grant=left<want?uint32_t(left):want;
   if(count.compare_exchange_weak(old,old+grant,cuda::memory_order_acq_rel,cuda::memory_order_relaxed))break;
  }
  if(grant<want)atomicOr(error,uint32_t(MGBFS_SEGMENTED_CAP_RETRY));
 }
 *g={base,total,grant};*spill_count=total-grant;
}
template<class T,class Input>
__global__ void materialize(Input in,const uint64_t* keys,uint32_t n,const uint32_t* word_first,
 const uint32_t* representative,const uint64_t* deltas,const uint32_t* prefix,const Grant* grant,
 MgbfsSegment active,uint32_t bank,uint32_t* spill,const uint32_t* error){
 if(fatal(*error))return;
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
  uint64_t id=keys[i];if(id==UINT64_MAX||(i&&keys[i-1]==id))continue;
  uint32_t first=word_first[i];uint64_t bit=1ull<<(id&63),delta=deltas[first];
  if(!(delta&bit))continue;
  uint32_t ordinal=prefix[first]+uint32_t(__popcll(delta&(bit-1)));
  uint32_t child=representative[i];
  if(ordinal>=grant->granted){spill[ordinal-grant->granted]=child;continue;}
  uint64_t row=grant->base+ordinal,stride=active.capacity*active.banks,offset=uint64_t(bank)*active.capacity+row;
  T* out=static_cast<T*>(active.states);
  for(uint32_t e=0;e<in.elements;e++)out[uint64_t(e)*stride+offset]=T(in.value(child,e));
 }
}
__global__ void commit_words(const uint64_t* keys,uint32_t n,const uint64_t* deltas,const uint64_t* snapshots,
 const uint32_t* counts,const uint32_t* prefix,const Grant* grant,uint64_t* bits,const uint32_t* error){
 if(fatal(*error))return;
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
  uint64_t id=keys[i];if(id==UINT64_MAX||(i&&keys[i-1]!=UINT64_MAX&&(keys[i-1]>>6)==(id>>6)))continue;
  uint32_t available=grant->granted>prefix[i]?grant->granted-prefix[i]:0;
  uint64_t delta=deltas[i],mask=0;
  if(available>=counts[i])mask=delta;
  else for(uint32_t k=0;k<available;k++){uint64_t bit=delta&(~delta+1);mask|=bit;delta-=bit;}
  bits[id>>6]=snapshots[i]|mask;
 }
}
__global__ void publish(uint64_t* accepted,const Grant* grant,const uint32_t* error){
 if(!blockIdx.x&&!threadIdx.x&&!fatal(*error))*accepted=grant->base+grant->granted;
}
template<class T,bool Packed>
int launch(uint32_t elements,const T* local,const T* remote,const GenericRouteRecord* lm,const GenericRouteRecord* rm,
 const uint32_t* lc,const uint32_t* rc,uint32_t rank,uint32_t world,uint32_t owner,uint64_t total,uint64_t owner_len,uint64_t* bits,
 uint32_t shard,uint32_t shards,uint32_t q,uint32_t visible,const uint32_t* selection,uint32_t selected,uint32_t raw_begin,MgbfsSegment active,uint32_t bank,
 uint64_t* accepted,uint32_t* spill,uint32_t* spill_count,uint32_t* error,uint64_t* global_count,uint64_t limit,
 void* scratch,const Workspace& w,cudaStream_t st){
 if(!selected)return int(cudaSuccess);
 char* base=static_cast<char*>(scratch);
 auto* ki=reinterpret_cast<uint64_t*>(base+w.keys_in);auto* ko=reinterpret_cast<uint64_t*>(base+w.keys_out);
 auto* oi=reinterpret_cast<uint32_t*>(base+w.origins_in);auto* oo=reinterpret_cast<uint32_t*>(base+w.origins_out);
 auto* word=reinterpret_cast<uint32_t*>(base+w.word_first);auto* rep=reinterpret_cast<uint32_t*>(base+w.representative);
 auto* delta=reinterpret_cast<uint64_t*>(base+w.deltas);auto* snap=reinterpret_cast<uint64_t*>(base+w.snapshots);
 auto* counts=reinterpret_cast<uint32_t*>(base+w.counts);auto* prefix=reinterpret_cast<uint32_t*>(base+w.prefix);
 auto* grant=reinterpret_cast<Grant*>(base+w.grant);uint32_t blocks=(selected+255)/256;if(blocks>65535)blocks=65535;
 cudaError_t rc0=cudaMemsetAsync(counts,0,size_t(selected)*4,st);if(rc0!=cudaSuccess)return int(rc0);
 MgbfsBoundedIncomingAction<T,Packed> in{{elements,world*q,q,1,rank,world,shard,shards,local,remote,lm,rm,lc,rc,nullptr},visible};
 prepare<<<blocks,256,0,st>>>(in,selection,raw_begin,selected,world,owner,total,owner_len,shard,shards,ki,oi,error);
 if((rc0=cudaGetLastError())!=cudaSuccess)return int(rc0);
 size_t tmp=w.temp_bytes;
 rc0=cub::DeviceRadixSort::SortPairs(base+w.temp,tmp,ki,ko,oi,oo,int(selected),0,64,st);if(rc0!=cudaSuccess)return int(rc0);
 group_words<<<blocks,256,0,st>>>(ko,oo,selected,bits,word,rep,delta,snap,counts,error);
 if((rc0=cudaGetLastError())!=cudaSuccess)return int(rc0);
 tmp=w.temp_bytes;rc0=cub::DeviceScan::ExclusiveSum(base+w.temp,tmp,counts,prefix,int(selected),st);if(rc0!=cudaSuccess)return int(rc0);
 reserve_rows<<<1,1,0,st>>>(counts,prefix,selected,accepted,active.capacity,global_count,limit,grant,spill_count,error);
 if((rc0=cudaGetLastError())!=cudaSuccess)return int(rc0);
 materialize<T><<<blocks,256,0,st>>>(in,ko,selected,word,rep,delta,prefix,grant,active,bank,spill,error);
 if((rc0=cudaGetLastError())!=cudaSuccess)return int(rc0);
 commit_words<<<blocks,256,0,st>>>(ko,selected,delta,snap,counts,prefix,grant,bits,error);
 if((rc0=cudaGetLastError())!=cudaSuccess)return int(rc0);
 publish<<<1,1,0,st>>>(accepted,grant,error);
 return int(cudaGetLastError());
}
}
extern "C" int mgbfs_bitmap_accept_batch_scratch_bytes(uint32_t selected,size_t* bytes){
 if(!bytes)return int(cudaErrorInvalidValue);Workspace w{};cudaError_t rc=layout(selected,&w);
 if(rc==cudaSuccess)*bytes=w.total;return int(rc);
}
extern "C" int mgbfs_bitmap_accept_batch(uint32_t state_bytes,uint32_t packed,uint32_t elements,const void* local,const void* remote,
 const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,uint32_t rank,uint32_t world,
 uint32_t owner,uint64_t total,uint64_t owner_len,uint64_t* bits,uint32_t shard,uint32_t shards,uint32_t q,uint32_t visible,
 const uint32_t* selection,uint32_t selected,uint32_t raw_begin,MgbfsSegment active,uint32_t bank,uint64_t* accepted,uint32_t* spill,
 uint32_t* spill_count,uint32_t* error,uint64_t* global_count,uint64_t limit,void* scratch,size_t scratch_bytes,void* stream){
 uint64_t bound=uint64_t(world)*q;
 if((state_bytes!=1&&state_bytes!=8)||packed>1||(packed&&(state_bytes!=1||elements>24))||!elements||
  !world||world>128||rank>=world||owner>=world||!total||!bits||!shards||shard>=shards||!q||
  bound>=0x7fffffffULL||visible>q||selected>bound||(!selection&&uint64_t(raw_begin)+selected>uint64_t(world)*visible)||(selection&&raw_begin!=0)||
  !lm||!rm||!lc||!rc||(!packed&&(!local||!remote))||!active.states||
  !(active.banks==2||active.banks==3)||!active.capacity||active.capacity>(uint64_t(1)<<32)||
  bank>=active.banks||!accepted||!spill||!spill_count||!error)return int(cudaErrorInvalidValue);
 Workspace w{};cudaError_t status=layout(selected,&w);if(status!=cudaSuccess)return int(status);
 if(selected&&(!scratch||uintptr_t(scratch)%kAlign||scratch_bytes<w.total))return int(cudaErrorInvalidValue);
 cudaStream_t st=static_cast<cudaStream_t>(stream);
 if(state_bytes==1&&packed)return launch<uint8_t,true>(elements,static_cast<const uint8_t*>(local),static_cast<const uint8_t*>(remote),lm,rm,lc,rc,rank,world,owner,total,owner_len,bits,shard,shards,q,visible,selection,selected,raw_begin,active,bank,accepted,spill,spill_count,error,global_count,limit,scratch,w,st);
 if(state_bytes==1)return launch<uint8_t,false>(elements,static_cast<const uint8_t*>(local),static_cast<const uint8_t*>(remote),lm,rm,lc,rc,rank,world,owner,total,owner_len,bits,shard,shards,q,visible,selection,selected,raw_begin,active,bank,accepted,spill,spill_count,error,global_count,limit,scratch,w,st);
 return launch<int64_t,false>(elements,static_cast<const int64_t*>(local),static_cast<const int64_t*>(remote),lm,rm,lc,rc,rank,world,owner,total,owner_len,bits,shard,shards,q,visible,selection,selected,raw_begin,active,bank,accepted,spill,spill_count,error,global_count,limit,scratch,w,st);
}
