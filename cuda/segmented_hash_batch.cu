// Experimental bounded exact sorted-run owner, one exclusive shard lane.
// Index = one u64 key + one u32 local row per live state; states stay in SoA.
// No old hash table, no global u32 rank addressing, and no per-state CPU work.
#include "generic_action.cuh"
#include "segmented_incoming.cuh"
#include "segmented_history.cuh"
#include "segmented_hash_batch.cuh"
#include "generic_sorted_origin_exact.cuh"
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {
constexpr size_t kAlign=256;
struct Grant {uint64_t base;uint32_t total,granted,run_index;};
struct Workspace {size_t hashes,origins,sorted,flags,prefix,grant,temp,temp_bytes,total;};
bool region(size_t& c,size_t n,size_t* off){
 if(c>std::numeric_limits<size_t>::max()-(kAlign-1))return false;
 c=(c+kAlign-1)&~(kAlign-1);*off=c;
 if(n>std::numeric_limits<size_t>::max()-c)return false;c+=n;return true;
}
template<class State,bool Packed> struct SelectedAction {
 uint32_t elements;
 MgbfsBoundedIncomingAction<State,Packed> in;
 const uint32_t* selection;
 uint32_t raw_begin;
 __device__ uint32_t child(uint32_t i)const{return selection?selection[i]:in.child(raw_begin+i);}
 __device__ bool valid(uint32_t c,uint32_t* e)const{return in.valid(c,e);}
 __device__ int64_t value(uint32_t c,uint32_t e)const{return in.value(c,e);}
 __device__ uint64_t hash(uint32_t c,uint64_t seed,uint32_t bits)const{return in.hash(c,seed,bits);}
};
template<class Action> cudaError_t layout(uint32_t n,Workspace* w){
 if(!w||n>=0x7fffffffU)return cudaErrorInvalidValue;*w={};if(!n)return cudaSuccess;
 size_t sort=0,scan=0;Action empty{};GenericSortedOriginLess<Action> cmp{empty,nullptr};
 cudaError_t rc=cub::DeviceMergeSort::SortKeysCopy(nullptr,sort,
  static_cast<const uint32_t*>(nullptr),static_cast<uint32_t*>(nullptr),int(n),cmp);
 if(rc!=cudaSuccess)return rc;
 rc=cub::DeviceScan::ExclusiveSum(nullptr,scan,static_cast<const uint32_t*>(nullptr),
  static_cast<uint32_t*>(nullptr),int(n));if(rc!=cudaSuccess)return rc;
 size_t c=0,temp=sort>scan?sort:scan;
 if(!region(c,size_t(n)*8,&w->hashes)||!region(c,size_t(n)*4,&w->origins)||
    !region(c,size_t(n)*4,&w->sorted)||!region(c,size_t(n)*4,&w->flags)||
    !region(c,size_t(n)*4,&w->prefix)||!region(c,sizeof(Grant),&w->grant)||
    !region(c,temp,&w->temp))return cudaErrorInvalidValue;
 w->temp_bytes=temp;w->total=c;return cudaSuccess;
}
__device__ inline bool fatal(uint32_t x){return (x&~uint32_t(MGBFS_SEGMENTED_CAP_RETRY|MGBFS_SEGMENTED_RUN_GROW))!=0;}
template<class Action>
__global__ void prepare(Action action,uint32_t n,uint64_t* hashes,uint32_t* origins,uint32_t* error){
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
  uint32_t child=action.child(i);bool valid=action.valid(child,error);
  origins[i]=valid?i:UINT32_MAX;hashes[i]=valid?action.hash(child,0,64):0;
 }
}
template<class T,class Action>
__device__ int compare_candidate(Action action,uint32_t origin,uint64_t hash,
 const MgbfsBatchHashSegment& segment,const MgbfsBatchHashRun& run,uint64_t mid,uint32_t* error){
 uint64_t at=uint64_t(run.bank)*segment.capacity+run.begin+mid;
 uint64_t stored_hash=segment.keys[at];
 if(hash<stored_hash)return -1;if(hash>stored_hash)return 1;
 uint32_t row=segment.rows[at];
 if(uint64_t(row)>=segment.capacity){atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return -1;}
 const T* states=static_cast<const T*>(segment.states);
 uint64_t stride=segment.capacity*segment.banks,offset=uint64_t(run.bank)*segment.capacity+row;
 uint32_t child=action.child(origin);
 for(uint32_t e=0;e<action.elements;e++){
  int64_t x=action.value(child,e),y=int64_t(states[uint64_t(e)*stride+offset]);
  if(x<y)return -1;if(x>y)return 1;
 }
 return 0;
}
template<class T,class Action>
__device__ bool contains(Action action,uint32_t origin,uint64_t hash,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,const MgbfsBatchHashRun* runs,
 uint32_t run_count,uint32_t* error){
 for(uint32_t j=0;j<run_count;j++){
  MgbfsBatchHashRun run=runs[j];
  if(run.segment>=segment_count){atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return false;}
  MgbfsBatchHashSegment s=segments[run.segment];
  if(!s.states||!s.keys||!s.rows||!s.capacity||s.capacity>(uint64_t(1)<<32)||
   !(s.banks==1||s.banks==3)||run.bank>=s.banks||run.begin>s.capacity||run.length>s.capacity-run.begin||
   !run.length||run.min_hash>run.max_hash){atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return false;}
  if(hash<run.min_hash||hash>run.max_hash)continue;
  uint64_t lo=0,hi=run.length;
  while(lo<hi){uint64_t mid=lo+(hi-lo)/2;int cmp=compare_candidate<T>(action,origin,hash,s,run,mid,error);
   if(cmp>0)lo=mid+1;else hi=mid;}
  if(lo<run.length&&compare_candidate<T>(action,origin,hash,s,run,lo,error)==0)return true;
 }
 return false;
}
template<class T,class Action>
__global__ void unique_and_probe(GenericSortedOriginLess<Action> order,const uint32_t* sorted,uint32_t n,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,const MgbfsBatchHashRun* runs,
 const uint32_t* run_count,uint32_t run_capacity,uint32_t* flags,uint32_t* error){
 uint32_t nr=*run_count;
 if(nr>run_capacity){if(!blockIdx.x&&!threadIdx.x)atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;}
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
  uint32_t origin=sorted[i];uint32_t fresh=0;
  if(origin!=UINT32_MAX&&(!i||order.exact_compare(sorted[i-1],origin)!=0))
   fresh=!contains<T>(order.action,origin,order.hashes[origin],segments,segment_count,runs,nr,error);
  flags[i]=fresh;
 }
}
__global__ void reserve(const uint32_t* flags,const uint32_t* prefix,uint32_t n,
 const MgbfsBatchHashSegment* segments,uint32_t active,uint32_t bank,uint32_t segment_count,
 uint64_t* accepted,const uint32_t* run_count,uint32_t run_capacity,uint64_t* global_count,uint64_t limit,
 Grant* grant,uint32_t* spill_count,uint32_t* error){
 if(blockIdx.x||threadIdx.x||fatal(*error))return;
 if(active>=segment_count){atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;}
 MgbfsBatchHashSegment seg=segments[active];
 uint64_t base=*accepted;
 if(!seg.states||!seg.keys||!seg.rows||!seg.capacity||seg.capacity>(uint64_t(1)<<32)||
  !(seg.banks==1||seg.banks==3)||bank>=seg.banks||base>seg.capacity){
  atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;
 }
 uint32_t total=prefix[n-1]+flags[n-1];
 uint64_t free=seg.capacity-base;uint32_t want=free<total?uint32_t(free):total,got=want;
 uint32_t index=*run_count;
 if(want&&index>=run_capacity){*grant={base,total,0,index};*spill_count=total;atomicOr(error,uint32_t(MGBFS_SEGMENTED_RUN_GROW));return;}
 if(global_count&&want){
  cuda::atomic_ref<uint64_t,cuda::thread_scope_device> count(*global_count);uint64_t old=count.load(cuda::memory_order_relaxed);
  for(;;){if(old>=limit){got=0;break;}uint64_t remaining=limit-old;got=remaining<want?uint32_t(remaining):want;
   if(count.compare_exchange_weak(old,old+got,cuda::memory_order_acq_rel,cuda::memory_order_relaxed))break;}
  if(got<want)atomicOr(error,uint32_t(MGBFS_SEGMENTED_CAP_RETRY));
 }
 *grant={base,total,got,index};*spill_count=total-got;
}
template<class T,class Action>
__global__ void materialize(Action action,const uint64_t* hashes,const uint32_t* sorted,const uint32_t* flags,
 const uint32_t* prefix,uint32_t n,const Grant* grant,const MgbfsBatchHashSegment* segments,uint32_t active,
 uint32_t bank,uint32_t* spill,const uint32_t* error){
 if(fatal(*error))return;
 MgbfsBatchHashSegment seg=segments[active];
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)if(flags[i]){
  uint32_t origin=sorted[i],ordinal=prefix[i],child=action.child(origin);
  if(ordinal>=grant->granted){spill[ordinal-grant->granted]=child;continue;}
  uint64_t row=grant->base+ordinal,at=uint64_t(bank)*seg.capacity+row,stride=seg.capacity*seg.banks;
  T* states=static_cast<T*>(seg.states);
  for(uint32_t e=0;e<action.elements;e++)states[uint64_t(e)*stride+at]=T(action.value(child,e));
  seg.keys[at]=hashes[origin];seg.rows[at]=uint32_t(row);
 }
}
__global__ void publish(const MgbfsBatchHashSegment* segments,uint32_t active,uint32_t bank,
 MgbfsBatchHashRun* runs,uint32_t* run_count,uint64_t* accepted,const Grant* g,const uint32_t* error){
 if(blockIdx.x||threadIdx.x||fatal(*error))return;
 if(g->granted){
  MgbfsBatchHashSegment seg=segments[active];uint64_t at=uint64_t(bank)*seg.capacity+g->base;
  runs[g->run_index]={active,bank,g->base,g->granted,seg.keys[at],seg.keys[at+g->granted-1]};
  *run_count=g->run_index+1;
 }
 *accepted=g->base+g->granted;
}
template<class T,bool Packed>
int launch(uint32_t elements,const T* local,const T* remote,const GenericRouteRecord* lm,const GenericRouteRecord* rm,
 const uint32_t* lc,const uint32_t* rc,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint32_t visible,
 const uint32_t* selection,uint32_t selected,uint32_t raw_begin,const MgbfsBatchHashSegment* segments,uint32_t segment_count,
 uint32_t active,uint32_t bank,MgbfsBatchHashRun* runs,uint32_t* run_count,uint32_t run_capacity,uint64_t* accepted,
 uint32_t* spill,uint32_t* spill_count,uint32_t* error,uint64_t* global_count,uint64_t limit,void* scratch,const Workspace& w,cudaStream_t st){
 if(!selected)return int(cudaSuccess);
 char* base=static_cast<char*>(scratch);auto* hashes=reinterpret_cast<uint64_t*>(base+w.hashes);
 auto* origins=reinterpret_cast<uint32_t*>(base+w.origins);auto* sorted=reinterpret_cast<uint32_t*>(base+w.sorted);
 auto* flags=reinterpret_cast<uint32_t*>(base+w.flags);auto* prefix=reinterpret_cast<uint32_t*>(base+w.prefix);
 auto* grant=reinterpret_cast<Grant*>(base+w.grant);
 MgbfsBoundedIncomingAction<T,Packed> in{{elements,world*q,q,1,rank,world,shard,shards,local,remote,lm,rm,lc,rc,nullptr},visible};
 SelectedAction<T,Packed> action{elements,in,selection,raw_begin};
 uint32_t blocks=(selected+255)/256;if(blocks>65535)blocks=65535;
 prepare<<<blocks,256,0,st>>>(action,selected,hashes,origins,error);
 cudaError_t status=cudaGetLastError();if(status!=cudaSuccess)return int(status);
 GenericSortedOriginLess<SelectedAction<T,Packed>> cmp{action,hashes};size_t tmp=w.temp_bytes;
 status=cub::DeviceMergeSort::SortKeysCopy(base+w.temp,tmp,origins,sorted,int(selected),cmp,st);
 if(status!=cudaSuccess)return int(status);
 unique_and_probe<T><<<blocks,256,0,st>>>(cmp,sorted,selected,segments,segment_count,runs,run_count,run_capacity,flags,error);
 if((status=cudaGetLastError())!=cudaSuccess)return int(status);
 tmp=w.temp_bytes;status=cub::DeviceScan::ExclusiveSum(base+w.temp,tmp,flags,prefix,int(selected),st);
 if(status!=cudaSuccess)return int(status);
 reserve<<<1,1,0,st>>>(flags,prefix,selected,segments,active,bank,segment_count,accepted,run_count,run_capacity,global_count,limit,grant,spill_count,error);
 if((status=cudaGetLastError())!=cudaSuccess)return int(status);
 materialize<T><<<blocks,256,0,st>>>(action,hashes,sorted,flags,prefix,selected,grant,segments,active,bank,spill,error);
 if((status=cudaGetLastError())!=cudaSuccess)return int(status);
 publish<<<1,1,0,st>>>(segments,active,bank,runs,run_count,accepted,grant,error);
 return int(cudaGetLastError());
}
}
extern "C" int mgbfs_hash_batch_scratch_bytes(uint32_t n,uint32_t state_bytes,uint32_t packed,size_t* bytes){
 if(!bytes||(state_bytes!=1&&state_bytes!=8)||packed>1||(packed&&state_bytes!=1))return int(cudaErrorInvalidValue);
 Workspace w{};cudaError_t status=state_bytes==1?(packed?layout<SelectedAction<uint8_t,true>>(n,&w):layout<SelectedAction<uint8_t,false>>(n,&w)):layout<SelectedAction<int64_t,false>>(n,&w);
 if(status==cudaSuccess)*bytes=w.total;return int(status);
}
extern "C" int mgbfs_hash_batch_accept(uint32_t state_bytes,uint32_t packed,uint32_t elements,const void* local,const void* remote,
 const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,
 uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint32_t visible,const uint32_t* selection,uint32_t selected,uint32_t raw_begin,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,uint32_t active,uint32_t bank,MgbfsBatchHashRun* runs,
 uint32_t* run_count,uint32_t run_capacity,uint64_t* accepted,uint32_t* spill,uint32_t* spill_count,uint32_t* error,
 uint64_t* global_count,uint64_t limit,void* scratch,size_t scratch_bytes,void* stream){
 uint64_t bound=uint64_t(world)*q;
 if((state_bytes!=1&&state_bytes!=8)||packed>1||(packed&&(state_bytes!=1||elements>24))||!elements||
  !world||world>128||rank>=world||!shards||shard>=shards||!q||bound>=0x7fffffffULL||visible>q||selected>bound||
  (!selection&&uint64_t(raw_begin)+selected>uint64_t(world)*visible)||(selection&&raw_begin!=0)||!lm||!rm||!lc||!rc||(!packed&&(!local||!remote))||
  !segments||!segment_count||active>=segment_count||!runs||!run_count||!run_capacity||!accepted||!spill||!spill_count||!error)
  return int(cudaErrorInvalidValue);
 Workspace w{};cudaError_t status=state_bytes==1?(packed?layout<SelectedAction<uint8_t,true>>(selected,&w):layout<SelectedAction<uint8_t,false>>(selected,&w)):layout<SelectedAction<int64_t,false>>(selected,&w);
 if(status!=cudaSuccess)return int(status);
 if(selected&&(!scratch||uintptr_t(scratch)%kAlign||scratch_bytes<w.total))return int(cudaErrorInvalidValue);
 cudaStream_t st=static_cast<cudaStream_t>(stream);
 if(state_bytes==1&&packed)return launch<uint8_t,true>(elements,static_cast<const uint8_t*>(local),static_cast<const uint8_t*>(remote),lm,rm,lc,rc,rank,world,shard,shards,q,visible,selection,selected,raw_begin,segments,segment_count,active,bank,runs,run_count,run_capacity,accepted,spill,spill_count,error,global_count,limit,scratch,w,st);
 if(state_bytes==1)return launch<uint8_t,false>(elements,static_cast<const uint8_t*>(local),static_cast<const uint8_t*>(remote),lm,rm,lc,rc,rank,world,shard,shards,q,visible,selection,selected,raw_begin,segments,segment_count,active,bank,runs,run_count,run_capacity,accepted,spill,spill_count,error,global_count,limit,scratch,w,st);
 return launch<int64_t,false>(elements,static_cast<const int64_t*>(local),static_cast<const int64_t*>(remote),lm,rm,lc,rc,rank,world,shard,shards,q,visible,selection,selected,raw_begin,segments,segment_count,active,bank,runs,run_count,run_capacity,accepted,spill,spill_count,error,global_count,limit,scratch,w,st);
}



// Seed the unique canonical start using the same exact run/index contract as
// accepted tiles. The caller has already admitted both segment and catalog.
namespace {
template<class T>
__global__ void seed_exact(const T* input,uint32_t elements,MgbfsBatchHashSegment seg,uint64_t hash,
 MgbfsBatchHashRun* runs,uint32_t* run_count,uint64_t* accepted){
 if(blockIdx.x||threadIdx.x)return;
 T* states=static_cast<T*>(seg.states);uint64_t stride=seg.capacity*seg.banks;
 for(uint32_t e=0;e<elements;e++)states[uint64_t(e)*stride]=input[e];
 seg.keys[0]=hash;seg.rows[0]=0;runs[0]={0,0,0,1,hash,hash};*run_count=1;*accepted=1;
}
__global__ void retire_bank_runs(MgbfsBatchHashRun* runs,uint32_t* count,uint32_t capacity,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,uint32_t bank,uint32_t* error){
 if(blockIdx.x||threadIdx.x)return;
 uint32_t n=*count;if(n>capacity){atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;}
 for(uint32_t i=0;i<n;i++){
  auto r=runs[i];if(r.segment>=segment_count||r.bank>=segments[r.segment].banks){
   atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;
  }
 }
 uint32_t out=0;for(uint32_t i=0;i<n;i++)if(runs[i].bank!=bank)runs[out++]=runs[i];
 *count=out;
}
}
extern "C" int mgbfs_hash_batch_seed(uint32_t state_bytes,uint32_t elements,const void* input,
 MgbfsBatchHashSegment seg,uint64_t hash,MgbfsBatchHashRun* runs,uint32_t* run_count,
 uint64_t* accepted,void* stream){
 if((state_bytes!=1&&state_bytes!=8)||!elements||!input||!seg.states||!seg.keys||!seg.rows||
  !seg.capacity||seg.capacity>(uint64_t(1)<<32)||!(seg.banks==1||seg.banks==3)||!runs||!run_count||!accepted)
  return int(cudaErrorInvalidValue);
 auto st=static_cast<cudaStream_t>(stream);
 if(state_bytes==1)seed_exact<uint8_t><<<1,1,0,st>>>(static_cast<const uint8_t*>(input),elements,seg,hash,runs,run_count,accepted);
 else seed_exact<int64_t><<<1,1,0,st>>>(static_cast<const int64_t*>(input),elements,seg,hash,runs,run_count,accepted);
 return int(cudaGetLastError());
}
extern "C" int mgbfs_hash_batch_retire_bank(MgbfsBatchHashRun* runs,uint32_t* count,uint32_t capacity,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,uint32_t bank,uint32_t* error,void* stream){
 if(!runs||!count||!capacity||!segments||!segment_count||bank>=3||!error)return int(cudaErrorInvalidValue);
 retire_bank_runs<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(runs,count,capacity,segments,segment_count,bank,error);
 return int(cudaGetLastError());
}

// Adjacent-run compaction is optional. Insufficient merge workspace is a
// successful deferred result: both original published runs stay searchable.
namespace {
struct MergeControl {MgbfsBatchHashRun left,right;uint32_t run_index,count,status;};
struct MergeLayout {size_t keys,rows,control,total;};
bool merge_layout(uint32_t entries,MergeLayout* w){
 if(!w||entries>=0x7fffffffU)return false;*w={};size_t c=0;
 if(!region(c,size_t(entries)*8,&w->keys)||!region(c,size_t(entries)*4,&w->rows)||
    !region(c,sizeof(MergeControl),&w->control))return false;
 w->total=c;return true;
}
__global__ void merge_prepare(const MgbfsBatchHashSegment* segments,uint32_t segment_count,
 MgbfsBatchHashRun* runs,const uint32_t* count,uint32_t max_entries,MergeControl* control,uint32_t* result,uint32_t* error){
 if(blockIdx.x||threadIdx.x||fatal(*error))return;
 *result=0;control->status=0;uint32_t n=*count;if(n<2)return;
 MgbfsBatchHashRun a=runs[n-2],b=runs[n-1];
 if(a.segment!=b.segment||a.bank!=b.bank)return;
 if(a.segment>=segment_count){atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;}
 MgbfsBatchHashSegment seg=segments[a.segment];
 if(!seg.states||!seg.keys||!seg.rows||!seg.capacity||seg.capacity>(uint64_t(1)<<32)||
  !(seg.banks==1||seg.banks==3)||a.bank>=seg.banks||!a.length||!b.length||
  a.begin>seg.capacity||a.length>seg.capacity-a.begin||b.begin>seg.capacity||b.length>seg.capacity-b.begin||
  a.min_hash>a.max_hash||b.min_hash>b.max_hash){
  atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;
 }
 if(a.begin+a.length!=b.begin)return;
 if(b.length<=UINT64_MAX/2 && a.length>2*b.length)return;
 uint64_t total=a.length+b.length;
 if(total>max_entries){*result=2;return;}
 *control={a,b,n-2,uint32_t(total),1};
}
template<class T>
__device__ int compare_rows(MgbfsBatchHashSegment seg,uint32_t bank,uint64_t a,uint64_t b,uint32_t elements,uint32_t* error){
 uint64_t base=uint64_t(bank)*seg.capacity,ka=seg.keys[base+a],kb=seg.keys[base+b];
 if(ka<kb)return -1;if(ka>kb)return 1;
 uint32_t ra=seg.rows[base+a],rb=seg.rows[base+b];
 if(uint64_t(ra)>=seg.capacity||uint64_t(rb)>=seg.capacity){atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return 0;}
 const T* states=static_cast<const T*>(seg.states);uint64_t stride=seg.capacity*seg.banks;
 for(uint32_t e=0;e<elements;e++){
  int64_t x=int64_t(states[uint64_t(e)*stride+base+ra]),y=int64_t(states[uint64_t(e)*stride+base+rb]);
  if(x<y)return -1;if(x>y)return 1;
 }
 return 0;
}
template<class T>
__global__ void merge_path(const MgbfsBatchHashSegment* segments,uint32_t elements,
 const MergeControl* control,uint32_t max_entries,uint64_t* out_keys,uint32_t* out_rows,uint32_t* error){
 if(fatal(*error)||control->status!=1)return;
 MgbfsBatchHashRun a=control->left,b=control->right;MgbfsBatchHashSegment seg=segments[a.segment];
 for(uint32_t k=blockIdx.x*blockDim.x+threadIdx.x;k<max_entries;k+=blockDim.x*gridDim.x){
  if(k>=control->count)continue;
  uint64_t low=k>b.length?uint64_t(k)-b.length:0,high=uint64_t(k)<a.length?k:a.length;
  while(low<high){
   uint64_t amid=low+(high-low)/2,bmid=uint64_t(k)-amid;
   if(amid<a.length&&bmid>0&&compare_rows<T>(seg,a.bank,b.begin+bmid-1,a.begin+amid,elements,error)>=0)low=amid+1;
   else high=amid;
  }
  uint64_t ai=low,bi=uint64_t(k)-ai;
  bool from_a=ai<a.length&&(bi>=b.length||compare_rows<T>(seg,a.bank,a.begin+ai,b.begin+bi,elements,error)<=0);
  uint64_t source=uint64_t(a.bank)*seg.capacity+(from_a?a.begin+ai:b.begin+bi);
  out_keys[k]=seg.keys[source];out_rows[k]=seg.rows[source];
 }
}
__global__ void merge_copy(const MgbfsBatchHashSegment* segments,const MergeControl* control,uint32_t max_entries,
 const uint64_t* keys,const uint32_t* rows,const uint32_t* error){
 if(fatal(*error)||control->status!=1)return;
 MgbfsBatchHashRun a=control->left;MgbfsBatchHashSegment seg=segments[a.segment];
 for(uint32_t k=blockIdx.x*blockDim.x+threadIdx.x;k<max_entries;k+=blockDim.x*gridDim.x)if(k<control->count){
  uint64_t at=uint64_t(a.bank)*seg.capacity+a.begin+k;seg.keys[at]=keys[k];seg.rows[at]=rows[k];
 }
}
__global__ void merge_publish(const MgbfsBatchHashSegment* segments,MgbfsBatchHashRun* runs,uint32_t* count,
 const MergeControl* control,uint32_t* result,const uint32_t* error){
 if(blockIdx.x||threadIdx.x||fatal(*error)||control->status!=1)return;
 MgbfsBatchHashRun a=control->left;MgbfsBatchHashSegment seg=segments[a.segment];
 uint64_t at=uint64_t(a.bank)*seg.capacity+a.begin;
 a.length=control->count;a.min_hash=seg.keys[at];a.max_hash=seg.keys[at+a.length-1];
 runs[control->run_index]=a;*count=control->run_index+1;*result=1;
}
}
extern "C" int mgbfs_hash_batch_compact_scratch_bytes(uint32_t entries,size_t* bytes){
 MergeLayout w{};if(!bytes||!merge_layout(entries,&w))return int(cudaErrorInvalidValue);
 *bytes=w.total;return int(cudaSuccess);
}
extern "C" int mgbfs_hash_batch_compact_tail(uint32_t state_bytes,uint32_t elements,const MgbfsBatchHashSegment* segments,
 uint32_t segment_count,MgbfsBatchHashRun* runs,uint32_t* count,uint32_t max_entries,void* scratch,
 size_t scratch_bytes,uint32_t* result,uint32_t* error,void* stream){
 MergeLayout w{};
 if((state_bytes!=1&&state_bytes!=8)||!elements||!segments||!segment_count||!runs||!count||!scratch||
  uintptr_t(scratch)%kAlign||!result||!error||!merge_layout(max_entries,&w)||scratch_bytes<w.total)
  return int(cudaErrorInvalidValue);
 char* base=static_cast<char*>(scratch);auto* keys=reinterpret_cast<uint64_t*>(base+w.keys);
 auto* rows=reinterpret_cast<uint32_t*>(base+w.rows);auto* control=reinterpret_cast<MergeControl*>(base+w.control);
 cudaStream_t st=static_cast<cudaStream_t>(stream);
 merge_prepare<<<1,1,0,st>>>(segments,segment_count,runs,count,max_entries,control,result,error);
 cudaError_t rc=cudaGetLastError();if(rc!=cudaSuccess)return int(rc);
 uint32_t blocks=(max_entries+255)/256;if(!blocks)blocks=1;if(blocks>65535)blocks=65535;
 if(state_bytes==1)merge_path<uint8_t><<<blocks,256,0,st>>>(segments,elements,control,max_entries,keys,rows,error);
 else merge_path<int64_t><<<blocks,256,0,st>>>(segments,elements,control,max_entries,keys,rows,error);
 if((rc=cudaGetLastError())!=cudaSuccess)return int(rc);
 merge_copy<<<blocks,256,0,st>>>(segments,control,max_entries,keys,rows,error);
 if((rc=cudaGetLastError())!=cudaSuccess)return int(rc);
 merge_publish<<<1,1,0,st>>>(segments,runs,count,control,result,error);
 return int(cudaGetLastError());
}
