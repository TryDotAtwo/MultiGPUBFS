#include "generic_visited.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda/atomic>

static constexpr uint64_t EMPTY=~uint64_t(0);
#include "generic_action.cuh"
#include "generic_route.h"
template<class State,bool Shared=false> __global__ void seed_table(uint32_t elements,const State* states,uint32_t stride,uint32_t count,uint64_t* slots,
 uint32_t capacity,uint64_t seed,uint32_t bits,uint32_t* error){
 for(uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;row<count;row+=blockDim.x*gridDim.x){
  uint64_t h=seed;for(uint32_t e=0;e<elements;e++)h=mix64(h^uint64_t(states[uint64_t(e)*stride+row])^uint64_t(e));h=finish_hash(h,bits);
  uint64_t token=(h&0xffffffff00000000ULL)|row;uint32_t p=Shared?uint32_t((uint64_t(uint32_t(h))*capacity)>>32):uint32_t(h&(capacity-1));bool done=false;
  for(uint32_t i=0;i<capacity;i++,p=(p+1)&(capacity-1))if(atomicCAS(reinterpret_cast<unsigned long long*>(slots+p),EMPTY,token)==EMPTY){done=true;break;}
  if(!done)atomicOr(error,1u);
 }
}
template<class Candidate,class State>
__global__ void accept_candidates(Candidate action,uint64_t* slots,uint32_t slot_capacity,State* visited,uint32_t visited_capacity,
 uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,uint32_t* future_count,uint64_t seed,uint32_t bits,uint32_t* error){
 const uint32_t children=action.count*action.generators;
 for(uint32_t iteration=blockIdx.x*blockDim.x+threadIdx.x;iteration<children;iteration+=blockDim.x*gridDim.x){
  const uint32_t child=action.child(iteration);
  if(!action.valid(child,error))continue;
  if(cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(*error).load(cuda::memory_order_relaxed))return;
  const uint64_t hash=action.hash(child,seed,bits),prefix=hash&0xffffffff00000000ULL;
  const uint64_t pending=prefix|0x80000000ULL|child;uint32_t slot=action.bucket(hash,slot_capacity);bool done=false;
  for(uint32_t probe=0;probe<slot_capacity;probe++,slot=(slot+1)&(slot_capacity-1)){
   cuda::atomic_ref<uint64_t,cuda::thread_scope_device> published(slots[slot]);
   uint64_t observed=EMPTY;
   published.compare_exchange_strong(observed,pending,cuda::memory_order_acq_rel,cuda::memory_order_acquire);
   if(observed==EMPTY){
    // Publish the origin atomically before any payload allocation. Peers can
    // compare that immutable origin by regeneration, without spinning on a
    // writer or relying on an incompletely written state.
    const uint32_t row=atomicAdd(visited_count,1u);
    if(row>=visited_capacity){atomicOr(error,2u);done=true;break;}
    for(uint32_t e=0;e<action.elements;e++)visited[uint64_t(e)*visited_capacity+row]=action.value(child,e);
    // Release/acquire publishes the full canonical payload, including on GPUs
    // whose state loads use L1. No reader spins on an unpublished payload.
    published.store(prefix|row,cuda::memory_order_release);
    const uint32_t position=atomicAdd(future_count,1u);
    if(position>=future_capacity){atomicOr(error,4u);done=true;break;}
    future[position]=row;done=true;break;
   }
   if((observed&0xffffffff00000000ULL)!=prefix)continue;
   const uint32_t ref=uint32_t(observed),other=ref&0x7fffffffU;const bool transient=ref&0x80000000U;
   if((transient && other>=action.origin_limit())||(!transient && other>=visited_capacity)){atomicOr(error,8u);done=true;break;}
   bool equal=true;
   for(uint32_t e=0;e<action.elements;e++){
    const int64_t old=transient?action.value(other,e):visited[uint64_t(e)*visited_capacity+other];
    if(old!=action.value(child,e)){equal=false;break;}
   }
   if(equal){done=true;break;}
  }
  if(!done)atomicOr(error,1u);
 }
}
template<class State> __global__ void gather(uint32_t elements,const State* source,uint32_t source_stride,const uint32_t* indices,
 uint32_t count,State* output,uint32_t output_stride){
 for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<uint64_t(elements)*count;i+=uint64_t(blockDim.x)*gridDim.x){
  uint32_t e=i/count,row=i%count;output[uint64_t(e)*output_stride+row]=source[uint64_t(e)*source_stride+indices[row]];
 }
}
static uint32_t grid(uint64_t work){uint64_t n=(work+255)/256;return uint32_t(n>65535?65535:n);}
extern "C" int mgbfs_generic_seed_i64(uint32_t elements,const int64_t* states,uint32_t stride,uint32_t count,uint64_t* slots,
 uint32_t capacity,uint64_t seed,uint32_t bits,uint32_t* error,void* stream){
 if(!elements||count>stride||!capacity||(capacity&(capacity-1))||bits>64||!error||!slots||(!states&&count)||count>=0x80000000U)return int(cudaErrorInvalidValue);
 if(!count)return 0;seed_table<<<grid(count),256,0,static_cast<cudaStream_t>(stream)>>>(elements,states,stride,count,slots,capacity,seed,bits,error);return int(cudaGetLastError());
}
extern "C" int mgbfs_generic_expand_i64(uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const int64_t* parents,uint32_t count,uint32_t stride,const uint32_t* permutations,const int64_t* matrices,const uint32_t* moduli,
 uint64_t* slots,uint32_t slot_capacity,int64_t* visited,uint32_t visited_capacity,uint32_t* visited_count,
 uint32_t* future,uint32_t future_capacity,uint32_t* future_count,uint64_t seed,uint32_t bits,uint32_t* error,void* stream){
 if(kind>1||!elements||!n||!m||!generators||count>stride||uint64_t(count)*generators>=0x7fffffffULL||
  !slot_capacity||(slot_capacity&(slot_capacity-1))||!visited_capacity||visited_capacity>=0x80000000U||bits>64||
  !visited_count||!future_count||!error||!slots||!visited||!future||!parents||
  (kind==0&&(!permutations||elements!=n||m!=1))||(kind==1&&(!matrices||!moduli||uint64_t(n)*m!=elements)))return int(cudaErrorInvalidValue);
 if(!count)return 0;
 Action action{kind,elements,n,m,generators,count,stride,parents,permutations,matrices,moduli};
 accept_candidates<<<grid(uint64_t(count)*generators),256,0,static_cast<cudaStream_t>(stream)>>>(action,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,seed,bits,error);
 return int(cudaGetLastError());
}
extern "C" int mgbfs_generic_gather_i64(uint32_t elements,const int64_t* source,uint32_t stride,const uint32_t* indices,
 uint32_t count,int64_t* output,uint32_t output_stride,void* stream){
 if(!elements||count>output_stride||(!source&&count)||(!indices&&count)||(!output&&count))return int(cudaErrorInvalidValue);
 if(!count)return 0;gather<<<grid(uint64_t(elements)*count),256,0,static_cast<cudaStream_t>(stream)>>>(elements,source,stride,indices,count,output,output_stride);return int(cudaGetLastError());
}

template<class State> struct IncomingActionT {
 __device__ uint32_t child(uint32_t i)const{return i;}
 __device__ uint32_t origin_limit()const{return count;}
 __device__ uint32_t bucket(uint64_t h,uint32_t capacity)const{return h&(capacity-1);}
 uint32_t elements,count,stride,generators=1;const State* states;
 const GenericRouteRecord* metadata;const uint32_t* received;
 __device__ bool valid(uint32_t row,uint32_t* error)const{
  uint32_t actual=*received;if(actual>count){atomicOr(error,32u);return false;}return row<actual;
 }
 __device__ int64_t value(uint32_t row,uint32_t element)const{return states[uint64_t(element)*stride+row];}
 __device__ uint64_t hash(uint32_t row,uint64_t,uint32_t)const{return metadata[row].hash;}
};
extern "C" int mgbfs_generic_accept_i64(uint32_t elements,const int64_t* incoming,uint32_t stride,
 const GenericRouteRecord* metadata,const uint32_t* received,uint32_t bound,uint64_t* slots,
 uint32_t slot_capacity,int64_t* visited,uint32_t visited_capacity,uint32_t* visited_count,
 uint32_t* future,uint32_t future_capacity,uint32_t* future_count,uint32_t* error,void* stream){
 if(!elements||!bound||bound>stride||bound>=0x7fffffffU||!slot_capacity||(slot_capacity&(slot_capacity-1))||
  !visited_capacity||visited_capacity>=0x80000000U||!future_capacity||!incoming||!metadata||!received||
  !slots||!visited||!visited_count||!future||!future_count||!error)return int(cudaErrorInvalidValue);
 IncomingActionT<int64_t> action{elements,bound,stride,1,incoming,metadata,received};
 accept_candidates<<<grid(bound),256,0,static_cast<cudaStream_t>(stream)>>>(action,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,0,64,error);
 return int(cudaGetLastError());
}

// Compact permutation payload ABI shares the exact same kernel logic.
extern "C" int mgbfs_generic_seed_u8(uint32_t elements,const uint8_t* states,uint32_t stride,uint32_t count,uint64_t* slots,
 uint32_t capacity,uint64_t seed,uint32_t bits,uint32_t* error,void* stream){
 if(!elements||count>stride||!capacity||(capacity&(capacity-1))||bits>64||!error||!slots||(!states&&count)||count>=0x80000000U)return int(cudaErrorInvalidValue);
 if(!count)return 0;seed_table<<<grid(count),256,0,static_cast<cudaStream_t>(stream)>>>(elements,states,stride,count,slots,capacity,seed,bits,error);return int(cudaGetLastError());
}
extern "C" int mgbfs_generic_expand_u8(uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const uint8_t* parents,uint32_t count,uint32_t stride,const uint32_t* permutations,const int64_t* matrices,const uint32_t* moduli,
 uint64_t* slots,uint32_t slot_capacity,uint8_t* visited,uint32_t visited_capacity,uint32_t* visited_count,
 uint32_t* future,uint32_t future_capacity,uint32_t* future_count,uint64_t seed,uint32_t bits,uint32_t* error,void* stream){
 if(kind>0||!elements||!n||!m||!generators||count>stride||uint64_t(count)*generators>=0x7fffffffULL||
  !slot_capacity||(slot_capacity&(slot_capacity-1))||!visited_capacity||visited_capacity>=0x80000000U||bits>64||
  !visited_count||!future_count||!error||!slots||!visited||!future||!parents||
  (kind==0&&(!permutations||elements!=n||m!=1))||(kind==1&&(!matrices||!moduli||uint64_t(n)*m!=elements)))return int(cudaErrorInvalidValue);
 if(!count)return 0;
 ActionT<uint8_t> action{kind,elements,n,m,generators,count,stride,parents,permutations,matrices,moduli};
 accept_candidates<<<grid(uint64_t(count)*generators),256,0,static_cast<cudaStream_t>(stream)>>>(action,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,seed,bits,error);
 return int(cudaGetLastError());
}
extern "C" int mgbfs_generic_gather_u8(uint32_t elements,const uint8_t* source,uint32_t stride,const uint32_t* indices,
 uint32_t count,uint8_t* output,uint32_t output_stride,void* stream){
 if(!elements||count>output_stride||(!source&&count)||(!indices&&count)||(!output&&count))return int(cudaErrorInvalidValue);
 if(!count)return 0;gather<<<grid(uint64_t(elements)*count),256,0,static_cast<cudaStream_t>(stream)>>>(elements,source,stride,indices,count,output,output_stride);return int(cudaGetLastError());
}
extern "C" int mgbfs_generic_accept_u8(uint32_t elements,const uint8_t* incoming,uint32_t stride,
 const GenericRouteRecord* metadata,const uint32_t* received,uint32_t bound,uint64_t* slots,
 uint32_t slot_capacity,uint8_t* visited,uint32_t visited_capacity,uint32_t* visited_count,
 uint32_t* future,uint32_t future_capacity,uint32_t* future_count,uint32_t* error,void* stream){
 if(!elements||!bound||bound>stride||bound>=0x7fffffffU||!slot_capacity||(slot_capacity&(slot_capacity-1))||
  !visited_capacity||visited_capacity>=0x80000000U||!future_capacity||!incoming||!metadata||!received||
  !slots||!visited||!visited_count||!future||!future_count||!error)return int(cudaErrorInvalidValue);
 IncomingActionT<uint8_t> action{elements,bound,stride,1,incoming,metadata,received};
 accept_candidates<<<grid(bound),256,0,static_cast<cudaStream_t>(stream)>>>(action,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,0,64,error);
 return int(cudaGetLastError());
}

// All source inboxes for one shard share one exclusive owner launch. Pending
// origins identify source+row in immutable banks, so exact equality remains
// independent of thread scheduling and full hash collisions.
template<class State,bool Packed=false,bool Shared=false> struct IncomingAllAction {
 uint32_t elements,count,stride,generators,rank,world,shard,shards;
 const State* local;const State* remote;const GenericRouteRecord* local_meta;const GenericRouteRecord* remote_meta;
 const uint32_t* local_counts;const uint32_t* remote_counts;
 __device__ uint32_t child(uint32_t i)const{return Shared?((i/stride)*shards+shard)*stride+i%stride:i;}
 __device__ uint32_t origin_limit()const{return Shared?count*shards:count;}
 __device__ uint32_t bucket(uint64_t h,uint32_t capacity)const{return Shared?uint32_t((uint64_t(uint32_t(h))*capacity)>>32):uint32_t(h&(capacity-1));}
 __device__ uint32_t source(uint32_t child)const{return Shared?(child/stride)/shards:child/stride;}
 __device__ uint64_t queue(uint32_t child)const{return Shared?child/stride:uint64_t(source(child))*shards+shard;}
 __device__ bool valid(uint32_t child,uint32_t* error)const{
  const uint32_t* counts=source(child)==rank?local_counts:remote_counts;
  uint32_t actual=counts[queue(child)];if(actual>stride){atomicOr(error,32u);return false;}return child%stride<actual;
 }
 __device__ int64_t value(uint32_t child,uint32_t e)const{
  if constexpr(Packed){
   const GenericRouteRecord* meta=source(child)==rank?local_meta:remote_meta;
   const auto v=meta[queue(child)*stride+child%stride];
   const uint64_t word=e<8?v.parent:(e<16?(uint64_t(v.source)|(uint64_t(v.generator)<<32)):(uint64_t(v.shard)|(uint64_t(v.reserved)<<32)));
   return int64_t((word>>(8*(e%8)))&255u);
  }else{
   const State* states=source(child)==rank?local:remote;
   return states[queue(child)*uint64_t(elements)*stride+uint64_t(e)*stride+child%stride];
  }
 }
 __device__ uint64_t hash(uint32_t child,uint64_t,uint32_t)const{
  const GenericRouteRecord* meta=source(child)==rank?local_meta:remote_meta;return meta[queue(child)*stride+child%stride].hash;
 }
};
template<class State,bool Packed=false,bool Shared=false> int accept_all(uint32_t elements,const State* local,const State* remote,
 const GenericRouteRecord* local_meta,const GenericRouteRecord* remote_meta,const uint32_t* local_counts,const uint32_t* remote_counts,
 uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t stride,uint64_t* slots,uint32_t slot_capacity,
 State* visited,uint32_t visited_capacity,uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,uint32_t* future_count,uint32_t* error,void* stream){
 if((Shared&&uint64_t(world)*shards*stride>=0x7fffffffULL)||(Packed&&elements>24)||!elements||!world||world>128||rank>=world||!shards||shards>4096||shard>=shards||!stride||uint64_t(world)*stride>=0x7fffffffULL||
  !slot_capacity||(slot_capacity&(slot_capacity-1))||!visited_capacity||visited_capacity>=0x80000000U||!future_capacity||
  !local||!remote||!local_meta||!remote_meta||!local_counts||!remote_counts||!slots||!visited||!visited_count||!future||!future_count||!error)return int(cudaErrorInvalidValue);
 IncomingAllAction<State,Packed,Shared> action{elements,world*stride,stride,1,rank,world,shard,shards,local,remote,local_meta,remote_meta,local_counts,remote_counts};
 accept_candidates<<<grid(uint64_t(world)*stride),256,0,static_cast<cudaStream_t>(stream)>>>(action,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,0,64,error);return int(cudaGetLastError());
}
extern "C" int mgbfs_generic_accept_all_i64(uint32_t elements,const int64_t* local,const int64_t* remote,const GenericRouteRecord* local_meta,const GenericRouteRecord* remote_meta,
 const uint32_t* local_counts,const uint32_t* remote_counts,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t stride,
 uint64_t* slots,uint32_t slot_capacity,int64_t* visited,uint32_t visited_capacity,uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,
 uint32_t* future_count,uint32_t* error,void* stream){return accept_all(elements,local,remote,local_meta,remote_meta,local_counts,remote_counts,rank,world,shard,shards,stride,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,error,stream);}
extern "C" int mgbfs_generic_accept_all_u8(uint32_t elements,const uint8_t* local,const uint8_t* remote,const GenericRouteRecord* local_meta,const GenericRouteRecord* remote_meta,
 const uint32_t* local_counts,const uint32_t* remote_counts,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t stride,
 uint64_t* slots,uint32_t slot_capacity,uint8_t* visited,uint32_t visited_capacity,uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,
 uint32_t* future_count,uint32_t* error,void* stream){return accept_all(elements,local,remote,local_meta,remote_meta,local_counts,remote_counts,rank,world,shard,shards,stride,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,error,stream);}

extern "C" int mgbfs_generic_accept_all_packed_u8(uint32_t elements,const uint8_t* local,const uint8_t* remote,const GenericRouteRecord* local_meta,const GenericRouteRecord* remote_meta,
 const uint32_t* local_counts,const uint32_t* remote_counts,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t stride,
 uint64_t* slots,uint32_t slot_capacity,uint8_t* visited,uint32_t visited_capacity,uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,
 uint32_t* future_count,uint32_t* error,void* stream){return accept_all<uint8_t,true>(elements,local,remote,local_meta,remote_meta,local_counts,remote_counts,rank,world,shard,shards,stride,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,error,stream);}

extern "C" int mgbfs_generic_accept_all_shared_i64(uint32_t elements,const int64_t* local,const int64_t* remote,const GenericRouteRecord* local_meta,const GenericRouteRecord* remote_meta,
 const uint32_t* local_counts,const uint32_t* remote_counts,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t stride,
 uint64_t* slots,uint32_t slot_capacity,int64_t* visited,uint32_t visited_capacity,uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,
 uint32_t* future_count,uint32_t* error,void* stream){return accept_all<int64_t,false,true>(elements,local,remote,local_meta,remote_meta,local_counts,remote_counts,rank,world,shard,shards,stride,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,error,stream);}

extern "C" int mgbfs_generic_accept_all_shared_u8(uint32_t elements,const uint8_t* local,const uint8_t* remote,const GenericRouteRecord* local_meta,const GenericRouteRecord* remote_meta,
 const uint32_t* local_counts,const uint32_t* remote_counts,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t stride,
 uint64_t* slots,uint32_t slot_capacity,uint8_t* visited,uint32_t visited_capacity,uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,
 uint32_t* future_count,uint32_t* error,void* stream){return accept_all<uint8_t,false,true>(elements,local,remote,local_meta,remote_meta,local_counts,remote_counts,rank,world,shard,shards,stride,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,error,stream);}

extern "C" int mgbfs_generic_accept_all_shared_packed_u8(uint32_t elements,const uint8_t* local,const uint8_t* remote,const GenericRouteRecord* local_meta,const GenericRouteRecord* remote_meta,
 const uint32_t* local_counts,const uint32_t* remote_counts,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t stride,
 uint64_t* slots,uint32_t slot_capacity,uint8_t* visited,uint32_t visited_capacity,uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,
 uint32_t* future_count,uint32_t* error,void* stream){return accept_all<uint8_t,true,true>(elements,local,remote,local_meta,remote_meta,local_counts,remote_counts,rank,world,shard,shards,stride,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,error,stream);}

extern "C" int mgbfs_generic_seed_shared_i64(uint32_t elements,const int64_t* states,uint32_t stride,uint32_t count,uint64_t* slots,
 uint32_t capacity,uint64_t seed,uint32_t bits,uint32_t* error,void* stream){
 if(!elements||count>stride||!capacity||(capacity&(capacity-1))||bits>64||!error||!slots||(!states&&count)||count>=0x80000000U)return int(cudaErrorInvalidValue);
 if(!count)return 0;seed_table<int64_t,true><<<grid(count),256,0,static_cast<cudaStream_t>(stream)>>>(elements,states,stride,count,slots,capacity,seed,bits,error);return int(cudaGetLastError());
}
extern "C" int mgbfs_generic_seed_shared_u8(uint32_t elements,const uint8_t* states,uint32_t stride,uint32_t count,uint64_t* slots,
 uint32_t capacity,uint64_t seed,uint32_t bits,uint32_t* error,void* stream){
 if(!elements||count>stride||!capacity||(capacity&(capacity-1))||bits>64||!error||!slots||(!states&&count)||count>=0x80000000U)return int(cudaErrorInvalidValue);
 if(!count)return 0;seed_table<uint8_t,true><<<grid(count),256,0,static_cast<cudaStream_t>(stream)>>>(elements,states,stride,count,slots,capacity,seed,bits,error);return int(cudaGetLastError());
}
#include "generic_rolling.cuh"
