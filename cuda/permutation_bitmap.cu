#include <cuda/atomic>
#include <cuda_runtime.h>
#include <cstdint>
#include "generic_route.h"
#include "generic_action.cuh"
#include "segmented_incoming.cuh"
#include "permutation_bitmap.cuh"
#include "segmented_history.cuh"
#include "segmented_warp_route.cuh"

// Rank is the exact state identifier; ownership is rank % world, and the
// bitmap index on that owner is rank / world. The map is a rank bijection.
template<class T,bool Packed,bool Grouped=false>
__global__ void bitmap_route(ActionT<T> action,BitmapDeviceDomain domain,uint32_t world,uint32_t source,
 uint32_t shards,uint32_t capacity,uint64_t begin,const uint32_t* map,
 GenericRouteRecord* output,uint32_t* counts,uint32_t* error){
 uint32_t children=action.count*action.generators;
 for(uint32_t child=blockIdx.x*blockDim.x+threadIdx.x;child<children;child+=blockDim.x*gridDim.x){
  uint64_t id=0;if(!bitmap_rank(domain,action,child,&id)){atomicOr(error,8u);continue;}
  uint32_t owner=uint32_t(id%world),target=map?map[owner]:owner;
  if(target>=world){atomicOr(error,2u);continue;}
  uint32_t shard=(uint64_t(uint32_t(mix64(id)))*shards)>>32;
  uint32_t queue=target*shards+shard,row;
  if constexpr(Grouped)row=mgbfs_warp_route_reserve(counts,queue);else row=atomicAdd(counts+queue,1u);
  if(row>=capacity){atomicOr(error,1u);continue;}
  if constexpr(Packed){
   uint64_t lo=0,hi=0,tail=0;
   for(uint32_t e=0;e<action.elements;e++){uint64_t v=uint64_t(action.value(child,e));if(e<8)lo|=v<<(8*e);else if(e<16)hi|=v<<(8*(e-8));else tail|=v<<(8*(e-16));}
   output[uint64_t(queue)*capacity+row]={id,lo,uint32_t(hi),uint32_t(hi>>32),uint32_t(tail),uint32_t(tail>>32)};
  }else output[uint64_t(queue)*capacity+row]={id,begin+child/action.generators,source,child%action.generators,shard,0};
 }
}
template<class T,bool Packed,bool Grouped=false>
int launch_route(BitmapDeviceDomain domain,uint32_t elements,uint32_t generators,const T* parents,uint32_t count,uint64_t stride,
 const uint32_t* permutations,uint32_t world,uint32_t source,uint32_t shards,uint32_t capacity,uint64_t begin,
 const uint32_t* map,GenericRouteRecord* output,uint32_t* counts,uint32_t* error,void* stream){
 if(!elements||elements!=domain.elements||!generators||!parents||count>stride||!permutations||
  !world||world>128||source>=world||!shards||shards>4096||!capacity||!output||!counts||!error||
  uint64_t(count)*generators>=0x7fffffffULL||begin>UINT64_MAX-count||(Packed&&elements>24))return int(cudaErrorInvalidValue);
 if(!count)return 0;
 ActionT<T> action{0,elements,elements,1,generators,count,stride,parents,permutations,nullptr,nullptr};
 uint64_t blocks=(uint64_t(count)*generators+255)/256;
 bitmap_route<T,Packed,Grouped><<<uint32_t(blocks>65535?65535:blocks),256,0,static_cast<cudaStream_t>(stream)>>>(action,domain,world,source,shards,capacity,begin,map,output,counts,error);
 return int(cudaGetLastError());
}
extern "C" int mgbfs_bitmap_route(uint32_t state_bytes,BitmapDeviceDomain domain,uint32_t elements,uint32_t generators,const void* parents,
 uint32_t count,uint64_t stride,const uint32_t* permutations,uint32_t world,uint32_t source,uint32_t shards,uint32_t capacity,uint64_t begin,
 const uint32_t* map,GenericRouteRecord* output,uint32_t* counts,uint32_t* error,void* stream){
 if(state_bytes==1){
  if(elements<=24)return launch_route<uint8_t,true>(domain,elements,generators,static_cast<const uint8_t*>(parents),count,stride,permutations,world,source,shards,capacity,begin,map,output,counts,error,stream);
  return launch_route<uint8_t,false>(domain,elements,generators,static_cast<const uint8_t*>(parents),count,stride,permutations,world,source,shards,capacity,begin,map,output,counts,error,stream);
 }
 if(state_bytes==8)return launch_route<int64_t,false>(domain,elements,generators,static_cast<const int64_t*>(parents),count,stride,permutations,world,source,shards,capacity,begin,map,output,counts,error,stream);
 return int(cudaErrorInvalidValue);
}

template<class T>
__global__ void bitmap_seed(const T* input,uint32_t elements,MgbfsSegment active,uint64_t* bits,uint64_t local,uint64_t* accepted){
 if(blockIdx.x||threadIdx.x)return;
 for(uint32_t e=0;e<elements;e++)static_cast<T*>(active.states)[uint64_t(e)*active.capacity*active.banks]=input[e];
 bits[local/64]|=1ull<<(local%64);*accepted=1;
}
extern "C" int mgbfs_bitmap_seed(uint32_t state_bytes,const void* input,uint32_t elements,MgbfsSegment active,uint64_t* bits,uint64_t local,uint64_t* accepted,void* stream){
 if(!input||!elements||!active.states||!(active.banks==2||active.banks==3)||!bits||!accepted)return int(cudaErrorInvalidValue);
 if(state_bytes==1)bitmap_seed<uint8_t><<<1,1,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const uint8_t*>(input),elements,active,bits,local,accepted);
 else if(state_bytes==8)bitmap_seed<int64_t><<<1,1,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const int64_t*>(input),elements,active,bits,local,accepted);
 else return int(cudaErrorInvalidValue);
 return int(cudaGetLastError());
}

// Winners keep their bit only after acquiring a state row. If no row exists,
// release the bit and put the original input in the bounded spill queue.
// All same-id inputs route to the same owner/shard, so one owner lane controls
// their spill lifetime. A fatal/capacity failure terminates the partial layer.
template<class T,class Input>
__global__ void bitmap_accept(Input in,uint32_t world,uint32_t owner,uint64_t total,uint64_t owner_len,
 uint64_t* bits,MgbfsSegment active,uint32_t bank,uint64_t* accepted,const uint32_t* selection,uint32_t selected,
 uint32_t* spill,uint32_t* spill_count,uint32_t* error,uint64_t* global_count,uint64_t global_limit){
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<selected;i+=blockDim.x*gridDim.x){
  uint32_t child=selection?selection[i]:in.child(i);if(!in.valid(child,error))continue;
  if(cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(*error).load(cuda::memory_order_relaxed))return;
  uint64_t id=in.hash(child,0,64),local=id/world;
  if(id>=total||id%world!=owner||local>=owner_len){atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));continue;}
  auto* word=reinterpret_cast<unsigned long long*>(&bits[local/64]);unsigned long long mask=1ull<<(local%64);
  if(atomicOr(word,mask)&mask)continue;
  bool reserved=false;
  if(global_count){
   cuda::atomic_ref<uint64_t,cuda::thread_scope_device> n(*global_count);uint64_t v=n.load(cuda::memory_order_relaxed);
   while(v<global_limit)if(n.compare_exchange_weak(v,v+1,cuda::memory_order_acq_rel,cuda::memory_order_relaxed)){reserved=true;break;}
   if(!reserved){atomicAnd(word,~mask);atomicOr(error,uint32_t(MGBFS_SEGMENTED_CAPACITY_LIMIT));continue;}
  }
  cuda::atomic_ref<uint64_t,cuda::thread_scope_device> accepted_counter(*accepted);
  uint64_t row=accepted_counter.load(cuda::memory_order_relaxed);
  while(row<active.capacity&&!accepted_counter.compare_exchange_weak(row,row+1,cuda::memory_order_relaxed,cuda::memory_order_relaxed)){}
  if(row>=active.capacity){
   if(reserved){cuda::atomic_ref<uint64_t,cuda::thread_scope_device> n(*global_count);n.fetch_sub(1,cuda::memory_order_acq_rel);}
   spill[atomicAdd(spill_count,1u)]=child;atomicAnd(word,~mask);continue;
  }
  uint64_t stride=uint64_t(active.capacity)*active.banks,offset=uint64_t(bank)*active.capacity+row;
  auto* out=static_cast<T*>(active.states);
  for(uint32_t e=0;e<in.elements;e++)out[uint64_t(e)*stride+offset]=static_cast<T>(in.value(child,e));
 }
}

// One owner lane processes this shard. Within that lane, a warp combines
// claims to each bitmap word, reserves rows once, and keeps every unadmitted
// winning origin in spill until the caller has drained/retried the wave.
__device__ inline uint32_t mgbfs_bitmap_group_reserve(uint64_t* counter,uint64_t limit,uint32_t wanted,uint64_t* base){
 cuda::atomic_ref<uint64_t,cuda::thread_scope_device> total(*counter);
 uint64_t old=total.load(cuda::memory_order_relaxed);
 for(;;){
  if(old>=limit||wanted==0){*base=old;return 0;}
  uint64_t available=limit-old;
  uint32_t grant=available<uint64_t(wanted)?uint32_t(available):wanted;
  if(total.compare_exchange_weak(old,old+grant,cuda::memory_order_acq_rel,cuda::memory_order_relaxed)){*base=old;return grant;}
 }
}
template<class T,class Input>
__global__ void bitmap_accept_grouped(Input in,uint32_t world,uint32_t owner,uint64_t total,uint64_t owner_len,
 uint64_t* bits,MgbfsSegment active,uint32_t bank,uint64_t* accepted,const uint32_t* selection,uint32_t selected,
 uint32_t* spill,uint32_t* spill_count,uint32_t* error,uint64_t* global_count,uint64_t global_limit){
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<selected;i+=blockDim.x*gridDim.x){
  uint32_t child=selection?selection[i]:in.child(i);if(!in.valid(child,error))continue;
  if(cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(*error).load(cuda::memory_order_relaxed)&~uint32_t(MGBFS_SEGMENTED_CAP_RETRY))return;
  uint64_t id=in.hash(child,0,64),local=id/world;
  if(id>=total||id%world!=owner||local>=owner_len){atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));continue;}
  const unsigned live=__activemask(),lane=threadIdx.x&31u;
  const unsigned same_word=__match_any_sync(live,local>>6);
  const unsigned same_id=__match_any_sync(live,local);
  const bool representative=lane==unsigned(__ffs(int(same_id))-1);
  unsigned long long mask=1ull<<(local&63u),merged=0;
  for(unsigned pending=same_word;pending;pending&=pending-1){
   int source=__ffs(int(pending))-1;
   merged|=__shfl_sync(same_word,mask,source);
  }
  const int leader=__ffs(int(same_word))-1;
  auto* word=reinterpret_cast<unsigned long long*>(&bits[local>>6]);
  unsigned long long old_word=0;
  if(int(lane)==leader)old_word=atomicOr(word,merged);
  old_word=__shfl_sync(same_word,old_word,leader);
  const bool winner=representative&&!(old_word&mask);
  const unsigned winners=__ballot_sync(live,winner);
  if(!winners)continue;
  const int winner_leader=__ffs(int(winners))-1;
  const unsigned before=(uint32_t(1)<<lane)-1u;
  const uint32_t ordinal=uint32_t(__popc(winners&before)),wanted=uint32_t(__popc(winners));
  uint64_t base=0;uint32_t grant=0;
  if(int(lane)==winner_leader){
   uint32_t rank_grant=wanted;
   if(global_count){uint64_t ignored=0;rank_grant=mgbfs_bitmap_group_reserve(global_count,global_limit,wanted,&ignored);if(rank_grant<wanted)atomicOr(error,uint32_t(MGBFS_SEGMENTED_CAP_RETRY));}
   grant=mgbfs_bitmap_group_reserve(accepted,active.capacity,rank_grant,&base);
   if(global_count&&rank_grant>grant){
    cuda::atomic_ref<uint64_t,cuda::thread_scope_device> rank_count(*global_count);
    rank_count.fetch_sub(uint64_t(rank_grant-grant),cuda::memory_order_acq_rel);
   }
  }
  base=__shfl_sync(live,base,winner_leader);grant=__shfl_sync(live,grant,winner_leader);
  const bool deferred=winner&&ordinal>=grant;
  const unsigned deferred_lanes=__ballot_sync(live,deferred);
  uint32_t spill_base=0;
  if(deferred_lanes){
   const int spill_leader=__ffs(int(deferred_lanes))-1;
   if(int(lane)==spill_leader)spill_base=atomicAdd(spill_count,uint32_t(__popc(deferred_lanes)));
   spill_base=__shfl_sync(live,spill_base,spill_leader);
  }
  if(deferred){
   spill[spill_base+uint32_t(__popc(deferred_lanes&before))]=child;
   atomicAnd(word,~mask);
  }else if(winner){
   uint64_t row=base+ordinal,stride=uint64_t(active.capacity)*active.banks,offset=uint64_t(bank)*active.capacity+row;
   auto* out=static_cast<T*>(active.states);
   for(uint32_t e=0;e<in.elements;e++)out[uint64_t(e)*stride+offset]=static_cast<T>(in.value(child,e));
  }
 }
}

template<class T,bool Packed,bool Grouped=false>
int launch_accept(uint32_t elements,const T* local,const T* remote,const GenericRouteRecord* lm,const GenericRouteRecord* rm,
 const uint32_t* lc,const uint32_t* rc,uint32_t rank,uint32_t world,uint32_t owner,uint64_t total,uint64_t owner_len,uint64_t* bits,
 uint32_t shard,uint32_t shards,uint32_t q,uint32_t visible,const uint32_t* selection,uint32_t selected,MgbfsSegment active,uint32_t bank,
 uint64_t* accepted,uint32_t* spill,uint32_t* spill_count,uint32_t* error,uint64_t* global_count,uint64_t limit,void* stream){
 uint64_t bound=uint64_t(world)*q;
 if(!elements||(Packed&&elements>24)||!world||world>128||rank>=world||owner>=world||!total||!bits||!shards||shard>=shards||!q||
  bound>=0x80000000ULL||visible>q||selected>bound||(!selection&&selected!=uint64_t(world)*visible)||!lm||!rm||!lc||!rc||(!Packed&&(!local||!remote))||
  !active.states||!(active.banks==2||active.banks==3)||!active.capacity||active.capacity>(uint64_t(1)<<32)||bank>=active.banks||!accepted||!spill||!spill_count||!error)return int(cudaErrorInvalidValue);
 if(!selected)return 0;
 MgbfsBoundedIncomingAction<T,Packed> in{{elements,uint32_t(bound),q,1,rank,world,shard,shards,local,remote,lm,rm,lc,rc,nullptr},visible};
 uint32_t blocks=(selected+255)/256;if(blocks>65535)blocks=65535;
 if constexpr(Grouped)bitmap_accept_grouped<T><<<blocks,256,0,static_cast<cudaStream_t>(stream)>>>(in,world,owner,total,owner_len,bits,active,bank,accepted,selection,selected,spill,spill_count,error,global_count,limit);
 else bitmap_accept<T><<<blocks,256,0,static_cast<cudaStream_t>(stream)>>>(in,world,owner,total,owner_len,bits,active,bank,accepted,selection,selected,spill,spill_count,error,global_count,limit);
 return int(cudaGetLastError());
}
extern "C" int mgbfs_bitmap_accept(uint32_t state_bytes,uint32_t packed,uint32_t elements,const void* local,const void* remote,
 const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,uint32_t rank,uint32_t world,
 uint32_t owner,uint64_t total,uint64_t owner_len,uint64_t* bits,uint32_t shard,uint32_t shards,uint32_t q,uint32_t visible,
 const uint32_t* selection,uint32_t selected,MgbfsSegment active,uint32_t bank,uint64_t* accepted,uint32_t* spill,
 uint32_t* spill_count,uint32_t* error,uint64_t* global_count,uint64_t limit,void* stream){
 if(state_bytes==1&&packed)return launch_accept<uint8_t,true>(elements,static_cast<const uint8_t*>(local),static_cast<const uint8_t*>(remote),lm,rm,lc,rc,rank,world,owner,total,owner_len,bits,shard,shards,q,visible,selection,selected,active,bank,accepted,spill,spill_count,error,global_count,limit,stream);
 if(state_bytes==1)return launch_accept<uint8_t,false>(elements,static_cast<const uint8_t*>(local),static_cast<const uint8_t*>(remote),lm,rm,lc,rc,rank,world,owner,total,owner_len,bits,shard,shards,q,visible,selection,selected,active,bank,accepted,spill,spill_count,error,global_count,limit,stream);
 if(state_bytes==8&&!packed)return launch_accept<int64_t,false>(elements,static_cast<const int64_t*>(local),static_cast<const int64_t*>(remote),lm,rm,lc,rc,rank,world,owner,total,owner_len,bits,shard,shards,q,visible,selection,selected,active,bank,accepted,spill,spill_count,error,global_count,limit,stream);
 return int(cudaErrorInvalidValue);
}

extern "C" int mgbfs_bitmap_accept_grouped(uint32_t state_bytes,uint32_t packed,uint32_t elements,const void* local,const void* remote,
 const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,uint32_t rank,uint32_t world,
 uint32_t owner,uint64_t total,uint64_t owner_len,uint64_t* bits,uint32_t shard,uint32_t shards,uint32_t q,uint32_t visible,
 const uint32_t* selection,uint32_t selected,MgbfsSegment active,uint32_t bank,uint64_t* accepted,uint32_t* spill,
 uint32_t* spill_count,uint32_t* error,uint64_t* global_count,uint64_t limit,void* stream){
 if(state_bytes==1&&packed)return launch_accept<uint8_t,true,true>(elements,static_cast<const uint8_t*>(local),static_cast<const uint8_t*>(remote),lm,rm,lc,rc,rank,world,owner,total,owner_len,bits,shard,shards,q,visible,selection,selected,active,bank,accepted,spill,spill_count,error,global_count,limit,stream);
 if(state_bytes==1)return launch_accept<uint8_t,false,true>(elements,static_cast<const uint8_t*>(local),static_cast<const uint8_t*>(remote),lm,rm,lc,rc,rank,world,owner,total,owner_len,bits,shard,shards,q,visible,selection,selected,active,bank,accepted,spill,spill_count,error,global_count,limit,stream);
 if(state_bytes==8&&!packed)return launch_accept<int64_t,false,true>(elements,static_cast<const int64_t*>(local),static_cast<const int64_t*>(remote),lm,rm,lc,rc,rank,world,owner,total,owner_len,bits,shard,shards,q,visible,selection,selected,active,bank,accepted,spill,spill_count,error,global_count,limit,stream);
 return int(cudaErrorInvalidValue);
}

// Explicit entry for the separately verified warp-grouped bitmap producer.
extern "C" int mgbfs_bitmap_route_grouped(uint32_t state_bytes,BitmapDeviceDomain domain,uint32_t elements,uint32_t generators,const void* parents,
 uint32_t count,uint64_t stride,const uint32_t* permutations,uint32_t world,uint32_t source,uint32_t shards,uint32_t capacity,uint64_t begin,
 const uint32_t* map,GenericRouteRecord* output,uint32_t* counts,uint32_t* error,void* stream){
 if(state_bytes==1){
  if(elements<=24)return launch_route<uint8_t,true,true>(domain,elements,generators,static_cast<const uint8_t*>(parents),count,stride,permutations,world,source,shards,capacity,begin,map,output,counts,error,stream);
  return launch_route<uint8_t,false,true>(domain,elements,generators,static_cast<const uint8_t*>(parents),count,stride,permutations,world,source,shards,capacity,begin,map,output,counts,error,stream);
 }
 if(state_bytes==8)return launch_route<int64_t,false,true>(domain,elements,generators,static_cast<const int64_t*>(parents),count,stride,permutations,world,source,shards,capacity,begin,map,output,counts,error,stream);
 return int(cudaErrorInvalidValue);
}

// Strategy ABI covers grouped route and grouped owner symbols together.
extern "C" uint32_t mgbfs_segmented_strategy_abi_version(){return 1;}
