#include "generic_sorted_native.h"
#include "generic_parent_action.cuh"
#include "generic_sorted_origin_exact.cuh"
#include "generic_sorted_accept.cuh"
#include <type_traits>
static_assert(sizeof(void*)==8,"sorted native ABI requires 64-bit host");
static_assert(sizeof(GenericSortedNativeInput)==176,"input ABI");
static_assert(sizeof(GenericSortedNativeWorkspace)==80,"workspace ABI");
static_assert(sizeof(GenericSortedNativeDestination)==112,"destination ABI");
static bool geometry(const GenericSortedNativeInput& p,bool launch){
 if(!p.elements||!p.world||p.world>128||p.rank>=p.world||!p.shards||p.shards>4096||p.shard>=p.shards||!p.queue_capacity||
  uint64_t(p.world)*p.shards*p.queue_capacity>=0x7fffffffull||!(p.state_bytes==1||p.state_bytes==8)||p.transport>2||p.hash_bits>64||p.reserved)return false;
 if(p.transport==1&&(p.state_bytes!=1||p.elements>24))return false;
 if(p.transport==2&&(p.kind>1||!p.rows||!p.cols||!p.generators||!p.parent_stride||p.chunk>p.parent_stride||
  (p.kind==0&&(p.elements!=p.rows||p.cols!=1))||(p.kind==1&&uint64_t(p.rows)*p.cols!=p.elements)))return false;
 if(!launch)return true;
 if(!p.local_meta||!p.remote_meta||!p.local_counts||!p.remote_counts)return false;
 if(p.transport==0&&(!p.local||!p.remote))return false;
 if(p.transport==2&&(!p.parents||!p.cursors||!p.frontiers||(p.kind==0&&!p.permutations)||(p.kind==1&&(!p.matrices||!p.moduli))))return false;
 return true;
}
template<class State,class F> static cudaError_t dispatch_state(const GenericSortedNativeInput& p,F& f){
 IncomingAllAction<State,false,true> incoming{p.elements,p.world*p.queue_capacity,p.queue_capacity,1,p.rank,p.world,p.shard,p.shards,
  static_cast<const State*>(p.local),static_cast<const State*>(p.remote),static_cast<const GenericRouteRecord*>(p.local_meta),static_cast<const GenericRouteRecord*>(p.remote_meta),
  static_cast<const uint32_t*>(p.local_counts),static_cast<const uint32_t*>(p.remote_counts),nullptr};
 if(p.transport==0)return f(incoming);
 if(p.transport==1){if constexpr(sizeof(State)==1){IncomingAllAction<State,true,true> packed{p.elements,p.world*p.queue_capacity,p.queue_capacity,1,p.rank,p.world,p.shard,p.shards,
  nullptr,nullptr,static_cast<const GenericRouteRecord*>(p.local_meta),static_cast<const GenericRouteRecord*>(p.remote_meta),static_cast<const uint32_t*>(p.local_counts),static_cast<const uint32_t*>(p.remote_counts),nullptr};return f(packed);}else return cudaErrorInvalidValue;}
 ActionT<State> graph{p.kind,p.elements,p.rows,p.cols,p.generators,p.chunk,p.parent_stride,nullptr,
  static_cast<const uint32_t*>(p.permutations),static_cast<const int64_t*>(p.matrices),static_cast<const uint32_t*>(p.moduli)};
 ParentOriginAction<State> action{incoming,graph,static_cast<const State*>(p.parents),static_cast<const uint64_t*>(p.cursors),static_cast<const uint32_t*>(p.frontiers),p.parent_stride,p.chunk,p.begin};return f(action);
}
template<class F>static cudaError_t dispatch(const GenericSortedNativeInput& p,F& f){return p.state_bytes==1?dispatch_state<uint8_t>(p,f):dispatch_state<int64_t>(p,f);}
extern "C" int mgbfs_generic_sorted_native_shape(const GenericSortedNativeInput* p,uint64_t* total,uint64_t* temporary,uint64_t* n,void* stream){
 if(!p||!total||!temporary||!n||!geometry(*p,false))return int(cudaErrorInvalidValue);
 *total=*temporary=*n=0;
 auto f=[&](auto action){GenericSortedOriginShape shape{};auto status=generic_sorted_origin_shape(action,p->world*p->queue_capacity,&shape,static_cast<cudaStream_t>(stream));
  if(status==cudaSuccess){*total=shape.aligned_workspace_bytes;*temporary=shape.shared_temporary_bytes;*n=p->world*p->queue_capacity;}return status;};return int(dispatch(*p,f));
}
extern "C" int mgbfs_generic_sorted_native_origin(const GenericSortedNativeInput* p,const GenericSortedNativeWorkspace* w,void* error,void* stream){
 if(!p||!w||!error||!geometry(*p,true))return int(cudaErrorInvalidValue);
 auto f=[&](auto action){return generic_sorted_origin_exact(action,p->world*p->queue_capacity,p->seed,p->hash_bits,
  static_cast<uint64_t*>(w->hashes),static_cast<uint32_t*>(w->origins),static_cast<uint32_t*>(w->sorted),static_cast<uint32_t*>(w->flags),static_cast<uint32_t*>(w->prefix),
  static_cast<uint64_t*>(w->unique_hashes),static_cast<uint32_t*>(w->unique_origins),static_cast<uint32_t*>(w->unique_count),w->temporary,size_t(w->temporary_bytes),static_cast<uint32_t*>(error),static_cast<cudaStream_t>(stream));};return int(dispatch(*p,f));
}
extern "C" int mgbfs_generic_sorted_native_accept(const GenericSortedNativeInput* p,const GenericSortedNativeWorkspace* w,const GenericSortedNativeDestination* d,void* stream){
 if(!p||!w||!d||!d->pool||d->reserved||!geometry(*p,true))return int(cudaErrorInvalidValue);
 auto f=[&](auto action){
  using State=std::remove_cv_t<std::remove_pointer_t<decltype(action.local)>>;
  return generic_sorted_accept(action,static_cast<const uint64_t*>(w->unique_hashes),static_cast<const uint32_t*>(w->unique_origins),static_cast<const uint32_t*>(w->unique_count),p->world*p->queue_capacity,
   static_cast<const GenericSortedHistorySnapshot*>(d->snapshots),d->snapshot_count,*static_cast<const GenericSortedRunPool*>(d->pool),static_cast<uint64_t*>(d->run_hashes),static_cast<uint32_t*>(d->run_rows),static_cast<State*>(d->arena),d->stride,d->base,d->capacity,
   static_cast<uint32_t*>(d->row_count),static_cast<uint32_t*>(d->frontier_count),d->rolling,static_cast<uint32_t*>(d->future),static_cast<uint32_t*>(w->flags),static_cast<uint32_t*>(w->prefix),w->temporary,size_t(w->temporary_bytes),static_cast<GenericSortedAcceptReservation*>(d->reservation),static_cast<GenericSortedRunCarry*>(d->carry),static_cast<uint32_t*>(d->error),static_cast<cudaStream_t>(stream));};return int(dispatch(*p,f));
}
extern "C" int mgbfs_generic_sorted_native_metadata_shape(uint64_t* out,uint32_t count){if(!out||count!=5)return int(cudaErrorInvalidValue);out[0]=sizeof(GenericSortedRunDescriptor);out[1]=sizeof(GenericSortedRunTiers);out[2]=sizeof(GenericSortedRunCarry);out[3]=sizeof(GenericSortedHistorySnapshot);out[4]=sizeof(GenericSortedAcceptReservation);return 0;}
__global__ void snapshot_acquire_native(GenericSortedRunPool pool,const GenericSortedRunTiers* tiers,GenericSortedHistorySnapshot* snapshot,uint64_t* hashes,uint32_t* rows,uint32_t* error){if(!blockIdx.x&&!threadIdx.x)generic_sorted_history_snapshot_acquire(pool,tiers,snapshot,hashes,rows,error);}
__global__ void snapshot_release_native(GenericSortedRunPool pool,GenericSortedHistorySnapshot* snapshot,uint32_t* error){if(!blockIdx.x&&!threadIdx.x)generic_sorted_history_snapshot_release(pool,snapshot,error);}
__global__ void retire_native(GenericSortedRunPool pool,GenericSortedRunTiers* tiers,uint32_t* error){if(!blockIdx.x&&!threadIdx.x){for(uint32_t c=0;c<32;++c)if(tiers->present&(1u<<c))generic_sorted_run_release(pool,tiers->tokens[c],true,error);tiers->present=0;}}
__global__ void seed_native(GenericSortedRunPool pool,GenericSortedRunTiers* tiers,uint64_t* hashes,uint32_t* rows,uint64_t hash,uint32_t row,uint32_t* error){
 if(blockIdx.x||threadIdx.x)return;if(tiers->present){atomicOr(error,512u);return;}GenericSortedRunToken token{};
 if(generic_sorted_run_allocate(pool,0,0,&token,error)!=SORTED_RUN_ACQUIRED){atomicOr(error,2u);return;}
 uint64_t offset=uint64_t(token.slot)*pool.page_entries;hashes[offset]=hash;rows[offset]=row;
 if(!generic_sorted_run_publish(pool,token,1,error)){generic_sorted_run_release(pool,token,true,error);return;}tiers->tokens[0]=token;__threadfence();tiers->present=1;
}
extern "C" int mgbfs_generic_sorted_native_snapshot_acquire(const void* pool,const void* tiers,void* snapshot,void* hashes,void* rows,void* error,void* stream){if(!pool||!tiers||!snapshot||!hashes||!rows||!error)return int(cudaErrorInvalidValue);snapshot_acquire_native<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(*static_cast<const GenericSortedRunPool*>(pool),static_cast<const GenericSortedRunTiers*>(tiers),static_cast<GenericSortedHistorySnapshot*>(snapshot),static_cast<uint64_t*>(hashes),static_cast<uint32_t*>(rows),static_cast<uint32_t*>(error));return int(cudaGetLastError());}
extern "C" int mgbfs_generic_sorted_native_snapshot_release(const void* pool,void* snapshot,void* error,void* stream){if(!pool||!snapshot||!error)return int(cudaErrorInvalidValue);snapshot_release_native<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(*static_cast<const GenericSortedRunPool*>(pool),static_cast<GenericSortedHistorySnapshot*>(snapshot),static_cast<uint32_t*>(error));return int(cudaGetLastError());}
extern "C" int mgbfs_generic_sorted_native_retire(const void* pool,void* tiers,void* error,void* stream){if(!pool||!tiers||!error)return int(cudaErrorInvalidValue);retire_native<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(*static_cast<const GenericSortedRunPool*>(pool),static_cast<GenericSortedRunTiers*>(tiers),static_cast<uint32_t*>(error));return int(cudaGetLastError());}
extern "C" int mgbfs_generic_sorted_native_seed(const void* pool,void* tiers,void* hashes,void* rows,uint64_t hash,uint32_t row,void* error,void* stream){if(!pool||!tiers||!hashes||!rows||!error)return int(cudaErrorInvalidValue);seed_native<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(*static_cast<const GenericSortedRunPool*>(pool),static_cast<GenericSortedRunTiers*>(tiers),static_cast<uint64_t*>(hashes),static_cast<uint32_t*>(rows),hash,row,static_cast<uint32_t*>(error));return int(cudaGetLastError());}
