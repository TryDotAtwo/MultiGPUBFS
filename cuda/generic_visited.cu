#include "generic_visited.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cuda/atomic>

static constexpr uint64_t EMPTY=~uint64_t(0);
#include "generic_action.cuh"
#include "generic_route.h"
template<class State,bool Shared=false> __global__ void seed_table(uint32_t elements,const State* states,uint32_t stride,uint32_t count,uint64_t* slots,
 uint64_t capacity,uint64_t seed,uint32_t bits,uint32_t* error){
 for(uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;row<count;row+=blockDim.x*gridDim.x){
  uint64_t h=seed;for(uint32_t e=0;e<elements;e++)h=mix64(h^uint64_t(states[uint64_t(e)*stride+row])^uint64_t(e));h=finish_hash(h,bits);
  uint64_t token=(h&0x7fffffff00000000ULL)|row;uint64_t p=Shared?mgbfs_shared_table_bucket(h,capacity):uint64_t(h&(capacity-1));bool done=false;
  for(uint64_t i=0;i<capacity;i++,p=(p+1)&(capacity-1))if(atomicCAS(reinterpret_cast<unsigned long long*>(slots+p),EMPTY,token)==EMPTY){done=true;break;}
  if(!done)atomicOr(error,1u);
 }
}
template<class Candidate,class State,bool StoreParent=false>
__global__ void accept_candidates(Candidate action,uint64_t* slots,uint64_t slot_capacity,State* visited,uint32_t visited_capacity,
 uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,uint32_t* future_count,uint64_t seed,uint32_t bits,uint32_t* error,GenericPathParent* lineage,uint32_t path_generators){
 const uint32_t children=action.count*action.generators;
 for(uint32_t iteration=blockIdx.x*blockDim.x+threadIdx.x;iteration<children;iteration+=blockDim.x*gridDim.x){
  const uint32_t child=action.child(iteration);
  if(!action.valid(child,error))continue;
  if(cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(*error).load(cuda::memory_order_relaxed))return;
  const uint64_t hash=action.hash(child,seed,bits),prefix=hash&0x7fffffff00000000ULL;
  const uint64_t pending=prefix|0x8000000000000000ULL|child;uint64_t slot=action.bucket(hash,slot_capacity);bool done=false;
  for(uint64_t probe=0;probe<slot_capacity;probe++,slot=(slot+1)&(slot_capacity-1)){
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
    if constexpr(StoreParent){
     const auto& route=action.route(child);
     if(route.parent>=UINT32_MAX||route.source!=action.source(child)||route.generator>=path_generators){
      atomicOr(error,128u);return;
     }
     lineage[row]={uint32_t(route.parent),route.source*path_generators+route.generator};
    }
    // Release/acquire publishes the full canonical payload, including on GPUs
    // whose state loads use L1. No reader spins on an unpublished payload.
    published.store(prefix|row,cuda::memory_order_release);
    const uint32_t position=atomicAdd(future_count,1u);
    if(position>=future_capacity){atomicOr(error,4u);done=true;break;}
    future[position]=row;done=true;break;
   }
   if((observed&0x7fffffff00000000ULL)!=prefix)continue;
   const uint32_t ref=uint32_t(observed),other=ref;const bool transient=observed&0x8000000000000000ULL;
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
extern "C" int mgbfs_generic_seed_i64_wide(uint32_t elements,const int64_t* states,uint32_t stride,uint32_t count,uint64_t* slots,
 uint64_t capacity,uint64_t seed,uint32_t bits,uint32_t* error,void* stream){
 if(!elements||count>stride||!capacity||(capacity&(capacity-1))||bits>64||!error||!slots||(!states&&count)||count>=0x80000000U)return int(cudaErrorInvalidValue);
 if(!count)return 0;seed_table<<<grid(count),256,0,static_cast<cudaStream_t>(stream)>>>(elements,states,stride,count,slots,capacity,seed,bits,error);return int(cudaGetLastError());
}
extern "C" int mgbfs_generic_expand_i64_wide(uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const int64_t* parents,uint32_t count,uint32_t stride,const uint32_t* permutations,const int64_t* matrices,const uint32_t* moduli,
 uint64_t* slots,uint64_t slot_capacity,int64_t* visited,uint32_t visited_capacity,uint32_t* visited_count,
 uint32_t* future,uint32_t future_capacity,uint32_t* future_count,uint64_t seed,uint32_t bits,uint32_t* error,void* stream){
 if(kind>1||!elements||!n||!m||!generators||count>stride||uint64_t(count)*generators>=0x7fffffffULL||
  !slot_capacity||(slot_capacity&(slot_capacity-1))||!visited_capacity||visited_capacity>=0x80000000U||bits>64||
  !visited_count||!future_count||!error||!slots||!visited||!future||!parents||
  (kind==0&&(!permutations||elements!=n||m!=1))||(kind==1&&(!matrices||!moduli||uint64_t(n)*m!=elements)))return int(cudaErrorInvalidValue);
 if(!count)return 0;
 Action action{kind,elements,n,m,generators,count,stride,parents,permutations,matrices,moduli};
 accept_candidates<<<grid(uint64_t(count)*generators),256,0,static_cast<cudaStream_t>(stream)>>>(action,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,seed,bits,error,nullptr,0);
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
 __device__ uint64_t bucket(uint64_t h,uint64_t capacity)const{return h&(capacity-1);}
 uint32_t elements,count,stride,generators=1;const State* states;
 const GenericRouteRecord* metadata;const uint32_t* received;
 __device__ bool valid(uint32_t row,uint32_t* error)const{
  uint32_t actual=*received;if(actual>count){atomicOr(error,32u);return false;}return row<actual;
 }
 __device__ int64_t value(uint32_t row,uint32_t element)const{return states[uint64_t(element)*stride+row];}
 __device__ uint64_t hash(uint32_t row,uint64_t,uint32_t)const{return metadata[row].hash;}
};
extern "C" int mgbfs_generic_accept_i64_wide(uint32_t elements,const int64_t* incoming,uint32_t stride,
 const GenericRouteRecord* metadata,const uint32_t* received,uint32_t bound,uint64_t* slots,
 uint64_t slot_capacity,int64_t* visited,uint32_t visited_capacity,uint32_t* visited_count,
 uint32_t* future,uint32_t future_capacity,uint32_t* future_count,uint32_t* error,void* stream){
 if(!elements||!bound||bound>stride||bound>=0x7fffffffU||!slot_capacity||(slot_capacity&(slot_capacity-1))||
  !visited_capacity||visited_capacity>=0x80000000U||!future_capacity||!incoming||!metadata||!received||
  !slots||!visited||!visited_count||!future||!future_count||!error)return int(cudaErrorInvalidValue);
 IncomingActionT<int64_t> action{elements,bound,stride,1,incoming,metadata,received};
 accept_candidates<<<grid(bound),256,0,static_cast<cudaStream_t>(stream)>>>(action,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,0,64,error,nullptr,0);
 return int(cudaGetLastError());
}

// Compact permutation payload ABI shares the exact same kernel logic.
extern "C" int mgbfs_generic_seed_u8_wide(uint32_t elements,const uint8_t* states,uint32_t stride,uint32_t count,uint64_t* slots,
 uint64_t capacity,uint64_t seed,uint32_t bits,uint32_t* error,void* stream){
 if(!elements||count>stride||!capacity||(capacity&(capacity-1))||bits>64||!error||!slots||(!states&&count)||count>=0x80000000U)return int(cudaErrorInvalidValue);
 if(!count)return 0;seed_table<<<grid(count),256,0,static_cast<cudaStream_t>(stream)>>>(elements,states,stride,count,slots,capacity,seed,bits,error);return int(cudaGetLastError());
}
extern "C" int mgbfs_generic_expand_u8_wide(uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const uint8_t* parents,uint32_t count,uint32_t stride,const uint32_t* permutations,const int64_t* matrices,const uint32_t* moduli,
 uint64_t* slots,uint64_t slot_capacity,uint8_t* visited,uint32_t visited_capacity,uint32_t* visited_count,
 uint32_t* future,uint32_t future_capacity,uint32_t* future_count,uint64_t seed,uint32_t bits,uint32_t* error,void* stream){
 if(kind>0||!elements||!n||!m||!generators||count>stride||uint64_t(count)*generators>=0x7fffffffULL||
  !slot_capacity||(slot_capacity&(slot_capacity-1))||!visited_capacity||visited_capacity>=0x80000000U||bits>64||
  !visited_count||!future_count||!error||!slots||!visited||!future||!parents||
  (kind==0&&(!permutations||elements!=n||m!=1))||(kind==1&&(!matrices||!moduli||uint64_t(n)*m!=elements)))return int(cudaErrorInvalidValue);
 if(!count)return 0;
 ActionT<uint8_t> action{kind,elements,n,m,generators,count,stride,parents,permutations,matrices,moduli};
 accept_candidates<<<grid(uint64_t(count)*generators),256,0,static_cast<cudaStream_t>(stream)>>>(action,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,seed,bits,error,nullptr,0);
 return int(cudaGetLastError());
}
extern "C" int mgbfs_generic_gather_u8(uint32_t elements,const uint8_t* source,uint32_t stride,const uint32_t* indices,
 uint32_t count,uint8_t* output,uint32_t output_stride,void* stream){
 if(!elements||count>output_stride||(!source&&count)||(!indices&&count)||(!output&&count))return int(cudaErrorInvalidValue);
 if(!count)return 0;gather<<<grid(uint64_t(elements)*count),256,0,static_cast<cudaStream_t>(stream)>>>(elements,source,stride,indices,count,output,output_stride);return int(cudaGetLastError());
}
extern "C" int mgbfs_generic_accept_u8_wide(uint32_t elements,const uint8_t* incoming,uint32_t stride,
 const GenericRouteRecord* metadata,const uint32_t* received,uint32_t bound,uint64_t* slots,
 uint64_t slot_capacity,uint8_t* visited,uint32_t visited_capacity,uint32_t* visited_count,
 uint32_t* future,uint32_t future_capacity,uint32_t* future_count,uint32_t* error,void* stream){
 if(!elements||!bound||bound>stride||bound>=0x7fffffffU||!slot_capacity||(slot_capacity&(slot_capacity-1))||
  !visited_capacity||visited_capacity>=0x80000000U||!future_capacity||!incoming||!metadata||!received||
  !slots||!visited||!visited_count||!future||!future_count||!error)return int(cudaErrorInvalidValue);
 IncomingActionT<uint8_t> action{elements,bound,stride,1,incoming,metadata,received};
 accept_candidates<<<grid(bound),256,0,static_cast<cudaStream_t>(stream)>>>(action,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,0,64,error,nullptr,0);
 return int(cudaGetLastError());
}

// All source inboxes for one shard share one exclusive owner launch. Pending
// origins identify source+row in immutable banks, so exact equality remains
// independent of thread scheduling and full hash collisions.
#include "generic_incoming_action.cuh"

template<class State,bool Packed=false,bool Shared=false,bool StoreParent=false> int accept_all(uint32_t elements,const State* local,const State* remote,
 const GenericRouteRecord* local_meta,const GenericRouteRecord* remote_meta,const uint32_t* local_counts,const uint32_t* remote_counts,
 uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t stride,uint64_t* slots,uint64_t slot_capacity,
 State* visited,uint32_t visited_capacity,uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,uint32_t* future_count,uint32_t* error,void* stream,GenericPathParent* lineage=nullptr,uint32_t path_generators=0){
 if((Shared&&uint64_t(world)*shards*stride>=0x7fffffffULL)||(Packed&&elements>24)||!elements||!world||world>128||rank>=world||!shards||shards>4096||shard>=shards||!stride||uint64_t(world)*stride>=0x7fffffffULL||
  !slot_capacity||(slot_capacity&(slot_capacity-1))||!visited_capacity||visited_capacity>=0x80000000U||!future_capacity||
  !local||!remote||!local_meta||!remote_meta||!local_counts||!remote_counts||!slots||!visited||!visited_count||!future||!future_count||!error)return int(cudaErrorInvalidValue);
 IncomingAllAction<State,Packed,Shared> action{elements,world*stride,stride,1,rank,world,shard,shards,local,remote,local_meta,remote_meta,local_counts,remote_counts};
 accept_candidates<decltype(action),State,StoreParent><<<grid(uint64_t(world)*stride),256,0,static_cast<cudaStream_t>(stream)>>>(action,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,0,64,error,lineage,path_generators);return int(cudaGetLastError());
}
extern "C" int mgbfs_generic_accept_path_parents_v2(
 uint32_t state_bytes,uint32_t elements,uint32_t generators,const void* local,const void* remote,
 const GenericRouteRecord* local_meta,const GenericRouteRecord* remote_meta,
 const uint32_t* local_counts,const uint32_t* remote_counts,
 uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t stride,
 uint64_t* slots,uint64_t slot_capacity,void* visited,uint32_t visited_capacity,
 uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,uint32_t* future_count,
 GenericPathParent* lineage,uint32_t* error,void* stream){
 if(!lineage||!generators||uint64_t(world)*generators>UINT32_MAX)return int(cudaErrorInvalidValue);
 if(state_bytes==1)return accept_all<uint8_t,false,true,true>(elements,
  static_cast<const uint8_t*>(local),static_cast<const uint8_t*>(remote),
  local_meta,remote_meta,local_counts,remote_counts,rank,world,shard,shards,stride,
  slots,slot_capacity,static_cast<uint8_t*>(visited),visited_capacity,visited_count,
  future,future_capacity,future_count,error,stream,lineage,generators);
 if(state_bytes==8)return accept_all<int64_t,false,true,true>(elements,
  static_cast<const int64_t*>(local),static_cast<const int64_t*>(remote),
  local_meta,remote_meta,local_counts,remote_counts,rank,world,shard,shards,stride,
  slots,slot_capacity,static_cast<int64_t*>(visited),visited_capacity,visited_count,
  future,future_capacity,future_count,error,stream,lineage,generators);
 return int(cudaErrorInvalidValue);
}

static_assert(sizeof(MgbfsCompactPathInputV1) == 40);
static_assert(offsetof(MgbfsCompactPathInputV1, states) == 24);
static_assert(sizeof(MgbfsCompactPathOwnerV1) == 72);
static_assert(offsetof(MgbfsCompactPathOwnerV1, slot_count) == 8);
static_assert(offsetof(MgbfsCompactPathOwnerV1, error) == 64);

template<class State>
struct CompactPathAction {
    uint32_t elements, count, generators = 1;
    uint32_t source_rank, shard, path_generators;
    const State* states;
    const GenericRouteRecord* records;
    bool store_parent;

    __device__ uint32_t child(uint32_t i) const { return i; }
    __device__ uint32_t origin_limit() const { return count; }
    __device__ uint64_t bucket(uint64_t h, uint64_t capacity) const {
        return mgbfs_shared_table_bucket(h, capacity);
    }
    __device__ uint32_t source(uint32_t) const { return source_rank; }
    __device__ const GenericRouteRecord& route(uint32_t row) const { return records[row]; }
    __device__ uint64_t hash(uint32_t row, uint64_t, uint32_t) const { return route(row).hash; }
    __device__ int64_t value(uint32_t row, uint32_t e) const {
        return states[uint64_t(e) * count + row];
    }
    __device__ bool valid(uint32_t row, uint32_t* error) const {
        const auto& r = route(row);
        if (r.source == source_rank && r.shard == shard && r.generator < path_generators
            && (!store_parent || r.parent < UINT32_MAX)) return true;
        atomicOr(error, 128u);
        return false;
    }
};

template<class State>
static CompactPathAction<State> compact_path_action(const MgbfsCompactPathInputV1& in,
                                                    bool store_parent) {
    return {in.elements, in.rows, 1, in.source, in.shard, in.generators,
            static_cast<const State*>(in.states), in.records, store_parent};
}

static bool valid_compact_path(const MgbfsCompactPathInputV1& in,
                               const MgbfsCompactPathOwnerV1& owner) {
    return (in.state_bytes == 1 || in.state_bytes == 8)
        && in.elements && in.generators && in.rows && in.rows < 0x7fffffffU
        && in.source < 128 && in.shard < 4096 && in.states && in.records
        && owner.capacity && owner.capacity < 0x80000000U && owner.slot_count
        && !(owner.slot_count & (owner.slot_count - 1))
        && owner.slot_count <= SIZE_MAX / sizeof(uint64_t)
        && uint64_t(in.elements) * in.rows <= SIZE_MAX / in.state_bytes
        && uint64_t(in.elements) * owner.capacity <= SIZE_MAX / in.state_bytes
        && owner.slots && owner.states && owner.visited_count && owner.error;
}

template<class State, bool StoreParent>
static void accept_compact_path(const MgbfsCompactPathInputV1& in,
                                const MgbfsCompactPathOwnerV1& owner, cudaStream_t stream) {
    const auto action = compact_path_action<State>(in, StoreParent);
    accept_candidates<CompactPathAction<State>, State, StoreParent>
        <<<grid(in.rows), 256, 0, stream>>>(
            action, owner.slots, owner.slot_count, static_cast<State*>(owner.states),
            owner.capacity, owner.visited_count, owner.future, owner.capacity,
            owner.future_count, 0, 64, owner.error, owner.lineage, in.generators);
}

extern "C" int mgbfs_generic_accept_compact_path_v1(
    const MgbfsCompactPathInputV1* input, const MgbfsCompactPathOwnerV1* output, void* raw_stream) {
    if (!input || !output || !valid_compact_path(*input, *output)
        || !output->future || !output->future_count
        || (output->lineage && (uint64_t(input->source) + 1) * input->generators > UINT32_MAX)) {
        return cudaErrorInvalidValue;
    }
    const auto stream = static_cast<cudaStream_t>(raw_stream);
    const auto& in = *input;
    const auto& owner = *output;
    if (in.state_bytes == 1) {
        if (owner.lineage) accept_compact_path<uint8_t, true>(in, owner, stream);
        else accept_compact_path<uint8_t, false>(in, owner, stream);
    } else {
        if (owner.lineage) accept_compact_path<int64_t, true>(in, owner, stream);
        else accept_compact_path<int64_t, false>(in, owner, stream);
    }
    return cudaGetLastError();
}
extern "C" int mgbfs_generic_accept_all_i64_wide(uint32_t elements,const int64_t* local,const int64_t* remote,const GenericRouteRecord* local_meta,const GenericRouteRecord* remote_meta,
 const uint32_t* local_counts,const uint32_t* remote_counts,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t stride,
 uint64_t* slots,uint64_t slot_capacity,int64_t* visited,uint32_t visited_capacity,uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,
 uint32_t* future_count,uint32_t* error,void* stream){return accept_all(elements,local,remote,local_meta,remote_meta,local_counts,remote_counts,rank,world,shard,shards,stride,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,error,stream);}
extern "C" int mgbfs_generic_accept_all_u8_wide(uint32_t elements,const uint8_t* local,const uint8_t* remote,const GenericRouteRecord* local_meta,const GenericRouteRecord* remote_meta,
 const uint32_t* local_counts,const uint32_t* remote_counts,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t stride,
 uint64_t* slots,uint64_t slot_capacity,uint8_t* visited,uint32_t visited_capacity,uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,
 uint32_t* future_count,uint32_t* error,void* stream){return accept_all(elements,local,remote,local_meta,remote_meta,local_counts,remote_counts,rank,world,shard,shards,stride,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,error,stream);}

extern "C" int mgbfs_generic_accept_all_packed_u8_wide(uint32_t elements,const uint8_t* local,const uint8_t* remote,const GenericRouteRecord* local_meta,const GenericRouteRecord* remote_meta,
 const uint32_t* local_counts,const uint32_t* remote_counts,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t stride,
 uint64_t* slots,uint64_t slot_capacity,uint8_t* visited,uint32_t visited_capacity,uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,
 uint32_t* future_count,uint32_t* error,void* stream){return accept_all<uint8_t,true>(elements,local,remote,local_meta,remote_meta,local_counts,remote_counts,rank,world,shard,shards,stride,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,error,stream);}

extern "C" int mgbfs_generic_accept_all_shared_i64_wide(uint32_t elements,const int64_t* local,const int64_t* remote,const GenericRouteRecord* local_meta,const GenericRouteRecord* remote_meta,
 const uint32_t* local_counts,const uint32_t* remote_counts,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t stride,
 uint64_t* slots,uint64_t slot_capacity,int64_t* visited,uint32_t visited_capacity,uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,
 uint32_t* future_count,uint32_t* error,void* stream){return accept_all<int64_t,false,true>(elements,local,remote,local_meta,remote_meta,local_counts,remote_counts,rank,world,shard,shards,stride,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,error,stream);}

extern "C" int mgbfs_generic_accept_all_shared_u8_wide(uint32_t elements,const uint8_t* local,const uint8_t* remote,const GenericRouteRecord* local_meta,const GenericRouteRecord* remote_meta,
 const uint32_t* local_counts,const uint32_t* remote_counts,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t stride,
 uint64_t* slots,uint64_t slot_capacity,uint8_t* visited,uint32_t visited_capacity,uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,
 uint32_t* future_count,uint32_t* error,void* stream){return accept_all<uint8_t,false,true>(elements,local,remote,local_meta,remote_meta,local_counts,remote_counts,rank,world,shard,shards,stride,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,error,stream);}

extern "C" int mgbfs_generic_accept_all_shared_packed_u8_wide(uint32_t elements,const uint8_t* local,const uint8_t* remote,const GenericRouteRecord* local_meta,const GenericRouteRecord* remote_meta,
 const uint32_t* local_counts,const uint32_t* remote_counts,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t stride,
 uint64_t* slots,uint64_t slot_capacity,uint8_t* visited,uint32_t visited_capacity,uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,
 uint32_t* future_count,uint32_t* error,void* stream){return accept_all<uint8_t,true,true>(elements,local,remote,local_meta,remote_meta,local_counts,remote_counts,rank,world,shard,shards,stride,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,error,stream);}

extern "C" int mgbfs_generic_seed_shared_i64_wide(uint32_t elements,const int64_t* states,uint32_t stride,uint32_t count,uint64_t* slots,
 uint64_t capacity,uint64_t seed,uint32_t bits,uint32_t* error,void* stream){
 if(!elements||count>stride||!capacity||(capacity&(capacity-1))||bits>64||!error||!slots||(!states&&count)||count>=0x80000000U)return int(cudaErrorInvalidValue);
 if(!count)return 0;seed_table<int64_t,true><<<grid(count),256,0,static_cast<cudaStream_t>(stream)>>>(elements,states,stride,count,slots,capacity,seed,bits,error);return int(cudaGetLastError());
}
extern "C" int mgbfs_generic_seed_shared_u8_wide(uint32_t elements,const uint8_t* states,uint32_t stride,uint32_t count,uint64_t* slots,
 uint64_t capacity,uint64_t seed,uint32_t bits,uint32_t* error,void* stream){
 if(!elements||count>stride||!capacity||(capacity&(capacity-1))||bits>64||!error||!slots||(!states&&count)||count>=0x80000000U)return int(cudaErrorInvalidValue);
 if(!count)return 0;seed_table<uint8_t,true><<<grid(count),256,0,static_cast<cudaStream_t>(stream)>>>(elements,states,stride,count,slots,capacity,seed,bits,error);return int(cudaGetLastError());
}
#include "generic_rolling.cuh"

#include "generic_parent_origin.cuh"

#include "generic_sort_origin.cuh"

// Window replay runs after the owner kernels for this immutable queue tile
// complete. A duplicate can then use the published row of its representative.
using WindowEdge = MgbfsCompactWindowEdgeV1;
static_assert(sizeof(WindowEdge) == 16, "window edge ABI");

__device__ void window_edge(uint32_t& output, uint32_t child, const GenericRouteRecord&) {
    output = child;
}
__device__ void window_edge(WindowEdge& output, uint32_t child, const GenericRouteRecord& route) {
    output = {route.parent, child, route.generator};
}

template<class State, class Edge, class Candidate, bool ReadChildLimit = false>
__global__ void resolve_incoming_edges(
    Candidate action, const uint64_t* slots,
    uint64_t slot_count, const State* arena, uint32_t stride,
    uint32_t first_child, uint32_t child_limit, uint32_t begin, uint32_t length,
    Edge* children, uint32_t* error, const uint32_t* visited_count) {
    if constexpr (ReadChildLimit) {
        if (cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(*error)
                .load(cuda::memory_order_relaxed)) return;
        child_limit = *visited_count;
        if (child_limit > stride || first_child > child_limit) {
            atomicOr(error, 256u);
            return;
        }
    }
    constexpr uint64_t prefix_mask = 0x7fffffff00000000ULL;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < length; i += blockDim.x * gridDim.x) {
        window_edge(children[i], UINT32_MAX, {});
        const uint32_t origin = action.child(begin + i);
        if (!action.valid(origin, error)) continue;
        const auto& route = action.route(origin);
        if (route.source != action.source(origin)) {
            atomicOr(error, 256u);
            continue;
        }
        const uint64_t hash = route.hash;
        const uint64_t mask = slot_count - 1;
        uint64_t slot = action.bucket(hash, slot_count);
        uint32_t resolved = UINT32_MAX;
        for (uint64_t probe = 0; probe < slot_count; ++probe, slot = (slot + 1) & mask) {
            const uint64_t token = slots[slot];
            if (token == EMPTY) break;
            const uint32_t child = uint32_t(token);
            if ((token >> 63) || child < first_child || child >= child_limit
                || ((token ^ hash) & prefix_mask)) continue;
            bool equal = true;
            for (uint32_t coordinate = 0; coordinate < action.elements; ++coordinate) {
                if (arena[uint64_t(coordinate) * stride + child] != action.value(origin, coordinate)) {
                    equal = false;
                    break;
                }
            }
            if (equal) {
                resolved = child - first_child;
                break;
            }
        }
        // A valid candidate must have a representative in this exact depth.
        if (resolved == UINT32_MAX) atomicOr(error, 256u);
        else window_edge(children[i], resolved, route);
    }
}

template<class Edge>
static int resolve_window_edges(
    uint32_t state_bytes, uint32_t elements, const void* local, const void* remote,
    const GenericRouteRecord* local_meta, const GenericRouteRecord* remote_meta,
    const uint32_t* local_counts, const uint32_t* remote_counts, uint32_t rank,
    uint32_t world, uint32_t shard, uint32_t shards, uint32_t queue_capacity,
    const uint64_t* slots, uint64_t slot_count, const void* arena, uint32_t stride,
    uint32_t first_child, uint32_t child_limit, Edge* children, uint32_t output_capacity,
    uint32_t begin, uint32_t length, uint32_t* error, void* raw_stream) {
    if ((state_bytes != 1 && state_bytes != 8) || !elements || !world || world > 128
        || rank >= world || !shards || shards > 4096 || shard >= shards || !queue_capacity
        || length > output_capacity || !length
        || uint64_t(begin) + length > uint64_t(world) * queue_capacity
        || uint64_t(world) * shards * queue_capacity >= 0x7fffffffULL
        || !slot_count || (slot_count & (slot_count - 1)) || child_limit > stride
        || first_child > child_limit || !local || !remote || !local_meta || !remote_meta
        || !local_counts || !remote_counts || !slots || !arena || !children || !error) {
        return cudaErrorInvalidValue;
    }
    const auto stream = static_cast<cudaStream_t>(raw_stream);
    const uint32_t count = world * queue_capacity;
    if (state_bytes == 1) {
        const IncomingAllAction<uint8_t, false, true> action{
            elements, count, queue_capacity, 1, rank, world, shard, shards,
            static_cast<const uint8_t*>(local), static_cast<const uint8_t*>(remote),
            local_meta, remote_meta, local_counts, remote_counts};
        resolve_incoming_edges<uint8_t, Edge><<<grid(length), 256, 0, stream>>>(
            action, slots, slot_count, static_cast<const uint8_t*>(arena), stride,
            first_child, child_limit, begin, length, children, error, nullptr);
    } else {
        const IncomingAllAction<int64_t, false, true> action{
            elements, count, queue_capacity, 1, rank, world, shard, shards,
            static_cast<const int64_t*>(local), static_cast<const int64_t*>(remote),
            local_meta, remote_meta, local_counts, remote_counts};
        resolve_incoming_edges<int64_t, Edge><<<grid(length), 256, 0, stream>>>(
            action, slots, slot_count, static_cast<const int64_t*>(arena), stride,
            first_child, child_limit, begin, length, children, error, nullptr);
    }
    return cudaGetLastError();
}

extern "C" int mgbfs_generic_resolve_incoming_edges_v2(
    uint32_t state_bytes, uint32_t elements, const void* local, const void* remote,
    const GenericRouteRecord* local_meta, const GenericRouteRecord* remote_meta,
    const uint32_t* local_counts, const uint32_t* remote_counts, uint32_t rank,
    uint32_t world, uint32_t shard, uint32_t shards, uint32_t queue_capacity,
    const uint64_t* slots, uint64_t slot_count, const void* arena, uint32_t stride,
    uint32_t first_child, uint32_t child_limit, uint32_t* children, uint32_t output_capacity,
    uint32_t begin, uint32_t length, uint32_t* error, void* raw_stream) {
    return resolve_window_edges(state_bytes, elements, local, remote, local_meta, remote_meta, local_counts, remote_counts,
        rank, world, shard, shards, queue_capacity, slots, slot_count, arena, stride,
        first_child, child_limit, children, output_capacity, begin, length, error, raw_stream);
}

extern "C" int mgbfs_generic_resolve_incoming_edges_v3(
    uint32_t state_bytes, uint32_t elements, const void* local, const void* remote,
    const GenericRouteRecord* local_meta, const GenericRouteRecord* remote_meta,
    const uint32_t* local_counts, const uint32_t* remote_counts, uint32_t rank,
    uint32_t world, uint32_t shard, uint32_t shards, uint32_t queue_capacity,
    const uint64_t* slots, uint64_t slot_count, const void* arena, uint32_t stride,
    uint32_t first_child, uint32_t child_limit, WindowEdge* children, uint32_t output_capacity,
    uint32_t begin, uint32_t length, uint32_t* error, void* raw_stream) {
    return resolve_window_edges(state_bytes, elements, local, remote, local_meta, remote_meta, local_counts, remote_counts,
        rank, world, shard, shards, queue_capacity, slots, slot_count, arena, stride,
        first_child, child_limit, children, output_capacity, begin, length, error, raw_stream);
}

template<class State>
static void resolve_compact_window_edges(
    const MgbfsCompactPathInputV1& in, const MgbfsCompactPathOwnerV1& owner,
    uint32_t first_child, uint32_t begin, uint32_t length, WindowEdge* children,
    cudaStream_t stream) {
    const auto action = compact_path_action<State>(in, false);
    resolve_incoming_edges<State, WindowEdge, CompactPathAction<State>, true>
        <<<grid(length), 256, 0, stream>>>(
            action, owner.slots, owner.slot_count, static_cast<const State*>(owner.states),
            owner.capacity, first_child, 0, begin, length, children, owner.error,
            owner.visited_count);
}

extern "C" int mgbfs_generic_resolve_compact_window_edges_v1(
    const MgbfsCompactPathInputV1* input, const MgbfsCompactPathOwnerV1* output,
    uint32_t first_child, uint32_t begin, uint32_t length, WindowEdge* children,
    uint32_t output_capacity, void* raw_stream) {
    if (!input || !output || !valid_compact_path(*input, *output) || !children
        || first_child > output->capacity || !length || length > output_capacity
        || begin > input->rows || length > input->rows - begin) {
        return cudaErrorInvalidValue;
    }
    const auto stream = static_cast<cudaStream_t>(raw_stream);
    if (input->state_bytes == 1) {
        resolve_compact_window_edges<uint8_t>(*input, *output, first_child, begin, length, children, stream);
    } else {
        resolve_compact_window_edges<int64_t>(*input, *output, first_child, begin, length, children, stream);
    }
    return cudaGetLastError();
}

extern "C" int mgbfs_generic_resolve_incoming_edges_v1(
    uint32_t state_bytes, uint32_t elements, const void* local, const void* remote,
    const GenericRouteRecord* local_meta, const GenericRouteRecord* remote_meta,
    const uint32_t* local_counts, const uint32_t* remote_counts, uint32_t rank,
    uint32_t world, uint32_t shard, uint32_t shards, uint32_t queue_capacity,
    const uint64_t* slots, uint64_t slot_count, const void* arena, uint32_t stride,
    uint32_t first_child, uint32_t child_limit, uint32_t* children, uint32_t output_capacity,
    uint32_t* error, void* raw_stream) {
    const uint64_t length = uint64_t(world) * queue_capacity;
    if (length > UINT32_MAX) return cudaErrorInvalidValue;
    return mgbfs_generic_resolve_incoming_edges_v2(
        state_bytes, elements, local, remote, local_meta, remote_meta, local_counts, remote_counts,
        rank, world, shard, shards, queue_capacity, slots, slot_count, arena, stride,
        first_child, child_limit, children, output_capacity, 0, uint32_t(length), error, raw_stream);
}
