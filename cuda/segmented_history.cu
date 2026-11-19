#include "segmented_history.cuh"
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <stdint.h>
#include "generic_action.cuh"
#include "segmented_incoming.cuh"

namespace {
constexpr uint64_t EMPTY = ~uint64_t(0);
constexpr uint64_t RETIRED = EMPTY - 1;
constexpr uint64_t PENDING = uint64_t(1) << 63;
constexpr uint64_t PREFIX = 0x7fffffff00000000ULL;
constexpr uint32_t ROW_MASK = (1u << 24) - 1;
constexpr uint32_t LOCAL_ROWS = 1u << 24;
constexpr uint32_t PAYLOAD_MASK = (1u << 26) - 1;

template<class T> struct FlatInput {
    uint32_t elements,count;const T* input;const uint64_t* hashes;
    __device__ uint32_t child(uint32_t i)const{return i;}
    __device__ uint32_t origin_limit()const{return count;}
    __device__ bool valid(uint32_t i,uint32_t* error)const{if(i<count)return true;atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return false;}
    __device__ int64_t value(uint32_t i,uint32_t e)const{return int64_t(input[uint64_t(e)*count+i]);}
    __device__ uint64_t hash(uint32_t i,uint64_t,uint32_t)const{return hashes[i];}
};
template<class Input>
__device__ bool equal_input(const Input& input, uint32_t a, uint32_t b) {
    for (uint32_t e = 0; e < input.elements; ++e)
        if (input.value(a,e) != input.value(b,e)) return false;
    return true;
}
template<class T,class Input>
__device__ bool equal_row(const Input& input, uint32_t child,
                          const MgbfsSegment& seg, uint32_t payload, uint32_t* error) {
    const uint32_t row=payload & ROW_MASK, bank=(payload >> 24) & 3u;
    if ((payload & ~PAYLOAD_MASK) || bank >= seg.banks || row >= seg.capacity) {
        atomicOr(error, uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE)); return false;
    }
    const auto* states=static_cast<const T*>(seg.states);
    const uint64_t stride=uint64_t(seg.capacity)*seg.banks;
    const uint64_t local=uint64_t(bank)*seg.capacity+row;
    for (uint32_t e=0; e<input.elements; ++e)
        if (input.value(child,e) != int64_t(states[uint64_t(e)*stride+local])) return false;
    return true;
}

template<class T,class Input>
__global__ void accept(Input input, const MgbfsSegment* sealed, uint32_t sealed_count, MgbfsSegment active,
                       uint32_t bank, uint32_t* accepted, uint32_t* future, uint32_t* error,
                       const uint32_t* selection, uint32_t selected_count,
                       uint32_t* spill, uint32_t* spill_count,
                       uint64_t* global_count, uint64_t global_limit) {
    for (uint32_t selected=blockIdx.x*blockDim.x+threadIdx.x; selected<selected_count; selected+=blockDim.x*gridDim.x) {
        const uint32_t child=selection ? selection[selected] : input.child(selected);
        if(!input.valid(child,error)) continue;
        if (cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(*error).load(cuda::memory_order_relaxed)) return;
        const uint64_t hash=input.hash(child,0,64), prefix=hash & PREFIX;
        bool duplicate=false;
        for (uint32_t s=0; s<sealed_count && !duplicate; ++s) {
            MgbfsSegment seg=sealed[s];
            // Descriptor arrays are cached per shard and include the active
            // segment. Its pending rows must be inspected by the active path.
            if (seg.slots==active.slots) continue;
            if (!seg.slots || !seg.states || !seg.slots_count || (seg.slots_count & (seg.slots_count-1)) ||
                !seg.capacity || seg.capacity>LOCAL_ROWS || !(seg.banks==1 || seg.banks==3)) {
                atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE)); return;
            }
            uint64_t start=mgbfs_shared_table_bucket(hash,seg.slots_count);
            for (uint64_t probe=0; probe<seg.slots_count; ++probe) {
                uint64_t observed=seg.slots[(start+probe)&(seg.slots_count-1)];
                if (observed==EMPTY) break;
                if (observed==RETIRED) continue;
                if (observed & PENDING) { atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE)); return; }
                if ((observed&PREFIX)==prefix && equal_row<T>(input,child,seg,uint32_t(observed),error)) {
                    duplicate=true; break;
                }
            }
        }
        if (duplicate || cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(*error).load(cuda::memory_order_relaxed)) continue;
        const uint64_t pending=prefix|PENDING|child;
        for (;;) {
            uint64_t first=active.slots_count, available=EMPTY;
            const uint64_t start=mgbfs_shared_table_bucket(hash,active.slots_count);
            duplicate=false;
            for (uint64_t probe=0; probe<active.slots_count; ++probe) {
                uint64_t slot=(start+probe)&(active.slots_count-1);
                cuda::atomic_ref<uint64_t,cuda::thread_scope_device> cell(active.slots[slot]);
                uint64_t observed=cell.load(cuda::memory_order_acquire);
                if (observed==RETIRED) { if(first==active.slots_count){first=slot;available=RETIRED;} continue; }
                if (observed==EMPTY) { if(first==active.slots_count){first=slot;available=EMPTY;} break; }
                if ((observed&PREFIX)!=prefix) continue;
                uint32_t ref=uint32_t(observed);
                if (observed&PENDING) {
                    if (ref>=input.origin_limit()) { atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE)); return; }
                    if (equal_input(input,child,ref)) { duplicate=true; break; }
                } else if (equal_row<T>(input,child,active,ref,error)) { duplicate=true; break; }
            }
            if (duplicate || cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(*error).load(cuda::memory_order_relaxed)) break;
            if (first==active.slots_count) {
                if(spill) spill[atomicAdd(spill_count,1u)]=child;
                else atomicOr(error,uint32_t(MGBFS_SEGMENTED_TABLE_FULL));
                break;
            }
            cuda::atomic_ref<uint64_t,cuda::thread_scope_device> cell(active.slots[first]);
            uint64_t expected=available;
            if (!cell.compare_exchange_strong(expected,pending,cuda::memory_order_acq_rel,cuda::memory_order_acquire)) continue;
            // Shared rank-wide hard limit; release reservation on segment spill.
            bool reserved=false;
            if(global_count){
                cuda::atomic_ref<uint64_t,cuda::thread_scope_device> total(*global_count);
                uint64_t value=total.load(cuda::memory_order_relaxed);
                while(value<global_limit){
                    if(total.compare_exchange_weak(value,value+1,cuda::memory_order_acq_rel,cuda::memory_order_relaxed)){reserved=true;break;}
                }
                if(!reserved){cell.store(RETIRED,cuda::memory_order_release);atomicOr(error,uint32_t(MGBFS_SEGMENTED_CAPACITY_LIMIT));break;}
            }
            uint32_t row=atomicAdd(accepted,0u);
            while(row<active.capacity) {
                const uint32_t observed=atomicCAS(accepted,row,row+1u);
                if(observed==row) break;
                row=observed;
            }
            if (row>=active.capacity) {
                if(reserved){cuda::atomic_ref<uint64_t,cuda::thread_scope_device> total(*global_count);total.fetch_sub(1,cuda::memory_order_acq_rel);}
                if(spill) {
                    spill[atomicAdd(spill_count,1u)]=child;
                    // Keep the probe chain intact. The spilled input remains
                    // alive until the stream and subsequent segment finish.
                    cell.store(RETIRED,cuda::memory_order_release);
                } else atomicOr(error,uint32_t(MGBFS_SEGMENTED_OVERFLOW));
                break;
            }
            const uint32_t payload=(bank<<24)|row;
            const uint64_t local=uint64_t(bank)*active.capacity+row;
            const uint64_t stride=uint64_t(active.capacity)*active.banks;
            auto* states=static_cast<T*>(active.states);
            for (uint32_t e=0;e<input.elements;++e) states[uint64_t(e)*stride+local]=static_cast<T>(input.value(child,e));
            if(active.positions) active.positions[local]=first;
            if(future) future[row]=payload;
            cell.store(prefix|payload,cuda::memory_order_release);
            break;
        }
    }
}
template<class T>
__global__ void rehash(MgbfsSegment segment,uint32_t elements,uint32_t c0,uint32_t c1,uint32_t c2,
                       uint64_t seed,uint32_t bits,uint32_t* error) {
    const uint32_t total=c0+c1+c2;
    for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<total;i+=blockDim.x*gridDim.x){
        const uint32_t bank=i<c0?0:(i<c0+c1?1:2);
        const uint32_t row=bank==0?i:(bank==1?i-c0:i-c0-c1);
        const uint64_t local=uint64_t(bank)*segment.capacity+row;
        const uint64_t stride=uint64_t(segment.capacity)*segment.banks;
        const auto* states=static_cast<const T*>(segment.states);uint64_t h=seed;
        for(uint32_t e=0;e<elements;e++)h=mix64(h^uint64_t(states[uint64_t(e)*stride+local])^uint64_t(e));
        h=finish_hash(h,bits);const uint64_t token=(h&PREFIX)|(uint64_t(bank)<<24)|row;
        const uint64_t start=mgbfs_shared_table_bucket(h,segment.slots_count);bool done=false;
        for(uint64_t probe=0;probe<segment.slots_count;probe++){
            const uint64_t slot=(start+probe)&(segment.slots_count-1);
            if(atomicCAS(reinterpret_cast<unsigned long long*>(segment.slots+slot),EMPTY,token)==EMPTY){
                if(segment.positions)segment.positions[local]=slot;done=true;break;
            }
        }
        if(!done)atomicOr(error,uint32_t(MGBFS_SEGMENTED_TABLE_FULL));
    }
}
template<class T>
int launch_rehash(uint32_t elements,MgbfsSegment segment,uint32_t c0,uint32_t c1,uint32_t c2,
                  uint64_t seed,uint32_t bits,uint32_t* error,void* stream){
    if(!elements || bits>64 || !segment.slots || !segment.states || !segment.slots_count ||
       (segment.slots_count&(segment.slots_count-1)) || segment.slots_count>SIZE_MAX/8 ||
       !segment.capacity || segment.capacity>LOCAL_ROWS || !(segment.banks==1||segment.banks==3) ||
       (segment.banks==3&&!segment.positions) || (segment.banks==1&&(c1||c2)) ||
       c0>segment.capacity || c1>segment.capacity || c2>segment.capacity || !error)
       return int(cudaErrorInvalidValue);
    cudaStream_t st=static_cast<cudaStream_t>(stream);cudaError_t result=cudaMemsetAsync(segment.slots,255,size_t(segment.slots_count)*8,st);
    if(result!=cudaSuccess)return int(result);const uint32_t total=c0+c1+c2;if(!total)return int(cudaSuccess);
    uint32_t blocks=(total+255u)/256u;if(blocks>65535u)blocks=65535u;
    rehash<T><<<blocks,256,0,st>>>(segment,elements,c0,c1,c2,seed,bits,error);return int(cudaGetLastError());
}
__global__ void retire(MgbfsSegment segment,uint32_t bank,uint32_t count,uint32_t* error) {
    for(uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;row<count;row+=blockDim.x*gridDim.x) {
        uint64_t slot=segment.positions[uint64_t(bank)*segment.capacity+row];
        if(slot>=segment.slots_count) { atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE)); continue; }
        cuda::atomic_ref<uint64_t,cuda::thread_scope_device> cell(segment.slots[slot]);
        uint64_t observed=cell.load(cuda::memory_order_acquire);
        uint32_t payload=(bank<<24)|row;
        if(observed==EMPTY || observed==RETIRED || (observed&PENDING) || uint32_t(observed)!=payload ||
           !cell.compare_exchange_strong(observed,RETIRED,cuda::memory_order_acq_rel,cuda::memory_order_acquire))
            atomicOr(error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));
    }
}

template<class T>
int launch(const T* input,const uint64_t* hashes,uint32_t count,uint32_t elements,
           const MgbfsSegment* sealed,uint32_t sealed_count,MgbfsSegment active,uint32_t bank,
           uint32_t* accepted,uint32_t* future,uint32_t* error,void* stream) {
    if (!elements || count>=0x80000000u || (sealed_count && !sealed) || !active.slots ||
        !active.states || (active.banks==3 && !active.positions) || !active.slots_count || (active.slots_count&(active.slots_count-1)) ||
        !active.capacity || active.capacity>LOCAL_ROWS || !(active.banks==1 || active.banks==3) || bank>=active.banks ||
        !accepted || !error || (count && (!input || !hashes))) return int(cudaErrorInvalidValue);
    if (!count) return int(cudaSuccess);
    uint32_t blocks=(count+255u)/256u; if(blocks>65535u) blocks=65535u;
    FlatInput<T> source{elements,count,input,hashes};
    accept<T><<<blocks,256,0,static_cast<cudaStream_t>(stream)>>>(source,sealed,sealed_count,active,bank,accepted,future,error,nullptr,count,nullptr,nullptr,nullptr,0);
    return int(cudaGetLastError());
}
template<class T>
int launch_spill(const T* input,const uint64_t* hashes,uint32_t input_count,const uint32_t* selection,
                 uint32_t selected_count,uint32_t elements,const MgbfsSegment* sealed,uint32_t sealed_count,
                 MgbfsSegment active,uint32_t bank,uint32_t* accepted,uint32_t* future,uint32_t* spill,
                 uint32_t* spill_count,uint32_t* error,void* stream) {
    if(!elements || input_count>=0x80000000u || selected_count>=0x80000000u ||
       (!selection && selected_count>input_count) || (sealed_count && !sealed) || !active.slots ||
       !active.states || (active.banks==3 && !active.positions) || !active.slots_count || (active.slots_count&(active.slots_count-1)) ||
       !active.capacity || active.capacity>LOCAL_ROWS || !(active.banks==1 || active.banks==3) || bank>=active.banks ||
       !accepted || !error || !spill || !spill_count || (selected_count && (!input || !hashes))) return int(cudaErrorInvalidValue);
    if(!selected_count) return int(cudaSuccess);
    uint32_t blocks=(selected_count+255u)/256u; if(blocks>65535u) blocks=65535u;
    FlatInput<T> source{elements,input_count,input,hashes};
    accept<T><<<blocks,256,0,static_cast<cudaStream_t>(stream)>>>(source,sealed,sealed_count,active,bank,accepted,future,error,selection,selected_count,spill,spill_count,nullptr,0);
    return int(cudaGetLastError());
}
template<class T,bool Packed>
int launch_routes(uint32_t elements,const T* local,const T* remote,const GenericRouteRecord* local_meta,
                  const GenericRouteRecord* remote_meta,const uint32_t* local_counts,const uint32_t* remote_counts,
                  uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint32_t visible,
                  const uint32_t* selection,uint32_t selected_count,const MgbfsSegment* sealed,uint32_t sealed_count,
                  MgbfsSegment active,uint32_t bank,uint32_t* accepted,uint32_t* future,uint32_t* spill,
                  uint32_t* spill_count,uint32_t* error,uint64_t* global_count,uint64_t global_limit,void* stream) {
    const uint64_t bound=uint64_t(world)*q;
    if(!elements || (Packed&&elements>24) || !world || world>128 || rank>=world || !shards || shards>4096 || shard>=shards || !q ||
       bound>=0x80000000ULL || visible>q || selected_count>bound || (!selection && selected_count!=uint64_t(world)*visible) ||
       !local_meta || !remote_meta || !local_counts || !remote_counts || (!Packed && (!local || !remote)) ||
       (sealed_count&&!sealed) || !active.slots || !active.states || (active.banks==3 && !active.positions) || !active.slots_count ||
       (active.slots_count&(active.slots_count-1)) || !active.capacity || active.capacity>LOCAL_ROWS ||
       !(active.banks==1 || active.banks==3) || bank>=active.banks || !accepted || !spill || !spill_count || !error)
       return int(cudaErrorInvalidValue);
    if(!selected_count)return int(cudaSuccess);
    MgbfsBoundedIncomingAction<T,Packed> source{{elements,uint32_t(bound),q,1,rank,world,shard,shards,local,remote,local_meta,remote_meta,local_counts,remote_counts,nullptr},visible};
    uint32_t blocks=(selected_count+255u)/256u;if(blocks>65535u)blocks=65535u;
    accept<T><<<blocks,256,0,static_cast<cudaStream_t>(stream)>>>(source,sealed,sealed_count,active,bank,accepted,future,error,selection,selected_count,spill,spill_count,global_count,global_limit);
    return int(cudaGetLastError());
}
} // namespace
extern "C" int mgbfs_segmented_accept_u8(const uint8_t* input,const uint64_t* hashes,uint32_t n,uint32_t elements,
    const MgbfsSegment* sealed,uint32_t sealed_count,MgbfsSegment active,uint32_t bank,uint32_t* accepted,
    uint32_t* future,uint32_t* error,void* stream) {
    return launch(input,hashes,n,elements,sealed,sealed_count,active,bank,accepted,future,error,stream);
}
extern "C" int mgbfs_segmented_accept_i64(const int64_t* input,const uint64_t* hashes,uint32_t n,uint32_t elements,
    const MgbfsSegment* sealed,uint32_t sealed_count,MgbfsSegment active,uint32_t bank,uint32_t* accepted,
    uint32_t* future,uint32_t* error,void* stream) {
    return launch(input,hashes,n,elements,sealed,sealed_count,active,bank,accepted,future,error,stream);
}

extern "C" int mgbfs_segmented_accept_spill_u8(const uint8_t* input,const uint64_t* hashes,
    uint32_t n,const uint32_t* selection,uint32_t selected,uint32_t elements,const MgbfsSegment* sealed,
    uint32_t sealed_count,MgbfsSegment active,uint32_t bank,uint32_t* accepted,uint32_t* future,
    uint32_t* spill,uint32_t* spill_count,uint32_t* error,void* stream) {
    return launch_spill(input,hashes,n,selection,selected,elements,sealed,sealed_count,active,bank,accepted,future,spill,spill_count,error,stream);
}
extern "C" int mgbfs_segmented_accept_spill_i64(const int64_t* input,const uint64_t* hashes,
    uint32_t n,const uint32_t* selection,uint32_t selected,uint32_t elements,const MgbfsSegment* sealed,
    uint32_t sealed_count,MgbfsSegment active,uint32_t bank,uint32_t* accepted,uint32_t* future,
    uint32_t* spill,uint32_t* spill_count,uint32_t* error,void* stream) {
    return launch_spill(input,hashes,n,selection,selected,elements,sealed,sealed_count,active,bank,accepted,future,spill,spill_count,error,stream);
}

extern "C" int mgbfs_segmented_accept_routes_u8(uint32_t elements,const uint8_t* local,const uint8_t* remote,
    const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,
    uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint32_t packed,
    const uint32_t* selection,uint32_t selected,const MgbfsSegment* sealed,uint32_t ns,
    MgbfsSegment active,uint32_t bank,uint32_t* accepted,uint32_t* future,uint32_t* spill,uint32_t* sc,uint32_t* error,void* stream) {
    if(packed>1)return int(cudaErrorInvalidValue);
    if(packed)return launch_routes<uint8_t,true>(elements,local,remote,lm,rm,lc,rc,rank,world,shard,shards,q,q,selection,selected,sealed,ns,active,bank,accepted,future,spill,sc,error,nullptr,0,stream);
    return launch_routes<uint8_t,false>(elements,local,remote,lm,rm,lc,rc,rank,world,shard,shards,q,q,selection,selected,sealed,ns,active,bank,accepted,future,spill,sc,error,nullptr,0,stream);
}
extern "C" int mgbfs_segmented_accept_routes_i64(uint32_t elements,const int64_t* local,const int64_t* remote,
    const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,
    uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,
    const uint32_t* selection,uint32_t selected,const MgbfsSegment* sealed,uint32_t ns,
    MgbfsSegment active,uint32_t bank,uint32_t* accepted,uint32_t* future,uint32_t* spill,uint32_t* sc,uint32_t* error,void* stream) {
    return launch_routes<int64_t,false>(elements,local,remote,lm,rm,lc,rc,rank,world,shard,shards,q,q,selection,selected,sealed,ns,active,bank,accepted,future,spill,sc,error,nullptr,0,stream);
}

// Optional shared hard counter applies across every owner lane on this rank.
extern "C" int mgbfs_segmented_accept_routes_limited_u8(uint32_t elements,const uint8_t* local,const uint8_t* remote,
 const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,
 uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint32_t visible,uint32_t packed,
 const uint32_t* selection,uint32_t selected,const MgbfsSegment* sealed,uint32_t ns,
 MgbfsSegment active,uint32_t bank,uint32_t* accepted,uint32_t* future,uint32_t* spill,uint32_t* sc,uint32_t* error,
 uint64_t* global_count,uint64_t limit,void* stream){
 if(packed>1)return int(cudaErrorInvalidValue);
 if(packed)return launch_routes<uint8_t,true>(elements,local,remote,lm,rm,lc,rc,rank,world,shard,shards,q,visible,selection,selected,sealed,ns,active,bank,accepted,future,spill,sc,error,global_count,limit,stream);
 return launch_routes<uint8_t,false>(elements,local,remote,lm,rm,lc,rc,rank,world,shard,shards,q,visible,selection,selected,sealed,ns,active,bank,accepted,future,spill,sc,error,global_count,limit,stream);
}
extern "C" int mgbfs_segmented_accept_routes_limited_i64(uint32_t elements,const int64_t* local,const int64_t* remote,
 const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,
 uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint32_t visible,
 const uint32_t* selection,uint32_t selected,const MgbfsSegment* sealed,uint32_t ns,
 MgbfsSegment active,uint32_t bank,uint32_t* accepted,uint32_t* future,uint32_t* spill,uint32_t* sc,uint32_t* error,
 uint64_t* global_count,uint64_t limit,void* stream){
 return launch_routes<int64_t,false>(elements,local,remote,lm,rm,lc,rc,rank,world,shard,shards,q,visible,selection,selected,sealed,ns,active,bank,accepted,future,spill,sc,error,global_count,limit,stream);
}
extern "C" int mgbfs_segmented_rehash_u8(uint32_t elements,MgbfsSegment segment,uint32_t c0,uint32_t c1,uint32_t c2,uint64_t seed,uint32_t bits,uint32_t* error,void* stream){return launch_rehash<uint8_t>(elements,segment,c0,c1,c2,seed,bits,error,stream);}
extern "C" int mgbfs_segmented_rehash_i64(uint32_t elements,MgbfsSegment segment,uint32_t c0,uint32_t c1,uint32_t c2,uint64_t seed,uint32_t bits,uint32_t* error,void* stream){return launch_rehash<int64_t>(elements,segment,c0,c1,c2,seed,bits,error,stream);}

extern "C" int mgbfs_segmented_retire(MgbfsSegment segment,uint32_t bank,uint32_t count,uint32_t* error,void* stream) {
    if(!segment.slots || !segment.positions || !segment.slots_count ||
       (segment.slots_count&(segment.slots_count-1)) || !segment.capacity || segment.capacity>LOCAL_ROWS ||
       !(segment.banks==1 || segment.banks==3) || bank>=segment.banks || count>segment.capacity || !error)
        return int(cudaErrorInvalidValue);
    if(!count) return int(cudaSuccess);
    uint32_t blocks=(count+255u)/256u; if(blocks>65535u) blocks=65535u;
    retire<<<blocks,256,0,static_cast<cudaStream_t>(stream)>>>(segment,bank,count,error);
    return int(cudaGetLastError());
}
