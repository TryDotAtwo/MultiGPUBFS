#include "bitmap_pack4_owner.cuh"
#include "generic_action.cuh"
#include "segmented_incoming.cuh"
#include "segmented_history.cuh"
#include <cuda/atomic>
#include <cuda_runtime.h>

template<class State,bool Packed>
__global__ void accept_pack4(MgbfsPack4OwnerArgs a,MgbfsBoundedIncomingAction<State,Packed> in) {
#ifdef MGBFS_PACK4_OWNER_TEST_BARRIER
    // Diagnostic build only: prove two shard CTAs reached the same phase.
    if(a.test_gate) {
        if(threadIdx.x==0) {
            atomicAdd(a.test_gate,1u);
            const unsigned long long start=clock64();
            while(atomicAdd(a.test_gate,0u)<2u && clock64()-start<3000000000ull) {}
            if(atomicAdd(a.test_gate,0u)<2u) atomicOr(a.error,256u);
        }
        __syncthreads();
        if(*a.error)return;
    }
#endif
    for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<a.selected;i+=blockDim.x*gridDim.x) {
        uint32_t child=a.selection?a.selection[i]:in.child(i);
        if(!in.valid(child,a.error))continue;
        if(cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(*a.error).load(cuda::memory_order_relaxed)&~uint32_t(MGBFS_SEGMENTED_CAP_RETRY))return;
        uint64_t id=in.hash(child,0,64),local=id/a.world;
        uint32_t shard=(uint64_t(uint32_t(mix64(id)))*a.shards)>>32;
        if(id>=a.total||id%a.world!=a.owner||local>=a.owner_len||shard!=a.shard){
            atomicOr(a.error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));continue;
        }
        int64_t values[16];for(uint32_t e=0;e<a.elements;e++)values[e]=in.value(child,e);
        uint64_t packed_word=0;
        if(!mgbfs_pack4_encode(a.dictionary,values,&packed_word)){
            atomicOr(a.error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));continue;
        }
        auto* word=reinterpret_cast<unsigned long long*>(&a.bits[local/64]);
        const unsigned long long mask=1ull<<(local%64);
        if(atomicOr(word,mask)&mask)continue;
        bool global_reserved=false;
        if(a.global_count){
            cuda::atomic_ref<uint64_t,cuda::thread_scope_device> global(*a.global_count);
            uint64_t current=global.load(cuda::memory_order_relaxed);
            while(current<a.global_limit){
                if(global.compare_exchange_weak(current,current+1,cuda::memory_order_acq_rel,cuda::memory_order_relaxed)){
                    global_reserved=true;break;
                }
            }
            if(!global_reserved){
                a.spill[atomicAdd(a.spill_count,1u)]=child;
                atomicAnd(word,~mask);
                atomicOr(a.error,uint32_t(MGBFS_SEGMENTED_CAP_RETRY));
                continue;
            }
        }
        cuda::atomic_ref<uint64_t,cuda::thread_scope_device> accepted(*a.accepted);
        uint64_t row=accepted.load(cuda::memory_order_relaxed);
        while(row<a.capacity&&!accepted.compare_exchange_weak(row,row+1,cuda::memory_order_relaxed,cuda::memory_order_relaxed)){}
        if(row>=a.capacity){
            if(global_reserved){
                cuda::atomic_ref<uint64_t,cuda::thread_scope_device> global(*a.global_count);
                global.fetch_sub(1,cuda::memory_order_acq_rel);
            }
            a.spill[atomicAdd(a.spill_count,1u)]=child;
            atomicAnd(word,~mask);
            continue;
        }
        a.rows[uint64_t(a.bank)*a.capacity+row]=packed_word;
    }
}

template<class T,bool Packed>
int launch(MgbfsPack4OwnerArgs a,cudaStream_t stream) {
    MgbfsBoundedIncomingAction<T,Packed> in{{a.elements,a.world*a.q,a.q,1,a.rank,a.world,a.shard,a.shards,
        static_cast<const T*>(a.local),static_cast<const T*>(a.remote),a.local_meta,a.remote_meta,a.local_counts,a.remote_counts,nullptr},a.visible};
    uint32_t blocks=(a.selected+255)/256;if(blocks>65535)blocks=65535;
    if(blocks)accept_pack4<<<blocks,256,0,stream>>>(a,in);
    return int(cudaGetLastError());
}
extern "C" int mgbfs_pack4_owner_accept(MgbfsPack4OwnerArgs a,void* stream) {
    const uint64_t bound=uint64_t(a.world)*a.q;
    if(!mgbfs_pack4_valid(a.dictionary)||a.elements!=a.dictionary.elements||
       (a.state_bytes!=1&&a.state_bytes!=8)||a.packed>1||(a.packed&&a.state_bytes!=1)||
       !a.world||a.world>128||a.rank>=a.world||a.owner>=a.world||!a.total||!a.bits||
       !a.shards||a.shard>=a.shards||!a.q||bound>=0x7fffffffULL||a.visible>a.q||
       a.selected>bound||(!a.selection&&a.selected!=uint64_t(a.world)*a.visible)||
       !a.local_meta||!a.remote_meta||!a.local_counts||!a.remote_counts||
       (!a.packed&&(!a.local||!a.remote))||!a.rows||!(a.banks==2||a.banks==3)||
       !a.capacity||a.capacity>(uint64_t(1)<<32)||a.bank>=a.banks||!a.accepted||
       !a.spill||!a.spill_count||!a.error||(!a.global_count&&a.global_limit))return int(cudaErrorInvalidValue);
    auto st=static_cast<cudaStream_t>(stream);
    if(a.state_bytes==1&&a.packed)return launch<uint8_t,true>(a,st);
    if(a.state_bytes==1)return launch<uint8_t,false>(a,st);
    return launch<int64_t,false>(a,st);
}

__global__ void seed_pack4(MgbfsPack4SeedArgs a){
    if(blockIdx.x||threadIdx.x)return;
    uint64_t word=0,mask=1ull<<(a.local%64);
    if(*a.accepted || (a.bits[a.local/64]&mask) ||
       !mgbfs_pack4_encode(a.dictionary,a.input,&word)){
        atomicOr(a.error,uint32_t(MGBFS_SEGMENTED_BAD_REFERENCE));return;
    }
    *a.row=word;
    a.bits[a.local/64]|=mask;
    *a.accepted=1;
}
extern "C" int mgbfs_pack4_seed(MgbfsPack4SeedArgs a,void* stream){
    if(!mgbfs_pack4_valid(a.dictionary)||!a.input||!a.row||!a.bits||!a.accepted||!a.error||
       !a.owner_len||a.local>=a.owner_len)return int(cudaErrorInvalidValue);
    seed_pack4<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(a);
    return int(cudaGetLastError());
}
