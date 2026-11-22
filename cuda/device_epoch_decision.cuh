#pragma once
#include <cuda/atomic>
#include <cstdint>

// Shared building block for a GPU-driven job epoch. All fields belong to
// rank-owned registered memory, live until every rank has stopped, and must
// support system-scope peer atomics. One coordinator writes Decision.
// A stop request is provisional: each rank first stops all local writers and
// reports stopped_epoch with release ordering. Only then may the coordinator
// combine every rank's final error bits and publish a terminal decision.
// If a rank disappears the external supervisor must cancel the whole job;
// this helper does not claim progress or produce COMPLETE in that case.
namespace mgbfs_device_epoch {
using Word=unsigned long long;
using Atomic=cuda::atomic_ref<Word,cuda::thread_scope_system>;
enum : Word { ADVANCE=1, COMPLETE=2, INCOMPLETE=3, ERROR=4 };
enum : Word { RESOURCE=1, FATAL=2 };
struct alignas(8) RankStatus { Word failure,stopped_epoch; };
struct alignas(8) Decision { Word request,request_epoch,final,final_epoch; };

__device__ __forceinline__ Word load(Word& x) {
    return Atomic(x).load(cuda::memory_order_acquire);
}
__device__ __forceinline__ void store(Word& x,Word value) {
    Atomic(x).store(value,cuda::memory_order_release);
}
__device__ __forceinline__ void publish_stopped(RankStatus& local,Word epoch) {
    store(local.stopped_epoch,epoch);
}
__device__ __forceinline__ bool try_finalize(
    Decision& decision,RankStatus* const* ranks,uint32_t world,Word epoch) {
    if(!world||!epoch||load(decision.request_epoch)!=epoch)return false;
    Word failures=0;
    for(uint32_t rank=0;rank<world;rank++) {
        if(!ranks[rank]||load(ranks[rank]->stopped_epoch)!=epoch)return false;
        failures|=load(ranks[rank]->failure);
    }
    const Word request=decision.request;
    const bool known=request==ADVANCE||request==COMPLETE||request==INCOMPLETE||request==ERROR;
    const Word final=(!known||request==ERROR||(failures&~Word(RESOURCE))) ? ERROR :
                     (failures&RESOURCE) ? INCOMPLETE : request;
    decision.final=final;
    store(decision.final_epoch,epoch);
    return true;
}
}
