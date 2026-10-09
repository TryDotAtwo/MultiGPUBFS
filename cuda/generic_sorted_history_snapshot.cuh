#pragma once
#include "generic_sorted_run_tiers.cuh"
// Snapshot creation is ordered in the shard owner stream. Input state planes
// must be independently held by the caller until all snapshot users finish.
struct GenericSortedHistorySnapshot {
 GenericSortedHistoryRun runs[32];GenericSortedRunToken tokens[32];uint32_t count,held;
};
__device__ inline void generic_sorted_history_snapshot_acquire(GenericSortedRunPool pool,
 const GenericSortedRunTiers* tiers,GenericSortedHistorySnapshot* snapshot,
 uint64_t* hashes,uint32_t* rows,uint32_t* error){
 if(snapshot->held){atomicOr(error,512u);return;}
 snapshot->count=0;if(*error)return;uint32_t present=tiers->present;
 for(uint32_t cls=0;cls<32;++cls)if(present&(1u<<cls)){
  auto token=tiers->tokens[cls];
  if(!generic_sorted_run_read_acquire(pool,token,error)){
   for(uint32_t i=0;i<snapshot->count;++i)generic_sorted_run_release(pool,snapshot->tokens[i],false,error);
   snapshot->count=0;atomicOr(error,512u);return;
  }
  uint32_t i=snapshot->count++;snapshot->tokens[i]=token;snapshot->runs[i]=generic_sorted_pool_view(pool,token,hashes,rows);
 }
 __threadfence();snapshot->held=1;
}
__device__ inline void generic_sorted_history_snapshot_release(GenericSortedRunPool pool,
 GenericSortedHistorySnapshot* snapshot,uint32_t* error){
 if(!snapshot->held){atomicOr(error,512u);return;}
 for(uint32_t i=0;i<snapshot->count;++i)generic_sorted_run_release(pool,snapshot->tokens[i],false,error);
 snapshot->held=0;snapshot->count=0;
}
template<class Candidate,class State> __device__ bool generic_sorted_history_snapshot_contains(
 const Candidate& action,uint32_t child,uint64_t hash,const GenericSortedHistorySnapshot* snapshot,
 const State* arena,uint32_t stride,uint32_t* error){
 if(!snapshot->held||snapshot->count>32){atomicOr(error,512u);return false;}
 for(uint32_t i=0;i<snapshot->count;++i)
  if(generic_sorted_history_contains_exact_sorted(action,child,hash,snapshot->runs[i],arena,stride,error))return true;
 return false;
}
