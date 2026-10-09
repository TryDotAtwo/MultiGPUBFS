#pragma once
#include "generic_sorted_run_pool.cuh"
#include "generic_sorted_history.cuh"

// One stream owns each shard's tier state. Other shards own disjoint tier
// metadata and share pool credits. A ticket does not remove either input until
// its output has been committed; failed/pressured merges preserve both inputs.
struct GenericSortedRunTiers {
 GenericSortedRunToken tokens[32];uint32_t present;
};
enum GenericSortedCarryStage {
 SORTED_CARRY_IDLE=0,SORTED_CARRY_STORED=1,SORTED_CARRY_MERGE=2,
 SORTED_CARRY_PRESSURE=3,SORTED_CARRY_NEXT=4,SORTED_CARRY_FAILED=5
};
struct GenericSortedMergeTicket {
 GenericSortedRunToken left,right,destination;uint32_t size_class;
 GenericSortedHistoryRun left_view,right_view;
 uint64_t* destination_hashes;uint32_t* destination_rows;
 uint32_t destination_capacity;
};
struct GenericSortedRunCarry {
 GenericSortedRunToken token;uint32_t valid,stage;GenericSortedMergeTicket ticket;
};
__device__ bool generic_sorted_same_token(GenericSortedRunToken a,GenericSortedRunToken b){return a.slot==b.slot&&a.generation==b.generation;}
__device__ GenericSortedHistoryRun generic_sorted_pool_view(GenericSortedRunPool pool,
 GenericSortedRunToken token,uint64_t* hashes,uint32_t* rows){
 uint64_t offset=uint64_t(token.slot)*pool.page_entries;
 return {hashes+offset,rows+offset,pool.descriptors[token.slot].count};
}
__device__ void generic_sorted_tier_prepare(GenericSortedRunPool pool,
 GenericSortedRunTiers* tiers,GenericSortedRunCarry* carry,uint64_t* hashes,uint32_t* rows,uint32_t* error){
 if(!carry->valid){carry->stage=SORTED_CARRY_IDLE;return;}
 if(*error){carry->stage=SORTED_CARRY_FAILED;return;}
 if(carry->stage==SORTED_CARRY_MERGE){atomicOr(error,512u);return;}
 auto right=carry->token;
 if(right.slot>=uint64_t(pool.regions)*64){atomicOr(error,8u);carry->stage=SORTED_CARRY_FAILED;return;}
 auto* d=pool.descriptors+right.slot;auto lease=atomicCAS(&d->lease,0ull,0ull);
 uint32_t cls=d->size_class;
 if(uint32_t(lease>>32)!=right.generation||!(lease&SORTED_RUN_PUBLISHED)||(lease&SORTED_RUN_RETIRING)||!(lease&SORTED_RUN_REFS)||cls>31){atomicOr(error,512u);carry->stage=SORTED_CARRY_FAILED;return;}
 if(!d->count){generic_sorted_run_release(pool,right,true,error);carry->valid=0;carry->stage=SORTED_CARRY_IDLE;return;}
 uint32_t bit=1u<<cls;
 if(!(tiers->present&bit)){
  tiers->tokens[cls]=right;__threadfence();tiers->present|=bit;
  carry->valid=0;carry->stage=SORTED_CARRY_STORED;return;
 }
 if(cls==31){carry->stage=SORTED_CARRY_PRESSURE;return;}
 GenericSortedRunToken destination{};
 auto acquired=generic_sorted_run_allocate(pool,cls+1,right.slot/64,&destination,error);
 if(acquired!=SORTED_RUN_ACQUIRED){carry->stage=acquired==SORTED_RUN_PRESSURE?SORTED_CARRY_PRESSURE:SORTED_CARRY_FAILED;return;}
 auto left=tiers->tokens[cls];
 if(generic_sorted_same_token(left,right)||left.slot>=uint64_t(pool.regions)*64||pool.descriptors[left.slot].size_class!=cls){
  generic_sorted_run_release(pool,destination,true,error);atomicOr(error,512u);carry->stage=SORTED_CARRY_FAILED;return;
 }
 bool left_read=generic_sorted_run_read_acquire(pool,left,error);
 bool right_read=left_read&&generic_sorted_run_read_acquire(pool,right,error);
 if(!right_read){
  if(left_read)generic_sorted_run_release(pool,left,false,error);
  generic_sorted_run_release(pool,destination,true,error);
  atomicOr(error,512u);carry->stage=SORTED_CARRY_FAILED;return;
 }
 uint64_t destination_offset=uint64_t(destination.slot)*pool.page_entries;
 uint64_t capacity=uint64_t(pool.page_entries)*(1ull<<(cls+1));
 if(capacity>0xffffffffull){
  generic_sorted_run_release(pool,left,false,error);generic_sorted_run_release(pool,right,false,error);
  generic_sorted_run_release(pool,destination,true,error);atomicOr(error,32u);carry->stage=SORTED_CARRY_FAILED;return;
 }
 carry->ticket={left,right,destination,cls,generic_sorted_pool_view(pool,left,hashes,rows),
  generic_sorted_pool_view(pool,right,hashes,rows),hashes+destination_offset,rows+destination_offset,uint32_t(capacity)};
 __threadfence();carry->stage=SORTED_CARRY_MERGE;
}
__device__ void generic_sorted_tier_abort(GenericSortedRunPool pool,GenericSortedRunCarry* carry,uint32_t* error){
 if(carry->stage!=SORTED_CARRY_MERGE){atomicOr(error,512u);return;}
 auto ticket=carry->ticket;
 generic_sorted_run_release(pool,ticket.left,false,error);generic_sorted_run_release(pool,ticket.right,false,error);
 generic_sorted_run_release(pool,ticket.destination,true,error);
 carry->stage=SORTED_CARRY_PRESSURE; // original tier + carry ownership are intact
}
__device__ void generic_sorted_tier_commit(GenericSortedRunPool pool,
 GenericSortedRunTiers* tiers,GenericSortedRunCarry* carry,uint32_t count,uint32_t* error){
 if(carry->stage!=SORTED_CARRY_MERGE){atomicOr(error,512u);return;}
 auto ticket=carry->ticket;uint32_t bit=1u<<ticket.size_class;
 if(*error){generic_sorted_tier_abort(pool,carry,error);carry->stage=SORTED_CARRY_FAILED;return;}

 if(!(tiers->present&bit)||!generic_sorted_same_token(tiers->tokens[ticket.size_class],ticket.left)||
  !generic_sorted_same_token(carry->token,ticket.right)||!count||count>ticket.destination_capacity||
  count<ticket.left_view.count||count<ticket.right_view.count||uint64_t(count)>uint64_t(ticket.left_view.count)+ticket.right_view.count){
  generic_sorted_tier_abort(pool,carry,error);atomicOr(error,32u);carry->stage=SORTED_CARRY_FAILED;return;
 }
 if(count&&(!generic_sorted_run_trim(pool,ticket.destination,count,error)||!generic_sorted_run_publish(pool,ticket.destination,count,error))){
  generic_sorted_tier_abort(pool,carry,error);carry->stage=SORTED_CARRY_FAILED;return;
 }
 // Publication follows all merge/unique writes in the owner stream. Existing
 // input readers keep their credits while retired inputs leave this hierarchy.
 tiers->present&=~bit;__threadfence();
 generic_sorted_run_release(pool,ticket.left,false,error);generic_sorted_run_release(pool,ticket.right,false,error);
 generic_sorted_run_release(pool,ticket.left,true,error);generic_sorted_run_release(pool,ticket.right,true,error);
 if(count){carry->token=ticket.destination;carry->valid=1;carry->stage=SORTED_CARRY_NEXT;}
 else{generic_sorted_run_release(pool,ticket.destination,true,error);carry->valid=0;carry->stage=SORTED_CARRY_IDLE;}
}
