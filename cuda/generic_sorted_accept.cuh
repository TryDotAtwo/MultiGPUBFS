#pragma once
#include "generic_sorted_history_snapshot.cuh"
#include <cub/device/device_scan.cuh>
// Each invocation owns one shard stream. Immutable snapshots and incoming
// planes remain leased until this stream completes publication and carry.
struct GenericSortedAcceptReservation {
 GenericSortedRunToken token;uint32_t count,row_begin,frontier_begin,valid,status;
};
template<class Action,class State> __global__ void generic_sorted_accept_flags(
 Action action,const uint64_t* hashes,const uint32_t* origins,const uint32_t* unique_count,
 uint32_t capacity,const GenericSortedHistorySnapshot* snapshots,uint32_t snapshot_count,
 const State* arena,uint32_t stride,uint32_t* flags,uint32_t* error){
 uint32_t count=*unique_count;if(count>capacity){if(!blockIdx.x&&!threadIdx.x)atomicOr(error,32u);count=0;}
 for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<capacity;i+=uint64_t(blockDim.x)*gridDim.x){
  bool keep=i<count&&!*error;
  if(keep)for(uint32_t j=0;j<snapshot_count;++j)
   if(generic_sorted_history_snapshot_contains(action,origins[i],hashes[i],snapshots+j,arena,stride,error)){keep=false;break;}
  flags[i]=keep;
 }
}
static __global__ void generic_sorted_accept_reserve(GenericSortedRunPool pool,
 const uint32_t* flags,const uint32_t* prefix,uint32_t n,uint32_t capacity,
 uint32_t* row_count,uint32_t* frontier_count,uint32_t rolling,
 GenericSortedAcceptReservation* out,const GenericSortedRunCarry* carry,uint32_t* error){
 if(blockIdx.x||threadIdx.x)return;*out={};if(*error)return;
 if(carry->valid){atomicOr(error,512u);return;}
 uint32_t count=n?prefix[n-1]+flags[n-1]:0;out->count=count;if(!count)return;
 uint32_t cls=0;while(cls<31&&(uint64_t(pool.page_entries)<<cls)<count)++cls;
 auto status=generic_sorted_run_allocate(pool,cls,0,&out->token,error);
 out->status=status;if(status!=SORTED_RUN_ACQUIRED)return;
 uint32_t begin=atomicAdd(row_count,0u);
 for(;;){if(begin>capacity||count>capacity-begin){generic_sorted_run_release(pool,out->token,true,error);atomicOr(error,2u);return;}
  uint32_t old=atomicCAS(row_count,begin,begin+count);if(old==begin)break;begin=old;}
 out->row_begin=begin;out->frontier_begin=rolling?begin:atomicAdd(frontier_count,count);
 if(out->frontier_begin>capacity||count>capacity-out->frontier_begin){
  generic_sorted_run_release(pool,out->token,true,error);atomicOr(error,32u);return;}
 out->valid=1;
}
template<class Action,class State> __global__ void generic_sorted_accept_materialize(
 Action action,const uint64_t* hashes,const uint32_t* origins,const uint32_t* flags,
 const uint32_t* prefix,uint32_t n,GenericSortedRunPool pool,
 const GenericSortedAcceptReservation* reservation,uint64_t* run_hashes,uint32_t* run_rows,
 State* arena,uint32_t stride,uint32_t base,uint32_t* future){
 if(!reservation->valid)return;
 uint64_t offset=uint64_t(reservation->token.slot)*pool.page_entries;
 for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=uint64_t(blockDim.x)*gridDim.x)if(flags[i]){
  uint32_t position=prefix[i],row=base+reservation->row_begin+position,child=origins[i];
  for(uint32_t c=0;c<action.elements;++c)arena[uint64_t(c)*stride+row]=State(action.value(child,c));
  run_hashes[offset+position]=hashes[i];run_rows[offset+position]=row;
  future[reservation->frontier_begin+position]=row;
 }
}
static __global__ void generic_sorted_accept_publish(GenericSortedRunPool pool,
 GenericSortedAcceptReservation* reservation,GenericSortedRunCarry* carry,uint32_t* error){
 if(blockIdx.x||threadIdx.x)return;
 if(!reservation->valid)return;
 if(*error){generic_sorted_run_release(pool,reservation->token,true,error);reservation->valid=0;return;}
 if(!generic_sorted_run_publish(pool,reservation->token,reservation->count,error)){
  generic_sorted_run_release(pool,reservation->token,true,error);reservation->valid=0;return;}
 carry->token=reservation->token;carry->valid=1;carry->stage=SORTED_CARRY_NEXT;carry->retries_remaining=1;reservation->valid=0;
}
template<class Action,class State> inline cudaError_t generic_sorted_accept(
 Action action,const uint64_t* hashes,const uint32_t* origins,const uint32_t* count,uint32_t n,
 const GenericSortedHistorySnapshot* snapshots,uint32_t snapshot_count,
 GenericSortedRunPool pool,uint64_t* run_hashes,uint32_t* run_rows,State* arena,
 uint32_t stride,uint32_t base,uint32_t capacity,uint32_t* row_count,uint32_t* frontier_count,
 uint32_t rolling,uint32_t* future,uint32_t* flags,uint32_t* prefix,void* temporary,
 size_t temporary_bytes,GenericSortedAcceptReservation* reservation,GenericSortedRunCarry* carry,
 uint32_t* error,cudaStream_t stream=0){
 GenericSortedRunPoolShape shape{};
 if(!action.elements||!hashes||!origins||!count||n>=0x7fffffffu||snapshot_count>3||
  (snapshot_count&&!snapshots)||!generic_sorted_run_pool_shape(pool.regions,pool.page_entries,&shape)||
  !pool.occupied||!pool.descriptors||!run_hashes||!run_rows||!arena||!stride||base>stride||
  !capacity||capacity>stride-base||!row_count||!frontier_count||rolling>1||
  (rolling&&row_count!=frontier_count)||(!rolling&&row_count==frontier_count)||!future||
  !flags||!prefix||!temporary||!reservation||!carry||!error)return cudaErrorInvalidValue;
 // Scratch for this fixed geometry is admitted by generic_sorted_origin_shape.
 // CUB execution retains its own storage/error validation, as in origin_exact.
 cudaError_t status=cudaSuccess;
 uint32_t blocks=n/256+(n%256!=0);if(!blocks)blocks=1;if(blocks>65535)blocks=65535;
 generic_sorted_accept_flags<<<blocks,256,0,stream>>>(action,hashes,origins,count,n,snapshots,snapshot_count,arena,stride,flags,error);
 status=cudaGetLastError();if(status!=cudaSuccess)return status;
 if(n){status=cub::DeviceScan::ExclusiveSum(temporary,temporary_bytes,flags,prefix,n,stream);if(status!=cudaSuccess)return status;}
 generic_sorted_accept_reserve<<<1,1,0,stream>>>(pool,flags,prefix,n,capacity,row_count,frontier_count,rolling,reservation,carry,error);
 status=cudaGetLastError();if(status!=cudaSuccess)return status;
 generic_sorted_accept_materialize<<<blocks,256,0,stream>>>(action,hashes,origins,flags,prefix,n,pool,reservation,run_hashes,run_rows,arena,stride,base,future);
 status=cudaGetLastError();if(status!=cudaSuccess)return status;
 generic_sorted_accept_publish<<<1,1,0,stream>>>(pool,reservation,carry,error);return cudaGetLastError();
}
