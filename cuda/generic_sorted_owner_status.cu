#include <cuda_runtime.h>
#include "generic_sorted_history_snapshot.cuh"
struct OwnerReservation {GenericSortedRunToken token;uint32_t count,row_begin,frontier_begin,valid,status;};
static_assert(sizeof(OwnerReservation)==28);
__global__ void sorted_owner_status(const OwnerReservation* reservation,const GenericSortedRunCarry* carry,uint32_t* error){
 if(blockIdx.x||threadIdx.x)return;
 if(reservation->status==SORTED_RUN_PRESSURE||carry->valid)atomicOr(error,16u);
}
extern "C" int mgbfs_generic_sorted_native_owner_status(const void* reservation,const void* carry,void* error,void* stream){
 if(!reservation||!carry||!error)return cudaErrorInvalidValue;
 sorted_owner_status<<<1,1,0,(cudaStream_t)stream>>>((const OwnerReservation*)reservation,(const GenericSortedRunCarry*)carry,(uint32_t*)error);return cudaGetLastError();
}
__global__ void sorted_owner_release_checked(GenericSortedRunPool pool,GenericSortedHistorySnapshot* snapshot,uint32_t* error){
 if(blockIdx.x||threadIdx.x)return;
 // An earlier allocation/validation failure can skip snapshot acquisition.
 // Preserve its error while releasing every lease that WAS acquired.
 if(snapshot->held)generic_sorted_history_snapshot_release(pool,snapshot,error);
 else if(!*error)atomicOr(error,512u);
}
extern "C" int mgbfs_generic_sorted_native_owner_snapshot_release(const void* pool,void* snapshot,void* error,void* stream){
 if(!pool||!snapshot||!error)return cudaErrorInvalidValue;
 sorted_owner_release_checked<<<1,1,0,(cudaStream_t)stream>>>(*(const GenericSortedRunPool*)pool,(GenericSortedHistorySnapshot*)snapshot,(uint32_t*)error);return cudaGetLastError();
}
