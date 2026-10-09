#include "generic_sorted_run_tiers.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CUDA_CHECK(x) do{auto cuda_status_=(x);if(cuda_status_!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(cuda_status_));exit(2);}}while(0)
__device__ GenericSortedRunToken new_root(GenericSortedRunPool pool,uint32_t cls,uint32_t count,uint32_t* error,uint32_t* failures){
 GenericSortedRunToken token{};
 if(generic_sorted_run_allocate(pool,cls,0,&token,error)!=SORTED_RUN_ACQUIRED||!generic_sorted_run_publish(pool,token,count,error))++*failures;
 return token;
}
__device__ void retry_prepare(GenericSortedRunPool pool,GenericSortedRunTiers* tiers,
 GenericSortedRunCarry* carry,uint64_t* hashes,uint32_t* rows,uint32_t* error,uint32_t* failures){
 carry->stage=SORTED_CARRY_NEXT;generic_sorted_tier_prepare(pool,tiers,carry,hashes,rows,error);
 if(carry->stage!=SORTED_CARRY_MERGE)++*failures;
}
__global__ void tier_contract(GenericSortedRunPool pool,GenericSortedRunTiers* tiers,
 GenericSortedRunCarry* carry,uint64_t* hashes,uint32_t* rows,uint32_t* error,uint32_t* failures){
 if(blockIdx.x||threadIdx.x)return;
 auto first=new_root(pool,0,4,error,failures);carry->token=first;carry->valid=1;
 generic_sorted_tier_prepare(pool,tiers,carry,hashes,rows,error);
 if(carry->valid||carry->stage!=SORTED_CARRY_STORED||tiers->present!=1)++*failures;
 auto second=new_root(pool,0,4,error,failures);carry->token=second;carry->valid=1;
 retry_prepare(pool,tiers,carry,hashes,rows,error,failures);auto destination=carry->ticket.destination;
 generic_sorted_tier_abort(pool,carry,error);
 if(tiers->present!=1||!generic_sorted_same_token(tiers->tokens[0],first)||!generic_sorted_same_token(carry->token,second)||!carry->valid)++*failures;
 if((pool.descriptors[first.slot].lease&SORTED_RUN_REFS)!=1||(pool.descriptors[second.slot].lease&SORTED_RUN_REFS)!=1||(pool.descriptors[destination.slot].lease&0xffffffffull))++*failures;
 // Invalid union counts and an already failed merge preserve both input owners.
 retry_prepare(pool,tiers,carry,hashes,rows,error,failures);generic_sorted_tier_commit(pool,tiers,carry,0,error);
 if(*error!=32||carry->stage!=SORTED_CARRY_FAILED||tiers->present!=1)++*failures;*error=0;
 retry_prepare(pool,tiers,carry,hashes,rows,error,failures);generic_sorted_tier_commit(pool,tiers,carry,9,error);
 if(*error!=32||carry->stage!=SORTED_CARRY_FAILED||tiers->present!=1)++*failures;*error=0;
 retry_prepare(pool,tiers,carry,hashes,rows,error,failures);*error=8;generic_sorted_tier_commit(pool,tiers,carry,4,error);
 if(*error!=8||carry->stage!=SORTED_CARRY_FAILED||tiers->present!=1)++*failures;*error=0;
 retry_prepare(pool,tiers,carry,hashes,rows,error,failures);generic_sorted_tier_commit(pool,tiers,carry,4,error);
 if(carry->stage!=SORTED_CARRY_NEXT||tiers->present||!carry->valid||pool.descriptors[carry->token.slot].size_class!=0)++*failures;
 generic_sorted_tier_prepare(pool,tiers,carry,hashes,rows,error);
 if(carry->stage!=SORTED_CARRY_STORED||tiers->present!=1)++*failures;
 // Two disjoint groups of4 fill one page after trim; adding4 promotes to class1.
 carry->token=new_root(pool,0,4,error,failures);carry->valid=1;
 retry_prepare(pool,tiers,carry,hashes,rows,error,failures);generic_sorted_tier_commit(pool,tiers,carry,8,error);
 generic_sorted_tier_prepare(pool,tiers,carry,hashes,rows,error);
 carry->token=new_root(pool,0,4,error,failures);carry->valid=1;
 retry_prepare(pool,tiers,carry,hashes,rows,error,failures);generic_sorted_tier_commit(pool,tiers,carry,12,error);
 generic_sorted_tier_prepare(pool,tiers,carry,hashes,rows,error);
 if(tiers->present!=2||pool.descriptors[tiers->tokens[1].slot].count!=12)++*failures;
 generic_sorted_run_release(pool,tiers->tokens[1],true,error);tiers->present=0;
 // Empty input is dropped before occupying a tier.
 carry->token=new_root(pool,0,0,error,failures);carry->valid=1;carry->stage=SORTED_CARRY_NEXT;
 generic_sorted_tier_prepare(pool,tiers,carry,hashes,rows,error);
 if(carry->valid||tiers->present||carry->stage!=SORTED_CARRY_IDLE)++*failures;
 for(uint32_t i=0;i<pool.regions;++i)if(pool.occupied[i])++*failures;
}
__global__ void tier_pressure_contract(GenericSortedRunPool pool,GenericSortedRunTiers* tiers,
 GenericSortedRunCarry* carry,uint64_t* hashes,uint32_t* rows,uint32_t* error,uint32_t* failures){
 if(blockIdx.x||threadIdx.x)return;
 auto left=new_root(pool,4,64,error,failures);tiers->tokens[4]=left;tiers->present=1u<<4;
 auto right=new_root(pool,4,64,error,failures),blocker=new_root(pool,0,1,error,failures);
 carry->token=right;carry->valid=1;carry->stage=SORTED_CARRY_NEXT;
 generic_sorted_tier_prepare(pool,tiers,carry,hashes,rows,error);
 if(carry->stage!=SORTED_CARRY_PRESSURE||*error||tiers->present!=(1u<<4)||!carry->valid)++*failures;
 if((pool.descriptors[left.slot].lease&SORTED_RUN_REFS)!=1||(pool.descriptors[right.slot].lease&SORTED_RUN_REFS)!=1)++*failures;
 generic_sorted_run_release(pool,blocker,true,error);
 retry_prepare(pool,tiers,carry,hashes,rows,error,failures);generic_sorted_tier_abort(pool,carry,error);
 generic_sorted_run_release(pool,left,true,error);generic_sorted_run_release(pool,right,true,error);tiers->present=0;carry->valid=0;
 for(uint32_t i=0;i<pool.regions;++i)if(pool.occupied[i])++*failures;
}
int main(){int devices=0;CUDA_CHECK(cudaGetDeviceCount(&devices));if(devices<2)return 5;
 for(int device=0;device<2;++device)for(int pressure=0;pressure<2;++pressure){CUDA_CHECK(cudaSetDevice(device));
  unsigned long long* occupied;GenericSortedRunDescriptor* descriptors;GenericSortedRunTiers* tiers;GenericSortedRunCarry* carry;uint64_t* hashes;uint32_t *rows,*error,*failures;
  uint32_t regions=pressure?1:4;GenericSortedRunPoolShape shape{};if(!generic_sorted_run_pool_shape(regions,8,&shape))return 6;
  CUDA_CHECK(cudaMalloc(&occupied,shape.bitmap_bytes));CUDA_CHECK(cudaMalloc(&descriptors,shape.descriptor_bytes));CUDA_CHECK(cudaMalloc(&hashes,shape.hash_bytes));CUDA_CHECK(cudaMalloc(&rows,shape.row_bytes));
  CUDA_CHECK(cudaMalloc(&tiers,sizeof(*tiers)));CUDA_CHECK(cudaMalloc(&carry,sizeof(*carry)));CUDA_CHECK(cudaMalloc(&error,4));CUDA_CHECK(cudaMalloc(&failures,4));
  CUDA_CHECK(cudaMemset(occupied,0,shape.bitmap_bytes));CUDA_CHECK(cudaMemset(descriptors,0,shape.descriptor_bytes));CUDA_CHECK(cudaMemset(tiers,0,sizeof(*tiers)));CUDA_CHECK(cudaMemset(carry,0,sizeof(*carry)));CUDA_CHECK(cudaMemset(error,0,4));CUDA_CHECK(cudaMemset(failures,0,4));
  GenericSortedRunPool pool{occupied,descriptors,regions,8};
  if(pressure)tier_pressure_contract<<<1,1>>>(pool,tiers,carry,hashes,rows,error,failures);else tier_contract<<<1,1>>>(pool,tiers,carry,hashes,rows,error,failures);
  CUDA_CHECK(cudaDeviceSynchronize());uint32_t failed=0,e=0;CUDA_CHECK(cudaMemcpy(&failed,failures,4,cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(&e,error,4,cudaMemcpyDeviceToHost));if(failed||e){fprintf(stderr,"tier contract failed %u %u\n",failed,e);return 3;}
  CUDA_CHECK(cudaFree(occupied));CUDA_CHECK(cudaFree(descriptors));CUDA_CHECK(cudaFree(hashes));CUDA_CHECK(cudaFree(rows));CUDA_CHECK(cudaFree(tiers));CUDA_CHECK(cudaFree(carry));CUDA_CHECK(cudaFree(error));CUDA_CHECK(cudaFree(failures));printf("SORTED_TIER_METADATA_PASS device=%d pressure=%d\n",device,pressure);
 }
}
