#pragma once
#include "generic_sorted_run_merge.cuh"
#include "generic_sorted_run_tiers.cuh"
// One direct merge pass; only a fixed shared tile is needed, no global staging.
struct GenericSortedMergeTile {
 uint32_t a0,b0,a1,b1,total,valid;uint64_t hashes[256];uint32_t rows[256];
};
template<class State> __device__ uint32_t generic_sorted_block_unique_flag(
 GenericSortedHistoryRun a,GenericSortedHistoryRun b,const State* arena,uint32_t stride,
 uint32_t width,uint32_t capacity,uint32_t* error,GenericSortedMergeTile* tile){
 uint32_t lane=threadIdx.x,start=blockIdx.x*256;
 if(!lane){uint64_t total=uint64_t(a.count)+b.count;tile->valid=a.count<=capacity/2&&b.count<=capacity/2&&total<=capacity;
  if(!tile->valid){atomicOr(error,32u);tile->total=0;tile->a0=tile->b0=tile->a1=tile->b1=0;}
  else{tile->total=uint32_t(total);uint32_t end=start<tile->total?min(tile->total,start+256):start;
   if(start<tile->total){tile->a0=generic_sorted_run_partition(start,a,b,arena,stride,width,error);tile->b0=start-tile->a0;
    tile->a1=generic_sorted_run_partition(end,a,b,arena,stride,width,error);tile->b1=end-tile->a1;}
   else tile->a0=tile->b0=tile->a1=tile->b1=0;}}
 __syncthreads();bool valid=tile->valid&&start+lane<tile->total;uint64_t hash=0;uint32_t row=0;
 if(valid){GenericSortedHistoryRun at{a.hashes?a.hashes+tile->a0:nullptr,a.rows?a.rows+tile->a0:nullptr,tile->a1-tile->a0},bt{b.hashes?b.hashes+tile->b0:nullptr,b.rows?b.rows+tile->b0:nullptr,tile->b1-tile->b0};
  uint32_t ai=generic_sorted_run_partition(lane,at,bt,arena,stride,width,error),bi=lane-ai;
  bool left=ai<at.count&&(bi>=bt.count||generic_sorted_run_compare(at.hashes[ai],at.rows[ai],bt.hashes[bi],bt.rows[bi],arena,stride,width,error)<=0);
  hash=left?at.hashes[ai]:bt.hashes[bi];row=left?at.rows[ai]:bt.rows[bi];}
 tile->hashes[lane]=hash;tile->rows[lane]=row;__syncthreads();
 if(!valid)return 0;if(row>=stride){atomicOr(error,8u);return 0;}if(!start&&!lane)return 1;
 uint64_t prior_hash;uint32_t prior_row;
 if(lane){prior_hash=tile->hashes[lane-1];prior_row=tile->rows[lane-1];}
 else{uint32_t diagonal=start-1,ai=generic_sorted_run_partition(diagonal,a,b,arena,stride,width,error),bi=diagonal-ai;
  bool left=ai<a.count&&(bi>=b.count||generic_sorted_run_compare(a.hashes[ai],a.rows[ai],b.hashes[bi],b.rows[bi],arena,stride,width,error)<=0);
  prior_hash=left?a.hashes[ai]:b.hashes[bi];prior_row=left?a.rows[ai]:b.rows[bi];}
 return generic_sorted_run_compare(prior_hash,prior_row,hash,row,arena,stride,width,error)!=0;
}

static constexpr uint32_t GENERIC_SORTED_DISJOINT_DUPLICATE=2048u;
// Native acceptance filters each new run against every retained future run.
// This merge publishes the sum directly. Unexpected duplicates are an invariant
// failure: the destination is aborted and both input owners remain intact.
template<class State> __global__ void generic_sorted_disjoint_merge(
 GenericSortedRunCarry* carry,const GenericSortedHistoryRun* av,const GenericSortedHistoryRun* bv,
 const State* arena,uint32_t stride,uint32_t width,uint32_t capacity,uint32_t* count,uint32_t* error){
 __shared__ GenericSortedMergeTile tile;
 uint32_t flag=generic_sorted_block_unique_flag(*av,*bv,arena,stride,width,capacity,error,&tile);
 if(!blockIdx.x&&!threadIdx.x)*count=tile.total;
 uint32_t position=blockIdx.x*256+threadIdx.x;
 if(tile.valid&&position<tile.total){
  if(!flag)atomicOr(error,GENERIC_SORTED_DISJOINT_DUPLICATE);
  auto ticket=carry->ticket;
  if(position>=ticket.destination_capacity){atomicOr(error,32u);return;}
  ticket.destination_hashes[position]=tile.hashes[threadIdx.x];ticket.destination_rows[position]=tile.rows[threadIdx.x];
 }
}
