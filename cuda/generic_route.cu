#include "generic_route.h"
#include "generic_action.cuh"
#include <limits>
static_assert(sizeof(GenericRouteRecord)==32,"route ABI");
__global__ void route(Action action,uint64_t seed,uint32_t bits,uint32_t world,uint32_t source,
 uint32_t shards,uint32_t capacity,uint64_t begin,const uint32_t* map,const uint64_t* cuts,
 GenericRouteRecord* output,uint32_t* counts,uint32_t* error){
 uint32_t children=action.count*action.generators;
 for(uint32_t child=blockIdx.x*blockDim.x+threadIdx.x;child<children;child+=blockDim.x*gridDim.x){
  uint64_t hash=action.hash(child,seed,bits);uint32_t high=hash>>32,owner;
  if(cuts){
   if(cuts[0]!=0||cuts[world]!=(uint64_t(1)<<32)){atomicOr(error,4u);continue;}
   uint32_t low=0,upper=world;
   while(low+1<upper){uint32_t mid=(low+upper)/2;if(uint64_t(high)<cuts[mid])upper=mid;else low=mid;}
   owner=low;if(!(cuts[owner]<=high&&high<cuts[owner+1])){atomicOr(error,4u);continue;}
  }else owner=(uint64_t(high)*world)>>32;
  uint32_t rank=map?map[owner]:owner;if(rank>=world){atomicOr(error,2u);continue;}
  uint32_t shard=(uint64_t(uint32_t(hash))*shards)>>32;uint32_t queue=rank*shards+shard;
  uint32_t row=atomicAdd(counts+queue,1u);if(row>=capacity){atomicOr(error,1u);continue;}
  output[uint64_t(queue)*capacity+row]={hash,begin+child/action.generators,source,child%action.generators,shard,0};
 }
}
extern "C" int mgbfs_generic_route_i64(uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const int64_t* parents,uint32_t count,uint32_t stride,const uint32_t* permutations,const int64_t* matrices,
 const uint32_t* moduli,uint64_t seed,uint32_t bits,uint32_t world,uint32_t source,uint32_t shards,
 uint32_t capacity,uint64_t begin,const uint32_t* map,const uint64_t* cuts,GenericRouteRecord* queues,
 uint32_t* counts,uint32_t* error,void* stream){
 if(kind>1||!elements||!n||!m||!generators||count>stride||uint64_t(count)*generators>=0x7fffffffULL||
  bits>64||!world||world>128||source>=world||!shards||shards>4096||!capacity||begin>UINT64_MAX-count||
  !parents||!queues||!counts||!error||(kind==0&&(!permutations||elements!=n||m!=1))||
  (kind==1&&(!matrices||!moduli||uint64_t(n)*m!=elements)))return int(cudaErrorInvalidValue);
 if(!count)return 0;uint64_t blocks=(uint64_t(count)*generators+255)/256;
 Action action{kind,elements,n,m,generators,count,stride,parents,permutations,matrices,moduli};
 route<<<uint32_t(blocks>65535?65535:blocks),256,0,static_cast<cudaStream_t>(stream)>>>(action,seed,bits,world,source,shards,capacity,begin,map,cuts,queues,counts,error);
 return int(cudaGetLastError());
}

__global__ void regenerate_routes(Action action,uint32_t source,uint64_t begin,const GenericRouteRecord* requests,
 uint32_t count,int64_t* output,uint32_t stride,uint32_t* error){
 for(uint64_t at=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;at<uint64_t(count)*action.elements;at+=uint64_t(blockDim.x)*gridDim.x){
  uint32_t element=at/count,row=at%count;GenericRouteRecord origin=requests[row];
  if(origin.source!=source||origin.parent<begin||origin.parent-begin>=action.count||origin.generator>=action.generators||origin.reserved){atomicOr(error,2u);continue;}
  uint32_t child=uint32_t(origin.parent-begin)*action.generators+origin.generator;
  output[uint64_t(element)*stride+row]=action.value(child,element);
 }
}
extern "C" int mgbfs_generic_regenerate_routes_i64(uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const int64_t* parents,uint32_t count,uint32_t stride,const uint32_t* permutations,const int64_t* matrices,
 const uint32_t* moduli,uint32_t source,uint64_t begin,const GenericRouteRecord* requests,uint32_t request_count,
 int64_t* output,uint32_t output_stride,uint32_t* error,void* stream){
 if(kind>1||!elements||!n||!m||!generators||count>stride||uint64_t(count)*generators>=0x7fffffffULL||source>=128||
  request_count>output_stride||begin>UINT64_MAX-count||!parents||!requests||!output||!error||
  (kind==0&&(!permutations||elements!=n||m!=1))||(kind==1&&(!matrices||!moduli||uint64_t(n)*m!=elements)))return int(cudaErrorInvalidValue);
 if(!request_count)return 0;uint64_t blocks=(uint64_t(request_count)*elements+255)/256;
 Action action{kind,elements,n,m,generators,count,stride,parents,permutations,matrices,moduli};
 regenerate_routes<<<uint32_t(blocks>65535?65535:blocks),256,0,static_cast<cudaStream_t>(stream)>>>(action,source,begin,requests,request_count,output,output_stride,error);
 return int(cudaGetLastError());
}
