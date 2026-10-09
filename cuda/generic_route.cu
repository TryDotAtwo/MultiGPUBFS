#include "generic_route.h"
#include "generic_action.cuh"
#include <limits>
static_assert(sizeof(GenericRouteRecord)==32,"route ABI");
template<class State,bool Packed=false> __global__ void route(ActionT<State> action,uint64_t seed,uint32_t bits,uint32_t world,uint32_t source,
 uint32_t shards,uint32_t capacity,uint64_t begin,const uint32_t* map,const uint64_t* cuts,
 GenericRouteRecord* output,uint32_t* counts,uint32_t* error){
 uint32_t children=action.count*action.generators;
 for(uint32_t child=blockIdx.x*blockDim.x+threadIdx.x;child<children;child+=blockDim.x*gridDim.x){
  uint64_t hash,lo=0,hi=0,tail=0;
  if constexpr(Packed){
   uint64_t h=seed;
   for(uint32_t e=0;e<action.elements;e++){
    const uint64_t v=uint64_t(action.value(child,e));h=mix64(h^v^uint64_t(e));
    if(e<8)lo|=v<<(8*e);else if(e<16)hi|=v<<(8*(e-8));else tail|=v<<(8*(e-16));
   }
   hash=finish_hash(h,bits);
  }else hash=action.hash(child,seed,bits);
  uint32_t high=hash>>32,owner;
  if(cuts){
   if(cuts[0]!=0||cuts[world]!=(uint64_t(1)<<32)){atomicOr(error,4u);continue;}
   uint32_t low=0,upper=world;
   while(low+1<upper){uint32_t mid=(low+upper)/2;if(uint64_t(high)<cuts[mid])upper=mid;else low=mid;}
   owner=low;if(!(cuts[owner]<=high&&high<cuts[owner+1])){atomicOr(error,4u);continue;}
  }else owner=(uint64_t(high)*world)>>32;
  uint32_t rank=map?map[owner]:owner;if(rank>=world){atomicOr(error,2u);continue;}
  uint32_t shard=(uint64_t(uint32_t(hash))*shards)>>32;uint32_t queue=rank*shards+shard;
  uint32_t row=atomicAdd(counts+queue,1u);if(row>=capacity){atomicOr(error,1u);continue;}
  if constexpr(Packed){
   // Exact bytes, not a probabilistic identity. Origin is diagnostic only.
   const uint64_t origin=(begin+child/action.generators)*action.generators+child%action.generators;
   output[uint64_t(queue)*capacity+row]={hash,lo,uint32_t(hi),uint32_t(hi>>32),uint32_t(tail),uint32_t(tail>>32)};
  }else output[uint64_t(queue)*capacity+row]={hash,begin+child/action.generators,source,child%action.generators,shard,0};
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

template<class State> __global__ void regenerate_routes(ActionT<State> action,uint32_t source,uint64_t begin,const GenericRouteRecord* requests,
 uint32_t count,const uint32_t* device_count,State* output,uint32_t stride,uint32_t* error){
 for(uint64_t at=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;at<uint64_t(count)*action.elements;at+=uint64_t(blockDim.x)*gridDim.x){
  uint32_t element=at/count,row=at%count;
  uint32_t actual=device_count?*device_count:count;if(actual>count){atomicOr(error,4u);continue;}if(row>=actual)continue;
  GenericRouteRecord origin=requests[row];
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
 regenerate_routes<<<uint32_t(blocks>65535?65535:blocks),256,0,static_cast<cudaStream_t>(stream)>>>(action,source,begin,requests,request_count,nullptr,output,output_stride,error);
 return int(cudaGetLastError());
}

extern "C" int mgbfs_generic_regenerate_routes_count_i64(uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const int64_t* parents,uint32_t count,uint32_t stride,const uint32_t* permutations,const int64_t* matrices,
 const uint32_t* moduli,uint32_t source,uint64_t begin,const GenericRouteRecord* requests,uint32_t request_count,const uint32_t* device_count,
 int64_t* output,uint32_t output_stride,uint32_t* error,void* stream){
 if(kind>1||!elements||!n||!m||!generators||count>stride||uint64_t(count)*generators>=0x7fffffffULL||source>=128||
  request_count>output_stride||begin>UINT64_MAX-count||!parents||!requests||!device_count||!output||!error||
  (kind==0&&(!permutations||elements!=n||m!=1))||(kind==1&&(!matrices||!moduli||uint64_t(n)*m!=elements)))return int(cudaErrorInvalidValue);
 if(!request_count)return 0;uint64_t blocks=(uint64_t(request_count)*elements+255)/256;
 Action action{kind,elements,n,m,generators,count,stride,parents,permutations,matrices,moduli};
 regenerate_routes<<<uint32_t(blocks>65535?65535:blocks),256,0,static_cast<cudaStream_t>(stream)>>>(action,source,begin,requests,request_count,device_count,output,output_stride,error);
 return int(cudaGetLastError());
}

// Compact permutation payload ABI shares the exact same kernel logic.
extern "C" int mgbfs_generic_route_u8(uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const uint8_t* parents,uint32_t count,uint32_t stride,const uint32_t* permutations,const int64_t* matrices,
 const uint32_t* moduli,uint64_t seed,uint32_t bits,uint32_t world,uint32_t source,uint32_t shards,
 uint32_t capacity,uint64_t begin,const uint32_t* map,const uint64_t* cuts,GenericRouteRecord* queues,
 uint32_t* counts,uint32_t* error,void* stream){
 if(kind>0||!elements||!n||!m||!generators||count>stride||uint64_t(count)*generators>=0x7fffffffULL||
  bits>64||!world||world>128||source>=world||!shards||shards>4096||!capacity||begin>UINT64_MAX-count||
  !parents||!queues||!counts||!error||(kind==0&&(!permutations||elements!=n||m!=1))||
  (kind==1&&(!matrices||!moduli||uint64_t(n)*m!=elements)))return int(cudaErrorInvalidValue);
 if(!count)return 0;uint64_t blocks=(uint64_t(count)*generators+255)/256;
 ActionT<uint8_t> action{kind,elements,n,m,generators,count,stride,parents,permutations,matrices,moduli};
 route<<<uint32_t(blocks>65535?65535:blocks),256,0,static_cast<cudaStream_t>(stream)>>>(action,seed,bits,world,source,shards,capacity,begin,map,cuts,queues,counts,error);
 return int(cudaGetLastError());
}
extern "C" int mgbfs_generic_regenerate_routes_u8(uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const uint8_t* parents,uint32_t count,uint32_t stride,const uint32_t* permutations,const int64_t* matrices,
 const uint32_t* moduli,uint32_t source,uint64_t begin,const GenericRouteRecord* requests,uint32_t request_count,
 uint8_t* output,uint32_t output_stride,uint32_t* error,void* stream){
 if(kind>0||!elements||!n||!m||!generators||count>stride||uint64_t(count)*generators>=0x7fffffffULL||source>=128||
  request_count>output_stride||begin>UINT64_MAX-count||!parents||!requests||!output||!error||
  (kind==0&&(!permutations||elements!=n||m!=1))||(kind==1&&(!matrices||!moduli||uint64_t(n)*m!=elements)))return int(cudaErrorInvalidValue);
 if(!request_count)return 0;uint64_t blocks=(uint64_t(request_count)*elements+255)/256;
 ActionT<uint8_t> action{kind,elements,n,m,generators,count,stride,parents,permutations,matrices,moduli};
 regenerate_routes<<<uint32_t(blocks>65535?65535:blocks),256,0,static_cast<cudaStream_t>(stream)>>>(action,source,begin,requests,request_count,nullptr,output,output_stride,error);
 return int(cudaGetLastError());
}
extern "C" int mgbfs_generic_regenerate_routes_count_u8(uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const uint8_t* parents,uint32_t count,uint32_t stride,const uint32_t* permutations,const int64_t* matrices,
 const uint32_t* moduli,uint32_t source,uint64_t begin,const GenericRouteRecord* requests,uint32_t request_count,const uint32_t* device_count,
 uint8_t* output,uint32_t output_stride,uint32_t* error,void* stream){
 if(kind>0||!elements||!n||!m||!generators||count>stride||uint64_t(count)*generators>=0x7fffffffULL||source>=128||
  request_count>output_stride||begin>UINT64_MAX-count||!parents||!requests||!device_count||!output||!error||
  (kind==0&&(!permutations||elements!=n||m!=1))||(kind==1&&(!matrices||!moduli||uint64_t(n)*m!=elements)))return int(cudaErrorInvalidValue);
 if(!request_count)return 0;uint64_t blocks=(uint64_t(request_count)*elements+255)/256;
 ActionT<uint8_t> action{kind,elements,n,m,generators,count,stride,parents,permutations,matrices,moduli};
 regenerate_routes<<<uint32_t(blocks>65535?65535:blocks),256,0,static_cast<cudaStream_t>(stream)>>>(action,source,begin,requests,request_count,device_count,output,output_stride,error);
 return int(cudaGetLastError());
}

// Distinguish retryable source skew from owner/metadata errors. Owners from
// earlier successful rounds are never reset or confused with source overflow.
__global__ void classify_route_retry(const uint32_t* source,const uint32_t* owner,uint32_t* vote){
 if(threadIdx.x==0&&blockIdx.x==0){const uint32_t s=*source,o=*owner;*vote=o?0x100u|o:((s==1u||s==5u)?1u:(s?0x10000u|s:0u));}
}
extern "C" int mgbfs_generic_route_retry_vote(const uint32_t* source,const uint32_t* owner,uint32_t* vote,void* stream){
 if(!source||!owner||!vote)return 1;classify_route_retry<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(source,owner,vote);return cudaGetLastError()==cudaSuccess?0:2;
}

extern "C" int mgbfs_generic_route_packed_u8(uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const uint8_t* parents,uint32_t count,uint32_t stride,const uint32_t* permutations,const int64_t* matrices,
 const uint32_t* moduli,uint64_t seed,uint32_t bits,uint32_t world,uint32_t source,uint32_t shards,
 uint32_t capacity,uint64_t begin,const uint32_t* map,const uint64_t* cuts,GenericRouteRecord* queues,
 uint32_t* counts,uint32_t* error,void* stream){
 if(kind>0||elements>24||!elements||!n||!m||!generators||count>stride||uint64_t(count)*generators>=0x7fffffffULL||
  bits>64||!world||world>128||source>=world||!shards||shards>4096||!capacity||begin>UINT64_MAX-count||
  !parents||!queues||!counts||!error||(kind==0&&(!permutations||elements!=n||m!=1))||
  (kind==1&&(!matrices||!moduli||uint64_t(n)*m!=elements)))return int(cudaErrorInvalidValue);
 if(!count)return 0;uint64_t blocks=(uint64_t(count)*generators+255)/256;
 ActionT<uint8_t> action{kind,elements,n,m,generators,count,stride,parents,permutations,matrices,moduli};
 route<uint8_t,true><<<uint32_t(blocks>65535?65535:blocks),256,0,static_cast<cudaStream_t>(stream)>>>(action,seed,bits,world,source,shards,capacity,begin,map,cuts,queues,counts,error);
 return int(cudaGetLastError());
}
