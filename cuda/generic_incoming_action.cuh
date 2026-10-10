#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "generic_route.h"

template<class State,bool Packed=false,bool Shared=false> struct IncomingAllAction {
 uint32_t elements,count,stride,generators,rank,world,shard,shards;
 const State* local;const State* remote;const GenericRouteRecord* local_meta;const GenericRouteRecord* remote_meta;
 const uint32_t* local_counts;const uint32_t* remote_counts;const uint32_t* ordered=nullptr;
 __device__ uint32_t child(uint32_t i)const{if(ordered)return ordered[i];return Shared?((i/stride)*shards+shard)*stride+i%stride:i;}
 __device__ uint32_t origin_limit()const{return Shared?count*shards:count;}
 __device__ uint64_t bucket(uint64_t h,uint64_t capacity)const{return Shared?mgbfs_shared_table_bucket(h,capacity):uint64_t(h&(capacity-1));}
 __device__ uint32_t source(uint32_t child)const{return Shared?(child/stride)/shards:child/stride;}
 __device__ uint64_t queue(uint32_t child)const{return Shared?child/stride:uint64_t(source(child))*shards+shard;}
 __device__ bool valid(uint32_t child,uint32_t* error)const{
  // Validate an ordered origin before dereferencing queue counters. Full and
  // packed inputs share the same queue geometry; payload fields may be packed.
  if(child>=origin_limit()){atomicOr(error,128u);return false;}
  const uint32_t owner_source=source(child);const uint64_t owner_queue=queue(child);
  if(owner_source>=world){atomicOr(error,128u);return false;}
  if constexpr(Shared)if(owner_queue!=uint64_t(owner_source)*shards+shard){atomicOr(error,128u);return false;}
  const uint32_t* counts=owner_source==rank?local_counts:remote_counts;
  uint32_t actual=counts[owner_queue];if(actual>stride){atomicOr(error,32u);return false;}return child%stride<actual;
 }
 __device__ int64_t value(uint32_t child,uint32_t e)const{
  if constexpr(Packed){
   const GenericRouteRecord* meta=source(child)==rank?local_meta:remote_meta;
   const auto v=meta[queue(child)*stride+child%stride];
   const uint64_t word=e<8?v.parent:(e<16?(uint64_t(v.source)|(uint64_t(v.generator)<<32)):(uint64_t(v.shard)|(uint64_t(v.reserved)<<32)));
   return int64_t((word>>(8*(e%8)))&255u);
  }else{
   const State* states=source(child)==rank?local:remote;
   return states[queue(child)*uint64_t(elements)*stride+uint64_t(e)*stride+child%stride];
  }
 }
 __device__ uint64_t hash(uint32_t child,uint64_t,uint32_t)const{
  const GenericRouteRecord* meta=source(child)==rank?local_meta:remote_meta;return meta[queue(child)*stride+child%stride].hash;
 }
};
