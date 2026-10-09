#pragma once
#include "generic_action.cuh"
#include "generic_incoming_action.cuh"

template<class State> struct ParentOriginAction:IncomingAllAction<State,false,true>{
 ActionT<State> graph;const State* parents;const uint64_t* cursors;const uint32_t* frontiers;
 uint32_t parent_stride,chunk;uint64_t begin;
 __device__ const GenericRouteRecord& origin(uint32_t child)const{
  const auto* meta=this->source(child)==this->rank?this->local_meta:this->remote_meta;
  return meta[this->queue(child)*this->stride+child%this->stride];
 }
 __device__ bool valid(uint32_t child,uint32_t* error)const{
  if(!IncomingAllAction<State,false,true>::valid(child,error))return false;
  const uint32_t source=this->source(child);const auto& o=origin(child);
  if(source>=this->world||o.source!=source||o.generator>=graph.generators||o.shard!=this->shard||o.reserved||begin>frontiers[source]){
   atomicOr(error,128u);return false;
  }
  const uint32_t available=uint32_t(uint64_t(frontiers[source])-begin);
  if(cursors[source]>UINT64_MAX-begin||o.parent<cursors[source]+begin||o.parent-(cursors[source]+begin)>=uint64_t(min(available,chunk))){atomicOr(error,128u);return false;}
  return true;
 }
 __device__ int64_t value(uint32_t child,uint32_t e)const{
  const uint32_t source=this->source(child);const auto& o=origin(child);
  auto g=graph;g.parents=parents+uint64_t(source)*parent_stride*this->elements;g.stride=parent_stride;
  const uint32_t row=uint32_t(o.parent-cursors[source]-begin);
  return g.value(row*g.generators+o.generator,e);
 }
};
