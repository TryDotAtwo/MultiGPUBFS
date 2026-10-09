// Wide exact candidates carry origins, not materialized child planes.
// The all-gathered SoA parent chunk is immutable until all shard owners retire.
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
template<class State> __global__ void pack_parent_chunk(uint32_t elements,const State* source,uint32_t source_stride,uint32_t count,State* out,uint32_t stride){
 for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<uint64_t(elements)*count;i+=uint64_t(blockDim.x)*gridDim.x){
  const uint32_t e=i/count,row=i%count;out[uint64_t(e)*stride+row]=source[uint64_t(e)*source_stride+row];
 }
}
extern "C" int mgbfs_generic_pack_parent_chunk(uint32_t bytes,uint32_t elements,const void* source,uint32_t source_stride,uint32_t count,void* out,uint32_t stride,void* stream){
 if((bytes!=1&&bytes!=8)||!elements||count>source_stride||count>stride||!source||!out)return int(cudaErrorInvalidValue);
 if(!count)return 0;auto st=static_cast<cudaStream_t>(stream);
 if(bytes==1)pack_parent_chunk<<<grid(uint64_t(elements)*count),256,0,st>>>(elements,static_cast<const uint8_t*>(source),source_stride,count,static_cast<uint8_t*>(out),stride);
 else pack_parent_chunk<<<grid(uint64_t(elements)*count),256,0,st>>>(elements,static_cast<const int64_t*>(source),source_stride,count,static_cast<int64_t*>(out),stride);
 return int(cudaGetLastError());
}
__global__ void advance_parent_cursors(uint64_t* cursors,uint32_t* frontiers,const uint32_t* next,uint32_t world){
 const uint32_t r=blockIdx.x*blockDim.x+threadIdx.x;if(r<world){cursors[r]+=frontiers[r];frontiers[r]=next[r];}
}
extern "C" int mgbfs_generic_advance_parent_cursors(uint64_t* cursors,uint32_t* frontiers,const uint32_t* next,uint32_t world,void* stream){
 if(!cursors||!frontiers||!next||!world||world>128)return int(cudaErrorInvalidValue);
 advance_parent_cursors<<<1,128,0,static_cast<cudaStream_t>(stream)>>>(cursors,frontiers,next,world);return int(cudaGetLastError());
}
template<class State> int accept_parent_origin(uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,const uint32_t* permutations,const int64_t* matrices,const uint32_t* moduli,
 const State* parents,uint32_t parent_stride,uint32_t chunk,uint64_t begin,const uint64_t* cursors,const uint32_t* frontiers,
 const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,
 uint64_t* slots,uint32_t sc,State* arena,uint32_t stride,uint32_t base,uint32_t capacity,uint32_t* visited_count,uint32_t* future,uint32_t* accepted,uint32_t* positions,uint32_t* error,uint32_t rolling,const uint32_t* ordered,void* stream){
 if(kind>1||!elements||!n||!m||!generators||!parents||!parent_stride||!chunk||chunk>parent_stride||uint64_t(parent_stride)*generators>=0x7fffffffULL||!cursors||!frontiers||!lm||!rm||!lc||!rc||!world||world>128||rank>=world||!shards||shards>4096||shard>=shards||!q||uint64_t(world)*shards*q>=0x7fffffffULL||!sc||(sc&(sc-1))||!slots||!arena||!stride||stride>=0x80000000U||base>stride||!capacity||capacity>stride-base||!visited_count||!future||!accepted||!error||rolling>1||(rolling&&!positions)||(kind==0&&(!permutations||elements!=n||m!=1))||(kind==1&&(!matrices||!moduli||uint64_t(n)*m!=elements)))return int(cudaErrorInvalidValue);
 IncomingAllAction<State,false,true> incoming{elements,world*q,q,1,rank,world,shard,shards,nullptr,nullptr,lm,rm,lc,rc,ordered};
 ActionT<State> graph{kind,elements,n,m,generators,parent_stride,parent_stride,nullptr,permutations,matrices,moduli};
 ParentOriginAction<State> action{incoming,graph,parents,cursors,frontiers,parent_stride,chunk,begin};
 if(rolling)accept_rolling<<<grid(uint64_t(world)*q),256,0,static_cast<cudaStream_t>(stream)>>>(action,slots,sc,arena,stride,base,capacity,accepted,future,positions,error);
 else accept_candidates<<<grid(uint64_t(world)*q),256,0,static_cast<cudaStream_t>(stream)>>>(action,slots,sc,arena,stride,visited_count,future,capacity,accepted,0,64,error);
 return int(cudaGetLastError());
}
extern "C" int mgbfs_generic_accept_parent_origin(uint32_t bytes,uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,const uint32_t* permutations,const int64_t* matrices,const uint32_t* moduli,
 const void* parents,uint32_t parent_stride,uint32_t chunk,uint64_t begin,const uint64_t* cursors,const uint32_t* frontiers,
 const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,
 uint64_t* slots,uint32_t sc,void* arena,uint32_t stride,uint32_t base,uint32_t capacity,uint32_t* visited_count,uint32_t* future,uint32_t* accepted,uint32_t* positions,uint32_t* error,uint32_t rolling,const uint32_t* ordered,void* stream){
 if(bytes==1)return accept_parent_origin(kind,elements,n,m,generators,permutations,matrices,moduli,static_cast<const uint8_t*>(parents),parent_stride,chunk,begin,cursors,frontiers,lm,rm,lc,rc,rank,world,shard,shards,q,slots,sc,static_cast<uint8_t*>(arena),stride,base,capacity,visited_count,future,accepted,positions,error,rolling,ordered,stream);
 if(bytes==8)return accept_parent_origin(kind,elements,n,m,generators,permutations,matrices,moduli,static_cast<const int64_t*>(parents),parent_stride,chunk,begin,cursors,frontiers,lm,rm,lc,rc,rank,world,shard,shards,q,slots,sc,static_cast<int64_t*>(arena),stride,base,capacity,visited_count,future,accepted,positions,error,rolling,ordered,stream);
 return int(cudaErrorInvalidValue);
}
