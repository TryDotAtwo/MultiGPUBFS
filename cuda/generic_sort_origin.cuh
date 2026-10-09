// Sort only indices and coarse hash keys. All exact values stay in leased planes.
#include <cub/device/device_radix_sort.cuh>
__global__ void sort_origin_keys(const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint64_t* keys,uint32_t* origins,uint32_t* error){
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<world*q;i+=blockDim.x*gridDim.x){
  const uint32_t source=i/q,row=i%q,queue=source*shards+shard,child=queue*q+row;const auto* counts=source==rank?lc:rc;const auto* meta=source==rank?lm:rm;
  const uint32_t actual=counts[queue];if(actual>q)atomicOr(error,32u);
  // Every valid key is below2^63. Pads sort last even for hash UINT64_MAX;
  // dropping one ordering bit does not drop any equality/hash information.
  keys[i]=actual<=q&&row<actual?meta[child].hash>>1:UINT64_MAX;origins[i]=child;
 }
}
extern "C" int mgbfs_generic_sort_origins(const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint64_t* keys,uint64_t* sorted_keys,uint32_t* origins,uint32_t* sorted_origins,void* scratch,uint64_t scratch_bytes,uint32_t* error,void* stream){
 if(!lm||!rm||!lc||!rc||!world||world>128||rank>=world||!shards||shards>4096||shard>=shards||!q||uint64_t(world)*shards*q>=0x7fffffffULL||!keys||!sorted_keys||!origins||!sorted_origins||!scratch||!error)return int(cudaErrorInvalidValue);
 const uint32_t n=world*q;size_t required=0;auto st=static_cast<cudaStream_t>(stream);
 auto result=cub::DeviceRadixSort::SortPairs(nullptr,required,keys,sorted_keys,origins,sorted_origins,int(n),0,64,st);if(result!=cudaSuccess)return int(result);if(required>scratch_bytes)return int(cudaErrorInvalidValue);
 sort_origin_keys<<<grid(n),256,0,st>>>(lm,rm,lc,rc,rank,world,shard,shards,q,keys,origins,error);result=cudaGetLastError();if(result!=cudaSuccess)return int(result);
 return int(cub::DeviceRadixSort::SortPairs(scratch,required,keys,sorted_keys,origins,sorted_origins,int(n),0,64,st));
}
template<class State,bool Packed> int accept_sorted(uint32_t elements,const State* local,const State* remote,const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint64_t* slots,uint32_t sc,State* arena,uint32_t stride,uint32_t base,uint32_t capacity,uint32_t* vc,uint32_t* future,uint32_t* accepted,uint32_t* positions,uint32_t* error,uint32_t rolling,const uint32_t* ordered,void* stream){
 if(!elements||(Packed&&elements>24)||!local||!remote||!lm||!rm||!lc||!rc||!world||world>128||rank>=world||!shards||shards>4096||shard>=shards||!q||uint64_t(world)*shards*q>=0x7fffffffULL||!slots||!sc||(sc&(sc-1))||!arena||!stride||base>stride||!capacity||capacity>stride-base||!vc||!future||!accepted||!error||!ordered||rolling>1||(rolling&&!positions))return int(cudaErrorInvalidValue);
 IncomingAllAction<State,Packed,true> action{elements,world*q,q,1,rank,world,shard,shards,local,remote,lm,rm,lc,rc,ordered};
 if(rolling)accept_rolling<<<grid(uint64_t(world)*q),256,0,static_cast<cudaStream_t>(stream)>>>(action,slots,sc,arena,stride,base,capacity,accepted,future,positions,error);
 else accept_candidates<<<grid(uint64_t(world)*q),256,0,static_cast<cudaStream_t>(stream)>>>(action,slots,sc,arena,stride,vc,future,capacity,accepted,0,64,error);
 return int(cudaGetLastError());
}
extern "C" int mgbfs_generic_accept_sorted(uint32_t bytes,uint32_t elements,const void* local,const void* remote,const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint64_t* slots,uint32_t sc,void* arena,uint32_t stride,uint32_t base,uint32_t capacity,uint32_t* vc,uint32_t* future,uint32_t* accepted,uint32_t* positions,uint32_t* error,uint32_t rolling,const uint32_t* ordered,void* stream){
 if(bytes==1){if(elements<=24)return accept_sorted<uint8_t,true>(elements,static_cast<const uint8_t*>(local),static_cast<const uint8_t*>(remote),lm,rm,lc,rc,rank,world,shard,shards,q,slots,sc,static_cast<uint8_t*>(arena),stride,base,capacity,vc,future,accepted,positions,error,rolling,ordered,stream);return accept_sorted<uint8_t,false>(elements,static_cast<const uint8_t*>(local),static_cast<const uint8_t*>(remote),lm,rm,lc,rc,rank,world,shard,shards,q,slots,sc,static_cast<uint8_t*>(arena),stride,base,capacity,vc,future,accepted,positions,error,rolling,ordered,stream);}
 if(bytes==8)return accept_sorted<int64_t,false>(elements,static_cast<const int64_t*>(local),static_cast<const int64_t*>(remote),lm,rm,lc,rc,rank,world,shard,shards,q,slots,sc,static_cast<int64_t*>(arena),stride,base,capacity,vc,future,accepted,positions,error,rolling,ordered,stream);
 return int(cudaErrorInvalidValue);
}
