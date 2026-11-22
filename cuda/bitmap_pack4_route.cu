#include "bitmap_pack4_route.cuh"
#include "bitmap_pack4_action.cuh"
#include "generic_action.cuh"
#include "bitmap_owner_page.cuh"
#include "segmented_warp_route.cuh"
#include <cuda_runtime.h>

template<bool ByteWire,bool Grouped>
__global__ void route_pack4(MgbfsPack4RouteArgs a,BitmapPack4Action action){
    const uint32_t children=a.count*a.generators;
    for(uint32_t child=blockIdx.x*blockDim.x+threadIdx.x;child<children;child+=blockDim.x*gridDim.x){
        uint32_t invalid=0;
        if(!action.valid(child,&invalid)){atomicOr(a.error,8u);continue;}
        uint64_t id=0;
        if(!bitmap_rank(a.domain,action,child,&id)){atomicOr(a.error,8u);continue;}
        id=mgbfs_bitmap_permute_owner_id(id,a.world,a.owner_page,a.owner_boundary);
        uint32_t owner=uint32_t(id%a.world),target=a.map?a.map[owner]:owner;
        if(target>=a.world){atomicOr(a.error,2u);continue;}
        uint32_t shard=(uint64_t(uint32_t(mix64(id)))*a.shards)>>32;
        uint32_t queue=target*a.shards+shard;
        uint32_t row=Grouped?mgbfs_warp_route_reserve(a.counts,queue):atomicAdd(a.counts+queue,1u);
        if(row>=a.capacity){atomicOr(a.error,1u);continue;}
        if constexpr(ByteWire){
            uint64_t lo=0,hi=0;
            for(uint32_t e=0;e<action.elements;e++){
                uint64_t v=uint64_t(action.value(child,e));
                if(e<8)lo|=v<<(8*e);else hi|=v<<(8*(e-8));
            }
            a.records[uint64_t(queue)*a.capacity+row]={id,lo,uint32_t(hi),uint32_t(hi>>32),0,0};
        }else{
            a.records[uint64_t(queue)*a.capacity+row]={id,child/a.generators,a.source,child%a.generators,shard,0};
            auto* output=static_cast<int64_t*>(a.payload);
            for(uint32_t e=0;e<action.elements;e++)
                output[uint64_t(queue)*action.elements*a.capacity+uint64_t(e)*a.capacity+row]=action.value(child,e);
        }
    }
}
template<bool ByteWire,bool Grouped>
static int launch(MgbfsPack4RouteArgs a,cudaStream_t stream){
    if(!a.count)return int(cudaSuccess);
    BitmapPack4Action action{a.dictionary.elements,a.generators,a.count,a.parents,a.permutations,a.dictionary};
    uint64_t blocks=(uint64_t(a.count)*a.generators+255)/256;
    route_pack4<ByteWire,Grouped><<<uint32_t(blocks>65535?65535:blocks),256,0,stream>>>(a,action);
    return int(cudaGetLastError());
}
extern "C" int mgbfs_pack4_route(MgbfsPack4RouteArgs a,void* stream){
    const uint64_t page_span=uint64_t(a.owner_page)*a.world;
    if(!mgbfs_pack4_valid(a.dictionary)||a.domain.elements!=a.dictionary.elements||
       !a.domain.symbols||!a.domain.counts||!a.domain.total||!a.parents||!a.permutations||
       !a.generators||uint64_t(a.count)*a.generators>=0x7fffffffULL||
       !a.world||a.world>128||a.source>=a.world||!a.shards||a.shards>4096||!a.capacity||
       (a.state_bytes!=1&&a.state_bytes!=8)||a.grouped>1||!a.records||!a.counts||!a.error||
       (a.state_bytes==8&&!a.payload))return int(cudaErrorInvalidValue);
    if((!a.owner_page&&a.owner_boundary)||(a.owner_page&&
       ((a.owner_page&(a.owner_page-1))||a.owner_page>(1u<<20)||!page_span||
        a.owner_boundary!=a.domain.total-a.domain.total%page_span)))return int(cudaErrorInvalidValue);
    if(a.state_bytes==1)for(uint32_t i=0;i<a.dictionary.distinct;i++)
        if(a.dictionary.symbols[i]<0||a.dictionary.symbols[i]>255)return int(cudaErrorInvalidValue);
    auto st=static_cast<cudaStream_t>(stream);
    if(a.state_bytes==1)return a.grouped?launch<true,true>(a,st):launch<true,false>(a,st);
    return a.grouped?launch<false,true>(a,st):launch<false,false>(a,st);
}
