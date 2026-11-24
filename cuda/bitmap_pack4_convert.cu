#include "bitmap_pack4.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <type_traits>

// A codec-only ABI. This does not advertise a usable packed BFS backend.
// Buffers are separate, non-overlapping device allocations; in-place
// conversion and aliased status/output buffers are not supported.
extern "C" uint32_t mgbfs_pack4_codec_abi_version() { return 2; }

template<class T>
__global__ void pack_soa(MgbfsPack4Dictionary dict,const T* input,uint64_t stride,uint64_t count,
                         uint64_t* packed,uint8_t* valid) {
    uint64_t step=uint64_t(blockDim.x)*gridDim.x;
    for(uint64_t row=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;row<count;row+=step) {
        int64_t values[64];
        for(uint32_t e=0;e<dict.elements;++e) values[e]=int64_t(input[uint64_t(e)*stride+row]);
        uint64_t word=0;
        const bool ok=mgbfs_pack4_encode(dict,values,&word);
        if(ok) packed[row]=word;
        valid[row]=uint8_t(ok);
    }
}

template<class T>
__global__ void unpack_soa(MgbfsPack4Dictionary dict,const uint64_t* packed,uint64_t stride,uint64_t count,
                           T* output,uint8_t* valid) {
    uint64_t step=uint64_t(blockDim.x)*gridDim.x;
    for(uint64_t row=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;row<count;row+=step) {
        int64_t values[64];
        const bool ok=mgbfs_pack4_decode(dict,packed[row],values);
        if(ok) for(uint32_t e=0;e<dict.elements;++e) output[uint64_t(e)*stride+row]=T(values[e]);
        valid[row]=uint8_t(ok);
    }
}

template<class T>
static bool valid_arguments(const MgbfsPack4Dictionary& dict,uint64_t stride,uint64_t count) {
    if(!mgbfs_pack4_valid(dict) || count>stride ||
       stride>uint64_t(SIZE_MAX)/sizeof(T)/dict.elements ||
       count>uint64_t(SIZE_MAX)/sizeof(uint64_t)) return false;
    if constexpr(std::is_same<T,uint8_t>::value)
        for(uint32_t i=0;i<dict.distinct;++i) if(dict.symbols[i]<0||dict.symbols[i]>255) return false;
    return true;
}

static uint32_t blocks_for(uint64_t count) {
    const uint64_t blocks=count/256+(count%256!=0);
    return uint32_t(blocks>65535 ? 65535 : blocks);
}

template<class T>
static int pack_launch(MgbfsPack4Dictionary dict,const T* input,uint64_t stride,uint64_t count,
                       uint64_t* packed,uint8_t* valid,void* stream) {
    if(!valid_arguments<T>(dict,stride,count)) return int(cudaErrorInvalidValue);
    if(!count) return 0;
    if(!input||!packed||!valid) return int(cudaErrorInvalidValue);
    pack_soa<T><<<blocks_for(count),256,0,static_cast<cudaStream_t>(stream)>>>(dict,input,stride,count,packed,valid);
    return int(cudaGetLastError());
}

template<class T>
static int unpack_launch(MgbfsPack4Dictionary dict,const uint64_t* packed,uint64_t stride,uint64_t count,
                         T* output,uint8_t* valid,void* stream) {
    if(!valid_arguments<T>(dict,stride,count)) return int(cudaErrorInvalidValue);
    if(!count) return 0;
    if(!packed||!output||!valid) return int(cudaErrorInvalidValue);
    unpack_soa<T><<<blocks_for(count),256,0,static_cast<cudaStream_t>(stream)>>>(dict,packed,stride,count,output,valid);
    return int(cudaGetLastError());
}

extern "C" int mgbfs_pack4_pack_soa_i64(MgbfsPack4Dictionary d,const int64_t* input,uint64_t stride,uint64_t count,uint64_t* packed,uint8_t* valid,void* stream) {
    return pack_launch<int64_t>(d,input,stride,count,packed,valid,stream);
}
extern "C" int mgbfs_pack4_unpack_soa_i64(MgbfsPack4Dictionary d,const uint64_t* packed,uint64_t stride,uint64_t count,int64_t* output,uint8_t* valid,void* stream) {
    return unpack_launch<int64_t>(d,packed,stride,count,output,valid,stream);
}
extern "C" int mgbfs_pack4_pack_soa_u8(MgbfsPack4Dictionary d,const uint8_t* input,uint64_t stride,uint64_t count,uint64_t* packed,uint8_t* valid,void* stream) {
    return pack_launch<uint8_t>(d,input,stride,count,packed,valid,stream);
}
extern "C" int mgbfs_pack4_unpack_soa_u8(MgbfsPack4Dictionary d,const uint64_t* packed,uint64_t stride,uint64_t count,uint8_t* output,uint8_t* valid,void* stream) {
    return unpack_launch<uint8_t>(d,packed,stride,count,output,valid,stream);
}
