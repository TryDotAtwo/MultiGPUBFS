#include "../cuda/bitmap_pack4.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

extern "C" uint32_t mgbfs_pack4_codec_abi_version();
extern "C" int mgbfs_pack4_pack_soa_i64(MgbfsPack4Dictionary,const int64_t*,uint64_t,uint64_t,uint64_t*,uint8_t*,void*);
extern "C" int mgbfs_pack4_unpack_soa_i64(MgbfsPack4Dictionary,const uint64_t*,uint64_t,uint64_t,int64_t*,uint8_t*,void*);
extern "C" int mgbfs_pack4_pack_soa_u8(MgbfsPack4Dictionary,const uint8_t*,uint64_t,uint64_t,uint64_t*,uint8_t*,void*);
extern "C" int mgbfs_pack4_unpack_soa_u8(MgbfsPack4Dictionary,const uint64_t*,uint64_t,uint64_t,uint8_t*,uint8_t*,void*);

#define CHECK(expr) do { cudaError_t x=(expr); if(x!=cudaSuccess){std::cerr<<"CUDA "<<#expr<<" "<<cudaGetErrorString(x)<<"\n";std::exit(1);} } while(0)
#define ASSERT(expr) do {if(!(expr)){std::cerr<<"CHECK_FAILED line="<<__LINE__<<" "#expr"\n";std::exit(1);}}while(0)

template<class T>
static void check_case(int device, MgbfsPack4Dictionary d, bool bad_row) {
    CHECK(cudaSetDevice(device));
    constexpr uint64_t count=257,stride=count+3,invalid_row=17;
    const T sentinel=std::is_same<T,uint8_t>::value ? T(251) : T(-777777777);
    std::vector<T> input(stride*d.elements,sentinel),output(input.size(),sentinel);
    for(uint32_t e=0;e<d.elements;++e)
        for(uint64_t r=0;r<count;++r)
            input[uint64_t(e)*stride+r]=T(d.symbols[(r+3*e)%d.distinct]);
    if(bad_row) input[invalid_row]=std::is_same<T,uint8_t>::value ? T(250) : T(999999999);
    T* in_d=nullptr;T* out_d=nullptr;uint64_t* packed_d=nullptr;uint8_t* enc_d=nullptr;uint8_t* dec_d=nullptr;
    CHECK(cudaMalloc(&in_d,input.size()*sizeof(T)));CHECK(cudaMalloc(&out_d,output.size()*sizeof(T)));
    CHECK(cudaMalloc(&packed_d,count*sizeof(uint64_t)));CHECK(cudaMalloc(&enc_d,count));CHECK(cudaMalloc(&dec_d,count));
    CHECK(cudaMemcpy(in_d,input.data(),input.size()*sizeof(T),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(out_d,output.data(),output.size()*sizeof(T),cudaMemcpyHostToDevice));
    CHECK(cudaMemset(packed_d,0xff,count*sizeof(uint64_t)));CHECK(cudaMemset(enc_d,0xff,count));CHECK(cudaMemset(dec_d,0xff,count));
    auto pack=[&](uint64_t n,uint64_t st)->int {
        if constexpr(std::is_same<T,uint8_t>::value) return mgbfs_pack4_pack_soa_u8(d,in_d,st,n,packed_d,enc_d,nullptr);
        else return mgbfs_pack4_pack_soa_i64(d,in_d,st,n,packed_d,enc_d,nullptr);
    };
    auto unpack=[&](uint64_t n,uint64_t st)->int {
        if constexpr(std::is_same<T,uint8_t>::value) return mgbfs_pack4_unpack_soa_u8(d,packed_d,st,n,out_d,dec_d,nullptr);
        else return mgbfs_pack4_unpack_soa_i64(d,packed_d,st,n,out_d,dec_d,nullptr);
    };
    ASSERT(pack(count,stride)==0);ASSERT(unpack(count,stride)==0);
    CHECK(cudaDeviceSynchronize());
    std::vector<uint64_t> packed(count);std::vector<uint8_t> enc(count),dec(count);
    CHECK(cudaMemcpy(packed.data(),packed_d,count*sizeof(uint64_t),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(enc.data(),enc_d,count,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(dec.data(),dec_d,count,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(output.data(),out_d,output.size()*sizeof(T),cudaMemcpyDeviceToHost));
    for(uint64_t r=0;r<count;++r) {
        if(bad_row&&r==invalid_row) {
            ASSERT(enc[r]==0);ASSERT(dec[r]==0);ASSERT(packed[r]==UINT64_MAX);
            for(uint32_t e=0;e<d.elements;++e) ASSERT(output[uint64_t(e)*stride+r]==sentinel);
        } else {
            uint64_t expected=0;
            for(uint32_t e=0;e<d.elements;++e) expected|=((r+3*e)%d.distinct)<<(4*e);
            ASSERT(enc[r]==1);ASSERT(dec[r]==1);ASSERT(packed[r]==expected);
            for(uint32_t e=0;e<d.elements;++e) ASSERT(output[uint64_t(e)*stride+r]==input[uint64_t(e)*stride+r]);
        }
    }
    for(uint32_t e=0;e<d.elements;++e)
        for(uint64_t r=count;r<stride;++r) ASSERT(output[uint64_t(e)*stride+r]==sentinel);
    ASSERT(pack(stride+1,stride)==int(cudaErrorInvalidValue));
    CHECK(cudaFree(in_d));CHECK(cudaFree(out_d));CHECK(cudaFree(packed_d));CHECK(cudaFree(enc_d));CHECK(cudaFree(dec_d));
}

int main(){
    int devices=0;CHECK(cudaGetDeviceCount(&devices));ASSERT(devices>=2);
    ASSERT(mgbfs_pack4_codec_abi_version()==1);
    MgbfsPack4Dictionary d{};d.elements=8;d.distinct=8;
    const int64_t signed_symbols[]={-9000000000LL,-31,-1,0,9,25,789,40000000000LL};
    for(int i=0;i<8;++i)d.symbols[i]=signed_symbols[i];
    ASSERT(mgbfs_pack4_pack_soa_i64(d,nullptr,0,0,nullptr,nullptr,nullptr)==0);
    ASSERT(mgbfs_pack4_pack_soa_i64(d,nullptr,UINT64_MAX,1,nullptr,nullptr,nullptr)==int(cudaErrorInvalidValue));
    ASSERT(mgbfs_pack4_unpack_soa_i64(d,nullptr,UINT64_MAX,1,nullptr,nullptr,nullptr)==int(cudaErrorInvalidValue));
    for(int device=0;device<2;++device){
        check_case<int64_t>(device,d,true);
        MgbfsPack4Dictionary repeated{};repeated.elements=6;repeated.distinct=2;repeated.symbols[0]=-4;repeated.symbols[1]=17;
        check_case<int64_t>(device,repeated,true);
        MgbfsPack4Dictionary wide{};wide.elements=16;wide.distinct=16;
        for(int i=0;i<16;++i)wide.symbols[i]=int64_t(i)*1000000000LL-7000000000LL;
        check_case<int64_t>(device,wide,false);
        MgbfsPack4Dictionary byte{};byte.elements=8;byte.distinct=8;
        for(int i=0;i<8;++i)byte.symbols[i]=2*i;
        check_case<uint8_t>(device,byte,true);
        ASSERT(mgbfs_pack4_unpack_soa_u8(d,nullptr,0,0,nullptr,nullptr,nullptr)==int(cudaErrorInvalidValue));
        std::cout<<"DEVICE_CONVERT_PASS "<<device<<"\n";
    }
    std::cout<<"PACK4_GPU_CONVERT_PASS physical_gpus=2 cases=8 rows_per_case=257\n";
}
