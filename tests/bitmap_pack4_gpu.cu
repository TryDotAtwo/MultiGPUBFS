#include "../cuda/bitmap_pack4.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define CUDA_CHECK(expr) do { \
    cudaError_t err_ = (expr); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA_FAILURE " << #expr << ": " << cudaGetErrorString(err_) << "\n"; \
        std::exit(1); \
    } \
} while (0)

struct Row { int64_t v[8]; };
struct Result { uint64_t packed, child[7]; int64_t decoded[8]; uint32_t errors; };

__global__ void exercise(const Row* rows, Result* results, uint32_t count, MgbfsPack4Dictionary dict) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    Result r{};
    if (!mgbfs_pack4_encode(dict, rows[tid].v, &r.packed)) r.errors |= 1;
    if (!mgbfs_pack4_decode(dict, r.packed, r.decoded)) r.errors |= 2;
    for (uint32_t i = 0; i < 7; ++i) {
        uint8_t perm[8];
        for (uint32_t j = 0; j < 8; ++j) perm[j] = static_cast<uint8_t>(j);
        uint8_t tmp=perm[i]; perm[i]=perm[i+1]; perm[i+1]=tmp;
        if (!mgbfs_pack4_apply(dict, r.packed, perm, &r.child[i])) r.errors |= 4;
    }
    results[tid] = r;
}

__global__ void edge_cases(uint32_t* errors) {
    if (blockIdx.x || threadIdx.x) return;
    uint32_t e=0;
    MgbfsPack4Dictionary d{};
    d.elements=5; d.distinct=2; d.symbols[0]=-4; d.symbols[1]=17;
    int64_t row[5]={17,-4,17,-4,17};
    int64_t out[16]={};
    uint64_t word=0, changed=0;
    if (!mgbfs_pack4_encode(d,row,&word) || word != 0x10101ULL) e|=1u;
    const uint8_t reverse[5]={4,3,2,1,0};
    if (!mgbfs_pack4_apply(d,word,reverse,&changed) || changed != word) e|=2u;
    if (!mgbfs_pack4_decode(d,changed,out)) e|=4u;
    for (int i=0;i<5;++i) if (out[i]!=row[4-i]) e|=8u;
    out[0]=123456789;
    if (mgbfs_pack4_decode(d,word|(uint64_t(1)<<20),out) || out[0]!=123456789) e|=16u;
    if (mgbfs_pack4_valid_word(d,(word&~uint64_t(15))|2u)) e|=32u;
    uint8_t invalid_perm[5]={0,1,1,3,4};
    changed=0xfeedbeef;
    if (mgbfs_pack4_apply(d,word,invalid_perm,&changed) || changed!=0xfeedbeef) e|=64u;
    int64_t unknown[5]={17,-4,17,99,17};
    changed=0xfeedbeef;
    if (mgbfs_pack4_encode(d,unknown,&changed) || changed!=0xfeedbeef) e|=128u;
    MgbfsPack4Dictionary wide{};
    wide.elements=16; wide.distinct=16;
    int64_t wide_row[16];
    for (int i=0;i<16;++i) wide.symbols[i]=wide_row[i]=int64_t(i)*1000000000LL-7000000000LL;
    if (!mgbfs_pack4_encode(wide,wide_row,&word) || word!=0xfedcba9876543210ULL) e|=256u;
    if (!mgbfs_pack4_decode(wide,word,out)) e|=512u;
    for (int i=0;i<16;++i) if (out[i]!=wide_row[i]) e|=1024u;
    wide.elements=17;
    if (mgbfs_pack4_valid(wide)) e|=2048u;
    d.symbols[1]=-4;
    if (mgbfs_pack4_valid(d)) e|=4096u;
    *errors=e;
}

static uint64_t expected_word(const std::array<int,8>& a) {
    uint64_t v=0;
    for (int i=0;i<8;++i) v|=uint64_t(a[i])<<(4*i);
    return v;
}

int main(int argc,char** argv) {
    const bool sample=argc>1 && std::string(argv[1])=="--sample";
    int device_count=0; CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count<2) { std::cerr<<"NEED_TWO_PHYSICAL_GPUS\n"; return 2; }
    MgbfsPack4Dictionary d{}; d.elements=8; d.distinct=8;
    const int64_t symbols[]={-9000000000LL,-31,-1,0,9,25,789,40000000000LL};
    std::copy(symbols,symbols+8,d.symbols);
    std::vector<Row> rows; std::vector<std::array<int,8>> orders;
    std::array<int,8> order{}; std::iota(order.begin(),order.end(),0);
    do {
        Row r{}; for(int i=0;i<8;++i) r.v[i]=symbols[order[i]];
        rows.push_back(r); orders.push_back(order);
        if (sample && rows.size()==128) break;
    } while(std::next_permutation(order.begin(),order.end()));
    if(!sample && rows.size()!=40320) { std::cerr<<"ORACLE_ENUMERATION_FAILED\n"; return 1; }
    uint64_t checked=0;
    for (int dev=0;dev<2;++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        size_t begin=rows.size()*dev/2, end=rows.size()*(dev+1)/2, n=end-begin;
        Row* gpu_rows=nullptr; Result* gpu_out=nullptr; uint32_t* gpu_errors=nullptr;
        CUDA_CHECK(cudaMalloc(&gpu_rows,n*sizeof(Row)));
        CUDA_CHECK(cudaMalloc(&gpu_out,n*sizeof(Result)));
        CUDA_CHECK(cudaMalloc(&gpu_errors,sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(gpu_rows,rows.data()+begin,n*sizeof(Row),cudaMemcpyHostToDevice));
        exercise<<<unsigned((n+127)/128),128>>>(gpu_rows,gpu_out,unsigned(n),d);
        CUDA_CHECK(cudaGetLastError());
        edge_cases<<<1,1>>>(gpu_errors);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<Result> out(n); uint32_t errors=0;
        CUDA_CHECK(cudaMemcpy(out.data(),gpu_out,n*sizeof(Result),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&errors,gpu_errors,sizeof(errors),cudaMemcpyDeviceToHost));
        if(errors) { std::cerr<<"EDGE_FAILURE dev="<<dev<<" mask="<<errors<<"\n"; return 1; }
        for(size_t k=0;k<n;++k) {
            const auto& a=orders[begin+k]; const auto& got=out[k];
            if(got.errors || got.packed!=expected_word(a)) {
                std::cerr<<"ENCODE_FAILURE dev="<<dev<<" ordinal="<<begin+k<<"\n"; return 1;
            }
            for(int j=0;j<8;++j) if(got.decoded[j]!=rows[begin+k].v[j]) {
                std::cerr<<"DECODE_FAILURE dev="<<dev<<" ordinal="<<begin+k<<"\n"; return 1;
            }
            for(int j=0;j<7;++j) {
                auto child=a; std::swap(child[j],child[j+1]);
                if(got.child[j]!=expected_word(child)) {
                    std::cerr<<"ACTION_FAILURE dev="<<dev<<" ordinal="<<begin+k<<" generator="<<j<<"\n"; return 1;
                }
            }
            ++checked;
        }
        CUDA_CHECK(cudaFree(gpu_rows)); CUDA_CHECK(cudaFree(gpu_out)); CUDA_CHECK(cudaFree(gpu_errors));
        std::cout<<"DEVICE_PASS "<<dev<<" permutations="<<n<<"\n";
    }
    std::cout<<"PACK4_GPU_LEAF_PASS physical_gpus=2 permutations="<<checked<<" edges="<<checked*7
             <<" oracle=independent_cpu sample="<<(sample?1:0)<<"\n";
}
