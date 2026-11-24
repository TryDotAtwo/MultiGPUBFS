#include "../cuda/bitmap_pack4_action.cuh"
#include "../cuda/permutation_bitmap.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#define CK(expr) do { cudaError_t e=(expr);if(e!=cudaSuccess){std::cerr<<"CUDA "<<#expr<<": "<<cudaGetErrorString(e)<<"\n";std::exit(1);}}while(0)
#define REQUIRE(expr) do {if(!(expr)){std::cerr<<"ASSERT line="<<__LINE__<<" "#expr"\n";std::exit(1);}}while(0)

__global__ void rank_children(BitmapPack4Action action,BitmapDeviceDomain domain,uint64_t* ranks,uint32_t* errors,uint32_t count) {
    const uint32_t child=blockIdx.x*blockDim.x+threadIdx.x;
    if(child>=count)return;
    uint32_t err=0;
    uint64_t rank=0, reference=0;
    const bool expected=action.valid(child,&err) && bitmap_rank(domain,action,child,&reference);
    const bool ranked=action.rank_candidate(domain,child,&rank);
    if(ranked!=expected || (ranked && rank!=reference)){errors[child]=32u;return;}
    if(!ranked){errors[child]=8u;return;}
    ranks[child]=rank;
    errors[child]=0;
}

static void run(int dev,std::vector<int64_t> root,bool inject_bad) {
    CK(cudaSetDevice(dev));std::sort(root.begin(),root.end());
    std::vector<std::vector<int64_t>> states;
    do {states.push_back(root);} while(std::next_permutation(root.begin(),root.end()));
    const uint32_t n=root.size(),gens=n-1,num=states.size();
    MgbfsPack4Dictionary dict{};dict.elements=n;
    std::vector<uint32_t> counts;std::vector<int64_t> symbols;
    for(auto v:root){if(symbols.empty()||symbols.back()!=v){symbols.push_back(v);counts.push_back(1);}else counts.back()++;}
    dict.distinct=symbols.size();for(size_t i=0;i<symbols.size();++i)dict.symbols[i]=symbols[i];
    std::vector<uint64_t> parents(num);
    for(uint32_t r=0;r<num;r++){
        uint64_t word=0;
        for(uint32_t e=0;e<n;e++){
            auto it=std::lower_bound(symbols.begin(),symbols.end(),states[r][e]);
            REQUIRE(it!=symbols.end() && *it==states[r][e]);
            word|=uint64_t(it-symbols.begin())<<(4*e);
        }
        parents[r]=word;
    }
    const uint32_t begin=num*dev/2,end=num*(dev+1)/2,local=end-begin;
    std::vector<uint32_t> permutations(gens*n);
    for(uint32_t g=0;g<gens;g++){
        for(uint32_t e=0;e<n;e++)permutations[g*n+e]=e;
        std::swap(permutations[g*n+g],permutations[g*n+g+1]);
    }
    uint64_t *dparents,*dranks;uint32_t *dperms,*dcounts,*derr;int64_t *dsymbols;
    CK(cudaMalloc(&dparents,local*8));CK(cudaMalloc(&dranks,uint64_t(local)*gens*8));
    CK(cudaMalloc(&dperms,permutations.size()*4));CK(cudaMalloc(&dcounts,counts.size()*4));
    CK(cudaMalloc(&dsymbols,symbols.size()*8));CK(cudaMalloc(&derr,uint64_t(local)*gens*4));
    CK(cudaMemcpy(dparents,parents.data()+begin,local*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dperms,permutations.data(),permutations.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dcounts,counts.data(),counts.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsymbols,symbols.data(),symbols.size()*8,cudaMemcpyHostToDevice));
    BitmapDeviceDomain domain{dsymbols,dcounts,uint32_t(symbols.size()),n,num};
    BitmapPack4Action action{n,gens,local,dparents,dperms,dict};
    const uint32_t children=local*gens;
    rank_children<<<(children+127)/128,128>>>(action,domain,dranks,derr,children);
    CK(cudaGetLastError());CK(cudaDeviceSynchronize());
    std::vector<uint64_t> got(children);std::vector<uint32_t> errs(children);
    CK(cudaMemcpy(got.data(),dranks,got.size()*8,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(errs.data(),derr,errs.size()*4,cudaMemcpyDeviceToHost));
    for(uint32_t i=0;i<children;i++) {
        auto child=states[begin+i/gens];std::swap(child[i%gens],child[i%gens+1]);
        auto it=std::lower_bound(states.begin(),states.end(),child);
        REQUIRE(it!=states.end() && *it==child);
        if(errs[i]||got[i]!=uint64_t(it-states.begin())) {
            std::cerr<<"MISMATCH dev="<<dev<<" child="<<i<<" got="<<got[i]
                     <<" expected="<<(it-states.begin())<<" err="<<errs[i]<<"\n";std::exit(1);
        }
    }
    if(inject_bad) {
        uint64_t bad=UINT64_MAX;CK(cudaMemcpy(dparents,&bad,8,cudaMemcpyHostToDevice));
        CK(cudaMemset(dranks,0xff,gens*8));CK(cudaMemset(derr,0,gens*4));
        rank_children<<<1,32>>>(action,domain,dranks,derr,gens);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
        std::vector<uint64_t> badr(gens);std::vector<uint32_t> bade(gens);
        CK(cudaMemcpy(badr.data(),dranks,gens*8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(bade.data(),derr,gens*4,cudaMemcpyDeviceToHost));
        for(uint32_t i=0;i<gens;i++){REQUIRE(bade[i]==8u);REQUIRE(badr[i]==UINT64_MAX);}
        CK(cudaMemcpy(dparents,parents.data()+begin,8,cudaMemcpyHostToDevice));
        uint32_t invalid=n;CK(cudaMemcpy(dperms,&invalid,4,cudaMemcpyHostToDevice));
        CK(cudaMemset(dranks,0xff,8));CK(cudaMemset(derr,0,4));
        rank_children<<<1,1>>>(action,domain,dranks,derr,1);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
        uint32_t error=0;uint64_t rank=0;CK(cudaMemcpy(&error,derr,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&rank,dranks,8,cudaMemcpyDeviceToHost));
        REQUIRE(error==8u);REQUIRE(rank==UINT64_MAX);
    }
    CK(cudaFree(dparents));CK(cudaFree(dranks));CK(cudaFree(dperms));CK(cudaFree(dcounts));CK(cudaFree(dsymbols));CK(cudaFree(derr));
    std::cout<<"ACTION_RANK_PASS gpu="<<dev<<" domain="<<num<<" parents="<<local<<" children="<<children<<"\n";
}
int main(int argc,char** argv){
    const bool sample=argc==2 && std::string(argv[1])=="--sample";
    REQUIRE(argc==1 || sample);
    int devices=0;CK(cudaGetDeviceCount(&devices));REQUIRE(devices>=2);
    for(int dev=0;dev<2;dev++){
        if(sample)run(dev,{-9000000000LL,-31,-1,0,40000000000LL},true);
        else run(dev,{-9000000000LL,-31,-1,0,9,25,789,40000000000LL},true);
        run(dev,{-7,-7,0,99,99},true);
    }
    std::cout<<"PACK4_ACTION_RANK_PASS physical_gpus=2 sample="<<sample
             <<" primary_edges="<<(sample?480:282240)<<" repeated_domain=30\n";
}
