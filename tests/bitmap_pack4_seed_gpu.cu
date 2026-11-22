#include "../cuda/bitmap_pack4_owner.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#define CK(x) do{auto _cuda_status=(x);if(_cuda_status!=cudaSuccess){std::cerr<<"CUDA "<<__LINE__<<" "<<cudaGetErrorString(_cuda_status)<<"\n";std::exit(1);}}while(0)
#define REQUIRE(x) do{if(!(x)){std::cerr<<"FAIL "<<__LINE__<<" "#x"\n";std::exit(1);}}while(0)
template<class T>T* alloc(size_t n){T* p=nullptr;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
static void run(int gpu,bool repeated){
 CK(cudaSetDevice(gpu));std::vector<int64_t> root=repeated?std::vector<int64_t>{-50,-50,12,900}:std::vector<int64_t>{-50,-1,12,900};
 MgbfsPack4Dictionary d{};d.elements=4;for(auto x:root)if(!d.distinct||d.symbols[d.distinct-1]!=x)d.symbols[d.distinct++]=x;
 auto *input=alloc<int64_t>(4);auto *rows=alloc<uint64_t>(1),*bits=alloc<uint64_t>(2),*accepted=alloc<uint64_t>(1);auto *error=alloc<uint32_t>(1);
 CK(cudaMemcpy(input,root.data(),32,cudaMemcpyHostToDevice));
 MgbfsPack4SeedArgs a{};a.dictionary=d;a.input=input;a.row=rows;a.bits=bits;a.accepted=accepted;a.error=error;a.local=64;a.owner_len=70;
 auto reset=[&](){CK(cudaMemset(rows,0xff,8));CK(cudaMemset(bits,0,16));CK(cudaMemset(accepted,0,8));CK(cudaMemset(error,0,4));};
 auto read=[&](uint64_t* row,uint64_t* b,uint64_t* count,uint32_t* e){
  CK(cudaMemcpy(row,rows,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(b,bits,16,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(count,accepted,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(e,error,4,cudaMemcpyDeviceToHost));
 };
 reset();CK((cudaError_t)mgbfs_pack4_seed(a,nullptr));CK(cudaDeviceSynchronize());
 uint64_t row=0,b[2]={},count=0;uint32_t err=0;read(&row,b,&count,&err);
 std::vector<int64_t> decoded(4);REQUIRE(!err&&count==1&&!b[0]&&b[1]==1&&mgbfs_pack4_decode(d,row,decoded.data())&&decoded==root);
 // A duplicate seed must not silently overwrite an existing row or count.
 CK(cudaMemset(rows,0xff,8));CK(cudaMemset(accepted,0,8));CK(cudaMemset(error,0,4));
 CK((cudaError_t)mgbfs_pack4_seed(a,nullptr));CK(cudaDeviceSynchronize());read(&row,b,&count,&err);
 REQUIRE(err&&count==0&&row==UINT64_MAX&&b[1]==1);
 // Invalid symbol is rejected before publishing any bit or row.
 reset();int64_t bad=901;CK(cudaMemcpy(input+3,&bad,8,cudaMemcpyHostToDevice));
 CK((cudaError_t)mgbfs_pack4_seed(a,nullptr));CK(cudaDeviceSynchronize());read(&row,b,&count,&err);
 REQUIRE(err&&count==0&&row==UINT64_MAX&&!b[0]&&!b[1]);
 a.local=70;REQUIRE(mgbfs_pack4_seed(a,nullptr)==int(cudaErrorInvalidValue));
 CK(cudaFree(input));CK(cudaFree(rows));CK(cudaFree(bits));CK(cudaFree(accepted));CK(cudaFree(error));
 std::cout<<"PACK4_SEED_PASS gpu="<<gpu<<" repeated="<<repeated<<" boundary_word=1\n";
}
int main(){int n=0;CK(cudaGetDeviceCount(&n));REQUIRE(n>=2);for(int d=0;d<2;d++){run(d,false);run(d,true);}std::cout<<"PACK4_SEED_TEST_PASS physical_gpus=2\n";}
