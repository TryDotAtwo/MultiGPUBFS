#include "lossless_state_key.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>
#include <utility>
using mgbfs_lossless::Key;
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"FAIL line %d\n",__LINE__);return 1;}}while(0)
__global__ void keys(const uint8_t* s,Key* out,uint32_t n,uint32_t bits,uint32_t rows){uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;if(row<rows)out[row]=mgbfs_lossless::pack(s+uint64_t(row)*n,n,bits);}
int main(){REQUIRE(mgbfs_lossless::shape(25,5));REQUIRE(!mgbfs_lossless::shape(26,5));REQUIRE(!mgbfs_lossless::shape(1,0));REQUIRE(!mgbfs_lossless::shape(1,9));std::vector<uint8_t> p{0,1,2,3,4,5,6,7};std::set<std::pair<uint64_t,uint64_t>> seen;do{Key k=mgbfs_lossless::pack(p.data(),8,3);REQUIRE(seen.emplace(k.lo,k.hi).second);for(uint32_t j=0;j<8;j++)REQUIRE(mgbfs_lossless::element(k,j,3)==p[j]);}while(std::next_permutation(p.begin(),p.end()));REQUIRE(seen.size()==40320);
for(uint32_t bits=1;bits<=8;bits++){uint32_t n=128/bits,rows=4096;std::vector<uint8_t> in(uint64_t(n)*rows);uint64_t x=7;for(auto&v:in){x=mgbfs_lossless::diffuse(x+1);v=uint8_t(x&((1u<<bits)-1));}std::vector<Key> expected(rows),actual(rows);for(uint32_t row=0;row<rows;row++){expected[row]=mgbfs_lossless::pack(in.data()+uint64_t(row)*n,n,bits);for(uint32_t j=0;j<n;j++)REQUIRE(mgbfs_lossless::element(expected[row],j,bits)==in[uint64_t(row)*n+j]);}uint8_t*d;Key*o;REQUIRE(cudaMalloc(&d,in.size())==cudaSuccess);REQUIRE(cudaMalloc(&o,rows*sizeof(Key))==cudaSuccess);REQUIRE(cudaMemcpy(d,in.data(),in.size(),cudaMemcpyHostToDevice)==cudaSuccess);keys<<<(rows+255)/256,256>>>(d,o,n,bits,rows);REQUIRE(cudaGetLastError()==cudaSuccess);REQUIRE(cudaMemcpy(actual.data(),o,rows*sizeof(Key),cudaMemcpyDeviceToHost)==cudaSuccess);for(uint32_t row=0;row<rows;row++)REQUIRE(actual[row].lo==expected[row].lo&&actual[row].hi==expected[row].hi);REQUIRE(cudaFree(d)==cudaSuccess);REQUIRE(cudaFree(o)==cudaSuccess);}
puts("LOSSLESS_KEY_PRIMITIVE_PASS CPU40320 GPU32768 boundary128 reversible128");return 0;}
