// No child-state output is supplied: HASH_FIRST may only store hashes/origins.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../cuda/regenerate.h"
extern "C" int mgbfs_generate_hash_only(
 uint32_t n,uint32_t moves,uint32_t modulus,uint32_t stride,uint32_t parent_capacity,
 uint32_t candidate_capacity,uint32_t source,uint64_t parent_begin,
 const uint8_t* parents,const uint8_t* generators,const uint32_t* coefficients,
 const uint32_t* offsets,const uint32_t* parent_count,uint32_t* hashes,
 MgbfsRegenerateOrigin* origins,uint32_t* candidate_count,uint32_t* fatal,void* stream);
static void require(bool x){if(!x)std::abort();}
template<class T> T* upload(const std::vector<T>& v){T* p=nullptr;require(cudaMalloc(&p,v.size()*sizeof(T))==cudaSuccess);require(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)==cudaSuccess);return p;}
int main(){
 constexpr uint32_t P=4294967291u, parents_n=257, cap=514;
 constexpr uint64_t begin=0x100000001ULL;
 std::vector<uint8_t> parents(parents_n*16,0);
 for(unsigned i=0;i<parents_n;++i){parents[i*16]=parents[i*16+3]=1;if(i%2)parents[i*16+1]=2;}
 auto p=upload(parents),g=upload<uint8_t>({1,1,0,1,1,0,1,1});
 // Per state-byte, four affine projection coefficients. Last lane tests F_p.
 auto w=upload<uint32_t>({1,2,3,P-1,2,5,13,P-2,3,7,17,P-3,4,11,19,P-4});
 auto b=upload<uint32_t>({7,11,13,1});
 auto count=upload<uint32_t>({parents_n}),out_count=upload<uint32_t>({99}),fatal=upload<uint32_t>({0});
 auto hashes=upload<uint32_t>(std::vector<uint32_t>(cap*4,0));
 auto origins=upload<MgbfsRegenerateOrigin>(std::vector<MgbfsRegenerateOrigin>(cap));
 require(mgbfs_generate_hash_only(2,2,5,16,parents_n,cap,1,begin,p,g,w,b,count,hashes,origins,out_count,fatal,nullptr)==0);
 require(cudaDeviceSynchronize()==cudaSuccess);
 std::vector<uint32_t> actual(cap*4);std::vector<MgbfsRegenerateOrigin> refs(cap);
 require(cudaMemcpy(actual.data(),hashes,actual.size()*4,cudaMemcpyDeviceToHost)==cudaSuccess);
 require(cudaMemcpy(refs.data(),origins,refs.size()*16,cudaMemcpyDeviceToHost)==cudaSuccess);
 const uint32_t expected[4][4]={{14,29,48,P-6},{15,31,52,P-7},{18,39,74,P-10},{27,63,116,P-19}};
 for(unsigned i=0;i<cap;++i){
   unsigned kind=((i/2)%2)*2+i%2;
   for(unsigned lane=0;lane<4;++lane)require(actual[i*4+lane]==expected[kind][lane]);
   require(refs[i].source==1&&refs[i].move==i%2&&refs[i].reserved==0&&refs[i].parent==begin+i/2);
 }
 uint32_t value=99;
 require(cudaMemcpy(&value,out_count,4,cudaMemcpyDeviceToHost)==cudaSuccess);require(value==cap);
 require(cudaMemcpy(&value,fatal,4,cudaMemcpyDeviceToHost)==cudaSuccess);require(value==0);
 // Device count exceeds preallocated parents: no input read/output store allowed.
 value=parents_n+1;require(cudaMemcpy(count,&value,4,cudaMemcpyHostToDevice)==cudaSuccess);
 require(cudaMemset(hashes,0xcc,cap*16)==cudaSuccess);
 require(mgbfs_generate_hash_only(2,2,5,16,parents_n,cap,1,begin,p,g,w,b,count,hashes,origins,out_count,fatal,nullptr)==0);
 require(cudaDeviceSynchronize()==cudaSuccess);
 require(cudaMemcpy(&value,fatal,4,cudaMemcpyDeviceToHost)==cudaSuccess);require(value!=0);
 require(cudaMemcpy(actual.data(),hashes,actual.size()*4,cudaMemcpyDeviceToHost)==cudaSuccess);
 for(auto x:actual)require(x==0xccccccccu);
 cudaFree(origins);cudaFree(hashes);cudaFree(fatal);cudaFree(out_count);cudaFree(count);cudaFree(b);cudaFree(w);cudaFree(g);cudaFree(p);
 std::puts("HASH_FIRST_GENERATE_PASS");
}
