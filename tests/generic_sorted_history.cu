#include "generic_sorted_history.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <limits>
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void need(bool value,const char* what){if(!value)throw std::runtime_error(what);}
template<class T>struct Device{T* ptr;size_t n;Device(size_t count):n(count){ck(cudaMalloc(&ptr,count*sizeof(T)));}~Device(){cudaFree(ptr);}void put(const std::vector<T>& values){need(values.size()<=n,"UPLOAD_BOUNDS");ck(cudaMemcpy(ptr,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));}std::vector<T>get(){std::vector<T>v(n);ck(cudaMemcpy(v.data(),ptr,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}};
uint64_t mix(uint64_t value){value^=value>>30;value*=0xbf58476d1ce4e5b9ULL;value^=value>>27;value*=0x94d049bb133111ebULL;return value^(value>>31);}
template<class State>uint64_t hash_row(const std::vector<State>& data,uint32_t stride,uint32_t row,uint32_t width,unsigned mode){if(mode==0)return 0;if(mode==1)return UINT64_MAX;uint64_t value=17;for(uint32_t e=0;e<width;e++)value=mix(value^uint64_t(int64_t(data[uint64_t(e)*stride+row]))^e);return mode==2?value&15:value;}
template<class State>struct Candidates{uint32_t elements,stride;const State* states;__device__ int64_t value(uint32_t child,uint32_t e)const{return int64_t(states[uint64_t(e)*stride+child]);}};
template<class State>__global__ void contains_kernel(Candidates<State> action,const uint64_t* query_hashes,uint32_t queries,GenericSortedHistoryRun run,const State* arena,uint32_t stride,uint32_t* error,uint8_t* answers){
 for(uint32_t q=blockIdx.x*blockDim.x+threadIdx.x;q<queries;q+=blockDim.x*gridDim.x){answers[q]=*error?0:uint8_t(generic_sorted_history_contains(action,q,query_hashes[q],run,arena,stride,error));}}
template<class State>void fixture(uint32_t width,unsigned mode){
 constexpr uint32_t stride=509,count=233,queries=387;
 std::vector<State>arena(uint64_t(stride)*width),targets(uint64_t(queries)*width);
 for(uint32_t e=0;e<width;e++)for(uint32_t row=0;row<stride;row++){uint64_t bits=mix(uint64_t(row)*0x9e3779b97f4a7c15ULL+e);int64_t signed_bits;std::memcpy(&signed_bits,&bits,8);arena[uint64_t(e)*stride+row]=State(signed_bits);}
 if(sizeof(State)==8){arena[0]=State(INT64_MIN);arena[stride]=State(INT64_MAX);}
 std::vector<uint32_t>rows(count);for(uint32_t i=0;i<count;i++)rows[i]=(i*53)%stride;
 std::sort(rows.begin(),rows.end(),[&](auto a,auto b){auto ha=hash_row(arena,stride,a,width,mode),hb=hash_row(arena,stride,b,width,mode);return ha==hb?a<b:ha<hb;});std::vector<uint64_t>hashes(count);
 for(uint32_t i=0;i<count;i++)hashes[i]=hash_row(arena,stride,rows[i],width,mode);
 for(uint32_t q=0;q<queries;q++){uint32_t origin=q<count?rows[(q*17)%count]:(q*29)%stride;for(uint32_t e=0;e<width;e++)targets[uint64_t(e)*queries+q]=arena[uint64_t(e)*stride+origin];if(q>=count)targets[uint64_t(width-1)*queries+q]=State(uint64_t(targets[uint64_t(width-1)*queries+q])^1);}
 std::vector<uint64_t>query_hashes(queries);std::vector<uint8_t>expected(queries);
 for(uint32_t q=0;q<queries;q++){query_hashes[q]=hash_row(targets,queries,q,width,mode);for(uint32_t at=0;at<count;at++){if(hashes[at]!=query_hashes[q])continue;bool same=true;for(uint32_t e=0;e<width;e++)if(arena[uint64_t(e)*stride+rows[at]]!=targets[uint64_t(e)*queries+q]){same=false;break;}if(same){expected[q]=1;break;}}}
 Device<State>states(arena.size()),candidate(targets.size());states.put(arena);candidate.put(targets);Device<uint64_t>keys(count),qkeys(queries);keys.put(hashes);qkeys.put(query_hashes);Device<uint32_t>refs(count),error(1);refs.put(rows);error.put({0});Device<uint8_t>answers(queries);
 GenericSortedHistoryRun run{keys.ptr,refs.ptr,count};generic_sorted_history_validate<<<2,128>>>(run,count,stride,error.ptr);contains_kernel<<<4,128>>>(Candidates<State>{width,queries,candidate.ptr},qkeys.ptr,queries,run,states.ptr,stride,error.ptr,answers.ptr);ck(cudaGetLastError());ck(cudaDeviceSynchronize());need(error.get()[0]==0,"VALIDATION_FATAL");need(answers.get()==expected,"FULL_STATE_CPU_ORACLE");
 error.put({0});contains_kernel<<<4,128>>>(Candidates<State>{width,queries,candidate.ptr},qkeys.ptr,queries,GenericSortedHistoryRun{nullptr,nullptr,0},states.ptr,stride,error.ptr,answers.ptr);ck(cudaDeviceSynchronize());need(answers.get()==std::vector<uint8_t>(queries,0),"EMPTY_RUN");
 auto invalid=rows;invalid[count-1]=stride;refs.put(invalid);error.put({0});generic_sorted_history_validate<<<2,128>>>(run,count,stride,error.ptr);ck(cudaDeviceSynchronize());need(error.get()[0]==8,"BAD_ROW_REFERENCE");refs.put(rows);
 error.put({0});generic_sorted_history_validate<<<2,128>>>(GenericSortedHistoryRun{keys.ptr,refs.ptr,count+1},count,stride,error.ptr);ck(cudaDeviceSynchronize());need(error.get()[0]==32,"BAD_RUN_COUNT");
 auto unordered=hashes;unordered[0]=UINT64_MAX;unordered[count-1]=0;keys.put(unordered);error.put({0});generic_sorted_history_validate<<<2,128>>>(run,count,stride,error.ptr);ck(cudaDeviceSynchronize());need(error.get()[0]==256,"BAD_SORT_ORDER");
 std::cout<<"SORTED_HISTORY_EXACT_ORACLE bytes="<<sizeof(State)<<" width="<<width<<" mode="<<mode<<" queries="<<queries<<"\n";
}
int main(){try{int devices;ck(cudaGetDeviceCount(&devices));for(int device=0;device<devices;device++){ck(cudaSetDevice(device));for(uint32_t width:{2u,17u,25u,129u})for(unsigned mode=0;mode<4;mode++){fixture<uint8_t>(width,mode);fixture<int64_t>(width,mode);}std::cout<<"SORTED_HISTORY_PRIMITIVE_PASS device="<<device<<"\n";}return 0;}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
