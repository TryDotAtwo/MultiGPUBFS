#include <cstdint>
#include "mgbfs_cuda.h"
#include "allocation_shape.h"
#include "lossless_state_key.h"
#include <cstddef>
#include <cstdio>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include "cutlass/gemm/device/gemm.h"

using Gemm = cutlass::gemm::device::Gemm<
  uint8_t, cutlass::layout::RowMajor, uint8_t, cutlass::layout::ColumnMajor,
  int32_t, cutlass::layout::RowMajor, int32_t,
  cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
  cutlass::gemm::GemmShape<64,32,64>, cutlass::gemm::GemmShape<32,32,64>,
  cutlass::gemm::GemmShape<8,8,16>,
  cutlass::epilogue::thread::LinearCombination<int32_t,4,int32_t,int32_t>,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,2>;

struct HashPlan {
  uint32_t width{},stride{},capacity{},lossless_bits{};
  uint8_t* weights{};
  uint32_t* offsets{};
  int32_t* partials{};
  void* workspace{};
  Gemm gemm;
  ~HashPlan() {cudaFree(workspace);cudaFree(partials);cudaFree(offsets);cudaFree(weights);}
};
extern "C" int mgbfs_hash_query(uint32_t bytes,uint32_t capacity,MgbfsHashBytes* out) {
  if(!out)return 1;*out={};MgbfsHashBytes q{};
  if(hash_shape(bytes,capacity,&q))return 1;
  Gemm::Arguments args({int(capacity),16,int(q.stride)}, {nullptr,int(q.stride)},
    {nullptr,int(q.stride)}, {nullptr,16},{nullptr,16},{1,0},1);
  q.workspace=Gemm::get_workspace_size(args);
  if(q.workspace!=0||Gemm::can_implement(args)!=cutlass::Status::kSuccess)return 2;
  *out=q;return 0;
}

static void check(cudaError_t result) {
  if(result!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

__global__ void finish_hash(const int32_t* sums,const uint32_t* offsets,uint32_t* output,uint32_t count) {
  const uint32_t word=blockIdx.x*blockDim.x+threadIdx.x;
  if(word>=count*4) return;
  const uint32_t row=word/4,lane=word%4;
  uint64_t sum=offsets[lane];
  #pragma unroll
  for(int limb=0;limb<4;++limb) sum+=uint64_t(sums[row*16+lane*4+limb])<<(8*limb);
  output[word]=uint32_t(sum%4294967291ULL);
}

extern "C" int mgbfs_hash_create(uint32_t bytes,uint32_t capacity,const uint8_t* limbs,const uint32_t* offsets,void** out,char* error,size_t n) {
  if(!out) return 1;
  *out=nullptr;
  try {
    MgbfsHashBytes allocation{};
    if(!limbs||!offsets||mgbfs_hash_query(bytes,capacity,&allocation)) throw std::runtime_error("HASH_SHAPE_OR_ACCUMULATOR_BOUND");
    for(int j=0;j<4;++j) if(offsets[j]>=4294967291ULL) throw std::runtime_error("HASH_OFFSET_RANGE");
    int device;check(cudaGetDevice(&device));cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,device));
    if(prop.major*10+prop.minor<75) throw std::runtime_error("UNSUPPORTED_SM");
    auto p=std::make_unique<HashPlan>();p->width=bytes;p->stride=allocation.stride;p->capacity=capacity;
    std::vector<uint8_t> weights(allocation.weights,0);
    for(uint32_t i=0;i<bytes;++i) for(uint32_t j=0;j<16;++j) weights[j*p->stride+i]=limbs[i*16+j];
    check(cudaMalloc(&p->weights,allocation.weights));
    check(cudaMalloc(&p->offsets,allocation.offsets));
    check(cudaMalloc(&p->partials,allocation.partials_s32));
    check(cudaMemcpy(p->weights,weights.data(),weights.size(),cudaMemcpyHostToDevice));
    check(cudaMemcpy(p->offsets,offsets,16,cudaMemcpyHostToDevice));
    // Split-K is fixed to one, hence this kernel has no semaphore workspace.
    *out=p.release();return 0;
  } catch(const std::exception& e) {if(error&&n)std::snprintf(error,n,"%s",e.what());return 1;}
}

__global__ void lossless_state_keys(const uint8_t* input,uint32_t* output,uint32_t count,uint32_t n,uint32_t stride,uint32_t bits){
 uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=count)return;
 auto k=mgbfs_lossless::pack(input+uint64_t(row)*stride,n,bits);
 reinterpret_cast<uint4*>(output)[row]=make_uint4(uint32_t(k.lo),uint32_t(k.lo>>32),uint32_t(k.hi),uint32_t(k.hi>>32));
}
extern "C" int mgbfs_lossless_hash_create(uint32_t n,uint32_t capacity,uint32_t bits,void**out,char*error,size_t size){
 if(!out)return 1;*out=nullptr;
 try{if(!mgbfs_lossless::shape(n,bits)||!capacity)throw std::runtime_error("LOSSLESS_KEY_DOMAIN");auto p=std::make_unique<HashPlan>();p->width=n;p->stride=(n+15)&~15u;p->capacity=capacity;p->lossless_bits=bits;*out=p.release();return 0;}
 catch(const std::exception&e){if(error&&size)std::snprintf(error,size,"%s",e.what());return 1;}
}
extern "C" int mgbfs_hash_run(void* plan,const uint8_t* input,uint32_t* output,uint32_t count,void* raw_stream) {
  auto* p=static_cast<HashPlan*>(plan);
  if(!p||!input||!output||count>p->capacity) return 1;
  if(count==0)return 0;
  auto stream=static_cast<cudaStream_t>(raw_stream);
  if(p->lossless_bits){if(reinterpret_cast<uintptr_t>(output)&15u)return 1;lossless_state_keys<<<(count+255)/256,256,0,stream>>>(input,output,count,p->width,p->stride,p->lossless_bits);return cudaGetLastError()==cudaSuccess?0:6;}
  Gemm::Arguments args({int(count),16,int(p->stride)}, {input,int(p->stride)},
    {p->weights,int(p->stride)}, {p->partials,16},{p->partials,16},{1,0},1);
  if(Gemm::get_workspace_size(args)!=0)return 2;
  if(p->gemm.can_implement(args)!=cutlass::Status::kSuccess)return 3;
  if(p->gemm.initialize(args,p->workspace,stream)!=cutlass::Status::kSuccess)return 4;
  if(p->gemm(stream)!=cutlass::Status::kSuccess)return 5;
  finish_hash<<<(count*4+255)/256,256,0,stream>>>(p->partials,p->offsets,output,count);
  return cudaGetLastError()==cudaSuccess?0:6;
}
extern "C" void mgbfs_hash_destroy(void* p) {delete static_cast<HashPlan*>(p);}

// Compact permutation hash: compose each generator with the linear hash once.
// Repeated run has no state materialization, allocations, or host readback.
struct CompactHashPlan {
 uint32_t n{},moves{},capacity{},stride{},lossless_bits{};bool move_major{};
 uint8_t* weights{};uint32_t* offsets{};int32_t* partials{};Gemm gemm;
 ~CompactHashPlan(){cudaFree(partials);cudaFree(offsets);cudaFree(weights);}
};
extern "C" int mgbfs_compact_hash_query(uint32_t n,uint32_t moves,uint32_t capacity,uint64_t* bytes){
 if(!bytes)return 1;*bytes=0;
 if(!n||n>128||!moves||moves>128||!capacity||uint64_t(capacity)*moves>0x3fffffffu)return 1;
 uint32_t stride=(n+15)&~15u,cols=moves*16;
 Gemm::Arguments a({int(capacity),int(cols),int(stride)},{nullptr,int(stride)},{nullptr,int(stride)},{nullptr,int(cols)},{nullptr,int(cols)},{1,0},1);
 if(Gemm::get_workspace_size(a)!=0||Gemm::can_implement(a)!=cutlass::Status::kSuccess)return 2;
 *bytes=uint64_t(cols)*stride+16+uint64_t(capacity)*cols*4;return 0;
}
extern "C" int mgbfs_compact_hash_create(uint32_t n,uint32_t moves,uint32_t capacity,const uint8_t* permutation,const uint8_t* limbs,const uint32_t* offsets,uint32_t move_major,void** out,char* error,size_t error_capacity){
 if(!out)return 1;*out=nullptr;
 try{
  uint64_t bytes;if(!permutation||!limbs||!offsets||move_major>1||mgbfs_compact_hash_query(n,moves,capacity,&bytes))throw std::runtime_error("COMPACT_HASH_SHAPE");
  auto p=std::make_unique<CompactHashPlan>();p->n=n;p->moves=moves;p->capacity=capacity;p->stride=(n+15)&~15u;p->move_major=move_major;
  std::vector<uint8_t> w(size_t(moves)*16*p->stride,0);
  for(uint32_t move=0;move<moves;++move){
   std::vector<bool> used(n,false);
   for(uint32_t j=0;j<n;++j){uint32_t col=permutation[size_t(move)*n+j];if(col>=n||used[col])throw std::runtime_error("COMPACT_HASH_PERMUTATION");used[col]=true;
    for(uint32_t limb=0;limb<16;++limb)w[(size_t(move)*16+limb)*p->stride+col]=limbs[size_t(j)*16+limb];
   }
  }
  for(int i=0;i<4;++i)if(offsets[i]>=4294967291ULL)throw std::runtime_error("COMPACT_HASH_OFFSET");
  check(cudaMalloc(&p->weights,w.size()));check(cudaMalloc(&p->offsets,16));check(cudaMalloc(&p->partials,uint64_t(capacity)*moves*64));
  check(cudaMemcpy(p->weights,w.data(),w.size(),cudaMemcpyHostToDevice));check(cudaMemcpy(p->offsets,offsets,16,cudaMemcpyHostToDevice));
  *out=p.release();return 0;
 }catch(const std::exception&e){if(error&&error_capacity)std::snprintf(error,error_capacity,"%s",e.what());return 1;}
}
__global__ void finish_compact_hash(const int32_t* sums,const uint32_t* offsets,uint32_t* output,uint32_t count,uint32_t moves,bool move_major){
 uint64_t word=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(word>=uint64_t(count)*moves*4)return;
 uint32_t lane=word%4;uint64_t child=word/4,parent=child/moves,move=child%moves;
 uint64_t sum=offsets[lane];
 #pragma unroll
 for(int limb=0;limb<4;++limb)sum+=uint64_t(sums[child*16+lane*4+limb])<<(8*limb);
 uint64_t dst=move_major?move*count+parent:child;output[dst*4+lane]=uint32_t(sum%4294967291ULL);
}

__global__ void lossless_child_keys(const uint8_t* parents,const uint8_t* permutation,uint32_t* output,uint32_t count,uint32_t n,uint32_t stride,uint32_t moves,uint32_t bits,bool move_major){
 uint64_t child=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(child>=uint64_t(count)*moves)return;
 uint32_t parent=uint32_t(child/moves),move=uint32_t(child%moves);mgbfs_lossless::Key k{0,0};
 for(uint32_t j=0;j<n;++j){uint64_t x=parents[uint64_t(parent)*stride+permutation[uint64_t(move)*n+j]];uint32_t at=j*bits;if(at<64){k.lo|=x<<at;if(at+bits>64)k.hi|=x>>(64-at);}else k.hi|=x<<(at-64);}
 k=mgbfs_lossless::mix(k);uint64_t dst=move_major?uint64_t(move)*count+parent:child;
 reinterpret_cast<uint4*>(output)[dst]=make_uint4(uint32_t(k.lo),uint32_t(k.lo>>32),uint32_t(k.hi),uint32_t(k.hi>>32));
}
extern "C" int mgbfs_lossless_compact_hash_create(uint32_t n,uint32_t moves,uint32_t capacity,const uint8_t* permutation,uint32_t bits,uint32_t move_major,void**out,char*error,size_t size){
 if(!out)return 1;*out=nullptr;
 try{if(!permutation||!mgbfs_lossless::shape(n,bits)||!moves||moves>128||!capacity||uint64_t(capacity)*moves>0x3fffffffu||move_major>1)throw std::runtime_error("LOSSLESS_CHILD_DOMAIN");
 for(uint32_t move=0;move<moves;++move){std::vector<bool> used(n,false);for(uint32_t j=0;j<n;++j){uint32_t col=permutation[uint64_t(move)*n+j];if(col>=n||used[col])throw std::runtime_error("LOSSLESS_CHILD_PERMUTATION");used[col]=true;}}
 auto p=std::make_unique<CompactHashPlan>();p->n=n;p->moves=moves;p->capacity=capacity;p->stride=(n+15)&~15u;p->lossless_bits=bits;p->move_major=move_major;
 check(cudaMalloc(&p->weights,uint64_t(n)*moves));check(cudaMemcpy(p->weights,permutation,uint64_t(n)*moves,cudaMemcpyHostToDevice));*out=p.release();return 0;}
 catch(const std::exception&e){if(error&&size)std::snprintf(error,size,"%s",e.what());return 1;}
}
extern "C" int mgbfs_compact_hash_run(void* raw,const uint8_t* parents,uint32_t* output,uint32_t count,void* stream){
 auto*p=static_cast<CompactHashPlan*>(raw);if(!p||!parents||!output||count>p->capacity)return 1;if(!count)return 0;
 auto s=static_cast<cudaStream_t>(stream);
 if(p->lossless_bits){if(reinterpret_cast<uintptr_t>(output)&15u)return 1;lossless_child_keys<<<(uint64_t(count)*p->moves+255)/256,256,0,s>>>(parents,p->weights,output,count,p->n,p->stride,p->moves,p->lossless_bits,p->move_major);return cudaGetLastError()==cudaSuccess?0:4;}
 uint32_t cols=p->moves*16;
 Gemm::Arguments a({int(count),int(cols),int(p->stride)},{parents,int(p->stride)},{p->weights,int(p->stride)},{p->partials,int(cols)},{p->partials,int(cols)},{1,0},1);
 if(Gemm::get_workspace_size(a)!=0||p->gemm.can_implement(a)!=cutlass::Status::kSuccess)return 2;
 if(p->gemm.initialize(a,nullptr,s)!=cutlass::Status::kSuccess||p->gemm(s)!=cutlass::Status::kSuccess)return 3;
 finish_compact_hash<<<(uint64_t(count)*p->moves*4+255)/256,256,0,s>>>(p->partials,p->offsets,output,count,p->moves,p->move_major);
 return cudaGetLastError()==cudaSuccess?0:4;
}
extern "C" void mgbfs_compact_hash_destroy(void*p){delete static_cast<CompactHashPlan*>(p);}
