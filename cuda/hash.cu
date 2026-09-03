#include <cstdint>
#include "mgbfs_cuda.h"
#include "allocation_shape.h"
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
  uint32_t width{},stride{},capacity{};
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
extern "C" int mgbfs_hash_run(void* plan,const uint8_t* input,uint32_t* output,uint32_t count,void* raw_stream) {
  auto* p=static_cast<HashPlan*>(plan);
  if(!p||!input||!output||count>p->capacity) return 1;
  if(count==0)return 0;
  auto stream=static_cast<cudaStream_t>(raw_stream);
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
