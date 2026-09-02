#include <cstdint>
#include "mgbfs_cuda.h"
#include <cstddef>
#include <cstdio>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include "cutlass/gemm/device/gemm.h"

using MatrixGemm = cutlass::gemm::device::Gemm<
  uint8_t, cutlass::layout::RowMajor, uint8_t, cutlass::layout::ColumnMajor,
  int32_t, cutlass::layout::RowMajor, int32_t,
  cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
  cutlass::gemm::GemmShape<64,32,64>, cutlass::gemm::GemmShape<32,32,64>,
  cutlass::gemm::GemmShape<8,8,16>,
  cutlass::epilogue::thread::LinearCombination<int32_t,4,int32_t,int32_t>,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,2>;
struct GeneratePlan {
  uint32_t n{},moves{},modulus{},capacity{},k{},stride{};
  uint8_t* generators{};
  uint8_t* parents{};
  int32_t* products{};
  MatrixGemm gemm;
  ~GeneratePlan(){cudaFree(products);cudaFree(parents);cudaFree(generators);}
};
static void checked(cudaError_t r){if(r!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(r));}

// Concatenated parent columns: B[k, parent*n+column], column-major.
__global__ void pack_parents(const uint8_t* input,uint8_t* packed,uint32_t n,uint32_t k,uint32_t stride,uint32_t columns,uint32_t count){
  const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=size_t(columns)*k)return;
  const uint32_t col=i/k,row=i%k,parent=col/n,c=col%n;
  packed[i]=(row<n&&parent<count)?input[size_t(parent)*stride+row*n+c]:0;
}
// Bounded materialization into parent-major / move-major canonical rows.
// Zero padding makes this directly consumable by the following hash GEMM.
__global__ void modular_materialize(const int32_t* products,uint8_t* output,uint32_t n,uint32_t moves,uint32_t modulus,uint32_t stride,uint32_t columns,uint32_t count){
  const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=size_t(count)*moves*stride)return;
  const uint32_t byte=i%stride;const size_t child=i/stride;
  if(byte>=n*n){output[i]=0;return;}
  const uint32_t parent=child/moves,move=child%moves;
  output[i]=uint32_t(products[size_t(move*n+byte/n)*columns+parent*n+byte%n])%modulus;
}
extern "C" int mgbfs_generate_create(uint32_t n,uint32_t moves,uint32_t modulus,uint32_t capacity,const uint8_t* generators,void** out,char* error,size_t error_capacity){
  if(!out)return 1;*out=nullptr;
  try {
    if(n==0||uint64_t(n)*n>33025||moves==0||moves>65535||modulus<2||modulus>256||capacity==0||!generators||uint64_t(capacity)*n>INT32_MAX-3||uint64_t(n)*(modulus-1)*(modulus-1)>INT32_MAX)
      throw std::runtime_error("GENERATION_SHAPE_OR_ACCUMULATOR_BOUND");
    int device;checked(cudaGetDevice(&device));cudaDeviceProp prop;checked(cudaGetDeviceProperties(&prop,device));
    if(prop.major*10+prop.minor<75)throw std::runtime_error("UNSUPPORTED_SM");
    auto p=std::make_unique<GeneratePlan>();p->n=n;p->moves=moves;p->modulus=modulus;p->capacity=capacity;p->k=(n+15)&~15u;p->stride=(n*n+15)&~15u;
    const uint32_t columns=(capacity*n+3)&~3u;
    const size_t product_bytes=size_t(moves)*n*columns*sizeof(int32_t);
    if(product_bytes/sizeof(int32_t)/columns/n!=moves)throw std::runtime_error("GENERATION_BYTE_OVERFLOW");
    std::vector<uint8_t> stacked(size_t(moves)*n*p->k,0);
    for(size_t row=0;row<size_t(moves)*n;++row)for(uint32_t c=0;c<n;++c){
      if(generators[row*n+c]>=modulus)throw std::runtime_error("GENERATOR_NONCANONICAL");
      stacked[row*p->k+c]=generators[row*n+c];
    }
    checked(cudaMalloc(&p->generators,stacked.size()));
    checked(cudaMalloc(&p->parents,size_t(columns)*p->k));
    checked(cudaMalloc(&p->products,product_bytes));
    checked(cudaMemcpy(p->generators,stacked.data(),stacked.size(),cudaMemcpyHostToDevice));
    *out=p.release();return 0;
  }catch(const std::exception& e){if(error&&error_capacity)std::snprintf(error,error_capacity,"%s",e.what());return 1;}
}
extern "C" int mgbfs_generate_run(void* plan,const uint8_t* parents,uint8_t* children,uint32_t count,void* raw_stream){
  auto* p=static_cast<GeneratePlan*>(plan);
  if(!p||!parents||!children||count>p->capacity)return 1;
  if(count==0)return 0;
  auto stream=static_cast<cudaStream_t>(raw_stream);
  const uint32_t columns=(count*p->n+3)&~3u;
  pack_parents<<<(size_t(columns)*p->k+255)/256,256,0,stream>>>(parents,p->parents,p->n,p->k,p->stride,columns,count);
  if(cudaGetLastError()!=cudaSuccess)return 2;
  MatrixGemm::Arguments args({int(p->moves*p->n),int(columns),int(p->k)},
    {p->generators,int(p->k)},{p->parents,int(p->k)},
    {p->products,int(columns)},{p->products,int(columns)},{1,0},1);
  if(MatrixGemm::get_workspace_size(args)!=0||p->gemm.can_implement(args)!=cutlass::Status::kSuccess)return 3;
  if(p->gemm.initialize(args,nullptr,stream)!=cutlass::Status::kSuccess)return 4;
  if(p->gemm(stream)!=cutlass::Status::kSuccess)return 5;
  modular_materialize<<<(size_t(count)*p->moves*p->stride+255)/256,256,0,stream>>>(p->products,children,p->n,p->moves,p->modulus,p->stride,columns,count);
  return cudaGetLastError()==cudaSuccess?0:6;
}
extern "C" void mgbfs_generate_destroy(void* p){delete static_cast<GeneratePlan*>(p);}
