#include <cstdint>
#include "mgbfs_cuda.h"
#include <cstddef>
#include <cstdio>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include "cutlass/gemm/device/gemm.h"

template<int M,int K>
using MatrixGemm = cutlass::gemm::device::Gemm<
  uint8_t, cutlass::layout::RowMajor, uint8_t, cutlass::layout::ColumnMajor,
  int32_t, cutlass::layout::RowMajor, int32_t,
  cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
  cutlass::gemm::GemmShape<M,32,K>, cutlass::gemm::GemmShape<(M==128?64:32),32,K>,
  cutlass::gemm::GemmShape<8,8,16>,
  cutlass::epilogue::thread::LinearCombination<int32_t,4,int32_t,int32_t>,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,2>;
struct GeneratePlan {
  uint32_t n{},moves{},modulus{},capacity{},k{},stride{},variant{},generator_rows{};
  int max_grid_x{},max_grid_y{};
  uint8_t* generators{};
  uint8_t* parents{};
  int32_t* products{};
  MatrixGemm<64,64> gemm64k64;
  MatrixGemm<64,32> gemm64k32;
  MatrixGemm<128,32> gemm128k32;
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
template<bool Transposed>
__global__ void modular_materialize(const int32_t* products,uint8_t* output,uint32_t n,uint32_t moves,uint32_t modulus,uint32_t stride,uint32_t columns,uint32_t generator_rows,uint32_t count){
  const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=size_t(count)*moves*stride)return;
  const uint32_t byte=i%stride;const size_t child=i/stride;
  if(byte>=n*n){output[i]=0;return;}
  const uint32_t parent=child/moves,move=child%moves;
  const size_t index=Transposed
    ? size_t(parent*n+byte%n)*generator_rows+move*n+byte/n
    : size_t(move*n+byte/n)*columns+parent*n+byte%n;
  output[i]=uint32_t(products[index])%modulus;
}
// U4-only alternative: four aligned 16-byte loads, register transpose, one
// 16-byte state store per lane. Neighboring moves read adjacent vectors.
__device__ uint32_t pack_row(uint32_t a,uint32_t b,uint32_t c,uint32_t d,uint32_t modulus){
  return (a%modulus)|((b%modulus)<<8)|((c%modulus)<<16)|((d%modulus)<<24);
}
__global__ void materialize_u4_vectors(const uint4* products,uint4* output,uint32_t moves,uint32_t modulus,uint32_t count){
  const size_t child=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(child>=size_t(count)*moves)return;
  const size_t parent=child/moves,move=child%moves;
  const uint4 a=products[(parent*4+0)*moves+move],b=products[(parent*4+1)*moves+move];
  const uint4 c=products[(parent*4+2)*moves+move],d=products[(parent*4+3)*moves+move];
  output[child]=make_uint4(pack_row(a.x,b.x,c.x,d.x,modulus),pack_row(a.y,b.y,c.y,d.y,modulus),
    pack_row(a.z,b.z,c.z,d.z,modulus),pack_row(a.w,b.w,c.w,d.w,modulus));
}
extern "C" int mgbfs_generate_create(uint32_t n,uint32_t moves,uint32_t modulus,uint32_t capacity,const uint8_t* generators,void** out,char* error,size_t error_capacity){
  return mgbfs_generate_create_variant(n,moves,modulus,capacity,generators,0,out,error,error_capacity);
}
extern "C" int mgbfs_generate_create_variant(uint32_t n,uint32_t moves,uint32_t modulus,uint32_t capacity,const uint8_t* generators,uint32_t variant,void** out,char* error,size_t error_capacity){
  if(!out)return 1;*out=nullptr;
  try {
    if(variant>4||(variant==4&&n!=4))throw std::runtime_error("GENERATION_VARIANT");
    if(n==0||uint64_t(n)*n>33025||moves==0||moves>65535||modulus<2||modulus>256||capacity==0||!generators||uint64_t(capacity)*n>INT32_MAX-3||uint64_t(n)*(modulus-1)*(modulus-1)>INT32_MAX)
      throw std::runtime_error("GENERATION_SHAPE_OR_ACCUMULATOR_BOUND");
    int device;checked(cudaGetDevice(&device));cudaDeviceProp prop;checked(cudaGetDeviceProperties(&prop,device));
    if(prop.major*10+prop.minor<75)throw std::runtime_error("UNSUPPORTED_SM");
    auto p=std::make_unique<GeneratePlan>();p->n=n;p->moves=moves;p->modulus=modulus;p->capacity=capacity;p->k=(n+15)&~15u;p->stride=(n*n+15)&~15u;
    p->variant=variant;p->max_grid_x=prop.maxGridSize[0];p->max_grid_y=prop.maxGridSize[1];
    p->generator_rows=variant?((moves*n+3)&~3u):moves*n;
    const uint32_t columns=(capacity*n+3)&~3u;
    const size_t product_bytes=size_t(p->generator_rows)*columns*sizeof(int32_t);
    if(product_bytes/sizeof(int32_t)/columns!=p->generator_rows)throw std::runtime_error("GENERATION_BYTE_OVERFLOW");
    std::vector<uint8_t> stacked(size_t(p->generator_rows)*p->k,0);
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
template<class Gemm>
static int launch_gemm(GeneratePlan* p,Gemm& gemm,uint32_t columns,cudaStream_t stream){
  const bool transposed=p->variant!=0;
  const int m=transposed?columns:p->generator_rows,n=transposed?p->generator_rows:columns;
  typename Gemm::Arguments args({m,n,int(p->k)},
    {transposed?p->parents:p->generators,int(p->k)},
    {transposed?p->generators:p->parents,int(p->k)},
    {p->products,n},{p->products,n},{1,0},1);
  if(Gemm::get_workspace_size(args)!=0||gemm.can_implement(args)!=cutlass::Status::kSuccess)return 3;
  if(gemm.initialize(args,nullptr,stream)!=cutlass::Status::kSuccess)return 4;
  if(gemm(stream)!=cutlass::Status::kSuccess)return 5;
  return 0;
}
static int generate_run(void* plan,const uint8_t* parents,uint8_t* children,uint32_t count,void* raw_stream,void* const* marks){
  auto* p=static_cast<GeneratePlan*>(plan);
  if(!p||!parents||!children||count>p->capacity)return 1;
  if(count==0)return 0;
  auto stream=static_cast<cudaStream_t>(raw_stream);
  const uint32_t columns=(count*p->n+3)&~3u;
  const uint32_t tile_m=p->variant==2?128:64;
  const uint64_t grid_x=(uint64_t(p->variant?columns:p->generator_rows)+tile_m-1)/tile_m;
  const uint64_t grid_y=(uint64_t(p->variant?p->generator_rows:columns)+31)/32;
  if(grid_x>uint64_t(p->max_grid_x)||grid_y>uint64_t(p->max_grid_y))return 7;
  if(marks&&cudaEventRecord(static_cast<cudaEvent_t>(marks[0]),stream)!=cudaSuccess)return 8;
  pack_parents<<<(size_t(columns)*p->k+255)/256,256,0,stream>>>(parents,p->parents,p->n,p->k,p->stride,columns,count);
  if(cudaGetLastError()!=cudaSuccess)return 2;
  if(marks&&cudaEventRecord(static_cast<cudaEvent_t>(marks[1]),stream)!=cudaSuccess)return 8;
  int status=0;
  switch(p->variant){
    case 1:case 4:status=launch_gemm(p,p->gemm64k32,columns,stream);break;
    case 2:status=launch_gemm(p,p->gemm128k32,columns,stream);break;
    default:status=launch_gemm(p,p->gemm64k64,columns,stream);break;
  }
  if(status)return status;
  if(marks&&cudaEventRecord(static_cast<cudaEvent_t>(marks[2]),stream)!=cudaSuccess)return 8;
  const size_t blocks=(size_t(count)*p->moves*p->stride+255)/256;
  if(p->variant==4)materialize_u4_vectors<<<(size_t(count)*p->moves+255)/256,256,0,stream>>>(reinterpret_cast<const uint4*>(p->products),reinterpret_cast<uint4*>(children),p->moves,p->modulus,count);
  else if(p->variant)modular_materialize<true><<<blocks,256,0,stream>>>(p->products,children,p->n,p->moves,p->modulus,p->stride,columns,p->generator_rows,count);
  else modular_materialize<false><<<blocks,256,0,stream>>>(p->products,children,p->n,p->moves,p->modulus,p->stride,columns,p->generator_rows,count);
  if(marks&&cudaEventRecord(static_cast<cudaEvent_t>(marks[3]),stream)!=cudaSuccess)return 8;
  return cudaGetLastError()==cudaSuccess?0:6;
}
extern "C" int mgbfs_generate_run(void* plan,const uint8_t* parents,uint8_t* children,uint32_t count,void* stream){
  return generate_run(plan,parents,children,count,stream,nullptr);
}
extern "C" int mgbfs_generate_profile_run(void* plan,const uint8_t* parents,uint8_t* children,uint32_t count,void* stream,void* const* marks){
  if(!marks||!count)return 1;
  return generate_run(plan,parents,children,count,stream,marks);
}
extern "C" void mgbfs_generate_destroy(void* p){delete static_cast<GeneratePlan*>(p);}
