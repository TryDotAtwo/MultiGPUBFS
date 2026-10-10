#pragma once
#include <cutlass/gemm/device/gemm.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
using GenericGemmOp=cutlass::gemm::device::Gemm<uint8_t,cutlass::layout::RowMajor,uint8_t,cutlass::layout::ColumnMajor,int32_t,cutlass::layout::ColumnMajor,int32_t,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,cutlass::gemm::GemmShape<128,128,64>,cutlass::gemm::GemmShape<64,64,64>,cutlass::gemm::GemmShape<16,8,32>>;
struct GenericGemmContext{uint32_t n,m,g,batch,k,columns,ld;uint8_t *a=nullptr,*b=nullptr;int32_t*c=nullptr;};
extern "C" int mgbfs_generic_gemm_create(uint32_t n,uint32_t m,uint32_t g,uint32_t batch,const int64_t* matrices,void** result){
 if(!result||!matrices||!n||n>64||!m||m>64||!g||g>64||!batch||uint64_t(batch)*m>1048576)return int(cudaErrorInvalidValue);
 int device;cudaDeviceProp prop;cudaError_t e=cudaGetDevice(&device);if(e!=cudaSuccess)return int(e);e=cudaGetDeviceProperties(&prop,device);if(e!=cudaSuccess)return int(e);if(prop.major<8)return int(cudaErrorNotSupported);
 auto*x=new GenericGemmContext{n,m,g,batch,(n+15)/16*16,(g*n+7)/8*8,(batch*m+7)/8*8};
 auto cleanup=[&](){cudaFree(x->a);cudaFree(x->b);cudaFree(x->c);delete x;};
 std::vector<uint8_t> packed(uint64_t(x->k)*x->columns,0);
 for(uint32_t gen=0;gen<g;gen++)for(uint32_t row=0;row<n;row++)for(uint32_t k=0;k<n;k++){int64_t v=matrices[uint64_t(gen)*n*n+row*n+k];if(v<0||v>255){cleanup();return int(cudaErrorInvalidValue);}packed[uint64_t(gen*n+row)*x->k+k]=uint8_t(v);}
 e=cudaMalloc(&x->a,uint64_t(x->ld)*x->k);if(e==cudaSuccess)e=cudaMalloc(&x->b,packed.size());if(e==cudaSuccess)e=cudaMalloc(&x->c,uint64_t(x->ld)*x->columns*4);if(e==cudaSuccess)e=cudaMemcpy(x->b,packed.data(),packed.size(),cudaMemcpyHostToDevice);
 if(e!=cudaSuccess){cleanup();return int(e);}*result=x;return 0;
}
extern "C" int mgbfs_generic_gemm_destroy(void*raw){auto*x=static_cast<GenericGemmContext*>(raw);if(x){cudaFree(x->a);cudaFree(x->b);cudaFree(x->c);delete x;}return 0;}
__global__ void generic_gemm_pack(const int64_t*parents,uint32_t count,uint32_t stride,uint32_t n,uint32_t m,uint32_t k,uint8_t*out){
 __shared__ uint8_t tile[32][33];uint32_t p=blockIdx.x*32+threadIdx.x,base=blockIdx.y*32,col=blockIdx.z;
 for(uint32_t j=threadIdx.y;j<32;j+=8){uint32_t row=base+j;tile[j][threadIdx.x]=(p<count&&row<n)?uint8_t(parents[uint64_t(row*m+col)*stride+p]):0;}__syncthreads();
 for(uint32_t j=threadIdx.y;j<32;j+=8){uint32_t parent=blockIdx.x*32+j,row=base+threadIdx.x;if(parent<count&&row<k)out[(uint64_t(col)*count+parent)*k+row]=tile[threadIdx.x][j];}
}
static int generic_gemm_compute(GenericGemmContext*x,const int64_t*parents,uint32_t count,uint32_t stride,cudaStream_t stream){
 if(!x||count>x->batch||count>stride||(!parents&&count))return int(cudaErrorInvalidValue);if(!count)return 0;
 generic_gemm_pack<<<dim3((count+31)/32,(x->k+31)/32,x->m),dim3(32,8),0,stream>>>(parents,count,stride,x->n,x->m,x->k,x->a);
 auto e=cudaGetLastError();if(e!=cudaSuccess)return int(e);
 GenericGemmOp op;GenericGemmOp::Arguments args({int(count*x->m),int(x->columns),int(x->k)},{x->a,int(x->k)},{x->b,int(x->k)},{x->c,int(x->ld)},{x->c,int(x->ld)},{1,0});
 if(op.can_implement(args)!=cutlass::Status::kSuccess)return int(cudaErrorNotSupported);
 if(op.initialize(args,nullptr,stream)!=cutlass::Status::kSuccess||op(stream)!=cutlass::Status::kSuccess)return int(cudaErrorUnknown);return int(cudaGetLastError());
}
