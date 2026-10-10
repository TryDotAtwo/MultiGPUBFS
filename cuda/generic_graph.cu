#include "generic_graph.h"
#include <cuda_runtime.h>
#include <cstdint>

template<class State> __global__ void generic_successors(uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const State* parents,uint32_t parent_count,uint32_t parent_stride,const uint32_t* permutations,
 const int64_t* matrices,const uint32_t* moduli,const uint64_t* selected,uint32_t count,
 State* output,uint32_t output_stride,uint32_t* error) {
 const uint64_t work=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 const uint64_t step=uint64_t(blockDim.x)*gridDim.x;
 for(uint64_t i=work;i<uint64_t(elements)*count;i+=step) {
  const uint32_t element=i/count,row=i%count;
  const uint64_t child=selected?selected[row]:row;
  if(child>=uint64_t(parent_count)*generators){if(error)atomicCAS(error,0u,1u);output[uint64_t(element)*output_stride+row]=0;continue;}
  const uint32_t parent=child/generators,generator=child%generators;
  int64_t value;
  if(kind==0) {
   const uint32_t source=permutations[uint64_t(generator)*elements+element];
   if(source>=elements){if(error)atomicCAS(error,0u,2u);output[uint64_t(element)*output_stride+row]=0;continue;}
   value=parents[uint64_t(source)*parent_stride+parent];
  } else {
   const uint32_t matrix_row=element/m,column=element%m;
   const int64_t* matrix=matrices+uint64_t(generator)*n*n;
   uint64_t sum=0;
   // Unsigned arithmetic makes int64 two's-complement wrap explicit, without
   // signed C++ overflow. Reduction occurs after wrapping, like CayleyPy.
   for(uint32_t k=0;k<n;k++)sum+=uint64_t(matrix[uint64_t(matrix_row)*n+k])*uint64_t(parents[uint64_t(k*m+column)*parent_stride+parent]);
   value=static_cast<int64_t>(sum);
   const uint32_t modulo=moduli[generator];
   if(modulo){value%=int64_t(modulo);if(value<0)value+=modulo;}
  }
  output[uint64_t(element)*output_stride+row]=value;
 }
}
extern "C" int mgbfs_generic_generate_i64(uint32_t kind,uint32_t elements,uint32_t rows,uint32_t cols,uint32_t generators,
 const int64_t* parents,uint32_t parent_count,uint32_t parent_stride,const uint32_t* permutation_tables,
 const int64_t* matrix_tables,const uint32_t* moduli,const uint64_t* selected_children,uint32_t output_count,
 int64_t* output,uint32_t output_stride,uint32_t* device_error,void* raw_stream) {
 if(kind>1 || !elements || !rows || !cols || !generators || parent_count>parent_stride || output_count>output_stride ||
  (kind==0 && (rows!=elements || cols!=1)) || (kind==1 && uint64_t(rows)*cols!=elements) ||
  (!selected_children && output_count>uint64_t(parent_count)*generators) ||
  (selected_children && !device_error))return int(cudaErrorInvalidValue);
 if(!output_count)return 0;
 if(!parents || !output || (kind==0 && !permutation_tables) || (kind==1 && (!matrix_tables || !moduli)))return int(cudaErrorInvalidValue);
 const uint64_t blocks=(uint64_t(elements)*output_count+255)/256;
 const uint32_t grid=uint32_t(blocks>65535?65535:blocks);
 generic_successors<<<grid,256,0,static_cast<cudaStream_t>(raw_stream)>>>(kind,elements,rows,cols,generators,parents,parent_count,parent_stride,
  permutation_tables,matrix_tables,moduli,selected_children,output_count,output,output_stride,device_error);
 return int(cudaGetLastError());
}

// Compact permutation payload ABI shares the exact same kernel logic.
extern "C" int mgbfs_generic_generate_u8(uint32_t kind,uint32_t elements,uint32_t rows,uint32_t cols,uint32_t generators,
 const uint8_t* parents,uint32_t parent_count,uint32_t parent_stride,const uint32_t* permutation_tables,
 const int64_t* matrix_tables,const uint32_t* moduli,const uint64_t* selected_children,uint32_t output_count,
 uint8_t* output,uint32_t output_stride,uint32_t* device_error,void* raw_stream) {
 if(kind>0 || !elements || !rows || !cols || !generators || parent_count>parent_stride || output_count>output_stride ||
  (kind==0 && (rows!=elements || cols!=1)) || (kind==1 && uint64_t(rows)*cols!=elements) ||
  (!selected_children && output_count>uint64_t(parent_count)*generators) ||
  (selected_children && !device_error))return int(cudaErrorInvalidValue);
 if(!output_count)return 0;
 if(!parents || !output || (kind==0 && !permutation_tables) || (kind==1 && (!matrix_tables || !moduli)))return int(cudaErrorInvalidValue);
 const uint64_t blocks=(uint64_t(elements)*output_count+255)/256;
 const uint32_t grid=uint32_t(blocks>65535?65535:blocks);
 generic_successors<<<grid,256,0,static_cast<cudaStream_t>(raw_stream)>>>(kind,elements,rows,cols,generators,parents,parent_count,parent_stride,
  permutation_tables,matrix_tables,moduli,selected_children,output_count,output,output_stride,device_error);
 return int(cudaGetLastError());
}

extern "C" int mgbfs_generic_hardware_info(int device,uint64_t* values){
 if(!values)return int(cudaErrorInvalidValue);auto e=cudaSetDevice(device);if(e!=cudaSuccess)return int(e);
 cudaDeviceProp p;e=cudaGetDeviceProperties(&p,device);if(e!=cudaSuccess)return int(e);
 values[0]=p.multiProcessorCount;values[1]=p.major;values[2]=p.minor;values[3]=p.l2CacheSize;values[4]=p.warpSize;values[5]=p.maxThreadsPerMultiProcessor;values[6]=p.maxThreadsPerBlock;values[7]=p.sharedMemPerMultiprocessor;return 0;
}
