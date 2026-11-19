// Exercise the real routed kernel with a >32-bit SoA stride. Only two CUDA
// virtual-memory pages are committed; the large gap is not GPU allocation.
#include "generic_route.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
extern "C" int mgbfs_segmented_route_storage(uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,const void*,uint32_t,uint64_t,const uint32_t*,const int64_t*,const uint32_t*,uint64_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint64_t,const uint32_t*,const uint64_t*,GenericRouteRecord*,uint32_t*,uint32_t*,void*);
#define D(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char* n=nullptr;cuGetErrorName(r,&n);fprintf(stderr,"driver %s line %d\n",n?n:"?",__LINE__);return 1;}}while(0)
#define C(x) do{cudaError_t r=(x);if(r!=cudaSuccess){fprintf(stderr,"runtime %s line %d\n",cudaGetErrorString(r),__LINE__);return 1;}}while(0)
int main(){
 int devices=0;C(cudaGetDeviceCount(&devices));if(!devices)return 77;
 C(cudaSetDevice(0));C(cudaFree(nullptr));D(cuInit(0));
 CUdevice dev;D(cuDeviceGet(&dev,0));int supported=0;
 D(cuDeviceGetAttribute(&supported,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,dev));
 if(!supported){printf("SKIP virtual-memory management unsupported\n");return 77;}
 CUmemAllocationProp prop{};prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
 prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=dev;
 size_t page=0;D(cuMemGetAllocationGranularity(&page,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
 const uint64_t stride=3ull*(1ull<<32);
 if(stride%page){printf("SKIP allocation granularity incompatible with test\n");return 77;}
 const size_t length=size_t(stride)+page;
 CUdeviceptr addr=0;D(cuMemAddressReserve(&addr,length,page,0,0));
 CUmemGenericAllocationHandle h0,h1;D(cuMemCreate(&h0,page,&prop,0));D(cuMemCreate(&h1,page,&prop,0));
 D(cuMemMap(addr,page,0,h0,0));D(cuMemMap(addr+stride,page,0,h1,0));
 CUmemAccessDesc access{};access.location=prop.location;access.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
 D(cuMemSetAccess(addr,page,&access,1));D(cuMemSetAccess(addr+stride,page,&access,1));
 uint8_t a=7,b=11;C(cudaMemcpy((void*)addr,&a,1,cudaMemcpyHostToDevice));C(cudaMemcpy((void*)(addr+stride),&b,1,cudaMemcpyHostToDevice));
 uint32_t *perms=nullptr,*counts=nullptr,*error=nullptr;GenericRouteRecord* records=nullptr;
 C(cudaMalloc((void**)&perms,8));C(cudaMalloc((void**)&counts,4));C(cudaMalloc((void**)&error,4));C(cudaMalloc((void**)&records,8*sizeof(GenericRouteRecord)));
 uint32_t perm[2]={1,0};C(cudaMemcpy(perms,perm,8,cudaMemcpyHostToDevice));C(cudaMemset(counts,0,4));C(cudaMemset(error,0,4));
 C((cudaError_t)mgbfs_segmented_route_storage(1,0,2,2,1,1,(void*)addr,1,stride,perms,nullptr,nullptr,123,64,1,0,1,8,0,nullptr,nullptr,records,counts,error,nullptr));
 C(cudaDeviceSynchronize());uint32_t n=0,e=0;GenericRouteRecord out{};
 C(cudaMemcpy(&n,counts,4,cudaMemcpyDeviceToHost));C(cudaMemcpy(&e,error,4,cudaMemcpyDeviceToHost));C(cudaMemcpy(&out,records,sizeof(out),cudaMemcpyDeviceToHost));
 if(n!=1||e!=0||(out.parent&0xffff)!=uint64_t(11|(7<<8))){fprintf(stderr,"FAIL large stride n=%u err=%u parent=%llx\n",n,e,(unsigned long long)out.parent);return 1;}
 C(cudaFree(perms));C(cudaFree(counts));C(cudaFree(error));C(cudaFree(records));
 D(cuMemUnmap(addr,page));D(cuMemUnmap(addr+stride,page));D(cuMemRelease(h0));D(cuMemRelease(h1));D(cuMemAddressFree(addr,length));
 printf("PASS routed SoA stride %llu with two committed pages of %zu bytes\n",(unsigned long long)stride,page);return 0;
}
