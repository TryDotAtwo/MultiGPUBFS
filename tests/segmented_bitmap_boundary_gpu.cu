// Exercise the real bitmap insertion at row 2^32-1 and the following spill.
// VMM reserves the address range but commits only one device-memory page.
#include "generic_route.h"
#include "segmented_history.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
extern "C" int mgbfs_bitmap_accept(uint32_t,uint32_t,uint32_t,const void*,const void*,
 const GenericRouteRecord*,const GenericRouteRecord*,const uint32_t*,const uint32_t*,uint32_t,uint32_t,
 uint32_t,uint64_t,uint64_t,uint64_t*,uint32_t,uint32_t,uint32_t,uint32_t,const uint32_t*,uint32_t,
 MgbfsSegment,uint32_t,uint64_t*,uint32_t*,uint32_t*,uint32_t*,uint64_t*,uint64_t,void*);
#define D(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char* n=nullptr;cuGetErrorName(r,&n);fprintf(stderr,"driver %s line %d\n",n?n:"?",__LINE__);return 1;}}while(0)
#define C(x) do{cudaError_t r=(x);if(r!=cudaSuccess){fprintf(stderr,"runtime %s line %d\n",cudaGetErrorString(r),__LINE__);return 1;}}while(0)
int main(){
 int devices=0;C(cudaGetDeviceCount(&devices));if(!devices)return 77;
 C(cudaSetDevice(0));C(cudaFree(nullptr));D(cuInit(0));
 CUdevice dev;D(cuDeviceGet(&dev,0));int supported=0;
 D(cuDeviceGetAttribute(&supported,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,dev));
 if(!supported){puts("SKIP virtual-memory management unsupported");return 77;}
 CUmemAllocationProp prop{};prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
 prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=dev;
 size_t page=0;D(cuMemGetAllocationGranularity(&page,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
 const uint64_t cap=1ull<<32, length=3*cap;
 if(cap%page){puts("SKIP granularity incompatible");return 77;}
 CUdeviceptr addr=0;D(cuMemAddressReserve(&addr,length,page,0,0));
 CUmemGenericAllocationHandle h;D(cuMemCreate(&h,page,&prop,0));
 D(cuMemMap(addr+cap-page,page,0,h,0));
 CUmemAccessDesc access{};access.location=prop.location;access.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
 D(cuMemSetAccess(addr+cap-page,page,&access,1));
 GenericRouteRecord *meta=nullptr;uint32_t *count=nullptr,*spill=nullptr,*spills=nullptr,*error=nullptr;
 uint64_t *accepted=nullptr,*bits=nullptr;
 C(cudaMalloc((void**)&meta,sizeof(*meta)));C(cudaMalloc((void**)&count,4));C(cudaMalloc((void**)&spill,4));
 C(cudaMalloc((void**)&spills,4));C(cudaMalloc((void**)&error,4));C(cudaMalloc((void**)&accepted,8));C(cudaMalloc((void**)&bits,8));
 GenericRouteRecord rec{0,42,0,0,0,0};uint32_t one=1;uint64_t initial=cap-1;
 C(cudaMemcpy(meta,&rec,sizeof(rec),cudaMemcpyHostToDevice));C(cudaMemcpy(count,&one,4,cudaMemcpyHostToDevice));
 C(cudaMemcpy(accepted,&initial,8,cudaMemcpyHostToDevice));C(cudaMemset(bits,0,8));C(cudaMemset(spills,0,4));C(cudaMemset(error,0,4));
 MgbfsSegment seg{nullptr,(void*)addr,nullptr,0,cap,3};
 auto accept=[&](){return (cudaError_t)mgbfs_bitmap_accept(1,1,1,nullptr,nullptr,meta,meta,count,count,0,1,0,2,2,bits,0,1,1,1,nullptr,1,seg,0,accepted,spill,spills,error,nullptr,0,nullptr);};
 C(accept());C(cudaDeviceSynchronize());
 uint64_t got=0,mask=0;uint32_t failed=0,overflow=0;uint8_t value=0;
 C(cudaMemcpy(&got,accepted,8,cudaMemcpyDeviceToHost));C(cudaMemcpy(&mask,bits,8,cudaMemcpyDeviceToHost));
 C(cudaMemcpy(&failed,error,4,cudaMemcpyDeviceToHost));C(cudaMemcpy(&value,(void*)(addr+cap-1),1,cudaMemcpyDeviceToHost));
 if(got!=cap||mask!=1||failed||value!=42){fprintf(stderr,"FAIL boundary count=%llu bits=%llu error=%u value=%u\n",(unsigned long long)got,(unsigned long long)mask,failed,value);return 1;}
 rec.hash=1;rec.parent=43;C(cudaMemcpy(meta,&rec,sizeof(rec),cudaMemcpyHostToDevice));
 C(accept());C(cudaDeviceSynchronize());
 C(cudaMemcpy(&got,accepted,8,cudaMemcpyDeviceToHost));C(cudaMemcpy(&mask,bits,8,cudaMemcpyDeviceToHost));
 C(cudaMemcpy(&failed,error,4,cudaMemcpyDeviceToHost));C(cudaMemcpy(&overflow,spills,4,cudaMemcpyDeviceToHost));
 if(got!=cap||mask!=1||failed||overflow!=1){fprintf(stderr,"FAIL spill count=%llu bits=%llu error=%u spill=%u\n",(unsigned long long)got,(unsigned long long)mask,failed,overflow);return 1;}
 C(cudaFree(meta));C(cudaFree(count));C(cudaFree(spill));C(cudaFree(spills));C(cudaFree(error));C(cudaFree(accepted));C(cudaFree(bits));
 D(cuMemUnmap(addr+cap-page,page));D(cuMemRelease(h));D(cuMemAddressFree(addr,length));
 printf("PASS bitmap row 2^32-1 and full-segment spill; committed %zu bytes\n",page);return 0;
}
