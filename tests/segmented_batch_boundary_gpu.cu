
// VMM proves addressing of local row 2^32-1; it does NOT allocate or validate
// 2^32 physical state rows. Only a last device-memory page of each plane is mapped.
#include "generic_route.h"
#include "segmented_history.cuh"
#include "segmented_hash_batch.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
extern "C" int mgbfs_bitmap_accept_batch_scratch_bytes(uint32_t,size_t*);
extern "C" int mgbfs_bitmap_accept_batch(uint32_t,uint32_t,uint32_t,const void*,const void*,
 const GenericRouteRecord*,const GenericRouteRecord*,const uint32_t*,const uint32_t*,uint32_t,uint32_t,
 uint32_t,uint64_t,uint64_t,uint64_t*,uint32_t,uint32_t,uint32_t,uint32_t,const uint32_t*,uint32_t,uint32_t,
 MgbfsSegment,uint32_t,uint64_t*,uint32_t*,uint32_t*,uint32_t*,uint64_t*,uint64_t,void*,size_t,void*);
extern "C" int mgbfs_hash_batch_scratch_bytes(uint32_t,uint32_t,uint32_t,size_t*);
extern "C" int mgbfs_hash_batch_accept(uint32_t,uint32_t,uint32_t,const void*,const void*,
 const GenericRouteRecord*,const GenericRouteRecord*,const uint32_t*,const uint32_t*,
 uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,const uint32_t*,uint32_t,uint32_t,
 const MgbfsBatchHashSegment*,uint32_t,uint32_t,uint32_t,MgbfsBatchHashRun*,uint32_t*,uint32_t,
 uint64_t*,uint32_t*,uint32_t*,uint32_t*,uint64_t*,uint64_t,void*,size_t,void*);
#define D(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char*n=nullptr;cuGetErrorName(r,&n);fprintf(stderr,"driver %s line %d\n",n?n:"?",__LINE__);exit(1);}}while(0)
#define C(x) do{cudaError_t r=(x);if(r!=cudaSuccess){fprintf(stderr,"runtime %s line %d\n",cudaGetErrorString(r),__LINE__);exit(1);}}while(0)
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);exit(1);}}while(0)
template<class T>T* alloc(size_t n=1){T*p=nullptr;C(cudaMalloc((void**)&p,n*sizeof(T)));return p;}
template<class T>void put(T*p,T v){C(cudaMemcpy(p,&v,sizeof(T),cudaMemcpyHostToDevice));}
template<class T>T get(T*p){T v{};C(cudaMemcpy(&v,p,sizeof(T),cudaMemcpyDeviceToHost));return v;}
struct Tail{
 CUdeviceptr addr{};uint64_t bytes{};size_t page{};CUmemGenericAllocationHandle handle{};
 Tail(uint64_t n,CUdevice device,size_t granularity):bytes(n),page(granularity){
  CHECK(n%page==0);CUmemAllocationProp prop{};prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=device;
  D(cuMemAddressReserve(&addr,bytes,page,0,0));D(cuMemCreate(&handle,page,&prop,0));
  D(cuMemMap(addr+bytes-page,page,0,handle,0));CUmemAccessDesc access{};
  access.location=prop.location;access.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  D(cuMemSetAccess(addr+bytes-page,page,&access,1));
 }
 ~Tail(){D(cuMemUnmap(addr+bytes-page,page));D(cuMemRelease(handle));D(cuMemAddressFree(addr,bytes));}
};
static void run(int gpu){
 C(cudaSetDevice(gpu));C(cudaFree(nullptr));CUdevice dev;D(cuDeviceGet(&dev,gpu));int supported=0;
 D(cuDeviceGetAttribute(&supported,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,dev));CHECK(supported);
 CUmemAllocationProp prop{};prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=dev;
 size_t page=0;D(cuMemGetAllocationGranularity(&page,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
 const uint64_t cap=uint64_t(1)<<32;Tail states(cap,dev,page),keys(cap*8,dev,page),rows(cap*4,dev,page);
 auto* meta=alloc<GenericRouteRecord>();auto* counts=alloc<uint32_t>();auto* spill=alloc<uint32_t>();
 auto* spills=alloc<uint32_t>();auto* error=alloc<uint32_t>();auto* accepted=alloc<uint64_t>();auto* bits=alloc<uint64_t>();
 GenericRouteRecord rec{0,42,0,0,0,0};C(cudaMemcpy(meta,&rec,sizeof(rec),cudaMemcpyHostToDevice));put(counts,uint32_t(1));
 put(accepted,cap-1);put(bits,uint64_t(0));put(spills,uint32_t(0));put(error,uint32_t(0));
 MgbfsSegment bitmap{nullptr,(void*)states.addr,nullptr,0,cap,3};
 size_t sz=0;C(cudaError_t(mgbfs_bitmap_accept_batch_scratch_bytes(1,&sz)));void* scratch=nullptr;C(cudaMalloc(&scratch,sz));
 auto bitmap_accept=[&](){return cudaError_t(mgbfs_bitmap_accept_batch(1,1,1,nullptr,nullptr,meta,meta,counts,counts,
  0,1,0,2,2,bits,0,1,1,1,nullptr,1,0,bitmap,0,accepted,spill,spills,error,nullptr,0,scratch,sz,nullptr));};
 C(bitmap_accept());C(cudaDeviceSynchronize());CHECK(get(accepted)==cap&&get(bits)==1&&get(error)==0);
 uint8_t value=0;C(cudaMemcpy(&value,(void*)(states.addr+cap-1),1,cudaMemcpyDeviceToHost));CHECK(value==42);
 rec.hash=1;rec.parent=43;C(cudaMemcpy(meta,&rec,sizeof(rec),cudaMemcpyHostToDevice));C(bitmap_accept());C(cudaDeviceSynchronize());
 CHECK(get(accepted)==cap&&get(bits)==1&&get(spills)==1&&get(error)==0);C(cudaFree(scratch));
 // Reinitialize the synthetic boundary for the sorted-index hash backend.
 MgbfsBatchHashSegment hs{(void*)states.addr,(uint64_t*)keys.addr,(uint32_t*)rows.addr,cap,1};
 auto* desc=alloc<MgbfsBatchHashSegment>();C(cudaMemcpy(desc,&hs,sizeof(hs),cudaMemcpyHostToDevice));
 auto* runs=alloc<MgbfsBatchHashRun>(2);auto* nruns=alloc<uint32_t>();put(nruns,uint32_t(0));
 rec.hash=0;rec.parent=42;C(cudaMemcpy(meta,&rec,sizeof(rec),cudaMemcpyHostToDevice));
 put(accepted,cap-1);put(spills,uint32_t(0));put(error,uint32_t(0));size_t hsbytes=0;
 C(cudaError_t(mgbfs_hash_batch_scratch_bytes(1,1,1,&hsbytes)));C(cudaMalloc(&scratch,hsbytes));
 auto hash_accept=[&](){return cudaError_t(mgbfs_hash_batch_accept(1,1,1,nullptr,nullptr,meta,meta,counts,counts,
  0,1,0,1,1,1,nullptr,1,0,desc,1,0,0,runs,nruns,2,accepted,spill,spills,error,nullptr,0,scratch,hsbytes,nullptr));};
 C(hash_accept());C(cudaDeviceSynchronize());CHECK(get(accepted)==cap&&get(nruns)==1&&get(error)==0);
 uint32_t last=0;C(cudaMemcpy(&last,(void*)(rows.addr+(cap-1)*4),4,cudaMemcpyDeviceToHost));CHECK(last==UINT32_MAX);
 MgbfsBatchHashRun observed{};C(cudaMemcpy(&observed,runs,sizeof(observed),cudaMemcpyDeviceToHost));CHECK(observed.begin==cap-1&&observed.length==1);
 rec.hash=1;rec.parent=43;C(cudaMemcpy(meta,&rec,sizeof(rec),cudaMemcpyHostToDevice));
 put(spills,uint32_t(0));put(error,uint32_t(0));C(hash_accept());C(cudaDeviceSynchronize());
 CHECK(get(accepted)==cap&&get(nruns)==1&&get(spills)==1&&get(error)==0);
 C(cudaFree(scratch));C(cudaFree(nruns));C(cudaFree(runs));C(cudaFree(desc));C(cudaFree(meta));C(cudaFree(counts));
 C(cudaFree(spill));C(cudaFree(spills));C(cudaFree(error));C(cudaFree(accepted));C(cudaFree(bits));
 printf("PASS gpu=%d bitmap/hash final u32 row and spill; virtual mappings %zu bytes per plane\n",gpu,page);
}
int main(){int n=0;C(cudaGetDeviceCount(&n));CHECK(n>=2);D(cuInit(0));run(0);run(1);puts("BATCH_BOUNDARY_TWO_GPU_PASS");}
