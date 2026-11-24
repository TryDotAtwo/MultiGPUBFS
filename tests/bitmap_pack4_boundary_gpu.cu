// Virtual-address boundary test, not a physical allocation of 2^32 rows.
#include "../cuda/bitmap_pack4_owner.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#define C(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
#define D(x) do{CUresult e=(x);if(e!=CUDA_SUCCESS){const char* name=nullptr;cuGetErrorName(e,&name);fprintf(stderr,"DRIVER %s line %d\n",name?name:"?",__LINE__);exit(1);}}while(0)
#define R(x) do{if(!(x)){fprintf(stderr,"FAIL line %d %s\n",__LINE__,#x);exit(1);}}while(0)
template<class T>T* alloc(size_t n){T* p=nullptr;C(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(){
 int devices=0;C(cudaGetDeviceCount(&devices));R(devices>=2);D(cuInit(0));
 for(int ordinal=0;ordinal<2;ordinal++){
  C(cudaSetDevice(ordinal));C(cudaFree(nullptr));CUdevice dev;D(cuDeviceGet(&dev,ordinal));int supported=0;
  D(cuDeviceGetAttribute(&supported,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,dev));
  if(!supported){printf("SKIP VMM gpu=%d\n",ordinal);return 77;}
  CUmemAllocationProp prop{};prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=dev;
  size_t page=0;D(cuMemGetAllocationGranularity(&page,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  constexpr uint64_t cap=1ull<<32,bank_bytes=cap*8,total_bytes=2*bank_bytes;
  R(bank_bytes%page==0);
  CUdeviceptr addr=0;D(cuMemAddressReserve(&addr,total_bytes,page,0,0));
  CUmemGenericAllocationHandle h;D(cuMemCreate(&h,page,&prop,0));
  D(cuMemMap(addr+bank_bytes-page,page,0,h,0));
  CUmemAccessDesc access{};access.location=prop.location;access.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  D(cuMemSetAccess(addr+bank_bytes-page,page,&access,1));
  auto *meta=alloc<GenericRouteRecord>(1);auto *counts=alloc<uint32_t>(1),*spill=alloc<uint32_t>(1),*spills=alloc<uint32_t>(1),*err=alloc<uint32_t>(1);
  auto *accepted=alloc<uint64_t>(1),*bits=alloc<uint64_t>(1);
  GenericRouteRecord rec{0,42,0,0,0,0};uint32_t one=1;uint64_t initial=cap-1;
  C(cudaMemcpy(meta,&rec,sizeof(rec),cudaMemcpyHostToDevice));C(cudaMemcpy(counts,&one,4,cudaMemcpyHostToDevice));
  C(cudaMemcpy(accepted,&initial,8,cudaMemcpyHostToDevice));C(cudaMemset(bits,0,8));C(cudaMemset(spills,0,4));C(cudaMemset(err,0,4));
  MgbfsPack4OwnerArgs a{};a.dictionary.elements=1;a.dictionary.distinct=1;a.dictionary.symbols[0]=42;
  a.elements=1;a.state_bytes=1;a.packed=1;a.local_meta=meta;a.remote_meta=meta;a.local_counts=counts;a.remote_counts=counts;
  a.world=1;a.owner=0;a.total=2;a.owner_len=2;a.bits=bits;a.shards=1;a.q=1;a.visible=1;a.selected=1;
  a.capacity=cap;a.banks=2;a.rows=reinterpret_cast<uint64_t*>(addr);a.accepted=accepted;a.spill=spill;a.spill_count=spills;a.error=err;
  C((cudaError_t)mgbfs_pack4_owner_accept(a,nullptr));C(cudaDeviceSynchronize());
  uint64_t cnt=0,mask=0,value=UINT64_MAX;uint32_t error=0;
  C(cudaMemcpy(&cnt,accepted,8,cudaMemcpyDeviceToHost));C(cudaMemcpy(&mask,bits,8,cudaMemcpyDeviceToHost));
  C(cudaMemcpy(&error,err,4,cudaMemcpyDeviceToHost));C(cudaMemcpy(&value,reinterpret_cast<void*>(addr+bank_bytes-8),8,cudaMemcpyDeviceToHost));
  R(cnt==cap&&mask==1&&!error&&value==0);
  rec.hash=1;C(cudaMemcpy(meta,&rec,sizeof(rec),cudaMemcpyHostToDevice));
  C((cudaError_t)mgbfs_pack4_owner_accept(a,nullptr));C(cudaDeviceSynchronize());
  uint32_t nspill=0;C(cudaMemcpy(&cnt,accepted,8,cudaMemcpyDeviceToHost));C(cudaMemcpy(&mask,bits,8,cudaMemcpyDeviceToHost));
  C(cudaMemcpy(&error,err,4,cudaMemcpyDeviceToHost));C(cudaMemcpy(&nspill,spills,4,cudaMemcpyDeviceToHost));
  R(cnt==cap&&mask==1&&!error&&nspill==1);
  C(cudaFree(meta));C(cudaFree(counts));C(cudaFree(spill));C(cudaFree(spills));C(cudaFree(err));C(cudaFree(accepted));C(cudaFree(bits));
  D(cuMemUnmap(addr+bank_bytes-page,page));D(cuMemRelease(h));D(cuMemAddressFree(addr,total_bytes));
  printf("PACK4_BOUNDARY_PASS gpu=%d row=2^32-1 spill_next=1 physical_page=%zu bytes\n",ordinal,page);
 }
}
