// Virtual-memory address test only: maps one physical page at the last row.
#include "device_pack4_accept.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
using namespace mgbfs_device_pack4;
#define C(x) do{auto s=(x);if(s!=cudaSuccess){std::fprintf(stderr,"%s line %d: %s\n",#x,__LINE__,cudaGetErrorString(s));std::exit(1);}}while(0)
#define D(x) do{auto s=(x);if(s!=CUDA_SUCCESS){const char* n=nullptr;cuGetErrorName(s,&n);std::fprintf(stderr,"%s line %d: %s\n",#x,__LINE__,n?n:"?");std::exit(1);}}while(0)
#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"CHECK %s line %d\n",#x,__LINE__);std::exit(1);}}while(0)
template<class T>T* managed(size_t n){T* p=nullptr;C(cudaMallocManaged(&p,n*sizeof(T)));return p;}
__global__ void insert(Owner owner,uint64_t id,uint64_t packed,Outcome* out) {
 if(!blockIdx.x&&!threadIdx.x)*out=accept(owner,id,packed);
}
int main(){
 int device_count=0;auto status=cudaGetDeviceCount(&device_count);
 if(status==cudaErrorNoDevice||(status==cudaSuccess&&!device_count))return 77;
 C(status);C(cudaSetDevice(0));C(cudaFree(nullptr));D(cuInit(0));
 CUdevice dev;D(cuDeviceGet(&dev,0));int supported=0;
 D(cuDeviceGetAttribute(&supported,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,dev));
 if(!supported){std::puts("SKIP VMM unsupported");return 77;}
 CUmemAllocationProp prop{};prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
 prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=dev;
 size_t page=0;D(cuMemGetAllocationGranularity(&page,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
 const uint64_t capacity=1ull<<32,bytes=capacity*sizeof(uint64_t);
 if(bytes%page){std::puts("SKIP incompatible VMM granularity");return 77;}
 CUdeviceptr addr=0;D(cuMemAddressReserve(&addr,bytes,page,0,0));
 CUmemGenericAllocationHandle handle;D(cuMemCreate(&handle,page,&prop,0));
 D(cuMemMap(addr+bytes-page,page,0,handle,0));
 CUmemAccessDesc access{};access.location=prop.location;access.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
 D(cuMemSetAccess(addr+bytes-page,page,&access,1));
 auto* symbols=managed<int64_t>(2);symbols[0]=-3;symbols[1]=9;
 auto* counts=managed<uint32_t>(2);counts[0]=1;counts[1]=1;
 auto* bits=managed<uint64_t>(1);*bits=0;
 auto* used0=managed<uint64_t>(1);*used0=capacity-1;
 auto* used1=managed<uint64_t>(1);*used1=0;
 auto* small=managed<uint64_t>(1);*small=0;
 auto* extents=managed<Extent>(2);
 extents[0]={reinterpret_cast<uint64_t*>(addr),used0,capacity,1,0};
 extents[1]={small,used1,1,1,0};
 Owner owner{};owner.domain={symbols,counts,2,2,2};
 owner.dictionary.elements=2;owner.dictionary.distinct=2;
 owner.dictionary.symbols[0]=-3;owner.dictionary.symbols[1]=9;
 owner.extents=extents;owner.extent_count=1;owner.bits=bits;owner.owner_len=2;
 owner.world=1;owner.owner=0;owner.shards=1;owner.shard=0;owner.bank=0;
 auto* out=managed<Outcome>(1);
 insert<<<1,1>>>(owner,0,0x10,out);C(cudaGetLastError());C(cudaDeviceSynchronize());
 CHECK(out->status==Status::accepted && out->row==UINT32_MAX && out->extent==0);
 CHECK(*used0==capacity && *bits==1);
 uint64_t last=0;C(cudaMemcpy(&last,reinterpret_cast<void*>(addr+bytes-8),8,cudaMemcpyDeviceToHost));CHECK(last==0x10);
 insert<<<1,1>>>(owner,1,0x01,out);C(cudaGetLastError());C(cudaDeviceSynchronize());
 CHECK(out->status==Status::extent_full && *used0==capacity && *bits==1);
 owner.extent_count=2;
 insert<<<1,1>>>(owner,1,0x01,out);C(cudaGetLastError());C(cudaDeviceSynchronize());
 CHECK(out->status==Status::accepted && out->extent==1 && out->row==0);
 CHECK(*used1==1 && *small==0x01 && *bits==3);
 insert<<<1,1>>>(owner,0,0x10,out);C(cudaGetLastError());C(cudaDeviceSynchronize());
 CHECK(out->status==Status::duplicate && *bits==3);
 D(cuMemUnmap(addr+bytes-page,page));D(cuMemRelease(handle));D(cuMemAddressFree(addr,bytes));
 std::printf("PASS last u32 row and next segment; virtual %llu bytes, physically mapped %zu bytes\n",
             static_cast<unsigned long long>(bytes),page);
}
