#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "generic_sorted_run_pool.cuh"
#define CUDA_CHECK(x) do { auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);} }while(0)
__global__ void lease_contract(GenericSortedRunPool pool,uint32_t* failures,uint32_t* errors) {
 if(blockIdx.x||threadIdx.x)return;
 GenericSortedRunToken first{},second{},blocked{};__shared__ uint32_t e; e=0;
 if(generic_sorted_run_allocate(pool,6,0,&first,&e)!=SORTED_RUN_ACQUIRED)++*failures;
 if(generic_sorted_run_read_acquire(pool,first,&e))++*failures;
 if(!generic_sorted_run_publish(pool,first,64*pool.page_entries,&e))++*failures;
 if(!generic_sorted_run_read_acquire(pool,first,&e))++*failures;
 if(!generic_sorted_run_release(pool,first,true,&e))++*failures;
 if(generic_sorted_run_read_acquire(pool,first,&e))++*failures;
 if(generic_sorted_run_allocate(pool,6,0,&blocked,&e)!=SORTED_RUN_PRESSURE)++*failures;
 if(!generic_sorted_run_release(pool,first,false,&e))++*failures;
 if(generic_sorted_run_allocate(pool,6,0,&second,&e)!=SORTED_RUN_ACQUIRED)++*failures;
 if(second.slot!=first.slot||second.generation!=first.generation+1)++*failures;
 // Stale publication cannot touch the new generation's count.
 __shared__ uint32_t expected_error; expected_error=0;
 if(generic_sorted_run_publish(pool,first,123,&expected_error)||expected_error!=512u)++*failures;
 if(pool.descriptors[second.slot].count!=0)++*failures;
 expected_error=0;
 if(generic_sorted_run_release(pool,first,true,&expected_error)||expected_error!=512u)++*failures;
 expected_error=0;
 if(generic_sorted_run_publish(pool,second,64*pool.page_entries+1,&expected_error)||expected_error!=32u)++*failures;
 // Invalid publication returns the sole writer credit; it can abort cleanly.
 if(!generic_sorted_run_release(pool,second,true,&e))++*failures;
 if(*pool.occupied)++*failures;
 expected_error=0;
 if(generic_sorted_run_allocate(pool,7,0,&blocked,&expected_error)!=SORTED_RUN_INVALID||expected_error!=32u)++*failures;
 *errors=e;
}
__global__ void fanout_prepare(GenericSortedRunPool pool,GenericSortedRunToken* token,uint32_t* errors) {
 if(blockIdx.x||threadIdx.x)return;
 if(generic_sorted_run_allocate(pool,6,0,token,errors)==SORTED_RUN_ACQUIRED)
  generic_sorted_run_publish(pool,*token,pool.page_entries*64,errors);
 else atomicOr(errors,1024u);
}
__global__ void fanout_acquire(GenericSortedRunPool pool,const GenericSortedRunToken* token,uint32_t* errors) {
 if(!generic_sorted_run_read_acquire(pool,*token,errors))atomicOr(errors,1024u);
}
__global__ void fanout_retire(GenericSortedRunPool pool,const GenericSortedRunToken* token,uint32_t* errors) {
 if(blockIdx.x||threadIdx.x)return;
 generic_sorted_run_release(pool,*token,true,errors);
 GenericSortedRunToken blocked{};
 if(generic_sorted_run_read_acquire(pool,*token,errors)||
  generic_sorted_run_allocate(pool,6,0,&blocked,errors)!=SORTED_RUN_PRESSURE)atomicOr(errors,1024u);
}
__global__ void fanout_release(GenericSortedRunPool pool,const GenericSortedRunToken* token,uint32_t* errors) {
 if(!generic_sorted_run_release(pool,*token,false,errors))atomicOr(errors,1024u);
}
__global__ void concurrent_allocate(GenericSortedRunPool pool,GenericSortedRunToken* tokens,
 uint32_t* classes,uint32_t* status,uint32_t* errors,uint32_t n) {
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 uint32_t cls=i%7;classes[i]=cls;
 status[i]=generic_sorted_run_allocate(pool,cls,0,tokens+i,errors);
 if(status[i]==SORTED_RUN_ACQUIRED)generic_sorted_run_publish(pool,tokens[i],pool.page_entries*(1u<<cls),errors);
}
__global__ void concurrent_retire(GenericSortedRunPool pool,GenericSortedRunToken* tokens,
 const uint32_t* status,uint32_t* errors,uint32_t n) {
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&status[i]==SORTED_RUN_ACQUIRED)
  generic_sorted_run_release(pool,tokens[i],true,errors);
}
static void shape_oracle() {
 for(uint32_t regions: {1u,2u,16u,32768u})for(uint32_t page: {1u,4096u,65536u}) {
  GenericSortedRunPoolShape shape{};uint64_t entries=uint64_t(regions)*64*page;
  bool okay=generic_sorted_run_pool_shape(regions,page,&shape);
  if(okay!=(entries<=0xffffffffull))exit(6);
  if(okay && (shape.entries!=entries || shape.total_bytes!=uint64_t(regions)*8+
    uint64_t(regions)*64*sizeof(GenericSortedRunDescriptor)+entries*12))exit(6);
 }
 GenericSortedRunPoolShape shape{};
 if(generic_sorted_run_pool_shape(0,1,&shape)||generic_sorted_run_pool_shape(1,0,&shape)||
  generic_sorted_run_pool_shape(0xffffffffu,1,&shape)||generic_sorted_run_pool_shape(1,0xffffffffu,&shape))exit(6);
 puts("SORTED_RUN_POOL_SHAPE_ORACLE_PASS");
}
static void run_device(int device) {
 CUDA_CHECK(cudaSetDevice(device));constexpr uint32_t regions=16,n=512,page=4096;
 unsigned long long* occupied;GenericSortedRunDescriptor* descriptors;
 GenericSortedRunToken* tokens;uint32_t *classes,*status,*errors,*failures;
 CUDA_CHECK(cudaMalloc(&occupied,regions*sizeof(*occupied)));
 CUDA_CHECK(cudaMalloc(&descriptors,regions*64*sizeof(*descriptors)));
 CUDA_CHECK(cudaMalloc(&tokens,n*sizeof(*tokens)));CUDA_CHECK(cudaMalloc(&classes,n*sizeof(*classes)));
 CUDA_CHECK(cudaMalloc(&status,n*sizeof(*status)));CUDA_CHECK(cudaMalloc(&errors,sizeof(*errors)));
 CUDA_CHECK(cudaMalloc(&failures,sizeof(*failures)));
 CUDA_CHECK(cudaMemset(occupied,0,regions*sizeof(*occupied)));
 CUDA_CHECK(cudaMemset(descriptors,0,regions*64*sizeof(*descriptors)));
 CUDA_CHECK(cudaMemset(errors,0,sizeof(*errors)));CUDA_CHECK(cudaMemset(failures,0,sizeof(*failures)));
 GenericSortedRunPool pool{occupied,descriptors,1,page};
 lease_contract<<<1,1>>>(pool,failures,errors);CUDA_CHECK(cudaDeviceSynchronize());
 uint32_t failed=0,error=0;CUDA_CHECK(cudaMemcpy(&failed,failures,sizeof(failed),cudaMemcpyDeviceToHost));
 CUDA_CHECK(cudaMemcpy(&error,errors,sizeof(error),cudaMemcpyDeviceToHost));
 if(failed||error){fprintf(stderr,"lease contract failure %u error %u\n",failed,error);exit(3);}
 fanout_prepare<<<1,1>>>(pool,tokens,errors);
 fanout_acquire<<<8,64>>>(pool,tokens,errors);
 fanout_retire<<<1,1>>>(pool,tokens,errors);
 fanout_release<<<8,64>>>(pool,tokens,errors);
 CUDA_CHECK(cudaDeviceSynchronize());
 unsigned long long remaining=1;
 CUDA_CHECK(cudaMemcpy(&remaining,occupied,sizeof(remaining),cudaMemcpyDeviceToHost));
 CUDA_CHECK(cudaMemcpy(&error,errors,sizeof(error),cudaMemcpyDeviceToHost));
 if(remaining||error){fprintf(stderr,"fanout readers retirement failed %llu %u\n",remaining,error);exit(7);}
 printf("SORTED_RUN_POOL_FANOUT_PASS device=%d readers=512\n",device);
 pool.regions=regions;
 std::vector<GenericSortedRunToken> ht(n);std::vector<uint32_t> hc(n),hs(n);
 std::vector<unsigned long long> masks(regions),actual(regions);
 for(uint32_t round=0;round<40;++round) {
  concurrent_allocate<<<8,64>>>(pool,tokens,classes,status,errors,n);CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(ht.data(),tokens,n*sizeof(*tokens),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hc.data(),classes,n*sizeof(*classes),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hs.data(),status,n*sizeof(*status),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(actual.data(),occupied,regions*sizeof(*occupied),cudaMemcpyDeviceToHost));
  std::fill(masks.begin(),masks.end(),0ull);uint32_t accepted=0;
  for(uint32_t i=0;i<n;++i) {
   if(hs[i]==SORTED_RUN_PRESSURE)continue;
   if(hs[i]!=SORTED_RUN_ACQUIRED){fprintf(stderr,"invalid allocation result\n");exit(4);}
   uint32_t region=ht[i].slot/64,start=ht[i].slot%64,pages=1u<<hc[i];
   if(region>=regions||start%pages||!ht[i].generation){fprintf(stderr,"bad geometry\n");exit(4);}
   auto mask=pages==64?~0ull:((1ull<<pages)-1ull)<<start;
   if(masks[region]&mask){fprintf(stderr,"overlapping live allocation\n");exit(4);}
   masks[region]|=mask;++accepted;
  }
  if(!accepted||masks!=actual){fprintf(stderr,"bitmap oracle mismatch\n");exit(4);}
  concurrent_retire<<<8,64>>>(pool,tokens,status,errors,n);CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(actual.data(),occupied,regions*sizeof(*occupied),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&error,errors,sizeof(error),cudaMemcpyDeviceToHost));
  for(auto value:actual)if(value){fprintf(stderr,"pool leak\n");exit(4);}
  if(error){fprintf(stderr,"unexpected pool error %u\n",error);exit(4);}
 }
 CUDA_CHECK(cudaFree(occupied));CUDA_CHECK(cudaFree(descriptors));CUDA_CHECK(cudaFree(tokens));
 CUDA_CHECK(cudaFree(classes));CUDA_CHECK(cudaFree(status));CUDA_CHECK(cudaFree(errors));CUDA_CHECK(cudaFree(failures));
 printf("SORTED_RUN_POOL_PASS device=%d rounds=40 concurrent_requests=512 shared_regions=16\n",device);
}
int main(){shape_oracle();int devices=0;CUDA_CHECK(cudaGetDeviceCount(&devices));if(devices<2)return 5;for(int i=0;i<2;++i)run_device(i);}
