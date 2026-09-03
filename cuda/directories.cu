#include "directories.h"
#include <cuda_runtime.h>
#include <climits>
namespace {
struct alignas(16) Key{uint32_t w[4];};
__global__ void check_count(const uint32_t*n,uint32_t cap,uint32_t*f){if(*n>cap)*f=30;}
__global__ void directory(const Key*keys,const uint32_t*count,uint32_t buckets,unsigned bits,MgbfsOwnerRange*out,const uint32_t*f){
 if(*f)return;unsigned b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=buckets)return;uint32_t n=*count;
 uint32_t lo=0,hi=n;while(lo<hi){uint32_t m=lo+(hi-lo)/2;uint32_t v=bits?keys[m].w[3]>>(32-bits):0;if(v<b)lo=m+1;else hi=m;}uint32_t begin=lo;
 hi=n;while(lo<hi){uint32_t m=lo+(hi-lo)/2;uint32_t v=bits?keys[m].w[3]>>(32-bits):0;if(v<=b)lo=m+1;else hi=m;}
 out[b]={begin,lo-begin};
}
__global__ void bind(MgbfsBucketJob*jobs,uint32_t n,const uint32_t*counts,uint32_t buckets){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&jobs[i].bucket<buckets)jobs[i].accepted_count=counts[jobs[i].bucket];}
__global__ void prefix(const uint32_t*counts,uint32_t b,uint32_t k,uint32_t cap,MgbfsOwnerRange*dir,uint32_t*n,uint32_t*f){
 if(*f)return;uint64_t sum=0;
 // FinalizeDepth control metadata only. No hash/state scan or old-layer merge.
 for(unsigned i=0;i<b;++i){if(counts[i]>k||sum+counts[i]>cap){*f=31;return;}sum+=counts[i];}
 sum=0;for(unsigned i=0;i<b;++i){dir[i]={sum,counts[i]};sum+=counts[i];}*n=unsigned(sum);
}
__global__ void copy(const Key*in,uint32_t k,Key*out,const MgbfsOwnerRange*dir,const uint32_t*f){
 if(*f)return;auto r=dir[blockIdx.x];for(uint64_t i=threadIdx.x;i<r.count;i+=blockDim.x)out[r.begin+i]=in[uint64_t(blockIdx.x)*k+i];
}
}
extern "C" int mgbfs_bucket_directory(const void*keys,const uint32_t*n,uint32_t cap,uint32_t b,MgbfsOwnerRange*out,uint32_t*f,void*stream){
 if(!keys||!n||!out||!f||!b||(b&(b-1))||b>INT_MAX||cap>INT_MAX)return 1;unsigned bits=0;for(unsigned v=b;v>1;v>>=1)++bits;auto s=static_cast<cudaStream_t>(stream);
 check_count<<<1,1,0,s>>>(n,cap,f);directory<<<(b+255)/256,256,0,s>>>(static_cast<const Key*>(keys),n,b,bits,out,f);return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_bind_owner_jobs(MgbfsBucketJob*jobs,uint32_t n,const uint32_t*counts,uint32_t b,void*stream){
 if(!jobs||!counts||!n||n>INT_MAX||!b)return 1;bind<<<(n+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(jobs,n,counts,b);return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_compact_hash_layer(const void*in,const uint32_t*counts,uint32_t b,uint32_t k,void*out,uint32_t cap,MgbfsOwnerRange*dir,uint32_t*n,uint32_t*f,void*stream){
 if(!in||!counts||!out||!dir||!n||!f||!b||b>INT_MAX||!k||cap>INT_MAX)return 1;auto s=static_cast<cudaStream_t>(stream);
 prefix<<<1,1,0,s>>>(counts,b,k,cap,dir,n,f);copy<<<b,256,0,s>>>(static_cast<const Key*>(in),k,static_cast<Key*>(out),dir,f);return cudaGetLastError()==cudaSuccess?0:2;
}
