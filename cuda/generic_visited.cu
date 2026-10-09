#include "generic_visited.h"
#include <cuda_runtime.h>
#include <cstdint>

static constexpr uint64_t EMPTY=~uint64_t(0);
__device__ uint64_t mix64(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
__device__ uint64_t finish_hash(uint64_t h,uint32_t bits){h=mix64(h);return bits==64?h:(bits? h&((uint64_t(1)<<bits)-1):0);}
struct Action {
 uint32_t kind,elements,n,m,generators,count,stride;
 const int64_t* parents;const uint32_t* permutations;const int64_t* matrices;const uint32_t* moduli;
 __device__ int64_t value(uint32_t child,uint32_t element)const {
  const uint32_t parent=child/generators,g=child%generators;
  if(kind==0)return parents[uint64_t(permutations[uint64_t(g)*elements+element])*stride+parent];
  const uint32_t row=element/m,col=element%m;const int64_t* matrix=matrices+uint64_t(g)*n*n;
  uint64_t sum=0;for(uint32_t k=0;k<n;k++)sum+=uint64_t(matrix[uint64_t(row)*n+k])*uint64_t(parents[uint64_t(k*m+col)*stride+parent]);
  int64_t v=static_cast<int64_t>(sum);uint32_t mod=moduli[g];if(mod){v%=int64_t(mod);if(v<0)v+=mod;}return v;
 }
 __device__ uint64_t hash(uint32_t child,uint64_t seed,uint32_t bits)const {
  uint64_t h=seed;for(uint32_t e=0;e<elements;e++)h=mix64(h^uint64_t(value(child,e))^uint64_t(e));return finish_hash(h,bits);
 }
};
__global__ void seed_table(uint32_t elements,const int64_t* states,uint32_t stride,uint32_t count,uint64_t* slots,
 uint32_t capacity,uint64_t seed,uint32_t bits,uint32_t* error){
 for(uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;row<count;row+=blockDim.x*gridDim.x){
  uint64_t h=seed;for(uint32_t e=0;e<elements;e++)h=mix64(h^uint64_t(states[uint64_t(e)*stride+row])^uint64_t(e));h=finish_hash(h,bits);
  uint64_t token=(h&0xffffffff00000000ULL)|row;uint32_t p=h&(capacity-1);bool done=false;
  for(uint32_t i=0;i<capacity;i++,p=(p+1)&(capacity-1))if(atomicCAS(reinterpret_cast<unsigned long long*>(slots+p),EMPTY,token)==EMPTY){done=true;break;}
  if(!done)atomicOr(error,1u);
 }
}
__global__ void expand(Action action,uint64_t* slots,uint32_t slot_capacity,int64_t* visited,uint32_t visited_capacity,
 uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,uint32_t* future_count,uint64_t seed,uint32_t bits,uint32_t* error){
 const uint32_t children=action.count*action.generators;
 for(uint32_t child=blockIdx.x*blockDim.x+threadIdx.x;child<children;child+=blockDim.x*gridDim.x){
  if(atomicAdd(error,0u))return;
  const uint64_t hash=action.hash(child,seed,bits),prefix=hash&0xffffffff00000000ULL;
  const uint64_t pending=prefix|0x80000000ULL|child;uint32_t slot=hash&(slot_capacity-1);bool done=false;
  for(uint32_t probe=0;probe<slot_capacity;probe++,slot=(slot+1)&(slot_capacity-1)){
   uint64_t observed=atomicCAS(reinterpret_cast<unsigned long long*>(slots+slot),EMPTY,pending);
   if(observed==EMPTY){
    // Publish the origin atomically before any payload allocation. Peers can
    // compare that immutable origin by regeneration, without spinning on a
    // writer or relying on an incompletely written state.
    const uint32_t row=atomicAdd(visited_count,1u);
    if(row>=visited_capacity){atomicOr(error,2u);done=true;break;}
    for(uint32_t e=0;e<action.elements;e++)visited[uint64_t(e)*visited_capacity+row]=action.value(child,e);
    __threadfence();atomicExch(reinterpret_cast<unsigned long long*>(slots+slot),prefix|row);
    const uint32_t position=atomicAdd(future_count,1u);
    if(position>=future_capacity){atomicOr(error,4u);done=true;break;}
    future[position]=row;done=true;break;
   }
   if((observed&0xffffffff00000000ULL)!=prefix)continue;
   const uint32_t ref=uint32_t(observed),other=ref&0x7fffffffU;const bool transient=ref&0x80000000U;
   if((transient && other>=children)||(!transient && other>=visited_capacity)){atomicOr(error,8u);done=true;break;}
   bool equal=true;
   for(uint32_t e=0;e<action.elements;e++){
    const int64_t old=transient?action.value(other,e):visited[uint64_t(e)*visited_capacity+other];
    if(old!=action.value(child,e)){equal=false;break;}
   }
   if(equal){done=true;break;}
  }
  if(!done)atomicOr(error,1u);
 }
}
__global__ void gather(uint32_t elements,const int64_t* source,uint32_t source_stride,const uint32_t* indices,
 uint32_t count,int64_t* output,uint32_t output_stride){
 for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<uint64_t(elements)*count;i+=uint64_t(blockDim.x)*gridDim.x){
  uint32_t e=i/count,row=i%count;output[uint64_t(e)*output_stride+row]=source[uint64_t(e)*source_stride+indices[row]];
 }
}
static uint32_t grid(uint64_t work){uint64_t n=(work+255)/256;return uint32_t(n>65535?65535:n);}
extern "C" int mgbfs_generic_seed_i64(uint32_t elements,const int64_t* states,uint32_t stride,uint32_t count,uint64_t* slots,
 uint32_t capacity,uint64_t seed,uint32_t bits,uint32_t* error,void* stream){
 if(!elements||count>stride||!capacity||(capacity&(capacity-1))||bits>64||!error||!slots||(!states&&count)||count>=0x80000000U)return int(cudaErrorInvalidValue);
 if(!count)return 0;seed_table<<<grid(count),256,0,static_cast<cudaStream_t>(stream)>>>(elements,states,stride,count,slots,capacity,seed,bits,error);return int(cudaGetLastError());
}
extern "C" int mgbfs_generic_expand_i64(uint32_t kind,uint32_t elements,uint32_t n,uint32_t m,uint32_t generators,
 const int64_t* parents,uint32_t count,uint32_t stride,const uint32_t* permutations,const int64_t* matrices,const uint32_t* moduli,
 uint64_t* slots,uint32_t slot_capacity,int64_t* visited,uint32_t visited_capacity,uint32_t* visited_count,
 uint32_t* future,uint32_t future_capacity,uint32_t* future_count,uint64_t seed,uint32_t bits,uint32_t* error,void* stream){
 if(kind>1||!elements||!n||!m||!generators||count>stride||uint64_t(count)*generators>=0x7fffffffULL||
  !slot_capacity||(slot_capacity&(slot_capacity-1))||!visited_capacity||visited_capacity>=0x80000000U||bits>64||
  !visited_count||!future_count||!error||!slots||!visited||!future||!parents||
  (kind==0&&(!permutations||elements!=n||m!=1))||(kind==1&&(!matrices||!moduli||uint64_t(n)*m!=elements)))return int(cudaErrorInvalidValue);
 if(!count)return 0;
 Action action{kind,elements,n,m,generators,count,stride,parents,permutations,matrices,moduli};
 expand<<<grid(uint64_t(count)*generators),256,0,static_cast<cudaStream_t>(stream)>>>(action,slots,slot_capacity,visited,visited_capacity,visited_count,future,future_capacity,future_count,seed,bits,error);
 return int(cudaGetLastError());
}
extern "C" int mgbfs_generic_gather_i64(uint32_t elements,const int64_t* source,uint32_t stride,const uint32_t* indices,
 uint32_t count,int64_t* output,uint32_t output_stride,void* stream){
 if(!elements||count>output_stride||(!source&&count)||(!indices&&count)||(!output&&count))return int(cudaErrorInvalidValue);
 if(!count)return 0;gather<<<grid(uint64_t(elements)*count),256,0,static_cast<cudaStream_t>(stream)>>>(elements,source,stride,indices,count,output,output_stride);return int(cudaGetLastError());
}
