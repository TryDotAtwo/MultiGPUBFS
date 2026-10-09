#include "shard_ab_job.h"
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cuda/std/tuple>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <memory>
#include <vector>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <stdexcept>
namespace {
struct alignas(16) Key { uint32_t w[4];
 __host__ __device__ bool operator==(Key const& b)const{return w[0]==b.w[0]&&w[1]==b.w[1]&&w[2]==b.w[2]&&w[3]==b.w[3];}
};
struct Less {
 __host__ __device__ bool operator()(Key a,Key b) const {
  for(int i=3;i>=0;--i)if(a.w[i]!=b.w[i])return a.w[i]<b.w[i];return false;
 }
};
struct Decompose {
 __host__ __device__ auto operator()(Key& key) const {return cuda::std::tie(key.w[3],key.w[2],key.w[1],key.w[0]);}
};
struct Equal {
 __host__ __device__ bool operator()(Key a,Key b) const {
  return a.w[0]==b.w[0]&&a.w[1]==b.w[1]&&a.w[2]==b.w[2]&&a.w[3]==b.w[3];
 }
};
struct Min {
 __host__ __device__ uint32_t operator()(uint32_t a,uint32_t b)const{return a<b?a:b;}
};
void check(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}

struct Binding {
 Key* keys;uint8_t* states;uint32_t* count;
 const Key* previous;uint32_t pn;const Key* current;uint32_t cn;
 uint32_t* fatal;const uint32_t* added_ranges;uint32_t ready_threshold;
};
uint32_t table_capacity(uint32_t n){uint32_t c=1;while(c<uint64_t(n)*2)c*=2;return c;}
struct CachedGraph { bool hash=false;uint32_t bound;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr; };

struct Job {
 uint32_t* hash_table=nullptr;uint32_t hash_capacity=0;
 Binding* binding=nullptr;Key* radix_keys=nullptr;uint32_t* radix_indices=nullptr;std::vector<CachedGraph> graphs;
 uint32_t capacity,stride;std::vector<uint32_t> supported_bounds;Key *sort_keys=nullptr,*unique_keys=nullptr;
 uint32_t *sort_indices=nullptr,*unique_indices=nullptr,*selected=nullptr,*runs=nullptr,*selected_count=nullptr;
 uint8_t *flags=nullptr,*state_scratch=nullptr;void* cub=nullptr;size_t cub_bytes=0;
 ~Job(){for(auto&g:graphs){cudaGraphExecDestroy(g.executable);cudaGraphDestroy(g.graph);}cudaFree(hash_table);cudaFree(binding);cudaFree(radix_keys);cudaFree(radix_indices);cudaFree(cub);cudaFree(state_scratch);cudaFree(flags);cudaFree(selected_count);
  cudaFree(runs);cudaFree(selected);cudaFree(unique_indices);cudaFree(sort_indices);cudaFree(unique_keys);cudaFree(sort_keys);}
};
size_t temp_bytes(uint32_t n){
 size_t a=0,b=0,c=0;
 check(cub::DeviceRadixSort::SortPairs(nullptr,a,(Key*)nullptr,(Key*)nullptr,(uint32_t*)nullptr,(uint32_t*)nullptr,int(n),Decompose{},0,128));
 check(cub::DeviceReduce::ReduceByKey(nullptr,b,(Key*)nullptr,(Key*)nullptr,(uint32_t*)nullptr,(uint32_t*)nullptr,
  (uint32_t*)nullptr,Min{},int(n)));
 check(cub::DeviceSelect::Flagged(nullptr,c,thrust::counting_iterator<uint32_t>{0},
  (uint8_t*)nullptr,(uint32_t*)nullptr,(uint32_t*)nullptr,int(n)));
size_t d=0;check(cub::DeviceSelect::Flagged(nullptr,d,thrust::counting_iterator<uint32_t>{0},(uint8_t*)nullptr,(uint32_t*)nullptr,(uint32_t*)nullptr,int(table_capacity(n))));
 return std::max(std::max(a,b),std::max(c,d));
}
size_t max_temp_bytes(uint32_t n){
 size_t bytes=temp_bytes(n);
 for(uint32_t bound=1;bound<n;bound*=2){bytes=std::max(bytes,temp_bytes(bound));if(bound>UINT32_MAX/2)break;}
 return bytes;
}
__device__ bool contains(const Key* a,uint32_t n,Key key){
 uint32_t lo=0,hi=n;while(lo<hi){uint32_t m=lo+(hi-lo)/2;if(Less{}(a[m],key))lo=m+1;else hi=m;}
 return lo<n&&Equal{}(a[lo],key);
}
__global__ void prepare(const Key* in,const uint32_t* count,uint32_t capacity,Key* out,uint32_t* indices,uint32_t* fatal){
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=capacity)return;
 if(*count>capacity){atomicCAS(fatal,0u,101u);out[i]={{UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX}};indices[i]=UINT32_MAX;return;}
 if(i<*count){
  Key k=in[i];
  // Padding is identified by its invalid row reference, not key value.
  out[i]=k;indices[i]=i;
 }else{out[i]={{UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX}};indices[i]=UINT32_MAX;}
}
__global__ void mark(const Key* keys,const uint32_t* indices,const uint32_t* runs,uint32_t capacity,
 const Key* previous,uint32_t pn,const Key* current,uint32_t cn,uint8_t* flags,const uint32_t* fatal){
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=capacity)return;
 bool keep=!*fatal&&i<*runs&&indices[i]!=UINT32_MAX;
 if(keep)keep=!contains(previous,pn,keys[i])&&!contains(current,cn,keys[i]);
 flags[i]=keep;
}
__global__ void gather(const Key* keys,const uint32_t* refs,const uint32_t* selected,
 const uint32_t* count,const uint8_t* input,uint32_t chunks,uint32_t capacity,Key* out,uint8_t* states,const uint32_t* fatal){
 uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(*fatal||i>=uint64_t(capacity)*chunks)return;
 uint32_t row=i/chunks,part=i%chunks;if(row>=*count)return;
 uint32_t at=selected[row],source=refs[at];
 reinterpret_cast<uint4*>(states)[i]=reinterpret_cast<const uint4*>(input)[uint64_t(source)*chunks+part];
 if(part==0)out[row]=keys[at];
}
__global__ void publish(const Key* keys,const uint8_t* states,const uint32_t* n,uint32_t capacity,
 uint32_t chunks,Key* out,uint8_t* destination,uint32_t* count,const uint32_t* fatal){
 uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(*fatal||i>=uint64_t(capacity)*chunks)return;
 uint32_t row=i/chunks,part=i%chunks;
 if(row<*n){reinterpret_cast<uint4*>(destination)[i]=reinterpret_cast<const uint4*>(states)[i];if(part==0)out[row]=keys[row];}
 // Publish count in a separate ordered kernel, not here.
}
__global__ void publish_count(const uint32_t* n,uint32_t* count,const uint32_t* fatal){if(!*fatal)*count=*n;}

__global__ void bind_job(Binding* dst,Binding value){*dst=value;}
// No conditional-node kernel: the captured jobs use ordinary ordered kernels.
__global__ void graph_prepare(const Binding* b,uint32_t capacity,Key* out,uint32_t* indices){
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=capacity)return;
 if(*b->count>capacity){atomicCAS(b->fatal,0u,101u);out[i]={{UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX}};indices[i]=UINT32_MAX;return;}
 if(i<*b->count){Key k=b->keys[i];out[i]=k;indices[i]=i;
 }else{out[i]={{UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX}};indices[i]=UINT32_MAX;}
}
__global__ void graph_mark(const Binding* b,const Key* keys,const uint32_t* indices,const uint32_t* runs,uint32_t capacity,uint8_t* flags){
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=capacity)return;
 bool keep=!*b->fatal&&i<*runs&&indices[i]!=UINT32_MAX;
 if(keep)keep=!contains(b->previous,b->pn,keys[i])&&!contains(b->current,b->cn,keys[i]);flags[i]=keep;
}
__global__ void graph_gather(const Binding* b,const Key* keys,const uint32_t* refs,const uint32_t* selected,
 const uint32_t* count,uint32_t chunks,uint32_t capacity,Key* out,uint8_t* states){
 uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(*b->fatal||i>=uint64_t(capacity)*chunks)return;
 uint32_t row=i/chunks,part=i%chunks;if(row>=*count)return;
 uint32_t at=selected[row],source=refs[at];
 reinterpret_cast<uint4*>(states)[i]=reinterpret_cast<const uint4*>(b->states)[uint64_t(source)*chunks+part];if(part==0)out[row]=keys[at];
}
__global__ void graph_publish(const Binding* b,const Key* keys,const uint8_t* states,const uint32_t* n,uint32_t capacity,uint32_t chunks){
 uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(*b->fatal||i>=uint64_t(capacity)*chunks)return;
 uint32_t row=i/chunks,part=i%chunks;
 if(row<*n){reinterpret_cast<uint4*>(b->states)[i]=reinterpret_cast<const uint4*>(states)[i];if(part==0)b->keys[row]=keys[row];}
}
__global__ void graph_publish_count(const Binding* b,const uint32_t* n){if(!*b->fatal)*b->count=*n;}

__global__ void small_sort(Key* keys,uint32_t* refs,uint32_t n){
 __shared__ Key k[1024];__shared__ uint32_t r[1024];
 uint32_t span=1;while(span<n)span*=2;
 for(uint32_t i=threadIdx.x;i<span;i+=blockDim.x){if(i<n){k[i]=keys[i];r[i]=refs[i];}else{k[i]={{UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX}};r[i]=UINT32_MAX;}}__syncthreads();
 for(uint32_t width=2;width<=span;width*=2)for(uint32_t step=width/2;step;step/=2){
  for(uint32_t i=threadIdx.x;i<span;i+=blockDim.x){uint32_t j=i^step;if(j>i){
   bool left_less=Less{}(k[i],k[j])||(Equal{}(k[i],k[j])&&r[i]<r[j]);
   if(((i&width)==0&&!left_less)||((i&width)!=0&&left_less)){Key a=k[i];k[i]=k[j];k[j]=a;uint32_t b=r[i];r[i]=r[j];r[j]=b;}
  }}__syncthreads();
 }
 for(uint32_t i=threadIdx.x;i<n;i+=blockDim.x){keys[i]=k[i];refs[i]=r[i];}
}

void create_graph(Job* p,uint32_t active){
 CachedGraph cache{};cache.bound=active;cudaStream_t stream=nullptr;
 check(cudaGraphCreate(&cache.graph,0));
 check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 check(cudaStreamBeginCaptureToGraph(stream,cache.graph,nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal));
 unsigned blocks=(active+255)/256;
 graph_prepare<<<blocks,256,0,stream>>>(p->binding,active,p->sort_keys,p->sort_indices);
 size_t t=p->cub_bytes;if(active<=1024)small_sort<<<1,256,0,stream>>>(p->sort_keys,p->sort_indices,active);else {check(cub::DeviceRadixSort::SortPairs(p->cub,t,p->sort_keys,p->radix_keys,p->sort_indices,p->radix_indices,int(active),Decompose{},0,128,stream));check(cudaMemcpyAsync(p->sort_keys,p->radix_keys,uint64_t(active)*16,cudaMemcpyDeviceToDevice,stream));check(cudaMemcpyAsync(p->sort_indices,p->radix_indices,uint64_t(active)*4,cudaMemcpyDeviceToDevice,stream));}
 t=p->cub_bytes;check(cub::DeviceReduce::ReduceByKey(p->cub,t,p->sort_keys,p->unique_keys,p->sort_indices,p->unique_indices,p->runs,Min{},int(active),stream));
 graph_mark<<<blocks,256,0,stream>>>(p->binding,p->unique_keys,p->unique_indices,p->runs,active,p->flags);
 t=p->cub_bytes;check(cub::DeviceSelect::Flagged(p->cub,t,thrust::counting_iterator<uint32_t>{0},p->flags,p->selected,p->selected_count,int(active),stream));
 uint32_t chunks=p->stride/16;unsigned grids=unsigned((uint64_t(active)*chunks+255)/256);
 graph_gather<<<grids,256,0,stream>>>(p->binding,p->unique_keys,p->unique_indices,p->selected,p->selected_count,chunks,active,p->sort_keys,p->state_scratch);
 graph_publish<<<grids,256,0,stream>>>(p->binding,p->sort_keys,p->state_scratch,p->selected_count,active,chunks);
 graph_publish_count<<<1,1,0,stream>>>(p->binding,p->selected_count);
 cudaGraph_t captured;check(cudaStreamEndCapture(stream,&captured));check(cudaStreamDestroy(stream));
 check(cudaGraphInstantiate(&cache.executable,cache.graph,0));p->graphs.push_back(cache);
}


__device__ bool added(const Binding* b){return (!b->added_ranges||*b->added_ranges!=0)&&*b->count>=b->ready_threshold;}
__device__ uint32_t key_hash(Key k){
 uint32_t h=k.w[0]*0x9e3779b1u^k.w[1]*0x85ebca77u^k.w[2]*0xc2b2ae3du^k.w[3];
 h^=h>>16;h*=0x7feb352du;h^=h>>15;h*=0x846ca68bu;return h^(h>>16);
}
__global__ void hash_clear(const Binding* b,uint32_t* table,uint32_t n,uint32_t bound,uint32_t* output_count){
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
 if(!added(b))return;
 if(*b->count>bound){if(i==0)atomicCAS(b->fatal,0u,101u);return;}
 if(i==0)*output_count=0;
 if(i<n)table[i]=UINT32_MAX;
}
__global__ void hash_insert(const Binding* b,uint32_t* table,uint32_t span,uint32_t bound){
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
 if(!added(b)||*b->fatal||i>=bound||i>=*b->count)return;
 Key key=b->keys[i];
 if(contains(b->previous,b->pn,key)||contains(b->current,b->cn,key))return;
 uint32_t slot=key_hash(key)&(span-1);
 for(uint32_t probe=0;probe<span;++probe){
  uint32_t old=atomicCAS(table+slot,UINT32_MAX,i);
  if(old==UINT32_MAX)return;
  if(Equal{}(b->keys[old],key)){atomicMin(table+slot,i);return;}
  slot=(slot+1)&(span-1);
 }
 atomicCAS(b->fatal,0u,103u);
}

__global__ void hash_compact(const Binding* b,const uint32_t* table,uint32_t* n,uint32_t span,
 uint32_t chunks,Key* keys,uint8_t* states){
 __shared__ uint32_t blocked;
 if(threadIdx.x==0)blocked=!added(b)||*b->fatal;__syncthreads();if(blocked)return;
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x&31;
 uint32_t source=i<span?table[i]:UINT32_MAX;bool valid=source!=UINT32_MAX;
 uint32_t mask=__ballot_sync(0xffffffffu,valid);uint32_t base=0;
 if(mask){uint32_t leader=__ffs(mask)-1;if(lane==leader)base=atomicAdd(n,__popc(mask));base=__shfl_sync(0xffffffffu,base,leader);}
 if(valid){uint32_t at=base+__popc(mask&((1u<<lane)-1));
  keys[at]=b->keys[source];for(uint32_t part=0;part<chunks;++part)
   reinterpret_cast<uint4*>(states)[uint64_t(at)*chunks+part]=reinterpret_cast<const uint4*>(b->states)[uint64_t(source)*chunks+part];
 }
}
__global__ void hash_publish(const Binding* b,const Key* keys,const uint8_t* states,const uint32_t* n,
 uint32_t chunks,uint32_t bound){
 uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(!added(b)||*b->fatal||i>=uint64_t(bound)*chunks)return;
 uint32_t row=i/chunks,part=i%chunks;
 if(row<*n){reinterpret_cast<uint4*>(b->states)[i]=reinterpret_cast<const uint4*>(states)[i];if(part==0)b->keys[row]=keys[row];}
}
__global__ void hash_publish_count(const Binding* b,const uint32_t* n){if(added(b)&&!*b->fatal)*b->count=*n;}
void create_hash_graph(Job* p,uint32_t active){
 CachedGraph cache{};cache.bound=active;cache.hash=true;cudaStream_t stream=nullptr;
 check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
 uint32_t span=table_capacity(active);unsigned blocks=(span+255)/256;
 hash_clear<<<blocks,256,0,stream>>>(p->binding,p->hash_table,span,active,p->selected_count);
 hash_insert<<<(active+255)/256,256,0,stream>>>(p->binding,p->hash_table,span,active);
 uint32_t chunks=p->stride/16;unsigned grids=unsigned((uint64_t(active)*chunks+255)/256);
 hash_compact<<<blocks,256,0,stream>>>(p->binding,p->hash_table,p->selected_count,span,chunks,p->sort_keys,p->state_scratch);
 hash_publish<<<grids,256,0,stream>>>(p->binding,p->sort_keys,p->state_scratch,p->selected_count,chunks,active);
 hash_publish_count<<<1,1,0,stream>>>(p->binding,p->selected_count);
 check(cudaStreamEndCapture(stream,&cache.graph));check(cudaStreamDestroy(stream));check(cudaGraphInstantiate(&cache.executable,cache.graph,0));p->graphs.push_back(cache);
}

}
extern "C" int mgbfs_shard_ab_job_query(uint32_t capacity,uint32_t stride,uint64_t* bytes){
 if(bytes)*bytes=0;if(!bytes||!capacity||capacity>(1u<<29)||!stride||stride%16||uint64_t(capacity)*(stride/16)>uint64_t(UINT32_MAX)*256)return 1;
 try{*bytes=uint64_t(capacity)*(52+12+1+stride)+8+sizeof(Binding)+uint64_t(table_capacity(capacity))*4+max_temp_bytes(capacity);return 0;}catch(...){return 2;}
}
extern "C" int mgbfs_shard_ab_job_create(uint32_t capacity,uint32_t stride,void** output){
 if(!output)return 1;*output=nullptr;
 try{
  uint64_t bytes;if(mgbfs_shard_ab_job_query(capacity,stride,&bytes))return 1;
  auto p=std::make_unique<Job>();p->capacity=capacity;p->stride=stride;p->cub_bytes=max_temp_bytes(capacity);p->supported_bounds.push_back(capacity);
  for(uint32_t bound=1;bound<capacity;bound*=2){
   p->supported_bounds.push_back(bound);
   if(bound>UINT32_MAX/2)break;
  }
  check(cudaMalloc(&p->sort_keys,uint64_t(capacity)*16));check(cudaMalloc(&p->unique_keys,uint64_t(capacity)*16));
  check(cudaMalloc(&p->sort_indices,uint64_t(capacity)*4));check(cudaMalloc(&p->unique_indices,uint64_t(capacity)*4));
  check(cudaMalloc(&p->selected,uint64_t(capacity)*4));check(cudaMalloc(&p->runs,4));check(cudaMalloc(&p->selected_count,4));
  check(cudaMalloc(&p->flags,capacity));check(cudaMalloc(&p->state_scratch,uint64_t(capacity)*stride));check(cudaMalloc(&p->cub,p->cub_bytes));
  p->hash_capacity=table_capacity(capacity);check(cudaMalloc(&p->hash_table,uint64_t(p->hash_capacity)*4));
  check(cudaMalloc(&p->radix_keys,uint64_t(capacity)*16));check(cudaMalloc(&p->radix_indices,uint64_t(capacity)*4));
  check(cudaMalloc(&p->binding,sizeof(Binding)));
  if(!std::getenv("MGBFS_TEST_AB_NO_GRAPH"))for(auto bound:p->supported_bounds){create_graph(p.get(),bound);create_hash_graph(p.get(),bound);}
  *output=p.release();return 0;
 }catch(...){return 2;}
}
extern "C" int mgbfs_shard_ab_job_run(void* raw,void* keys,uint8_t* states,uint32_t* count,
 const void* previous,uint32_t pn,const void* current,uint32_t cn,uint32_t* fatal,void* raw_stream){
 auto*p=static_cast<Job*>(raw);
 if(!p)return 1;
 return mgbfs_shard_ab_job_run_bounded(raw,p->capacity,keys,states,count,previous,pn,current,cn,fatal,raw_stream);
}
extern "C" int mgbfs_shard_ab_job_run_bounded(void* raw,uint32_t active,void* keys,uint8_t* states,uint32_t* count,
 const void* previous,uint32_t pn,const void* current,uint32_t cn,uint32_t* fatal,void* raw_stream){
 auto*p=static_cast<Job*>(raw);
 if(!p||!active||active>p->capacity||!keys||!states||!count||!fatal||(pn&&!previous)||(cn&&!current))return 1;
 try{
  if(std::find(p->supported_bounds.begin(),p->supported_bounds.end(),active)==p->supported_bounds.end())return 3;
  auto s=static_cast<cudaStream_t>(raw_stream);
  if(!p->graphs.empty()){
  Binding binding{static_cast<Key*>(keys),states,count,static_cast<const Key*>(previous),pn,static_cast<const Key*>(current),cn,fatal,nullptr,0};
  bind_job<<<1,1,0,s>>>(p->binding,binding);
  for(auto&g:p->graphs)if(g.bound==active&&!g.hash){check(cudaGraphLaunch(g.executable,s));check(cudaGetLastError());return 0;}
  return 3;}
  unsigned blocks=(active+255)/256;
  prepare<<<blocks,256,0,s>>>(static_cast<Key*>(keys),count,active,p->sort_keys,p->sort_indices,fatal);
  size_t t=p->cub_bytes;
  if(active<=1024)small_sort<<<1,256,0,s>>>(p->sort_keys,p->sort_indices,active);else {check(cub::DeviceRadixSort::SortPairs(p->cub,t,p->sort_keys,p->radix_keys,p->sort_indices,p->radix_indices,int(active),Decompose{},0,128,s));check(cudaMemcpyAsync(p->sort_keys,p->radix_keys,uint64_t(active)*16,cudaMemcpyDeviceToDevice,s));check(cudaMemcpyAsync(p->sort_indices,p->radix_indices,uint64_t(active)*4,cudaMemcpyDeviceToDevice,s));}
  t=p->cub_bytes;
  check(cub::DeviceReduce::ReduceByKey(p->cub,t,p->sort_keys,p->unique_keys,p->sort_indices,p->unique_indices,p->runs,Min{},int(active),s));
  mark<<<blocks,256,0,s>>>(p->unique_keys,p->unique_indices,p->runs,active,
   static_cast<const Key*>(previous),pn,static_cast<const Key*>(current),cn,p->flags,fatal);
  t=p->cub_bytes;
  check(cub::DeviceSelect::Flagged(p->cub,t,thrust::counting_iterator<uint32_t>{0},p->flags,p->selected,p->selected_count,int(active),s));
  uint32_t chunks=p->stride/16;uint64_t work=uint64_t(active)*chunks;unsigned grids=unsigned((work+255)/256);
  gather<<<grids,256,0,s>>>(p->unique_keys,p->unique_indices,p->selected,p->selected_count,states,chunks,active,p->sort_keys,p->state_scratch,fatal);
  publish<<<grids,256,0,s>>>(p->sort_keys,p->state_scratch,p->selected_count,p->capacity,chunks,static_cast<Key*>(keys),states,count,fatal);
  publish_count<<<1,1,0,s>>>(p->selected_count,count,fatal);check(cudaGetLastError());return 0;
 }catch(...){return 2;}
}
extern "C" int mgbfs_shard_ab_job_destroy(void* raw){delete static_cast<Job*>(raw);return 0;}

extern "C" int mgbfs_shard_ab_job_run_added_ready(void* raw,uint32_t active,void* keys,uint8_t* states,uint32_t* count,
 const void* previous,uint32_t pn,const void* current,uint32_t cn,uint32_t* fatal,const uint32_t* added,uint32_t threshold,void* raw_stream){
 auto*p=static_cast<Job*>(raw);
 if(!p||!keys||!states||!count||!fatal||!added||(pn&&!previous)||(cn&&!current))return 1;
 if(p->graphs.empty())return mgbfs_shard_ab_job_run_bounded(raw,active,keys,states,count,previous,pn,current,cn,fatal,raw_stream);
 try{auto s=static_cast<cudaStream_t>(raw_stream);
 Binding binding{static_cast<Key*>(keys),states,count,static_cast<const Key*>(previous),pn,static_cast<const Key*>(current),cn,fatal,added,threshold};
 bind_job<<<1,1,0,s>>>(p->binding,binding);
 for(auto&g:p->graphs)if(g.bound==active&&g.hash){check(cudaGraphLaunch(g.executable,s));check(cudaGetLastError());return 0;}
 return 3;}catch(...){return 2;}
}

extern "C" int mgbfs_shard_ab_job_run_added(void* raw,uint32_t active,void* keys,uint8_t* states,uint32_t* count,
 const void* previous,uint32_t pn,const void* current,uint32_t cn,uint32_t* fatal,const uint32_t* added,void* stream){
 return mgbfs_shard_ab_job_run_added_ready(raw,active,keys,states,count,previous,pn,current,cn,fatal,added,0,stream);
}
