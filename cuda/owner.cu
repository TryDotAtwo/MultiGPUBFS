#include "mgbfs_cuda.h"
#include <cuda_runtime.h>
#include <cub/device/device_select.cuh>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <stdexcept>

namespace {
struct alignas(16) Key {uint32_t word[4];};
struct alignas(16) Record {Key key;uint64_t ref;uint32_t kind,pad;};
static_assert(sizeof(Key)==16 && sizeof(Record)==32);
static_assert(sizeof(MgbfsOwnerState)==24 && offsetof(MgbfsOwnerState,count)==8);
__device__ bool less(Key a,Key b){
  for(int i=3;i>=0;--i){if(a.word[i]<b.word[i])return true;if(a.word[i]>b.word[i])return false;}return false;
}
__device__ bool equal(Key a,Key b){return !less(a,b)&&!less(b,a);}
// Each view is one contiguous run, not a per-record pointer container.
struct Run {
  const Key* keys{};const uint64_t* refs{};const Record* records{};
  uint32_t fixed{};const uint32_t* extra1{};const uint32_t* extra2{};uint32_t kind{};
  __device__ uint32_t size() const{return fixed+(extra1?*extra1:0)+(extra2?*extra2:0);}
  __device__ Record at(uint32_t i) const {return records?records[i]:Record{keys[i],refs?refs[i]:0,kind,0};}
};
// Stable merge partition: the left run precedes the right run for equal keys.
__device__ uint32_t partition(Run a,Run b,uint32_t na,uint32_t nb,uint32_t diagonal){
  uint32_t low=diagonal>nb?diagonal-nb:0,high=diagonal<na?diagonal:na;
  while(low<high){
    const uint32_t i=low+(high-low)/2,j=diagonal-i;
    if(j>0 && i<na && !less(b.at(j-1).key,a.at(i).key))low=i+1;
    else high=i;
  }
  return low;
}
constexpr uint32_t tile=256,threads=128;
__global__ void merge_tiles(Run a,Run b,Record* output,const MgbfsOwnerState* state){
  if(state->fatal)return;
  const uint32_t na=a.size(),nb=b.size(),start=blockIdx.x*tile;
  if(start>=na+nb)return;
  const uint32_t end=min(start+tile,na+nb);
  __shared__ uint32_t a0,b0,ac,bc;
  __shared__ Record sa[tile],sb[tile];
  if(threadIdx.x==0){
    a0=partition(a,b,na,nb,start);b0=start-a0;
    const uint32_t a1=partition(a,b,na,nb,end);ac=a1-a0;bc=end-a1-b0;
  }
  __syncthreads();
  for(uint32_t i=threadIdx.x;i<ac;i+=blockDim.x)sa[i]=a.at(a0+i);
  for(uint32_t i=threadIdx.x;i<bc;i+=blockDim.x)sb[i]=b.at(b0+i);
  __syncthreads();
  Run ar{};ar.records=sa;Run br{};br.records=sb;
  for(uint32_t d=threadIdx.x;d<end-start;d+=blockDim.x){
    const uint32_t i=partition(ar,br,ac,bc,d),j=d-i;
    output[start+d]=(i<ac && (j==bc || !less(sb[j].key,sa[i].key)))?sa[i]:sb[j];
  }
}
struct OwnerPlan {
  uint32_t candidate_capacity{},bucket_capacity{},bound{};
  Record *a{},*b{};uint32_t *indices{},*next_indices{},*new_indices{},*counts{};
  uint8_t *next_flags{},*new_flags{};void* scratch{};size_t scratch_bytes{};
  ~OwnerPlan(){cudaFree(scratch);cudaFree(new_flags);cudaFree(next_flags);cudaFree(counts);cudaFree(new_indices);cudaFree(next_indices);cudaFree(indices);cudaFree(b);cudaFree(a);}
};
void check(cudaError_t r){if(r!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(r));}
__global__ void begin_epoch(MgbfsOwnerState* s,const uint32_t* count,uint32_t cc,uint32_t bc,uint64_t epoch,uint32_t* output_count){
  *output_count=0;
  if(s->fatal)return;
  if(s->count>bc || *count>cc){s->fatal=1;return;}
  if(s->initialized && epoch<=s->last_epoch){s->fatal=2;return;}
}
__global__ void validate_sorted(Run prev,Run curr,Run accepted,Run incoming,MgbfsOwnerState* s,uint32_t cc,uint32_t bc){
  // Do not read fatal while other threads atomically set it in this kernel.
  if(accepted.size()>bc || incoming.size()>cc)return;
  const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
  const Run runs[4]={prev,curr,accepted,incoming};
  for(int r=0;r<4;++r)if(i>0 && i<runs[r].size() && less(runs[r].at(i).key,runs[r].at(i-1).key))atomicCAS(&s->fatal,0u,3u);
}
__global__ void flag_survivors(Run merged,uint32_t bound,const MgbfsOwnerState* state,uint32_t* indices,uint8_t* next_flags,uint8_t* new_flags){
  const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=bound)return;
  indices[i]=i;bool fresh=false,keep=false;
  if(!state->fatal && i<merged.size()){
    const Record record=merged.at(i);
    fresh=record.kind==2 && (i==0 || !equal(record.key,merged.at(i-1).key));
    keep=record.kind==1 || fresh;
  }
  next_flags[i]=keep;new_flags[i]=fresh;
}
__global__ void guard_commit(MgbfsOwnerState* state,const uint32_t* counts,uint32_t bc){
  if(!state->fatal && counts[0]>bc)state->fatal=1;
}
__global__ void copy_commit(const Record* merged,const uint32_t* next_indices,const uint32_t* new_indices,const uint32_t* counts,
  Key* accepted,Key* survivors,uint64_t* refs,const MgbfsOwnerState* state,uint32_t bound){
  if(state->fatal)return;
  const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=bound)return;
  if(i<counts[0])accepted[i]=merged[next_indices[i]].key;
  if(i<counts[1]){const Record r=merged[new_indices[i]];survivors[i]=r.key;refs[i]=r.ref;}
}
__global__ void publish_commit(MgbfsOwnerState* state,const uint32_t* counts,uint32_t* output_count,uint64_t epoch){
  if(state->fatal)return;
  state->count=counts[0];state->last_epoch=epoch;state->initialized=1;*output_count=counts[1];
}
}
extern "C" int mgbfs_owner_create(uint32_t cc,uint32_t bc,void** out,char* error,size_t n){
  if(!out)return 1;*out=nullptr;
  try{
    if(!cc||!bc||uint64_t(bc)*3+cc>INT32_MAX)throw std::runtime_error("OWNER_CAPACITY_BOUND");
    auto p=std::make_unique<OwnerPlan>();p->candidate_capacity=cc;p->bucket_capacity=bc;p->bound=3*bc+cc;
    check(cudaMalloc(&p->a,size_t(p->bound)*sizeof(Record)));check(cudaMalloc(&p->b,size_t(3)*bc*sizeof(Record)));
    check(cudaMalloc(&p->indices,size_t(p->bound)*4));check(cudaMalloc(&p->next_indices,size_t(p->bound)*4));check(cudaMalloc(&p->new_indices,size_t(p->bound)*4));check(cudaMalloc(&p->counts,8));
    check(cudaMalloc(&p->next_flags,p->bound));check(cudaMalloc(&p->new_flags,p->bound));
    check(cub::DeviceSelect::Flagged(nullptr,p->scratch_bytes,p->indices,p->next_flags,p->next_indices,p->counts,int(p->bound)));
    check(cudaMalloc(&p->scratch,p->scratch_bytes));*out=p.release();return 0;
  }catch(const std::exception& e){if(error&&n)std::snprintf(error,n,"%s",e.what());return 1;}
}
extern "C" int mgbfs_owner_run(void* plan,const void* prev,uint32_t pn,const void* curr,uint32_t cn,void* accepted,MgbfsOwnerState* state,
  const void* candidates,const uint64_t* refs,const uint32_t* count,void* survivors,uint64_t* survivor_refs,uint32_t* survivor_count,uint64_t epoch,void* raw_stream){
  auto* p=static_cast<OwnerPlan*>(plan);
  if(!p||pn>p->bucket_capacity||cn>p->bucket_capacity||(pn&&!prev)||(cn&&!curr)||!accepted||!state||!candidates||!refs||!count||!survivors||!survivor_refs||!survivor_count)return 1;
  const auto stream=static_cast<cudaStream_t>(raw_stream);const uint32_t bound=pn+cn+p->bucket_capacity+p->candidate_capacity;
  Run pr{static_cast<const Key*>(prev),nullptr,nullptr,pn,nullptr,nullptr,0};
  Run cr{static_cast<const Key*>(curr),nullptr,nullptr,cn,nullptr,nullptr,0};
  Run ar{static_cast<const Key*>(accepted),nullptr,nullptr,0,&state->count,nullptr,1};
  Run in{static_cast<const Key*>(candidates),refs,nullptr,0,count,nullptr,2};
  Run old1{nullptr,nullptr,p->a,pn+cn,nullptr,nullptr,0};
  Run old2{nullptr,nullptr,p->b,pn+cn,&state->count,nullptr,0};
  Run merged{nullptr,nullptr,p->a,pn+cn,&state->count,count,0};
  begin_epoch<<<1,1,0,stream>>>(state,count,p->candidate_capacity,p->bucket_capacity,epoch,survivor_count);
  validate_sorted<<<(p->bound+threads-1)/threads,threads,0,stream>>>(pr,cr,ar,in,state,p->candidate_capacity,p->bucket_capacity);
  if(pn+cn)merge_tiles<<<(pn+cn+tile-1)/tile,threads,0,stream>>>(pr,cr,p->a,state);
  merge_tiles<<<(pn+cn+p->bucket_capacity+tile-1)/tile,threads,0,stream>>>(old1,ar,p->b,state);
  merge_tiles<<<(bound+tile-1)/tile,threads,0,stream>>>(old2,in,p->a,state);
  flag_survivors<<<(bound+threads-1)/threads,threads,0,stream>>>(merged,bound,state,p->indices,p->next_flags,p->new_flags);
  if(cudaGetLastError()!=cudaSuccess)return 2;
  size_t bytes=p->scratch_bytes;
  if(cub::DeviceSelect::Flagged(p->scratch,bytes,p->indices,p->next_flags,p->next_indices,p->counts,int(bound),stream)!=cudaSuccess)return 3;
  bytes=p->scratch_bytes;
  if(cub::DeviceSelect::Flagged(p->scratch,bytes,p->indices,p->new_flags,p->new_indices,p->counts+1,int(bound),stream)!=cudaSuccess)return 3;
  guard_commit<<<1,1,0,stream>>>(state,p->counts,p->bucket_capacity);
  copy_commit<<<(bound+threads-1)/threads,threads,0,stream>>>(p->a,p->next_indices,p->new_indices,p->counts,static_cast<Key*>(accepted),static_cast<Key*>(survivors),survivor_refs,state,bound);
  publish_commit<<<1,1,0,stream>>>(state,p->counts,survivor_count,epoch);
  return cudaGetLastError()==cudaSuccess?0:4;
}
extern "C" void mgbfs_owner_destroy(void* p){delete static_cast<OwnerPlan*>(p);}
