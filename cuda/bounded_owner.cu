#include "bounded_owner.h"
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#include <memory>
#include <climits>
namespace {
constexpr unsigned T=256;
struct alignas(16) Key { uint32_t w[4]; };
static_assert(sizeof(MgbfsBucketJob)==64 && alignof(MgbfsBucketJob)==64);
static_assert(sizeof(MgbfsOwnerCounts)==32 && sizeof(MgbfsOwnerControl)==64);
struct Plan {
  uint32_t i,j,k; uint8_t* flags=nullptr; uint32_t* indices=nullptr; Key* merged=nullptr;
  ~Plan(){cudaFree(merged);cudaFree(indices);cudaFree(flags);}
};
__device__ bool less(Key a,Key b){for(int w=3;w>=0;--w){if(a.w[w]!=b.w[w])return a.w[w]<b.w[w];}return false;}
__device__ bool equal(Key a,Key b){return !less(a,b)&&!less(b,a);}
struct Read {
  const Key* keys; const uint32_t* indices; uint64_t offset;
  __device__ Key operator[](uint32_t x)const{return keys[indices?indices[offset+x]:offset+x];}
};
// Stable merge-path: equal A precedes B. Global searches only at tile
// boundaries; per-lane searches run on cooperatively loaded shared tiles.
__device__ uint32_t rank_a(uint64_t d,Read a,uint32_t m,Read b,uint32_t n){
  uint32_t lo=d>n?uint32_t(d-n):0,hi=uint32_t(d<m?d:m);
  while(lo<=hi){uint32_t x=lo+(hi-lo)/2,y=uint32_t(d-x);
    if(x&&y<n&&less(b[y],a[x-1])) hi=x-1;
    else if(y&&x<m&&!less(b[y-1],a[x])) lo=x+1;
    else return x;
  }return lo;
}
__global__ void validate(const MgbfsBucketJob* jobs,uint32_t count,uint32_t rows,
    uint32_t k,const uint32_t* lengths,uint32_t buckets,uint32_t per_shard,
    uint32_t lane,uint32_t generation,uint64_t pn,uint64_t cn,MgbfsOwnerControl* c){
  *c={};uint64_t end=0;uint32_t shard=jobs[0].bucket/per_shard;
  for(uint32_t j=0;j<count;++j){auto d=jobs[j];
    if(d.bucket>=buckets||d.bucket/per_shard!=shard||d.lane!=lane||d.generation!=generation||
       (j&&d.bucket<=jobs[j-1].bucket)||d.incoming.begin!=end||!d.incoming.count||
       d.incoming.count>rows-end){c->error=1;return;}
    end+=d.incoming.count;
    if(d.prev.count>k||d.curr.count>k||d.accepted_count>k){c->error=2;return;}
    if(d.prev.begin>pn||d.prev.count>pn-d.prev.begin||d.curr.begin>cn||d.curr.count>cn-d.curr.begin||
       d.accepted_count!=lengths[d.bucket]){c->error=1;return;}
  }
  if(end!=rows)c->error=1;
}
__global__ void initial(const MgbfsBucketJob* jobs,const Key* in,uint8_t* flags,const MgbfsOwnerControl* c){
  if(c->error)return;auto d=jobs[blockIdx.x];
  for(uint64_t x=threadIdx.x;x<d.incoming.count;x+=T){uint64_t r=d.incoming.begin+x;flags[r]=x&&equal(in[r],in[r-1])?1:0;}
}
template<bool Commit> __global__ void merge_tiles(const MgbfsBucketJob* jobs,const Key* in,
    const Key* old,uint32_t k,uint8_t* flags,const uint32_t* indices,Key* merged,
    const MgbfsOwnerCounts* counts,const MgbfsOwnerControl* c,unsigned category){
  if(c->error)return;auto d=jobs[blockIdx.x];Read a{},b{};uint32_t m,n;
  if constexpr(Commit){a={old,nullptr,uint64_t(d.bucket)*k};m=d.accepted_count;
    b={in,indices,d.incoming.begin};n=counts[blockIdx.x].survivors;
  }else{a={in,nullptr,d.incoming.begin};m=uint32_t(d.incoming.count);
    auto r=category==2?d.prev:category==3?d.curr:MgbfsOwnerRange{uint64_t(d.bucket)*k,d.accepted_count};
    b={old,nullptr,r.begin};n=uint32_t(r.count);
  }
  __shared__ Key sa[T],sb[T];__shared__ uint32_t ab[4];
  for(uint64_t base=0;base<uint64_t(m)+n;base+=T){
    uint64_t end=base+T<uint64_t(m)+n?base+T:uint64_t(m)+n;
    if(threadIdx.x==0){ab[0]=rank_a(base,a,m,b,n);ab[1]=uint32_t(base-ab[0]);
      ab[2]=rank_a(end,a,m,b,n);ab[3]=uint32_t(end-ab[2]);}
    __syncthreads();unsigned am=ab[2]-ab[0],bn=ab[3]-ab[1];
    if(threadIdx.x<am)sa[threadIdx.x]=a[ab[0]+threadIdx.x];
    if(threadIdx.x<bn)sb[threadIdx.x]=b[ab[1]+threadIdx.x];
    __syncthreads();
    if(threadIdx.x<end-base){unsigned x=rank_a(threadIdx.x,{sa,nullptr,0},am,{sb,nullptr,0},bn),y=threadIdx.x-x;
      bool take_a=y==bn||(x<am&&!less(sb[y],sa[x]));
      if constexpr(Commit){merged[uint64_t(blockIdx.x)*k+base+threadIdx.x]=take_a?sa[x]:sb[y];}
      else if(take_a){uint64_t row=d.incoming.begin+ab[0]+x;unsigned gy=ab[1]+y;
        if(!flags[row]&&gy<n&&equal(sa[x],b[gy]))flags[row]=uint8_t(category);}
    }__syncthreads();
  }
}
__global__ void compact(const MgbfsBucketJob* jobs,const uint8_t* flags,uint32_t* indices,
    MgbfsOwnerCounts* out,const MgbfsOwnerControl* control){
  if(control->error)return;auto d=jobs[blockIdx.x];
  using Scan=cub::BlockScan<unsigned,T>;using Reduce=cub::BlockReduce<unsigned,T>;
  __shared__ typename Scan::TempStorage scan;__shared__ typename Reduce::TempStorage reduce;
  __shared__ unsigned carry;unsigned categories[5]={};if(threadIdx.x==0)carry=0;__syncthreads();
  for(uint64_t base=0;base<d.incoming.count;base+=T){uint64_t r=base+threadIdx.x;bool live=r<d.incoming.count;
    unsigned f=live?flags[d.incoming.begin+r]:0,keep=live&&f==0,prefix,total;
    if(live)++categories[f];Scan(scan).ExclusiveSum(keep,prefix,total);
    if(keep)indices[d.incoming.begin+carry+prefix]=uint32_t(d.incoming.begin+r);
    __syncthreads();if(threadIdx.x==0)carry+=total;__syncthreads();
  }
  unsigned totals[5];for(unsigned f=0;f<5;++f){totals[f]=Reduce(reduce).Sum(categories[f]);__syncthreads();}
  if(threadIdx.x==0)out[blockIdx.x]={totals[1],totals[2],totals[3],totals[4],totals[0],0,0};
}
__global__ void finish_compare(const MgbfsBucketJob* jobs,uint32_t j,uint32_t k,MgbfsOwnerCounts* counts,MgbfsOwnerControl* c){
  if(c->error)return;uint32_t sum=0;
  for(unsigned b=0;b<j;++b){auto& x=counts[b];if(x.survivors>k-jobs[b].accepted_count){c->error=2;return;}
    x.new_count=jobs[b].accepted_count+x.survivors;x.output_offset=sum;sum+=x.survivors;}
  c->survivors=sum;c->stage=1;
}
__global__ void check_grant(const uint32_t* grant,MgbfsOwnerControl* c){
  if(c->error)return;if(c->stage!=1){c->error=3;return;}if(*grant<c->survivors)c->error=4;
}
__global__ void publish(const MgbfsBucketJob* jobs,uint32_t k,const Key* merged,Key* accepted,
    uint32_t* lengths,const uint32_t* local_indices,uint32_t* indices,const MgbfsOwnerCounts* counts,const MgbfsOwnerControl* c){
  if(c->error)return;auto d=jobs[blockIdx.x];auto x=counts[blockIdx.x];
  for(uint32_t r=threadIdx.x;r<x.new_count;r+=T)accepted[uint64_t(d.bucket)*k+r]=merged[uint64_t(blockIdx.x)*k+r];
  for(uint32_t r=threadIdx.x;r<x.survivors;r+=T)indices[x.output_offset+r]=local_indices[d.incoming.begin+r];
  __syncthreads();if(threadIdx.x==0)lengths[d.bucket]=x.new_count;
}
__global__ void finish_commit(MgbfsOwnerControl* c){if(!c->error)c->stage=2;}
}
extern "C" int mgbfs_bounded_owner_create(uint32_t i,uint32_t j,uint32_t k,void** out){
  if(!out)return 1;*out=nullptr;
  if(!i||i>INT_MAX||!j||j>i||!k||k>INT_MAX||uint64_t(j)*k>SIZE_MAX/sizeof(Key))return 1;
  auto p=std::make_unique<Plan>();p->i=i;p->j=j;p->k=k;
  if(cudaMalloc(&p->flags,i)!=cudaSuccess||cudaMalloc(&p->indices,uint64_t(i)*4)!=cudaSuccess||
     cudaMalloc(&p->merged,uint64_t(j)*k*sizeof(Key))!=cudaSuccess)return 2;
  *out=p.release();return 0;
}
extern "C" void mgbfs_bounded_owner_destroy(void* p){delete static_cast<Plan*>(p);}
extern "C" int mgbfs_bounded_owner_compare(void* raw,const MgbfsBucketJob* jobs,uint32_t j,uint32_t rows,
    const void* in,const void* prev,uint64_t pn,const void* curr,uint64_t cn,const void* accepted,
    const uint32_t* lengths,uint32_t buckets,uint32_t per_shard,uint32_t lane,uint32_t generation,
    MgbfsOwnerCounts* counts,MgbfsOwnerControl* control,void* stream){
  auto p=static_cast<Plan*>(raw);auto s=static_cast<cudaStream_t>(stream);
  if(!p||!jobs||!j||j>p->j||!rows||rows>p->i||!in||!prev||!curr||!accepted||!lengths||!counts||!control||
     !buckets||!per_shard||(per_shard&(per_shard-1))||buckets%per_shard)return 1;
  validate<<<1,1,0,s>>>(jobs,j,rows,p->k,lengths,buckets,per_shard,lane,generation,pn,cn,control);
  initial<<<j,T,0,s>>>(jobs,static_cast<const Key*>(in),p->flags,control);
  for(unsigned tag=2;tag<=4;++tag)merge_tiles<false><<<j,T,0,s>>>(jobs,static_cast<const Key*>(in),
    static_cast<const Key*>(tag==2?prev:tag==3?curr:accepted),p->k,p->flags,p->indices,p->merged,counts,control,tag);
  compact<<<j,T,0,s>>>(jobs,p->flags,p->indices,counts,control);
  finish_compare<<<1,1,0,s>>>(jobs,j,p->k,counts,control);
  return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_bounded_owner_commit(void* raw,const MgbfsBucketJob* jobs,uint32_t j,const void* in,
    void* accepted,uint32_t* lengths,const MgbfsOwnerCounts* counts,MgbfsOwnerControl* control,
    const uint32_t* grant,uint32_t* selected,void* stream){
  auto p=static_cast<Plan*>(raw);auto s=static_cast<cudaStream_t>(stream);
  if(!p||!jobs||!j||j>p->j||!in||!accepted||!lengths||!counts||!control||!grant||!selected)return 1;
  check_grant<<<1,1,0,s>>>(grant,control);
  merge_tiles<true><<<j,T,0,s>>>(jobs,static_cast<const Key*>(in),static_cast<const Key*>(accepted),p->k,p->flags,p->indices,p->merged,counts,control,0);
  publish<<<j,T,0,s>>>(jobs,p->k,p->merged,static_cast<Key*>(accepted),lengths,p->indices,selected,counts,control);
  finish_commit<<<1,1,0,s>>>(control);
  return cudaGetLastError()==cudaSuccess?0:2;
}
