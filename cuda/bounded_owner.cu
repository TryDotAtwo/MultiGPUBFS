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
  uint32_t backend=0,tile_limit=0; uint32_t* refinement_errors=nullptr;
  ~Plan(){cudaFree(refinement_errors);cudaFree(merged);cudaFree(indices);cudaFree(flags);}
};
unsigned tile_count(const Plan* p,unsigned j){
  unsigned tiles=(128+j-1)/j;if(tiles>64)tiles=64;
  // Fixed plan capacity bounds useful tile concurrency. Tiny validation/jobs
  // must not launch dozens of empty CTAs per bucket.
  unsigned capacity_tiles=(p->i+T-1)/T;
  return tiles<capacity_tiles?tiles:capacity_tiles;
}
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
struct RefinedRange { uint64_t a,b; uint32_t m,n; };
static_assert(sizeof(RefinedRange)==24);
__device__ uint32_t bit_boundary(const Key* keys,uint64_t begin,uint32_t count,unsigned bit){
  uint32_t lo=0,hi=count;
  while(lo<hi){uint32_t mid=lo+(hi-lo)/2;
    if((keys[begin+mid].w[bit/32]>>(bit%32))&1u)hi=mid;else lo=mid+1;
  }return lo;
}
// One bounded metadata stack per bucket, no pointer nodes/global append or
// allocation. A binary prefix path is at most 128 bits deep. Data work below
// uses all warps, coalescing four adjacent words per candidate/reference key.
__global__ void bmma_membership(const MgbfsBucketJob* jobs,const Key* in,const Key* old,
    uint32_t k,uint32_t tile_limit,uint8_t* flags,uint32_t* errors,
    const MgbfsOwnerControl* control,unsigned category){
  __shared__ RefinedRange stack[129],work;
  __shared__ unsigned top,action;
  const unsigned tid=threadIdx.x,lane=tid&31,warp=tid>>5;
  if(control->error){if(tid==0)errors[blockIdx.x]=0;return;}
  if(tid==0){auto d=jobs[blockIdx.x];
    auto r=category==2?d.prev:category==3?d.curr:MgbfsOwnerRange{uint64_t(d.bucket)*k,d.accepted_count};
    stack[0]={d.incoming.begin,r.begin,uint32_t(d.incoming.count),uint32_t(r.count)};
    top=1;errors[blockIdx.x]=0;
  }__syncthreads();
  while(true){
    if(tid==0){action=0;
      if(top){work=stack[--top];action=1;
        if(!work.m||!work.n)action=0;
        else if(work.m<=tile_limit&&work.n<=tile_limit)action=2;
        else {
          Key first=in[work.a],last=in[work.a+work.m-1];
          Key rf=old[work.b],rl=old[work.b+work.n-1];int bit=-1;
          // All four endpoints share every higher bit. Splitting at the first
          // difference skips long common prefixes without 128 array passes.
          for(int w=3;w>=0;--w){unsigned diff=(first.w[w]^last.w[w])|
              (first.w[w]^rf.w[w])|(first.w[w]^rl.w[w]);
            if(diff){bit=w*32+31-__clz(diff);break;}}
          if(bit<0)action=3; // One identical full Hash128 run: linear marking.
          else {
            unsigned am=bit_boundary(in,work.a,work.m,unsigned(bit));
            unsigned bn=bit_boundary(old,work.b,work.n,unsigned(bit));
            bool left=am&&bn,right=am<work.m&&bn<work.n;
            if(top+unsigned(left)+unsigned(right)>129){errors[blockIdx.x]=5;action=4;}
            else {
              if(right)stack[top++]={work.a+am,work.b+bn,work.m-am,work.n-bn};
              if(left)stack[top++]={work.a,work.b,am,bn};
            }
          }
        }
      }else action=4;
    }__syncthreads();
    if(action==4)break;
    if(action==3){
      for(uint32_t i=tid;i<work.m;i+=T){auto row=work.a+i;if(!flags[row])flags[row]=uint8_t(category);}
    }else if(action==2){
      for(uint32_t base=warp*8;base<work.m;base+=(T/32)*8){
        const uint32_t row=base+lane/4;bool found=false;
        const uint32_t a=row<work.m?in[work.a+row].w[lane%4]:0;
        for(uint32_t ref=0;ref<work.n;ref+=8){
          const uint32_t br=ref+lane/4;
          const uint32_t b=br<work.n?old[work.b+br].w[lane%4]:0;
          int d0=0,d1=0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 900
          asm volatile("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc "
            "{%0,%1}, {%2}, {%3}, {%4,%5};"
            : "=r"(d0),"=r"(d1):"r"(a),"r"(b),"r"(0),"r"(0));
#else
          asm volatile("trap;");
#endif
          unsigned col=ref+(lane%4)*2;
          found|=(col<work.n&&d0==0)||(col+1<work.n&&d1==0);
        }
        const unsigned mask=__ballot_sync(0xffffffffu,found);
        if(lane%4==0&&row<work.m&&(mask&(15u<<(lane&~3u)))){
          auto index=work.a+row;if(!flags[index])flags[index]=uint8_t(category);
        }
      }
    }
    __syncthreads();
  }
}
__global__ void refinement_status(const uint32_t* errors,uint32_t count,MgbfsOwnerControl* c){
  if(c->error)return;for(unsigned i=0;i<count;++i)if(errors[i]){c->error=errors[i];return;}
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
  // Independent merge-path tiles own disjoint output positions. Multiple CTAs
  // per bucket expose parallelism even when a job contains few large buckets.
  for(uint64_t base=uint64_t(blockIdx.y)*T;base<uint64_t(m)+n;base+=uint64_t(gridDim.y)*T){
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
  for(uint64_t r=uint64_t(blockIdx.y)*T+threadIdx.x;r<x.new_count;r+=uint64_t(gridDim.y)*T)accepted[uint64_t(d.bucket)*k+r]=merged[uint64_t(blockIdx.x)*k+r];
  for(uint64_t r=uint64_t(blockIdx.y)*T+threadIdx.x;r<x.survivors;r+=uint64_t(gridDim.y)*T)indices[x.output_offset+r]=local_indices[d.incoming.begin+r];
  // No concurrent consumer: stream completion of this whole kernel publishes
  // both data and count before finish_commit or the next owner's job can read.
  if(blockIdx.y==0&&threadIdx.x==0)lengths[d.bucket]=x.new_count;
}
__global__ void finish_commit(MgbfsOwnerControl* c){if(!c->error)c->stage=2;}
}
extern "C" int mgbfs_bounded_owner_create(uint32_t i,uint32_t j,uint32_t k,void** out){
  if(!out)return 1;*out=nullptr;
  MgbfsBoundedOwnerBytes q{};
  if(mgbfs_bounded_owner_query(i,j,k,0,0,0,&q))return 1;
  auto p=std::make_unique<Plan>();p->i=i;p->j=j;p->k=k;
  if(cudaMalloc(&p->flags,q.flags)!=cudaSuccess||cudaMalloc(&p->indices,q.indices)!=cudaSuccess||
     cudaMalloc(&p->merged,q.merged)!=cudaSuccess)return 2;
  *out=p.release();return 0;
}
extern "C" void mgbfs_bounded_owner_destroy(void* p){delete static_cast<Plan*>(p);}
extern "C" int mgbfs_bounded_owner_create_backend(uint32_t i,uint32_t j,uint32_t k,
    uint32_t backend,uint32_t refinement_capacity,uint32_t tile_limit,void** out){
  if(!out)return 1;*out=nullptr;
  if(backend==0)return mgbfs_bounded_owner_create(i,j,k,out);
  MgbfsBoundedOwnerBytes q{};
  if(mgbfs_bounded_owner_query(i,j,k,backend,refinement_capacity,tile_limit,&q))return 1;
  int device;cudaDeviceProp prop{};
  if(cudaGetDevice(&device)!=cudaSuccess||cudaGetDeviceProperties(&prop,device)!=cudaSuccess)return 2;
  // V1 hardware policy is explicit SM75, never a silent scalar/CUB fallback.
  if(prop.major!=7||prop.minor!=5)return 3;
  void* raw=nullptr;int status=mgbfs_bounded_owner_create(i,j,k,&raw);if(status)return status;
  std::unique_ptr<Plan> p(static_cast<Plan*>(raw));
  if(cudaMalloc(&p->refinement_errors,q.refinement_errors)!=cudaSuccess)return 2;
  p->backend=1;p->tile_limit=tile_limit;*out=p.release();return 0;
}
extern "C" int mgbfs_bounded_owner_compare(void* raw,const MgbfsBucketJob* jobs,uint32_t j,uint32_t rows,
    const void* in,const void* prev,uint64_t pn,const void* curr,uint64_t cn,const void* accepted,
    const uint32_t* lengths,uint32_t buckets,uint32_t per_shard,uint32_t lane,uint32_t generation,
    MgbfsOwnerCounts* counts,MgbfsOwnerControl* control,void* stream){
  auto p=static_cast<Plan*>(raw);auto s=static_cast<cudaStream_t>(stream);
  if(!p||!jobs||!j||j>p->j||!rows||rows>p->i||!in||!prev||!curr||!accepted||!lengths||!counts||!control||
     !buckets||!per_shard||(per_shard&(per_shard-1))||buckets%per_shard)return 1;
  validate<<<1,1,0,s>>>(jobs,j,rows,p->k,lengths,buckets,per_shard,lane,generation,pn,cn,control);
  initial<<<j,T,0,s>>>(jobs,static_cast<const Key*>(in),p->flags,control);
  unsigned tiles=tile_count(p,j);
  for(unsigned tag=2;tag<=4;++tag){
    const auto old=static_cast<const Key*>(tag==2?prev:tag==3?curr:accepted);
    if(p->backend==1){
      bmma_membership<<<j,T,0,s>>>(jobs,static_cast<const Key*>(in),old,p->k,p->tile_limit,
        p->flags,p->refinement_errors,control,tag);
      refinement_status<<<1,1,0,s>>>(p->refinement_errors,j,control);
    }else merge_tiles<false><<<dim3(j,tiles),T,0,s>>>(jobs,static_cast<const Key*>(in),
      old,p->k,p->flags,p->indices,p->merged,counts,control,tag);
  }
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
  unsigned tiles=tile_count(p,j);
  merge_tiles<true><<<dim3(j,tiles),T,0,s>>>(jobs,static_cast<const Key*>(in),static_cast<const Key*>(accepted),p->k,p->flags,p->indices,p->merged,counts,control,0);
  publish<<<dim3(j,tiles),T,0,s>>>(jobs,p->k,p->merged,static_cast<Key*>(accepted),lengths,p->indices,selected,counts,control);
  finish_commit<<<1,1,0,s>>>(control);
  return cudaGetLastError()==cudaSuccess?0:2;
}
