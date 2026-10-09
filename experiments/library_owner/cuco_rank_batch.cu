#include "cuco_rank_batch.cuh"
#include <cub/device/device_select.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda/std/tuple>
#include <thrust/iterator/counting_iterator.h>
#include <array>
#include <climits>
#include <type_traits>
#include <utility>

namespace mgbfs {
namespace {
using Set = cuco_owner_detail::Set;
using ContainsRef = decltype(std::declval<Set&>().ref(cuco::contains));
using InsertRef = decltype(std::declval<Set&>().ref(cuco::insert));
static_assert(std::is_trivially_copyable_v<ContainsRef> &&
              std::is_trivially_copyable_v<InsertRef>);

struct AcceptedView { uint32_t* words[4]; uint32_t stride; uint64_t* state_refs; };

__device__ void poison(MgbfsStateRingControl* ring,MgbfsOwnerControl* owner,uint32_t code){
  atomicCAS(&owner->error,0u,code);
  atomicCAS(&ring->fatal,0u,code);
}
__global__ void copy_candidates(MgbfsLibraryKeysV1 input,const uint32_t* valid,
    uint32_t cap,uint32_t* output,uint32_t stride,MgbfsStateRingControl* ring,
    MgbfsOwnerControl* owner){
  if(ring->fatal||owner->error)return;
  uint32_t n=*valid;
  if(n>cap){
    if(blockIdx.x==0&&threadIdx.x==0)poison(ring,owner,21);
    return;
  }
  for(uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;row<n;
      row+=gridDim.x*blockDim.x)
    for(unsigned word=0;word<4;++word)
      output[uint64_t(word)*stride+row]=input.words[word][row];
}
template<class Ref>
__global__ void insert_candidates(Ref set,const uint32_t* valid,uint32_t cap,
    const MgbfsOwnerControl* owner){
  if(owner->error)return;
  uint32_t n=*valid<cap?*valid:cap;
  for(uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;row<n;
      row+=gridDim.x*blockDim.x)set.insert(key_index(3,row));
}
template<class Ref>
__global__ void representatives(Ref set,const uint32_t* valid,uint32_t cap,
    uint32_t* first,uint32_t* minima,uint32_t* error,
    const MgbfsOwnerControl* owner){
  if(owner->error)return;
  uint32_t n=*valid<cap?*valid:cap;
  for(uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;row<n;
      row+=gridDim.x*blockDim.x){
    auto found=set.find(key_index(3,row));
    if(found==set.end()){first[row]=UINT32_MAX;atomicExch(error,1u);continue;}
    uint64_t index=*found,source=index&((uint64_t{1}<<62)-1);
    if((index>>62)!=3||source>=n){first[row]=UINT32_MAX;atomicExch(error,1u);continue;}
    first[row]=uint32_t(source);atomicMin(minima+source,row);
  }
}
__global__ void rank_flags(const ContainsRef* refs,const uint32_t* high,
    const uint32_t* valid,uint32_t cap,uint32_t stride,uint32_t shift,
    uint32_t logical_owner,uint32_t shards,const uint32_t* first,
    const uint32_t* minima,const uint32_t* workspace_error,uint8_t* flags,
    MgbfsStateRingControl* ring,MgbfsOwnerControl* owner){
  uint32_t n=*valid<cap?*valid:cap;
  for(uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;row<cap;
      row+=gridDim.x*blockDim.x){
    flags[row]=0;
    if(row>=n||atomicAdd(&owner->error,0u)||*workspace_error)return;
    uint32_t prefix=high[uint64_t(3)*stride+row]>>shift;
    if(prefix/shards!=logical_owner){poison(ring,owner,21);continue;}
    uint32_t representative=first[row];
    if(representative==UINT32_MAX||representative>=n){poison(ring,owner,21);continue;}
    if(minima[representative]==row)
      flags[row]=!refs[prefix&(shards-1)].contains(key_index(3,row));
  }
}
__global__ void finish_compare(const uint32_t* valid,uint32_t cap,
    const uint32_t* workspace_control,MgbfsStateRingControl* ring,
    MgbfsOwnerControl* owner){
  if(ring->fatal||owner->error){poison(ring,owner,ring->fatal?ring->fatal:owner->error);return;}
  if(*valid>cap||workspace_control[1]||workspace_control[0]>*valid){
    poison(ring,owner,21);return;
  }
  owner->stage=1;owner->survivors=0;
}
__global__ void guard_commit(const uint32_t* count,const uint32_t* shard_counts,
    const uint32_t* offsets,const uint32_t* accepted,const uint32_t* caps,
    uint32_t shards,const MgbfsStateExtent* extent,
    MgbfsStateRingControl* ring,MgbfsOwnerControl* owner){
  if(ring->fatal||owner->error){poison(ring,owner,ring->fatal?ring->fatal:owner->error);return;}
  uint32_t n=*count;
  if(owner->stage!=1||owner->survivors!=n||extent->count!=n||
      extent->granted_rows!=n||offsets[0]!=0||offsets[shards]!=n){
    poison(ring,owner,22);return;
  }
  for(uint32_t shard=0;shard<shards;++shard)
    if(offsets[shard]>offsets[shard+1]||
       offsets[shard+1]-offsets[shard]!=shard_counts[shard]||
       accepted[shard]>caps[shard]||
       shard_counts[shard]>caps[shard]-accepted[shard]){
      poison(ring,owner,22);return;
    }
}
__global__ void append_rank(const uint32_t* selected,const uint32_t* count,
    const uint32_t* candidate,uint32_t stride,uint32_t shift,uint32_t shards,
    const uint32_t* offsets,const uint32_t* accepted,const AcceptedView* views,
    const MgbfsStateExtent* extent,const MgbfsOwnerControl* owner){
  if(owner->error)return;
  uint32_t n=*count;
  for(uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;row<n;
      row+=gridDim.x*blockDim.x){
    uint32_t source=selected[row],shard=(candidate[uint64_t(3)*stride+source]>>shift)&(shards-1);
    uint32_t dest=accepted[shard]+row-offsets[shard];
    AcceptedView view=views[shard];
    for(unsigned word=0;word<4;++word)
      view.words[word][dest]=candidate[uint64_t(word)*stride+source];
    if(view.state_refs)view.state_refs[dest]=extent->sequence+row;
  }
}
__global__ void insert_rank(InsertRef* refs,const uint32_t* selected,
    const uint32_t* count,const uint32_t* candidate,uint32_t stride,
    uint32_t shift,uint32_t shards,const uint32_t* offsets,
    const uint32_t* accepted,const MgbfsOwnerControl* owner){
  if(owner->error)return;
  uint32_t n=*count;
  for(uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;row<n;
      row+=gridDim.x*blockDim.x){
    uint32_t source=selected[row],shard=(candidate[uint64_t(3)*stride+source]>>shift)&(shards-1);
    uint32_t dest=accepted[shard]+row-offsets[shard];
    refs[shard].insert(key_index(2,dest));
  }
}
__global__ void publish_counts(const uint32_t* counts,uint32_t* accepted,
    uint32_t shards,const MgbfsStateRingControl* ring,MgbfsOwnerControl* owner){
  if(ring->fatal||owner->error)return;
  for(uint32_t shard=0;shard<shards;++shard)accepted[shard]+=counts[shard];
  owner->stage=2;
}

struct alignas(16) ExportKey { uint32_t words[4]; };
struct ExportDecompose {
  __host__ __device__ auto operator()(ExportKey& k) const {
    return cuda::std::tie(k.words[3],k.words[2],k.words[1],k.words[0]);
  }
};
__global__ void export_prefix(const uint32_t* counts,const uint32_t* caps,
    uint32_t shards,uint32_t capacity,uint32_t* offsets,uint32_t* total,
    MgbfsStateRingControl* ring,MgbfsOwnerControl* owner){
  *total=0;offsets[0]=0;
  if(ring->fatal||owner->error)return;
  uint64_t n=0;
  for(uint32_t i=0;i<shards;++i){
    if(counts[i]>caps[i]){poison(ring,owner,22);return;}
    n+=counts[i];
    if(n>capacity){poison(ring,owner,22);return;}
    offsets[i+1]=uint32_t(n);
  }
  *total=uint32_t(n);
}
__global__ void export_fill(ExportKey* keys,uint64_t* refs,uint32_t capacity){
  for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<capacity;i+=gridDim.x*blockDim.x){
    keys[i]={{UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX}};
    refs[i]=UINT64_MAX;
  }
}
__global__ void export_pack(const AcceptedView* views,const uint32_t* counts,
    const uint32_t* offsets,uint32_t capacity,ExportKey* keys,uint64_t* refs,
    const MgbfsStateRingControl* ring,const MgbfsOwnerControl* owner){
  if(ring->fatal||owner->error)return;
  uint32_t shard=blockIdx.y;AcceptedView view=views[shard];
  for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<counts[shard];i+=gridDim.x*blockDim.x){
    uint32_t dest=offsets[shard]+i;
    if(dest>=capacity)return;
    for(unsigned w=0;w<4;++w)keys[dest].words[w]=view.words[w][i];
    refs[dest]=view.state_refs[i];
  }
}

uint32_t bits(uint32_t x){uint32_t n=0;for(;x>1;x>>=1)++n;return n;}
uint32_t* data(rmm::device_buffer& b){return static_cast<uint32_t*>(b.data());}
const uint32_t* data(const rmm::device_buffer& b){return static_cast<const uint32_t*>(b.data());}
}

struct CucoRankBatch::Impl {
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref resource;
  uint32_t incoming,shards,logical_owner,world,shift;
  uint64_t pending_epoch{0},committed_epoch{0};
  bool pending{false},needs_complete{false};
  bool sealed{false};
  bool empty_history{true};
  std::shared_ptr<CucoWorkspace> workspace;
  std::vector<uint32_t> capacities;
  std::vector<rmm::device_buffer> accepted_keys;
  std::vector<rmm::device_buffer> accepted_state_refs;
  std::vector<size_t> accepted_strides;
  std::vector<std::unique_ptr<Set>> sets;
  rmm::device_buffer contains_refs,insert_refs,accepted_views;
  rmm::device_buffer accepted_counts,accepted_capacities,shard_counts,shard_offsets;
  rmm::device_buffer export_keys,export_refs,export_scratch;
  uint32_t export_capacity{0};size_t export_scratch_bytes{0};

  Impl(std::vector<MgbfsLibraryKeysV1> previous,std::vector<MgbfsLibraryKeysV1> current,
      std::vector<uint32_t> caps,uint32_t cap,uint32_t logical,uint32_t world_size,
      rmm::cuda_stream_view s,rmm::device_async_resource_ref res,bool retain_refs,uint32_t settled_capacity)
      : stream(s),resource(res),incoming(cap),shards(uint32_t(caps.size())),
        logical_owner(logical),world(world_size),shift(32-bits(world_size*shards)),
        workspace(std::make_shared<CucoWorkspace>(cap,s,res)),capacities(std::move(caps)) {
    if(!incoming||incoming>INT32_MAX||!shards||shards>256||(shards&(shards-1))||
       previous.size()!=shards||current.size()!=shards||
       !world||world>128||(world&(world-1))||logical_owner>=world)
      throw std::runtime_error("RANK_OWNER_SHAPE");
    auto allocate=[&](size_t bytes){return rmm::device_buffer{bytes,s,res};};
    accepted_counts=allocate(shards*4);accepted_capacities=allocate(shards*4);
    shard_counts=allocate(shards*4);shard_offsets=allocate((shards+1)*4);
    cuco_owner_detail::check(cudaMemsetAsync(accepted_counts.data(),0,shards*4,s.value()));
    cuco_owner_detail::check(cudaMemcpyAsync(accepted_capacities.data(),capacities.data(),
        shards*4,cudaMemcpyHostToDevice,s.value()));
    std::vector<ContainsRef> contains;
    std::vector<InsertRef> inserts;
    std::vector<AcceptedView> views;
    contains.reserve(shards);inserts.reserve(shards);views.reserve(shards);
    accepted_keys.reserve(shards);accepted_state_refs.reserve(shards);accepted_strides.reserve(shards);sets.reserve(shards);
    for(uint32_t shard=0;shard<shards;++shard){
      auto old=previous[shard],now=current[shard];uint32_t capacity=capacities[shard];
      empty_history=empty_history&&old.rows==0&&now.rows==0;
      if(!capacity||capacity>INT32_MAX||old.reserved||now.reserved||
         old.rows>INT32_MAX||now.rows>INT32_MAX||
         uint64_t(old.rows)+now.rows+capacity>INT32_MAX)
        throw std::runtime_error("RANK_OWNER_CAPACITY");
      for(unsigned word=0;word<4;++word)
        if((old.rows&&!old.words[word])||(now.rows&&!now.words[word]))
          throw std::runtime_error("RANK_OWNER_HISTORY");
      size_t stride=(size_t(capacity)+63)&~size_t{63};
      accepted_keys.push_back(allocate(stride*16));accepted_strides.push_back(stride);
      AcceptedView accepted{};accepted.stride=uint32_t(stride);
      if(retain_refs){
        accepted_state_refs.push_back(allocate(stride*sizeof(uint64_t)));
        accepted.state_refs=static_cast<uint64_t*>(accepted_state_refs.back().data());
      }
      IndexKeyViews keys{};
      for(unsigned word=0;word<4;++word){
        accepted.words[word]=data(accepted_keys.back())+word*stride;
        keys.planes[0][word]=old.words[word];
        keys.planes[1][word]=now.words[word];
        keys.planes[2][word]=accepted.words[word];
        keys.planes[3][word]=data(workspace->candidates)+word*workspace->stride;
      }
      sets.push_back(cuco_owner_detail::make_set(uint64_t(old.rows)+now.rows+capacity,
          keys,res,s));
      if(old.rows)cuco_owner_detail::insert_rows<<<cuco_owner_detail::grid(old.rows),256,0,s.value()>>>(
          sets.back()->ref(cuco::insert),0,0,old.rows);
      if(now.rows)cuco_owner_detail::insert_rows<<<cuco_owner_detail::grid(now.rows),256,0,s.value()>>>(
          sets.back()->ref(cuco::insert),1,0,now.rows);
      contains.push_back(sets.back()->ref(cuco::contains));
      inserts.push_back(sets.back()->ref(cuco::insert));
      views.push_back(accepted);
    }
    cuco_owner_detail::check(cudaGetLastError());
    contains_refs=allocate(contains.size()*sizeof(ContainsRef));
    insert_refs=allocate(inserts.size()*sizeof(InsertRef));
    accepted_views=allocate(views.size()*sizeof(AcceptedView));
    cuco_owner_detail::check(cudaMemcpyAsync(contains_refs.data(),contains.data(),
        contains_refs.size(),cudaMemcpyHostToDevice,s.value()));
    cuco_owner_detail::check(cudaMemcpyAsync(insert_refs.data(),inserts.data(),
        insert_refs.size(),cudaMemcpyHostToDevice,s.value()));
    cuco_owner_detail::check(cudaMemcpyAsync(accepted_views.data(),views.data(),
        accepted_views.size(),cudaMemcpyHostToDevice,s.value()));
    if(retain_refs){
      uint64_t total=0;for(auto capacity:capacities)total+=capacity;
      if(!settled_capacity||settled_capacity>INT32_MAX||settled_capacity>total)
        throw std::runtime_error("RANK_OWNER_EXPORT_CAPACITY");
      // Shared settlement buffers are layer-sized, not sum-of-shard-capacities sized.
      export_capacity=settled_capacity;
      export_keys=allocate(size_t(export_capacity)*16);
      export_refs=allocate(size_t(export_capacity)*8);
      cuco_owner_detail::check(cub::DeviceRadixSort::SortPairs(nullptr,export_scratch_bytes,
          static_cast<const ExportKey*>(nullptr),static_cast<ExportKey*>(nullptr),
          static_cast<const uint64_t*>(nullptr),static_cast<uint64_t*>(nullptr),
          int(export_capacity),ExportDecompose{},0,128,s.value()));
      export_scratch=allocate(export_scratch_bytes);
    }
    // Setup may synchronize: constructor-local host arrays back these uploads.
    s.synchronize();
  }
};

// Keep this next to the actual constructor/workspace allocations. Queries
// don't construct a Set, launch kernels, or change the current RMM resource.
extern "C" int mgbfs_library_rank_pool_query_v1(uint32_t layer,
    uint32_t accepted,uint32_t shards,uint32_t incoming,uint64_t* result) {
  if(result)*result=0;
  if(!result||!layer||!accepted||!incoming||!shards||shards>256||
      (shards&(shards-1))||incoming>INT32_MAX||
      uint64_t(layer)*2+accepted>INT32_MAX)return -1;
  try {
    uint64_t bytes=0;
    auto charge=[&](uint64_t n){
      if(n>UINT64_MAX-255||bytes>UINT64_MAX-((n+255)&~uint64_t{255}))
        throw std::runtime_error("POOL_QUERY_OVERFLOW");
      bytes+=(n+255)&~uint64_t{255};
    };
    auto table=[&](uint64_t rows){
      using Storage=typename Set::storage_ref_type;
      auto extent=cuco::make_valid_extent<typename Set::probing_scheme_type,Storage>(
          cuco::extent<size_t>{size_t(rows*2)});
      charge(uint64_t(size_t(extent))*sizeof(uint64_t));
    };
    uint64_t stride=(uint64_t(incoming)+63)&~uint64_t{63};
    charge(stride*16); // workspace candidate SoA
    charge(uint64_t(incoming)*4); // minima
    charge(uint64_t(incoming)*4); // representatives
    charge(incoming); // flags
    charge(uint64_t(incoming)*4); // selected
    charge(uint64_t(incoming)*4); // sources
    charge(8); // control
    size_t scratch=0;
    cuco_owner_detail::check(cub::DeviceSelect::Flagged(nullptr,scratch,
        thrust::counting_iterator<uint32_t>{0},static_cast<uint8_t*>(nullptr),
        static_cast<uint32_t*>(nullptr),static_cast<uint32_t*>(nullptr),
        int(incoming),cudaStream_t{nullptr}));
    charge(scratch);
    table(incoming);
    for(uint32_t shard=0;shard<shards;++shard){
      charge(((uint64_t(accepted)+63)&~uint64_t{63})*16);
      // Histories may skew arbitrarily. Each shard can see up to two complete
      // rank layers; charge that safe bound rather than assuming uniform keys.
      table(uint64_t(layer)*2+accepted);
    }
    charge(uint64_t(shards)*sizeof(ContainsRef));
    charge(uint64_t(shards)*sizeof(InsertRef));
    charge(uint64_t(shards)*sizeof(AcceptedView));
    charge(uint64_t(shards)*4);charge(uint64_t(shards)*4);
    charge(uint64_t(shards)*4);charge(uint64_t(shards+1)*4);
    *result=bytes;
    return 0;
  } catch (...) {return -1;}
}

CucoRankBatch::CucoRankBatch(std::vector<MgbfsLibraryKeysV1> previous,
    std::vector<MgbfsLibraryKeysV1> current,std::vector<uint32_t> caps,
    uint32_t incoming,uint32_t logical_owner,uint32_t world,
    rmm::cuda_stream_view stream,rmm::device_async_resource_ref resource,bool retain_refs,uint32_t settled_capacity)
    : impl_(std::make_unique<Impl>(std::move(previous),std::move(current),
          std::move(caps),incoming,logical_owner,world,stream,resource,retain_refs,settled_capacity)) {}
CucoRankBatch::~CucoRankBatch()=default;

CucoRankDeviceBatch CucoRankBatch::compare(uint64_t epoch,MgbfsLibraryCandidatesV1 input,
    const uint32_t* valid,MgbfsOwnerControl* owner,MgbfsStateRingControl* ring){
  auto& p=*impl_;
  if(p.sealed||p.pending||p.needs_complete||!valid||!owner||!ring||
     input.keys.rows!=p.incoming||input.keys.reserved||!input.source_indices||
     (p.committed_epoch&&epoch<=p.committed_epoch))
    throw std::runtime_error("RANK_OWNER_ORDER_OR_INPUT");
  for(auto word:input.keys.words)if(!word)throw std::runtime_error("RANK_OWNER_INPUT_KEYS");
  auto s=p.stream.value();uint32_t cap=p.incoming;
  p.workspace->transient->clear_async(cuda::stream_ref{s});
  copy_candidates<<<cuco_owner_detail::grid(cap),256,0,s>>>(input.keys,valid,cap,
      data(p.workspace->candidates),uint32_t(p.workspace->stride),ring,owner);
  cuco_owner_detail::check(cudaMemsetAsync(p.workspace->minima.data(),0xff,cap*4,s));
  cuco_owner_detail::check(cudaMemsetAsync(p.workspace->flags.data(),0,cap,s));
  cuco_owner_detail::check(cudaMemsetAsync(p.workspace->control.data(),0,8,s));
  insert_candidates<<<cuco_owner_detail::grid(cap),256,0,s>>>(
      p.workspace->transient->ref(cuco::insert),valid,cap,owner);
  representatives<<<cuco_owner_detail::grid(cap),256,0,s>>>(
      p.workspace->transient->ref(cuco::find),valid,cap,
      data(p.workspace->representatives),data(p.workspace->minima),
      data(p.workspace->control)+1,owner);
  rank_flags<<<cuco_owner_detail::grid(cap),256,0,s>>>(
      static_cast<ContainsRef const*>(p.contains_refs.data()),
      data(p.workspace->candidates),valid,cap,uint32_t(p.workspace->stride),
      p.shift,p.logical_owner,p.shards,data(p.workspace->representatives),
      data(p.workspace->minima),data(p.workspace->control)+1,
      static_cast<uint8_t*>(p.workspace->flags.data()),ring,owner);
  size_t scratch=p.workspace->scratch_bytes;
  cuco_owner_detail::check(cub::DeviceSelect::Flagged(p.workspace->scratch.data(),scratch,
      thrust::counting_iterator<uint32_t>{0},
      static_cast<uint8_t*>(p.workspace->flags.data()),data(p.workspace->selected),
      data(p.workspace->control),int(cap),s));
  gather_owner_sources<<<cuco_owner_detail::grid(cap),256,0,s>>>(
      data(p.workspace->selected),data(p.workspace->control),cap,input.source_indices,
      data(p.workspace->sources));
  finish_compare<<<1,1,0,s>>>(valid,cap,data(p.workspace->control),ring,owner);
  cuco_owner_detail::check(cudaGetLastError());
  p.pending=true;p.pending_epoch=epoch;
  return {data(p.workspace->candidates)+3*p.workspace->stride,valid,
      data(p.workspace->selected),data(p.workspace->control),
      data(p.workspace->sources),
      data(p.accepted_counts),data(p.accepted_capacities),
      data(p.shard_counts),data(p.shard_offsets)};
}

void CucoRankBatch::commit(uint64_t epoch,MgbfsOwnerControl* owner,
    MgbfsStateRingControl* ring,const MgbfsStateExtent* extent){
  auto& p=*impl_;
  if(!p.pending||p.pending_epoch!=epoch||!owner||!ring||!extent)
    throw std::runtime_error("RANK_OWNER_ORDER");
  auto s=p.stream.value();uint32_t cap=p.incoming;
  guard_commit<<<1,1,0,s>>>(data(p.workspace->control),data(p.shard_counts),
      data(p.shard_offsets),data(p.accepted_counts),data(p.accepted_capacities),
      p.shards,extent,ring,owner);
  append_rank<<<cuco_owner_detail::grid(cap),256,0,s>>>(
      data(p.workspace->selected),data(p.workspace->control),
      data(p.workspace->candidates),uint32_t(p.workspace->stride),p.shift,p.shards,
      data(p.shard_offsets),data(p.accepted_counts),
      static_cast<AcceptedView const*>(p.accepted_views.data()),extent,owner);
  insert_rank<<<cuco_owner_detail::grid(cap),256,0,s>>>(
      static_cast<InsertRef*>(p.insert_refs.data()),data(p.workspace->selected),
      data(p.workspace->control),data(p.workspace->candidates),
      uint32_t(p.workspace->stride),p.shift,p.shards,data(p.shard_offsets),
      data(p.accepted_counts),owner);
  publish_counts<<<1,1,0,s>>>(data(p.shard_counts),data(p.accepted_counts),p.shards,ring,owner);
  cuco_owner_detail::check(cudaGetLastError());
  p.pending=false;p.needs_complete=true;p.committed_epoch=epoch;
}

void CucoRankBatch::complete(uint64_t epoch){
  auto& p=*impl_;
  if(!p.needs_complete||p.committed_epoch!=epoch)throw std::runtime_error("RANK_OWNER_ORDER");
  p.needs_complete=false;
}



void CucoRankBatch::export_sorted(void* keys,uint64_t* refs,uint32_t* count,
    uint32_t capacity,MgbfsStateRingControl* ring,MgbfsOwnerControl* owner){
  auto& p=*impl_;
  if(p.pending||p.needs_complete||p.sealed||!p.export_capacity||
      !capacity||capacity>p.export_capacity||!keys||!refs||!count||!ring||!owner)
    throw std::runtime_error("RANK_OWNER_SORT_EXPORT_ORDER");
  // The preallocated export arena is the physical upper bound; callers may
  // impose a smaller logical layer budget. Let the GPU prefix guard report
  // overflow through shared owner/ring fatal state, not a host ABI exception.
  auto stream=p.stream.value();
  export_prefix<<<1,1,0,stream>>>(data(p.accepted_counts),data(p.accepted_capacities),
      p.shards,capacity,data(p.shard_offsets),count,ring,owner);
  export_fill<<<cuco_owner_detail::grid(capacity),256,0,stream>>>(
      static_cast<ExportKey*>(p.export_keys.data()),
      static_cast<uint64_t*>(p.export_refs.data()),capacity);
  export_pack<<<dim3(cuco_owner_detail::grid(capacity),p.shards),256,0,stream>>>(
      static_cast<const AcceptedView*>(p.accepted_views.data()),data(p.accepted_counts),
      data(p.shard_offsets),capacity,static_cast<ExportKey*>(p.export_keys.data()),
      static_cast<uint64_t*>(p.export_refs.data()),ring,owner);
  cuco_owner_detail::check(cudaGetLastError());
  size_t bytes=p.export_scratch_bytes;
  // Stable SortPairs keeps real maximum-valued keys before trailing padding.
  // Only *count valid records are passed to the existing weighted settlement.
  cuco_owner_detail::check(cub::DeviceRadixSort::SortPairs(p.export_scratch.data(),bytes,
      static_cast<const ExportKey*>(p.export_keys.data()),static_cast<ExportKey*>(keys),
      static_cast<const uint64_t*>(p.export_refs.data()),refs,int(capacity),
      ExportDecompose{},0,128,stream));
}

const uint64_t* CucoRankBatch::export_state_refs(uint32_t shard) const {
  auto const& p=*impl_;
  if(shard>=p.shards||p.accepted_state_refs.size()!=p.shards||p.pending||p.needs_complete)
    throw std::runtime_error("RANK_OWNER_STATE_REF_EXPORT");
  return static_cast<const uint64_t*>(p.accepted_state_refs[shard].data());
}

void CucoRankBatch::reset_empty_history(){
  auto& p=*impl_;
  // Caller must retire every exported reader before reuse on this owner stream.
  // Tables with borrowed old/current history cannot be reset as empty targets.
  if(!p.empty_history||p.sealed||p.pending||p.needs_complete)
    throw std::runtime_error("RANK_OWNER_RESET_ORDER");
  for(auto& set:p.sets)set->clear_async(cuda::stream_ref{p.stream.value()});
  cuco_owner_detail::check(cudaMemsetAsync(p.accepted_counts.data(),0,p.shards*4,p.stream.value()));
  // Epoch monotonicity is retained across slot generations.
}

void CucoRankBatch::seal(){
  auto& p=*impl_;
  if(p.sealed||p.pending||p.needs_complete)
    throw std::runtime_error("RANK_OWNER_SEAL_ORDER");
  // FinalizeDepth caller has drained the stream and all borrowed readers.
  // Accepted key planes are owned separately and remain exportable.
  p.sets.clear();
  p.sealed=true;
}

MgbfsLibraryKeysV1 CucoRankBatch::export_shard(uint32_t shard,uint32_t rows) const {
  auto const& p=*impl_;
  if(shard>=p.shards||rows>p.capacities[shard]||p.pending)
    throw std::runtime_error("RANK_OWNER_EXPORT");
  MgbfsLibraryKeysV1 out{};out.rows=rows;
  for(unsigned word=0;word<4;++word)
    out.words[word]=data(p.accepted_keys[shard])+word*p.accepted_strides[shard];
  return out;
}
}
