#include "cuco_rank_batch.cuh"
#include <cub/device/device_select.cuh>
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

struct AcceptedView { uint32_t* words[4]; uint32_t stride; };

__device__ void poison(MgbfsStateRingControl* ring,MgbfsOwnerControl* owner,uint32_t code){
  atomicCAS(&owner->error,0u,code);
  atomicCAS(&ring->fatal,0u,code);
}
__global__ void copy_candidates(MgbfsLibraryKeysV1 input,const uint32_t* valid,
    uint32_t cap,uint32_t* output,uint32_t stride,MgbfsStateRingControl* ring,
    MgbfsOwnerControl* owner){
  if(blockIdx.x==0&&threadIdx.x==0&&*valid>cap)poison(ring,owner,21);
  uint32_t n=*valid<cap?*valid:cap;
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
    const MgbfsOwnerControl* owner){
  if(owner->error)return;
  uint32_t n=*count;
  for(uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;row<n;
      row+=gridDim.x*blockDim.x){
    uint32_t source=selected[row],shard=(candidate[uint64_t(3)*stride+source]>>shift)&(shards-1);
    uint32_t dest=accepted[shard]+row-offsets[shard];
    AcceptedView view=views[shard];
    for(unsigned word=0;word<4;++word)
      view.words[word][dest]=candidate[uint64_t(word)*stride+source];
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
  std::shared_ptr<CucoWorkspace> workspace;
  std::vector<uint32_t> capacities;
  std::vector<rmm::device_buffer> accepted_keys;
  std::vector<size_t> accepted_strides;
  std::vector<std::unique_ptr<Set>> sets;
  rmm::device_buffer contains_refs,insert_refs,accepted_views;
  rmm::device_buffer accepted_counts,accepted_capacities,shard_counts,shard_offsets;

  Impl(std::vector<MgbfsLibraryKeysV1> previous,std::vector<MgbfsLibraryKeysV1> current,
      std::vector<uint32_t> caps,uint32_t cap,uint32_t logical,uint32_t world_size,
      rmm::cuda_stream_view s,rmm::device_async_resource_ref res)
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
    accepted_keys.reserve(shards);accepted_strides.reserve(shards);sets.reserve(shards);
    for(uint32_t shard=0;shard<shards;++shard){
      auto old=previous[shard],now=current[shard];uint32_t capacity=capacities[shard];
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
    // Setup may synchronize: constructor-local host arrays back these uploads.
    s.synchronize();
  }
};

CucoRankBatch::CucoRankBatch(std::vector<MgbfsLibraryKeysV1> previous,
    std::vector<MgbfsLibraryKeysV1> current,std::vector<uint32_t> caps,
    uint32_t incoming,uint32_t logical_owner,uint32_t world,
    rmm::cuda_stream_view stream,rmm::device_async_resource_ref resource)
    : impl_(std::make_unique<Impl>(std::move(previous),std::move(current),
          std::move(caps),incoming,logical_owner,world,stream,resource)) {}
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
      static_cast<AcceptedView const*>(p.accepted_views.data()),owner);
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
