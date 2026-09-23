#include "cuco_owner.cuh"
#include "cuco_rank_batch.cuh"
#include "owner_handle.hpp"
#include <rmm/mr/per_device_resource.hpp>

namespace {
struct WorkspaceHandle {
  rmm::device_async_resource_ref resource;
  std::shared_ptr<mgbfs::CucoWorkspace> workspace;
  WorkspaceHandle(uint32_t incoming, rmm::cuda_stream_view stream)
      : resource(rmm::mr::get_current_device_resource_ref()),
        workspace(std::make_shared<mgbfs::CucoWorkspace>(incoming, stream, resource)) {}
  void check() const {
    int device;
    mgbfs::cuco_owner_detail::check(cudaGetDevice(&device));
    if (device != workspace->device || resource != rmm::mr::get_current_device_resource_ref())
      throw std::runtime_error("WORKSPACE_DEVICE_RESOURCE");
  }
};
struct CucoHandle final : mgbfs::LibraryOwnerHandle {
  mgbfs::CucoOwner owner;
  CucoHandle(MgbfsLibraryKeysV1 previous, MgbfsLibraryKeysV1 current,
      uint32_t capacity, uint32_t incoming, rmm::cuda_stream_view stream,
      std::shared_ptr<mgbfs::CucoWorkspace> workspace = {})
      : owner(previous, current, capacity, incoming, stream,
              rmm::mr::get_current_device_resource_ref(), std::move(workspace)) {}
  MgbfsLibrarySurvivorsV1 compare(uint64_t epoch, MgbfsLibraryCandidatesV1 input) override {
    return owner.compare(epoch, input);
  }
  void commit(uint64_t epoch, uint32_t granted) override { owner.commit(epoch, granted); }
  void complete(uint64_t epoch) override { owner.complete(epoch); }
  MgbfsLibraryKeysV1 export_committed() override { return owner.export_committed(); }
  void seal() override { owner.seal(); }
};
}
extern "C" int mgbfs_library_cuco_workspace_create_v1(uint32_t incoming,
    void* stream, void** output) {
  if (!output) return -1;
  *output = nullptr;
  try {
    *output = new WorkspaceHandle(incoming,
        rmm::cuda_stream_view{static_cast<cudaStream_t>(stream)});
    return 0;
  } catch (...) { return -1; }
}
extern "C" int mgbfs_library_cuco_workspace_destroy_v1(void* workspace) {
  if (!workspace) return -1;
  try {
    auto* handle = static_cast<WorkspaceHandle*>(workspace);
    handle->check();
    if (handle->workspace.use_count() != 1) return -1;
    handle->workspace->lease.check_idle();
    delete handle;
    return 0;
  } catch (...) { return -1; }
}
extern "C" int mgbfs_library_owner_create_cuco_shared_v1(MgbfsLibraryKeysV1 previous,
    MgbfsLibraryKeysV1 current, uint32_t capacity, void* workspace, void** output) {
  if (!output) return -1;
  *output = nullptr;
  if (!workspace) return -1;
  try {
    auto* handle = static_cast<WorkspaceHandle*>(workspace);
    handle->check();
    auto shared = handle->workspace;
    *output = static_cast<mgbfs::LibraryOwnerHandle*>(new CucoHandle(previous, current,
        capacity, shared->incoming, shared->stream, shared));
    return 0;
  } catch (...) { return -1; }
}
extern "C" int mgbfs_library_owner_create_cuco_window_v1(MgbfsLibraryKeysV1 previous,
    MgbfsLibraryKeysV1 current, uint32_t capacity, uint32_t incoming,
    void* stream, void** output) {
  if (!output) return -1;
  *output = nullptr;
  try {
    *output = static_cast<mgbfs::LibraryOwnerHandle*>(new CucoHandle(previous, current,
        capacity, incoming, rmm::cuda_stream_view{static_cast<cudaStream_t>(stream)}));
    return 0;
  } catch (...) { return -1; }
}

namespace {
struct RankHandle {
  int device;
  rmm::device_async_resource_ref resource;
  mgbfs::CucoRankBatch batch;
  RankHandle(std::vector<MgbfsLibraryKeysV1> previous,
      std::vector<MgbfsLibraryKeysV1> current,std::vector<uint32_t> capacities,
      uint32_t incoming,uint32_t logical_owner,uint32_t world,
      rmm::cuda_stream_view stream)
      : device(current_device()),resource(rmm::mr::get_current_device_resource_ref()),
        batch(std::move(previous),std::move(current),std::move(capacities),
            incoming,logical_owner,world,stream,resource) {}
  static int current_device(){
    int id=-1;
    mgbfs::cuco_owner_detail::check(cudaGetDevice(&id));
    return id;
  }
  void check() const {
    if(current_device()!=device||resource!=rmm::mr::get_current_device_resource_ref())
      throw std::runtime_error("RANK_OWNER_DEVICE_RESOURCE");
  }
};
}
extern "C" int mgbfs_library_rank_create_cuco_v1(const MgbfsLibraryKeysV1* previous,
    const MgbfsLibraryKeysV1* current,const uint32_t* capacities,
    uint32_t shards,uint32_t incoming,uint32_t logical_owner,uint32_t world,
    void* stream,void** output){
  if(!output)return -1;
  *output=nullptr;
  if(!previous||!current||!capacities||!shards||shards>256)return -1;
  try {
    std::vector<MgbfsLibraryKeysV1> old(previous,previous+shards);
    std::vector<MgbfsLibraryKeysV1> now(current,current+shards);
    std::vector<uint32_t> caps(capacities,capacities+shards);
    *output=new RankHandle(std::move(old),std::move(now),std::move(caps),
        incoming,logical_owner,world,
        rmm::cuda_stream_view{static_cast<cudaStream_t>(stream)});
    return 0;
  } catch (...) {return -1;}
}
extern "C" int mgbfs_library_rank_compare_v1(void* handle,uint64_t epoch,
    MgbfsLibraryCandidatesV1 input,const uint32_t* valid_rows,
    MgbfsOwnerControl* owner,MgbfsStateRingControl* ring,
    MgbfsLibraryRankDeviceBatchV1* output){
  if(output)*output={};
  if(!handle||!output)return -1;
  try {
    auto* h=static_cast<RankHandle*>(handle);
    h->check();
    auto d=h->batch.compare(epoch,input,valid_rows,owner,ring);
    *output={d.high_words,d.valid_rows,d.selected,d.selected_count,
        d.source_indices,d.accepted_counts,d.accepted_capacities,
        d.shard_counts,d.shard_offsets};
    return 0;
  } catch (...) {return -1;}
}
extern "C" int mgbfs_library_rank_commit_v1(void* handle,uint64_t epoch,
    MgbfsOwnerControl* owner,MgbfsStateRingControl* ring,
    const MgbfsStateExtent* extent){
  if(!handle)return -1;
  try {
    auto* h=static_cast<RankHandle*>(handle);
    h->check();h->batch.commit(epoch,owner,ring,extent);
    return 0;
  } catch (...) {return -1;}
}
extern "C" int mgbfs_library_rank_complete_v1(void* handle,uint64_t epoch){
  if(!handle)return -1;
  try {
    auto* h=static_cast<RankHandle*>(handle);
    h->check();h->batch.complete(epoch);
    return 0;
  } catch (...) {return -1;}
}
extern "C" int mgbfs_library_rank_export_shard_v1(void* handle,uint32_t shard,
    uint32_t rows,MgbfsLibraryKeysV1* output){
  if(output)*output={};
  if(!handle||!output)return -1;
  try {
    auto* h=static_cast<RankHandle*>(handle);
    h->check();*output=h->batch.export_shard(shard,rows);
    return 0;
  } catch (...) {return -1;}
}
extern "C" int mgbfs_library_rank_destroy_v1(void* handle){
  if(!handle)return -1;
  try {
    auto* h=static_cast<RankHandle*>(handle);
    h->check();delete h;
    return 0;
  } catch (...) {return -1;}
}
