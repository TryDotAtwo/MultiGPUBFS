#include "cuco_owner.cuh"
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
