#include "cuco_owner.cuh"
#include "owner_handle.hpp"
#include <rmm/mr/per_device_resource.hpp>

namespace {
struct CucoHandle final : mgbfs::LibraryOwnerHandle {
  mgbfs::CucoOwner owner;
  CucoHandle(MgbfsLibraryKeysV1 previous, MgbfsLibraryKeysV1 current,
      uint32_t capacity, uint32_t incoming, rmm::cuda_stream_view stream)
      : owner(previous, current, capacity, incoming, stream,
              rmm::mr::get_current_device_resource_ref()) {}
  MgbfsLibrarySurvivorsV1 compare(uint64_t epoch, MgbfsLibraryCandidatesV1 input) override {
    return owner.compare(epoch, input);
  }
  void commit(uint64_t epoch, uint32_t granted) override { owner.commit(epoch, granted); }
  MgbfsLibraryKeysV1 export_committed() override { return owner.export_committed(); }
  void seal() override { owner.seal(); }
};
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
