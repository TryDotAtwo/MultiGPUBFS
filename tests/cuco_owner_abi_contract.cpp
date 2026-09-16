// Exercise the SAME compare/commit/export/destroy ABI with the cuco factory.
#include "../experiments/library_owner/owner_abi.h"
#include <stdexcept>
#include <iostream>
#include <cuda_runtime_api.h>
void require(bool value, char const* message) {
  if (!value) throw std::runtime_error(message);
}
int main() {
  void* pool = nullptr;
  require(mgbfs_library_pool_create_v1(64ULL << 20, 1ULL << 30, &pool) == 0, "POOL_CREATE");
  void* owner = reinterpret_cast<void*>(1);
  require(mgbfs_library_owner_create_cuco_window_v1({}, {}, 8, 0, nullptr, &owner) != 0 &&
          owner == nullptr, "INVALID_FACTORY_CLEARS_HANDLE");
  require(mgbfs_library_owner_create_cuco_window_v1({}, {}, 8, 16, nullptr, &owner) == 0 &&
          owner != nullptr, "CUCO_FACTORY");
  MgbfsLibrarySurvivorsV1 result{};
  require(mgbfs_library_owner_compare_v1(owner, 12, {}, &result) == 0 &&
          result.rows == 0 && result.epoch == 12, "CUCO_COMMON_COMPARE");
  require(mgbfs_library_owner_commit_v1(owner, 12, 0) == 0, "CUCO_COMMON_COMMIT");
  // Drain constructor initialization too; this is an ABI dispatch check, not
  // a replacement for the nonempty GPU owner fixture.
  require(cudaStreamSynchronize(nullptr) == cudaSuccess, "DRAIN");
  require(mgbfs_library_owner_seal_v1(owner) == 0, "CUCO_COMMON_SEAL");
  MgbfsLibraryKeysV1 keys{};
  require(mgbfs_library_owner_export_v1(owner, &keys) == 0 && keys.rows == 0,
          "CUCO_COMMON_EXPORT");
  mgbfs_library_owner_destroy_v1(owner);
  require(cudaStreamSynchronize(nullptr) == cudaSuccess, "TEARDOWN_DRAIN");
  require(mgbfs_library_pool_destroy_v1(pool) == 0, "CUCO_COMMON_DESTROY");
  std::cout << "CUCO_COMMON_ABI_PASS\n";
}
