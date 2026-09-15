#include "owner_abi.h"

// RED fixture only: replaced by the adapter after conformance failure is observed.
extern "C" int mgbfs_library_owner_create_v1(MgbfsLibraryKeysV1, uint32_t,
                                             void*, void** owner) {
  if (owner) *owner = nullptr;
  return -1;
}
extern "C" int mgbfs_library_owner_compare_v1(void*, uint64_t,
    MgbfsLibraryCandidatesV1, MgbfsLibrarySurvivorsV1*) { return -1; }
extern "C" int mgbfs_library_owner_commit_v1(void*, uint64_t, uint32_t) { return -1; }
extern "C" void mgbfs_library_owner_destroy_v1(void*) {}
