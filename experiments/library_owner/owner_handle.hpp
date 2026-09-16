#pragma once
#include "owner_abi.h"

namespace mgbfs {
// Internal C++ dispatch only; never crosses the versioned C ABI as a struct.
// All factories return this base pointer. Epoch/error semantics stay in the
// common boundary, so selecting a library does not change the runtime protocol.
struct LibraryOwnerHandle {
  uint64_t epoch{0}, last_epoch{0};
  bool pending{false}, has_last{false}, poisoned{false};
  bool completion_pending{false};
  virtual ~LibraryOwnerHandle() = default;
  virtual MgbfsLibrarySurvivorsV1 compare(uint64_t epoch, MgbfsLibraryCandidatesV1 input) = 0;
  virtual void commit(uint64_t epoch, uint32_t granted) = 0;
  virtual void complete(uint64_t) {}
  virtual MgbfsLibraryKeysV1 export_committed() = 0;
  virtual void seal() = 0;
};
}
