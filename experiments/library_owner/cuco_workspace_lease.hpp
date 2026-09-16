#pragma once
#include <cstdint>
#include <stdexcept>

namespace mgbfs {
// Host-side, externally serialized stream ownership. This does not observe
// device completion: the caller must drain every reader before complete().
class CucoWorkspaceLease {
 public:
  void acquire(void const* owner, uint64_t epoch) {
    check_idle();
    if (!owner) throw std::runtime_error("WORKSPACE_NULL_OWNER");
    owner_ = owner;
    epoch_ = epoch;
    committed_ = false;
  }
  void commit(void const* owner, uint64_t epoch) {
    check_owner(owner, epoch);
    if (committed_) throw std::runtime_error("WORKSPACE_ALREADY_COMMITTED");
    committed_ = true;
  }
  void complete(void const* owner, uint64_t epoch) {
    check_owner(owner, epoch);
    if (!committed_) throw std::runtime_error("WORKSPACE_NOT_COMMITTED");
    owner_ = nullptr;
    committed_ = false;
  }
  void check_idle() const {
    if (poisoned_ || owner_) throw std::runtime_error("WORKSPACE_UNAVAILABLE");
  }
  void abort(void const* owner) noexcept {
    if (owner && owner_ == owner) poisoned_ = true;
  }
 private:
  void check_owner(void const* owner, uint64_t epoch) const {
    if (poisoned_ || !owner || owner_ != owner || epoch_ != epoch)
      throw std::runtime_error("WORKSPACE_OWNER_EPOCH");
  }
  void const* owner_{nullptr};
  uint64_t epoch_{0};
  bool committed_{false}, poisoned_{false};
};
}
