#include "../experiments/library_owner/cuco_workspace_lease.hpp"
#include <stdexcept>

template<class F> void rejects(F fn) {
  bool rejected = false;
  try { fn(); } catch (std::runtime_error const&) { rejected = true; }
  if (!rejected) throw std::runtime_error("EXPECTED_REJECTION");
}

int main(int argc, char**) {
  // Harness self-check: an operation that unexpectedly succeeds must fail the
  // fixture even when Release defines NDEBUG. No production code is bypassed.
  if (argc > 1) {
    try { rejects([] {}); } catch (std::runtime_error const&) { return 2; }
    return 0;
  }
  mgbfs::CucoWorkspaceLease lease;
  int first, second;
  lease.acquire(&first, 10);
  rejects([&] { lease.acquire(&second, 11); });
  rejects([&] { lease.complete(&first, 10); }); // Compare is not commit.
  rejects([&] { lease.commit(&second, 10); });
  rejects([&] { lease.commit(&first, 9); });
  lease.commit(&first, 10);
  rejects([&] { lease.acquire(&second, 11); }); // Commit does not end readers.
  rejects([&] { lease.complete(&second, 10); });
  rejects([&] { lease.complete(&first, 11); });
  lease.complete(&first, 10); // Caller has drained every GPU reader.
  rejects([&] { lease.complete(&first, 10); });
  lease.acquire(&second, 11);
  lease.commit(&second, 11);
  lease.complete(&second, 11);
  rejects([&] { lease.acquire(nullptr, 12); });
  lease.acquire(&first, 12);
  lease.abort(&second); // An unrelated destructor cannot revoke a live lease.
  rejects([&] { lease.acquire(&second, 13); });
  lease.abort(&first);
  rejects([&] { lease.commit(&first, 12); });
  rejects([&] { lease.acquire(&second, 13); }); // Fatal, never recycle after error.
}
