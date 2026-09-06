// Compile the actual production wrapper against a deterministic NCCL test double.
// This tests wrapper cleanup, not NCCL correctness or GPU communication.
#include <cassert>
#include "../cuda/nccl_transport.cpp"
int fail_stage = 0, group_depth = 0, send_calls = 0, recv_calls = 0, end_calls = 0;
int abort_calls = 0, destroy_calls = 0;
int main() {
  void* comm = nullptr;
  ncclUniqueId id{};
  assert(mgbfs_nccl_create(0, 2, 0, &id, &comm, nullptr, 0) == 0);
  char byte = 0;
  for (int stage = 0; stage <= 4; ++stage) {
    fail_stage = stage;
    group_depth = send_calls = recv_calls = end_calls = 0;
    const int result = mgbfs_nccl_send_recv(comm, &byte, 1, 1, &byte, 1, nullptr);
    assert((result == 0) == (stage == 0));
    assert(group_depth == 0); // Every successful start must have one end.
    assert(end_calls == (stage == 1 ? 0 : 1));
    assert(send_calls == (stage == 1 ? 0 : 1));
    assert(recv_calls == (stage == 1 || stage == 2 ? 0 : 1));
  }
  assert(mgbfs_nccl_abort(comm) == 0);
  assert(abort_calls == 1);
  assert(mgbfs_nccl_abort(comm) == 0);
  assert(abort_calls == 1); // Repeated abort must not reuse a freed NCCL handle.
  group_depth = send_calls = recv_calls = end_calls = 0;
  assert(mgbfs_nccl_send_recv(comm, &byte, 1, 1, &byte, 1, nullptr) != 0);
  assert(send_calls == 0 && recv_calls == 0 && end_calls == 0);
  uint32_t word = 0;
  assert(mgbfs_nccl_all_gather_u32(comm, &word, &word, nullptr) != 0);
  assert(mgbfs_nccl_all_reduce_max_u32(comm, &word, &word, nullptr) != 0);
  mgbfs_nccl_destroy(comm);
  assert(destroy_calls == 0); // Wrapper deletion must not destroy an aborted handle.
  assert(mgbfs_nccl_abort(nullptr) != 0);
}
