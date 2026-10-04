// Compile the actual production wrapper against a deterministic NCCL test double.
// This tests wrapper cleanup, not NCCL correctness or GPU communication.
#include <cassert>
#include "../cuda/nccl_transport.cpp"
int fail_stage = 0, group_depth = 0, send_calls = 0, recv_calls = 0, end_calls = 0;
int abort_calls = 0, destroy_calls = 0, finalize_calls = 0;
int async_state = 0, async_query_status = 0;
int async_pending_queries = 0;
int group_end_override = -1;
int init_blocking = -1, init_calls = 0;
int last_send_peer = -1, last_recv_peer = -1;
int cancel_after = -1;
int cancellation_probe(void*) { return cancel_after == 0 || (cancel_after > 0 && --cancel_after == 0); }
struct RetirementFixture { int published=0, queries=0; };
int retirement_probe(void* context,int publish) {
  auto& fixture=*static_cast<RetirementFixture*>(context);
  assert(publish>=0);
  if(publish)++fixture.published;
  else ++fixture.queries;
  return fixture.queries>=3;
}
extern "C" int mgbfs_nccl_create_with_cancel(uint32_t,uint32_t,uint32_t,
    const void*,void**,char*,size_t,int (*)(void*),void*) __attribute__((weak));
int main() {
  void* comm = nullptr;
  ncclUniqueId id{};
  assert(mgbfs_nccl_create(0, 2, 0, &id, &comm, nullptr, 0) == 0);
  assert(init_calls == 1 && init_blocking == 0);
  // The registration diagnostic must preserve the terminal code, not the
  // initial ncclInProgress submission. This exercises the production waiter.
  ncclResult_t observed=ncclSuccess;
  async_state=9;
  assert(await_nccl(static_cast<Comm*>(comm),ncclInProgress,&observed)==9);
  assert(observed==9);
  async_state=ncclSuccess;
  async_query_status=8;
  assert(await_nccl(static_cast<Comm*>(comm),ncclInProgress,&observed)==8);
  assert(observed==8);
  async_query_status=0;
  assert(await_nccl(static_cast<Comm*>(comm),5,&observed)==6);
  assert(observed==5);
  assert(await_nccl(static_cast<Comm*>(comm),ncclSuccess,&observed)==0);
  assert(observed==ncclSuccess);
  char byte = 0;
  // Failed LSA retirement intentionally retains the communicator handle.
  // Terminal admission must therefore depend on state, not only a null handle.
  auto* terminal = static_cast<Comm*>(comm);
  terminal->terminal_started = true;
  // Failed retirement may retain a non-null handle. A progress waiter must
  // not treat that retained communicator as available for another operation.
  assert(await_nccl(terminal, ncclSuccess, &observed) != 0);
  uint32_t terminal_word = 0;
  const uint64_t terminal_sizes[] = {0, 1};
  group_depth = send_calls = recv_calls = end_calls = 0;
  assert(mgbfs_nccl_send_recv(comm, &byte, 1, 1, &byte, 1, nullptr) != 0);
  assert(mgbfs_nccl_scatter(comm, 0, &byte, 1, terminal_sizes, nullptr, 0, 0, nullptr) != 0);
  assert(mgbfs_nccl_all_gather_u32(comm, &terminal_word, &terminal_word, nullptr) != 0);
  assert(mgbfs_nccl_all_reduce_max_u32(comm, &terminal_word, &terminal_word, nullptr) != 0);
  assert(send_calls == 0 && recv_calls == 0 && end_calls == 0 && group_depth == 0);
  terminal->terminal_started = false; // Test fixture resumes the separate healthy cases.
  assert(mgbfs_nccl_poll(comm) == 0);
  async_state = ncclInProgress;
  assert(mgbfs_nccl_poll(comm) == 4); // In flight is not a terminal NCCL error.
  async_state = 9;
  assert(mgbfs_nccl_poll(comm) == 3);
  async_state = 0;
  async_query_status = 8;
  assert(mgbfs_nccl_poll(comm) != 0);
  async_query_status = 0;
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
  // An operation error does not authorize skipping GroupEnd's asynchronous
  // progress protocol. Preserve the first error, but settle the open group
  // before the dispatcher can issue another NCCL operation or abort.
  for (int stage : {2, 3}) {
    fail_stage = stage;
    group_end_override = ncclInProgress;
    async_state = ncclInProgress;
    async_pending_queries = 3;
    assert(mgbfs_nccl_send_recv(comm, &byte, 1, 1, &byte, 1, nullptr) == stage + 1);
    assert(async_pending_queries == 0);
    assert(group_depth == 0);
  }
  group_end_override = -1;
  async_state = ncclInProgress;
  async_pending_queries = 3;
  fail_stage = 0;
  assert(mgbfs_nccl_send_recv(comm, &byte, 1, 1, &byte, 1, nullptr) == 0);
  assert(async_pending_queries == 0);
  assert(mgbfs_nccl_bind_cancel(comm, cancellation_probe, &cancel_after) == 0);
  async_state = ncclInProgress;
  async_pending_queries = 0;
  cancel_after = 2;
  assert(mgbfs_nccl_send_recv(comm, &byte, 1, 1, &byte, 1, nullptr) == 7);
  assert(abort_calls == 1);
  assert(mgbfs_nccl_abort(comm) == 0);
  assert(abort_calls == 1);
  assert(mgbfs_nccl_abort(comm) == 0);
  assert(abort_calls == 1); // Repeated abort must not reuse a freed NCCL handle.
  assert(mgbfs_nccl_poll(comm) != 0);
  assert(mgbfs_nccl_poll(nullptr) != 0);
  group_depth = send_calls = recv_calls = end_calls = 0;
  assert(mgbfs_nccl_send_recv(comm, &byte, 1, 1, &byte, 1, nullptr) != 0);
  assert(send_calls == 0 && recv_calls == 0 && end_calls == 0);
  uint32_t word = 0;
  assert(mgbfs_nccl_all_gather_u32(comm, &word, &word, nullptr) != 0);
  assert(mgbfs_nccl_all_reduce_max_u32(comm, &word, &word, nullptr) != 0);
  mgbfs_nccl_destroy(comm);
  assert(destroy_calls == 0); // Wrapper deletion must not destroy an aborted handle.
  assert(mgbfs_nccl_abort(nullptr) != 0);
  fail_stage = 0;
  void* source = nullptr;
  async_state = ncclInProgress;
  async_pending_queries = 3;
  assert(mgbfs_nccl_create(2, 3, 2, &id, &source, nullptr, 0) == 0);
  assert(async_pending_queries == 0 && init_calls == 2 && init_blocking == 0);
  const uint64_t sizes[] = {2, 0, 3};
  char payload[5] = {};
  send_calls = recv_calls = end_calls = 0;
  assert(mgbfs_nccl_scatter(source, 2, payload, 5, sizes, nullptr, 0, 0, nullptr) == 0);
  assert(send_calls == 2 && recv_calls == 0 && end_calls == 1);
  assert(last_send_peer == 1 && group_depth == 0);
  send_calls = end_calls = 0;
  assert(mgbfs_nccl_scatter(source, 2, payload, 4, sizes, nullptr, 0, 0, nullptr) != 0);
  assert(send_calls == 0 && end_calls == 0);
  fail_stage = 2;
  group_end_override = ncclInProgress;
  async_state = ncclInProgress;
  async_pending_queries = 3;
  assert(mgbfs_nccl_scatter(source, 2, payload, 5, sizes, nullptr, 0, 0, nullptr) == 3);
  assert(async_pending_queries == 0 && group_depth == 0);
  fail_stage = 0;
  group_end_override = -1;
  async_state = ncclInProgress;
  async_pending_queries = 3;
  mgbfs_nccl_destroy(source);
  assert(async_pending_queries == 0);
  assert(finalize_calls == 1 && destroy_calls == 1);
  void* receiver = nullptr;
  assert(mgbfs_nccl_create(1, 3, 1, &id, &receiver, nullptr, 0) == 0);
  recv_calls = 0;
  assert(mgbfs_nccl_scatter(receiver, 2, nullptr, 0, nullptr, nullptr, 0, 0, nullptr) == 0);
  assert(recv_calls == 1 && last_recv_peer == 2 && group_depth == 0);
  recv_calls = end_calls = 0;
  assert(mgbfs_nccl_scatter(receiver, 2, nullptr, 0, nullptr, payload, 5, 4, nullptr) != 0);
  assert(recv_calls == 0 && end_calls == 0 && group_depth == 0);
  assert(mgbfs_nccl_scatter(receiver, 2, nullptr, 0, nullptr, payload, 5, 5, nullptr) == 0);
  assert(recv_calls == 1 && end_calls == 1 && group_depth == 0);
  fail_stage = 3;
  group_end_override = ncclInProgress;
  async_state = ncclInProgress;
  async_pending_queries = 3;
  assert(mgbfs_nccl_scatter(receiver, 2, nullptr, 0, nullptr, payload, 5, 5, nullptr) == 4);
  assert(async_pending_queries == 0 && group_depth == 0);
  fail_stage = 0;
  group_end_override = -1;
  // Exercise the dispatcher's explicit fatal abort, not only cancellation
  // encountered while polling an in-progress NCCL operation.
  const int aborts_before = abort_calls;
  const int destroys_before = destroy_calls;
  RetirementFixture retirement;
  assert(mgbfs_nccl_bind_retirement(receiver,retirement_probe,&retirement)==0);
  assert(mgbfs_nccl_abort(receiver) == 0);
  assert(retirement.published==1 && retirement.queries==3);
  assert(abort_calls == aborts_before + 1);
  assert(mgbfs_nccl_abort(receiver) == 0);
  assert(abort_calls == aborts_before + 1);
  assert(mgbfs_nccl_poll(receiver) != 0);
  mgbfs_nccl_destroy(receiver);
  assert(destroy_calls == destroys_before);
  // Cancellation during a failing group's asynchronous close must run the
  // same dispatcher's abort, without replacing the original send error or
  // permitting another NCCL operation on the released handle.
  void* failed_group = nullptr;
  assert(mgbfs_nccl_create(0,2,0,&id,&failed_group,nullptr,0)==0);
  cancel_after = 3;
  assert(mgbfs_nccl_bind_cancel(failed_group,cancellation_probe,&cancel_after)==0);
  fail_stage = 2;
  group_end_override = ncclInProgress;
  async_state = ncclInProgress;
  async_pending_queries = 0;
  const int failure_aborts = abort_calls;
  assert(mgbfs_nccl_send_recv(failed_group,&byte,1,1,&byte,1,nullptr)==3);
  assert(abort_calls==failure_aborts+1 && group_depth==0);
  const int failure_sends = send_calls;
  assert(mgbfs_nccl_send_recv(failed_group,&byte,1,1,&byte,1,nullptr)!=0);
  assert(send_calls==failure_sends);
  assert(mgbfs_nccl_abort(failed_group)==0 && abort_calls==failure_aborts+1);
  mgbfs_nccl_destroy(failed_group);
  fail_stage = 0;
  group_end_override = -1;
  // Cancellation must be installed before an in-progress initialization,
  // not only after a communicator has been returned to Rust.
  assert(mgbfs_nccl_create_with_cancel != nullptr);
  void* startup = nullptr;
  async_state = ncclInProgress;
  async_pending_queries = 0;
  cancel_after = 2;
  const int startup_aborts = abort_calls;
  assert(mgbfs_nccl_create_with_cancel(0,2,0,&id,&startup,nullptr,0,
      cancellation_probe,&cancel_after) != 0);
  assert(startup == nullptr && abort_calls == startup_aborts + 1);
  cancel_after = 0;
  const int calls_before = init_calls;
  assert(mgbfs_nccl_create_with_cancel(0,2,0,&id,&startup,nullptr,0,
      cancellation_probe,&cancel_after) == 7);
  assert(startup == nullptr && init_calls == calls_before);
}
