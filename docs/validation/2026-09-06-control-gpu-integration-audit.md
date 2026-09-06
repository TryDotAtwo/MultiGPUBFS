# Control/GPU integration audit, 2026-09-06

Source inspected: `0d15789`.

The CPU control pump passes six targeted tests, including real TCP causal
Candidate/Request/Response/Receipt ordering and deadline failure propagation.
It is **not connected to the native BFS data plane**.

## Concrete integration gaps

- `examples/distributed_bench.rs` retains `BootstrapGroup` as `_control_group`;
  its peers are not passed to `ControlPump`.
- `distributed_native.rs::advance_inner` still uses host-synchronous route and
  count reads, `all_max` admission checks and one candidate workspace. Replacing
  the launcher alone cannot create route-slot overlap.
- `cuda/nccl_transport.cpp::mgbfs_nccl_send_recv` returns immediately on failed
  `ncclSend` or `ncclRecv`, after a successful `ncclGroupStart`, without calling
  `ncclGroupEnd`. This error path needs an executable injected-error wrapper
  test and balanced group cleanup before relying on continued host progress.
- The communicator wrapper only exposes destruction, not abort or async-error
  polling. A caller deadline in Rust cannot interrupt a blocked NCCL call.

NVIDIA documents grouped calls and asynchronous error/abort handling in
[Group Calls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html)
and [Creating a Communicator](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2265/user-guide/docs/usage/communicators.html).
The installed Kaggle NCCL version must be recorded before choosing newer APIs.

## Required next implementation evidence

1. Executable C++ transport failure tests, not source-text assertions, covering
   group start/send/recv/end errors and communicator cleanup.
2. Bounded per-slot device buffers and events with count/byte capacity checks;
   wire actual readiness/completion/consumption into the pump.
3. Real two-T4 correctness and sanitizer regression, followed by a timeline
   proving overlap. CPU control tests do not satisfy these GPU gates.

The existing S11 one-T4 v12 benchmark was still reported RUNNING by Kaggle
during this audit. It was not restarted or overwritten. Published S13 was not
rerun. No new performance claim follows from this control work.

## Follow-up: group cleanup fix

`tests/nccl_transport_failure.cpp` compiles the actual wrapper against test-only
CUDA/NCCL doubles. The pre-fix executable failed `group_depth == 0` when send
failed. The wrapper now always ends a successfully opened group, skips recv
after failed send, and retains the first operation error. All five injected
stages (success/start/send/recv/end) pass with MSVC C++17 locally. Linux CI now
runs the same executable. This does not test real NCCL failure recovery; abort,
async-error polling and real hardware regression remain required.
