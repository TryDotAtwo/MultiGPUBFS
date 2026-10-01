# Archive descriptor rejection: cancellation before reader cleanup

## Change

In the distributed DENSE/HASH_FIRST archive producer, a failed
`PinnedArchive::submit` previously discarded its owned `Slot` before returning
to `advance_archived`'s failure notifier. `Slot::drop` synchronizes the recorded
D2H event. Thus an exhausted/disconnected descriptor queue could enter CUDA
cleanup before publishing cancellation to the rank group.

The existing queue is retained. `submit_notifying` now invokes the runtime's
failure callback while the rejected message still owns the pinned slot. The
callback records the original diagnostic, poisons the runtime, publishes the
existing sideband failure token, and calls abort on the existing NCCL owner
thread. Only then can slot cleanup run. Shape rejection and closed archive
follow the same order. Notification does not authorize early reuse/free.

Healthy submission still uses `try_send`: no wait, CUDA operation, allocation,
device readback or new queue is added. Other archive users retain their existing
submission interface. No CUDA/NCCL ABI or payload sizing changed.

## Evidence

Four tests use real bounded `std::sync::mpsc` channels and a reader lease with
observable destruction, not a mocked channel. The mechanical extraction of
the old policy produced three RED assertion failures: each observed
`reader_cleanup` without the preceding `group_failure`. The healthy test
already passed. With the correction all four tests pass in the complete
default-members `cargo test --locked` run (session 97177, exit 0).

The stronger `cargo test --locked --workspace` command failed before testing
the legacy root GPU prototype: its build requires `MULTIGPUBFS_CUDA_LIB_DIR`
on Windows. This is not reported as a passing all-workspace check. Existing
CPU dead-code warnings for ConstructorFailureReport remain.

Linux CUDA+library-owner CLI rebuilt and linked successfully with the pinned
local CUDA 12.9/RAPIDS/NCCL libraries. The first link omitted RAPIDS dependency
search paths and failed on kvikio/nvcomp references; restoring the existing
LD_LIBRARY_PATH resolved it without dependency edits. All 11 Linux CLI tests
pass with the actual local RTX3070 Laptop, including the production owner-DAG
capture test. The pre-existing reference_bench unused-mut warning remains.

An attempted NCCL-group change that waited before returning a member error
was rejected and removed: it could delay failure notification while peers
await missing operations. No such wait was left in the runtime. NCCL allows
nonblocking communicator abort during in-flight work; see the official
[fault-tolerance guide](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2297/user-guide/docs/usage/communicators.html#fault-tolerance).

## Not proved

This proves host-side notification/destruction order and compilation, not
bounded two-rank termination with an actual queued D2H reader. The fresh
two-T4 asymmetric descriptor/worker failure fixture, four sanitizer gates,
complete timeline and paired A/B remain open. Earlier sanitizer/timeline
results belong to the earlier source and are not promoted to this revision.

No paid instance or new Kaggle worker was started for this change. Existing
unrelated working-tree edits were preserved.
