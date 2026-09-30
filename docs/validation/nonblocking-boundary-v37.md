# Nonblocking boundary v37

Source: `c7488ca6d901afc1545edba4e25ea28f915f8206`.
Hardware: two independent rank processes on two actual T4s, bidirectional P2P.
Transport: NCCL LSA; owner: DENSE CUCO_RANK.

Full owner DAG capture probe: plain and all four Compute Sanitizer tools PASS.
This is not full distributed BFS/NCCL sanitizer acceptance.

The first injected owner host error on rank 0 failed bounded termination.
Rank 0 logged its originating error and `stage=nccl_abort_begin`, but never
logged `stage=nccl_abort_end`. Rank 1 logged remote cancellation, both abort
markers, owner drain completion and all four stream drain completions, then
exited with error. Torchrun killed rank 0 after its 30-second monitor interval.
Thus this stall is inside rank 0's `ncclCommAbort`, not the later owner or Rust
stream destructors. The internal NCCL/CUDA blocking operation is still unknown.
Five later asymmetric fault cases were NOT_RUN. No COMPLETE was accepted.

Logs: `test_results/kaggle_nonblocking_boundary_v37/lsa-bfs-gate/`.
Kaggle output download returned a Windows console encoding error after the
summary and relevant logs had been downloaded; those files were inspected.

Next diagnostic reuses the existing parent-launch debugger, follows the real
torchrun child processes, samples stacks at 10/20 seconds and bounds execution
at 27 seconds. It does not attach or change security settings, restore healthy
batch barriers, suppress sanitizer findings, or count debugger interruption as
graceful fault termination. A connected runtime fix remains OPEN.

Local MSVC NCCL-double regression PASS including explicit dispatcher abort,
repeat-abort idempotence, rejected polling after abort and no double destroy.
This validates wrapper control flow only, not the actual NCCL blocking call.
