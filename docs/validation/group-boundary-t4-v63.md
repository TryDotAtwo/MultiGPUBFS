# Two-T4 run-boundary gate, Kaggle v63

Pinned source: `befd0dc06d9dbf5f624a2d2cf3faf3044409a0f5`.
Private Kaggle kernel: `trydotatwo/mgbfs-lsa-full-bfs-gate-t4`, version 63.
Raw downloaded outputs: `test_results/kaggle_group_boundary_v63/lsa-bfs-gate/`.

The notebook reported `COMPLETE` on two distinct Tesla T4s with bidirectional
P2P. The Linux `mgbfs-cli --features library-owner` build and boundary CPU
tests passed. Both fresh two-process torchrun S4/CUCO_RANK/DENSE runs completed:

| Transport | Global per-depth counts | rank-0 search seconds | rank-1 search seconds | Archive/marker |
|---|---|---:|---:|---|
| HostSized NCCL | 1,3,5,6,5,3,1 | 0.054950121 | 0.054948155 | both rank archives verified, checksummed group marker |
| NCCL LSA | 1,3,5,6,5,3,1 | 0.01431629 | 0.014314375 | both rank archives verified, checksummed group marker |

The injected rank-0 `create_new` archive failure produced
`ARCHIVE_EXTENT: File exists` locally and `REMOTE_ARCHIVE_ADMISSION_FATAL`
on rank 1; no group-complete marker was written. This confirms bounded
peer-visible failure for this admission fixture, not for every setup failure.

The times are tiny one-off correctness samples, **not** a speed comparison.
This notebook checks archive structural hashes, group-marker digests, total
states, and global layer counts. It does not itself compare every archived
state against an independent CPU oracle. Earlier full-state tests are separate
evidence; repeat one at this pinned source before claiming a fresh full-state
gate. No new sanitizer or Nsight timeline was run in v63.

Version 64 reran the strengthened notebook on a different Kaggle pair but
reported `UNSUPPORTED_HOST`: both distinct T4s had `cudaDeviceCanAccessPeer`
equal to zero, so it correctly stopped before build/BFS. It is not a failed
algorithm test. Raw inventory and summary are in
`test_results/kaggle_group_boundary_v64/lsa-bfs-gate/`; v65 retries the same
pinned source and full-state oracle gate.

Version 65 passed on another distinct P2P-capable T4 pair using the same
source. HostSized and LSA torchrun S4 runs both passed per-depth counts
`1,3,5,6,5,3,1`, archive verification and group-marker checks; the asymmetric
archive-admission fault again stopped both ranks. The separate ignored
`cuco_rank_lsa_two_gpu_dense_layers_and_archives_match_oracle` test passed
(`1 passed`, 5.74 s) and checks full per-depth state sets against
`MatrixGroup::exact_layers` plus archives. This oracle is two rank threads on
two devices, not the torchrun process fixture. Raw logs are in
`test_results/kaggle_group_boundary_v65/lsa-bfs-gate/`. No new sanitizer or
large-graph timing claim follows from this small gate.
