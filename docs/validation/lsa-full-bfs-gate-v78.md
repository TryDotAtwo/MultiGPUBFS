# 2×T4 full-BFS boundary gate, Kaggle v78

- Kernel: private `trydotatwo/mgbfs-lsa-full-bfs-gate-t4`, status `COMPLETE` on 2026-09-26.
- Source SHA: `ed194012d8051c3f2f13f22a68be9efb170c6ee5` (older than current branch).
- Physical hardware: two Tesla T4, 15,360 MiB each; P2P allowed both directions.
- Raw downloaded logs and JSON: `test_results/kaggle_lsa_full_bfs_gate_v78/lsa-bfs-gate/` (locally ignored by Git).

All three completed S4 configurations had exact global layer counts
`[1, 3, 5, 6, 5, 3, 1]` and 24 states. `host_sized_nccl` and `nccl_lsa`
produced group markers and both committed archives; both rank archives
passed the archive checksum/count verifier. An independent full-state S4
oracle test passed. The `nccl_lsa` run with per-rank layer capacity 6 also
completed despite 24 total states, validating this bounded case rather
than proving arbitrary capacity planning.

Injected one-rank archive-admission and invalid-configuration cases reported
group-fatal outcomes. The constructor-fault GPU fixture reported
`CUDA_STATUS_-1` on rank 0 and `REMOTE_CONSTRUCTOR_FATAL` on rank 1.

Single-run search times were 0.0490 s (`host_sized_nccl`) and 0.0145 s
(`nccl_lsa`), using the slower rank. These are tiny-graph diagnostics, **not**
speed claims; the `cuda_allocated_used_bytes` sum was 497.9 vs 577.9 MiB,
and the recorded memory sampling was setup/final only, not full peak VRAM.
No four-tool Compute Sanitizer gate or CPU-free timeline is established here.
The current HEAD's later warmup-admission change (`5b04b2a`) was not in v78
and still needs a real two-process fault fixture.
