# Route-bank capacity failure after within-depth reuse

Base runtime: `ac24a0d` (`ce1e0e7` physical-bank implementation).
Added integration test:
`cuco_lsa_route_bank_capacity_failure_latches_after_reuse`.
No runtime algorithm change is claimed.

U4 over F2 has literal layer sizes 1,3,5,8,11,... . Next-layer capacity is
eight, batch size one, banks 2/3/4, K3, both DENSE and scalar HASH_FIRST.
The test checks full states of the successful prefix, rejects the attempt to
produce the eleven-state layer, requires actual within-depth bank reuse, and
requires a later `advance` to return `DISTRIBUTED_FAILED` rather than reopen
admission. Removing capacity enforcement or the failed-runtime latch must break
this behavioral test.

Real local RTX3070Laptop, world1, CUDA native12.8/sm86,
library-owner12.9/sm86 and architecture-isolated NCCL2.29.7:
all six cases passed in 6.49 seconds total fixture time (not BFS performance).
Each returned `GROUP_OWNER_OR_PRE_OWNER_FATAL:LSA_DEVICE_LOGICAL_FATAL`.
The existing teardown printed native abort status12; this test does not certify
that status as a successful API result or establish multi-rank abort correctness.

Full `cargo test --workspace` without CUDA features passed. Full
`library_bfs_gpu` GPU file: four passed, one failed. The failure was the existing
`library_bfs_layers_match_full_state_oracle_in_both_profiles` Tensor case,
rejected at construction with `HASH_FIRST_TC_DEVICE_UNSUPPORTED` on sm86.
It is retained and not counted as a green whole-GPU suite. The existing
unused-mut warning at reference_bench.rs:756 remains.

An initial linking attempt omitted installed RAPIDS transitive library paths
and failed before execution. The rerun used libcudf/libkvikio/librmm/nvcomp/
rapids_logger paths from `/linux-build/library-site`, with the same source and
cached native libraries.

This is one-rank, no-archive capacity evidence. It does not close asymmetric
two-rank faults, archive.finalize/RunCommit, sm75 Tensor generation, four target
sanitizers, the two-T4 timeline, or paired A/B.
