# N-peer T4 regression v13

Kaggle `trydotatwo/mgbfs-library-capacity-t4`, version 13 / scriptVersionId
350382295 completed on 2026-09-16. Downloaded summary status: PASS.
Source: `b95b1fbdb80f0141274279750f5c1e38a17ac823`.

Evidence: `test_results/library-capacity-v13/library-owner/summary.json` and
the downloaded structured `mgbfs-library-capacity-t4.log` alongside it.
Two distinct Tesla T4 UUIDs are recorded in the manifest. CUDA build targets
are `75;90`, including the cuCollections shared library. This proves a dual
architecture build, not H200 execution.

## Executed gates

All 15 BFS invocations passed: GPU0, GPU1, and the two-device NCCL test,
each plain plus memcheck/racecheck/initcheck/synccheck. The 12 instrumented
BFS invocations report zero errors; all three BFS racechecks report zero
warnings and hazards. The eight-device fixture was explicitly ignored, not run.

The two-device fixture took 13.88 seconds plain, 51.14 under memcheck,
646.48 under racecheck, 28.94 under initcheck and 19.92 under synccheck.
These are correctness-fixture durations, NOT production performance results.

The separate CLI gate used actual two-process torchrun launches for:

- cuDF and cuCollections;
- S4 DENSE, S4 HASH_FIRST, U4m2 DENSE.

All 12 per-rank archive verification receipts report VERIFIED with scope
`committed_archive_checksums_and_counts`. This receipt is not a full-state
oracle. Native CUB CLI and U4m2 HASH_FIRST CLI are not covered by these six
CLI cases. Full-state comparisons belong to the BFS oracle fixtures above.

## Remaining

Physical eight-H200 correctness/sanitizers, eight-process bootstrap, matched
H200 backend measurements, streamed HF end-to-end validation and full LRX13.
No rental or LRX13 completion is implied by this report.
