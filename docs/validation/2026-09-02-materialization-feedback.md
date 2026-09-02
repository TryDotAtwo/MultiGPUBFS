# GPU materialization and exhaustive feedback

Implementation: `af6b0a5de781b6d65d1a729eee7abedfdd43226d`.
Bounded sanitizer fixtures: `6049d9dac7a4af64cac50dc1c04ae96ba28320e2`.

This is a single-GPU, single-bucket development executor, not the production
archived multi-GPU runtime. There is no speedup claim or CayleyPy A/B result.

## Implemented contract

- Committed requests are sorted by source row; state chunks and associated
  hashes are appended densely in that order. No device allocation or host
  count read occurs in materialization.
- Capacity and invalid-reference failures are sticky, preserve the destination
  bytes/count, and prohibit publishing a partial next layer.
- GPU-generated/materialized states feed the next depth. The CPU oracle never
  supplies later frontiers. A bounded child slot is reused serially, with one
  stream and status/count readback only at depth finalization.
- Two state arenas and three sorted hash arenas implement the inverse-closed
  three-layer window. Source batches are bounded; the full candidate layer is
  never materialized at once. StateRing reclamation and overlap remain absent.

Full state sets at every depth, including exhaustion, match the independent
CPU oracle for U4 m=2..6, seeds 0/1/20260828 and pre-dedup OFF/ON. Batches are
7/64 for m=2..4 and 257 for m=5..6: 48 configurations. This does not turn
hash-only deduplication into an exact collision-resolving algorithm.

## Local evidence

RTX 3070 Laptop, existing CUDA 12.8/Rust 1.75 toolchain:

- 25 CPU contract tests and all existing CUDA primitive tests passed.
- Materialization append/order, invalid references, exact capacity, overflow
  and sticky failure passed all four Compute Sanitizer tools, zero errors.
- Complete GPU feedback and fatal-capacity tests passed. The smaller full-depth
  m=2..3 fixture additionally passed normally (24 configurations).
- Full m=2..6 memcheck passed with zero errors.
- Unrestricted m=2..6 racecheck was stopped after 15 minutes without a verdict.
  It is **not passed**. Its logs remain in `test_results/native-sanitizer/`.
- The bounded sanitizer run completed successfully, with zero errors/hazards/
  warnings in all four logs in `test_results/native-sanitizer-feedback-bounded/`.
  Full m=2..6 ran under memcheck/initcheck/synccheck; m=2..3 under racecheck.
  The latter took 853.45 seconds locally. This is instrumentation time, not BFS
  benchmark time.

The explicit sanitizer policy keeps all 48 configurations under
memcheck/initcheck/synccheck and uses the 24 full-depth m=2..3 configurations
under racecheck. Existing owner primitive tests cover tiled merge boundaries
independently. Both required test names are checked before applying filters;
a stale binary cannot pass by silently skipping all feedback tests.

## Kaggle attempts

Private kernel: `trydotatwo/mgbfs-native-matrix-primitives-t4`.

- Version 3: source-fetch failed before build, requesting GitHub authentication.
  Anonymous raw source access and PUBLIC repository visibility were verified;
  the reason for this individual fetch failure was not established. No GPU
  correctness result came from this attempt.
- Version 4: same immutable source fetched and compiled successfully. On the
  first physical T4, full m=2..6 feedback passed normally and under memcheck
  with zero errors. The run then hit its 900-second racecheck timeout. The
  second GPU and later primitive executables were not reached. Overall status
  is INCOMPLETE, not a successful two-GPU gate.
- Version 5: used source `6049d9dac7a4af64cac50dc1c04ae96ba28320e2` and the
  bounded racecheck policy. Plain and full m=2..6 memcheck passed on the first
  physical T4, with zero memcheck errors. Racecheck progressed to m=3, seed=1,
  batch=64, pre-dedup=true, then hit the 900-second test limit before finishing.
  No successful racecheck verdict exists for this attempt. The second T4 and
  later executables were not reached. Overall status is again INCOMPLETE.

Raw artifacts stay private under `test_results/kaggle_native_primitives_v3/`,
`test_results/kaggle_native_primitives_v4/` and
`test_results/kaggle_native_primitives_v5/`; all were downloaded and inspected.
Runtime and sanitizer timeouts are not performance
measurements and are not evidence of an algorithmic deadlock.

The two-T4 gate for this implementation remains OPEN. A next validation run
needs a larger preselected racecheck timeout (the fixture is retained, not
silently skipped). No further Kaggle run was launched after version 5.

## Still required

Shard scheduling, overlapping route slots/streams, GPU StateRing reclamation,
pinned asynchronous archive, full allocation/reserve planning, native NCCL,
HASH_FIRST, BMMA, CLI, RunCommit and the prescribed performance/memory A/B.
CayleyPy baseline remains unchanged at `f0f2b8e`.
