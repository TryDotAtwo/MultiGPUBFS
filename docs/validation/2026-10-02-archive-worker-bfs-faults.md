# Real BFS archive worker I/O failure fixtures

## Change

The existing debug reference launcher now admits two rank-selective disk
faults: `MGBFS_TEST_ARCHIVE_WORKER_WRITE_FAULT_RANK` and
`MGBFS_TEST_ARCHIVE_WORKER_SYNC_FAULT_RANK`. These wrap the actual disk extent,
not the producer or `archive.finish` result. Reservation and the 48-byte header
remain real. The write fixture rejects subsequent frame writes on the archive
worker. The sync fixture rejects the worker's final durable sync, after actual
search and record/layer writes. Neither fixture exists in release builds.

Both enter the failure token and existing group vote integrated by 9d124a0.
There is no new transport, queue, count readback, healthy-batch wait, payload
padding or GPU/pinned allocation. The existing independent-rank replay
supervisor runs each new fault on rank 0 and rank 1 and requires the original
fault marker, nonzero exits for both ranks, bounded cleanup and no group
COMPLETE. Adding these cases is not evidence that remote replay ran.

## Verification

RED: with identity extent wrapping, both fault tests failed because record
writes and durable RunCommit succeeded; the healthy case passed. GREEN:
three real-file codec tests passed, plus all 17 CUDA/library-owner runtime unit
tests and all 12 CUDA/library-owner CLI integration tests. The new CLI test
runs the real CUCO_RANK + LSA S4 BFS for DENSE/HASH_FIRST and both faults.
Every case exits 1 within the 30-second subprocess deadline, reaches the
worker I/O diagnostic, and publishes neither rank result nor group COMPLETE.
The existing healthy native owner-DAG capture test also passes.

The complete default-members `cargo test --locked` passed on Windows (session
58329). The ordinary sandbox invocation initially failed opening the target
build lock; the authorized elevated invocation passed. This is not a claim
that the legacy all-workspace GPU build works on Windows. Existing unrelated
`unused_mut` warning in reference_bench remains. Three Python owner-environment
tests pass; owned-file `git diff --check` reports no whitespace errors.

Four unfiltered Compute Sanitizer tools were then run against the actual CLI
BFS, not the worker unit fixture: 4 tools x 2 profiles x 2 I/O faults = 16 runs.
Each used `--target-processes all --error-exitcode 97`, no kernel filters or API
error suppression, and a 45-second timeout. Every application exits 1 for the
injected error, never 97/124; all tools report zero errors (racecheck also zero
warnings/hazards), and no group COMPLETE exists. The enclosing loop exits 0
(session 99726). Logs and environment passport are retained locally under
`test_results/archive_worker_bfs_20261002/<tool>-<profile>-<WRITE|SYNC>/run.log`.

Hardware: one actual RTX 3070 Laptop, sm86, host driver 572.70. Existing pinned
CUDA 12.9 runtime/library-owner dependencies and local diagnostic NCCL
architecture-guard build were reused; sanitizer 2025.1.0.0. CLI SHA-256:
`2ec8bd3b78ff967791430a1630f3d54dafdf4818de51bf0b8f2d5644f4744541`.
Reference settings: S4, batch 7, pre-dedup ON, fixed 64-record capacity,
128-record StateRing, four shards/eight buckets, fixed 64 MiB library pool,
mandatory matrix_u8 archive, three rows/128 pinned slots, no warmup/fallback.

## Remaining gates

This is single-rank integrated BFS evidence, not independent-rank cancellation
or two-T4 acceptance. No remote lease/worker was launched by this change.
Fresh asymmetric two-rank replay, NCCL registration initcheck on actual T4,
full two-GPU timeline and paired A/B remain open. The macro-depth distributed
path, DB/framework analysis and HF publication remain part of the full goal.
