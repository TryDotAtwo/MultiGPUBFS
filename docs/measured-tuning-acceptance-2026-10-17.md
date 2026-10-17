# Measured autotuning acceptance — 2026-10-17

Release **0.2.0.dev29**. Python source `9ab794acb1ae78614517ca8a8c06748e9613ed51`; wheel SHA-256 `ab5d6452dd1588bf4c1a003ad911b4cea03d3e6e61e7a6ea3cc18d2d40bf7a51`.
Native CLI source `1067171010ae0a27e283aa0b19365da768affb28`; executable SHA-256 `10e423aee53f450d5a14f064db0cb1c7784e8080e7b94045adbad378ad304fce`.
Native index SHA-256 `2d8104d9428e94cf133ad5e8cf93498709077f5c11c46c2e85090151fa0b7eec`. Per-SM CUDA libraries, CUDA runtime, NCCL and BFS mathematics are unchanged. Only the Rust CLI's handling of read-only plan refusals changed in native code.

## What is fixed

dev24 contained an optional CUDA999 using an unmeasured session-wide fallback. dev29 removes that fallback and latch. A captured, identified optional candidate failure preserves its primary native error and durable evidence, closes the failed rank group, invalidates stale admission inventory and rejects that candidate without retry. The other configurations are actually admitted and measured on fresh workers on the same selected GPUs. A captured failure with fewer than two successful sufficiently measured survivors raises `AUTOTUNE_NO_MEASURED_PROFILE_AFTER_FAILURE`. Unknown errors, timeouts and nonmatching CUDA faults remain errors. Production BFS is not automatically restarted.

Every successful candidate's completed layer vector is validated, even if sampling was short. Candidates below the existing depth/work requirements are recorded in `excluded_prefix_candidates` and cannot shorten the comparison window of sufficiently measured survivors. Sufficient INCOMPLETE probes remain eligible; comparisons use only their common completed prefix after warmup. Mode/geometry measurement (`selection_measured`) is separate from frontier-size coverage (`size_profiles_measured`). No unmeasured size tier is labelled measured.

After a measured admitted choice, recognized resource refusals for an additional size-tier generator are recorded in `size_generator_rejections`. Unadmitted generators are excluded from live and reused tiers. The selected generator's admission failure, unknown failures and timeouts still propagate. Disk cache identity is verified after a real JSON round-trip.

The GPU check uncovered a second half of that protocol defect: after Python caught an expected plan refusal, the Rust resident loop unconditionally exited on the original error. Both layers are now corrected. Python records the submitted command, validates schema/sequence/response identity and drains all submitted peer plans under the original deadline. Rust preserves the truthful error response and continues only for `graph-plan`/`graph-local-plan` with exact `REQUESTED_CAPACITY_EXCEEDS_SORTED_ADMISSION`, `REQUESTED_CAPACITY_EXCEEDS_ADMISSION` or `GEMM_BATCH_EXCEEDS_ADMISSION`, and only if no native fault was recorded. BFS/bench failures, unknown errors and dead workers remain fatal. `key-info` exposes `resident_plan_rejection_v1` for this CLI capability.

## Physical installed-package acceptance

Hardware: NVIDIA GeForce RTX 3060, 8.6, 595.71.05, NVIDIA GeForce RTX 3060, 8.6, 595.71.05. Two six-generator isomorphic matrix fixtures use 4×4/i64 and 8×8/i64 representations of `SL2(17) × C17`. Independent CPU factor BFS and a layer convolution establish all **83,232 states** and every layer. This definition is included in the raw receipt.

Every case completes two exact BFS calculations, compares every layer with the CPU oracle, requires measured selection with at least two sufficient candidates, and verifies identical healthy worker PIDs and a session profile cache hit on the repeat. Each ordinary 8×8 case includes a measured CUDA/GEMM candidate comparison. Instrumentation exists only in the developer acceptance test, never in the released wheel or native binary.

| Case | Matrix elements | Measured candidates | Generator / history / shards | Common depth | Cold tuning, s | Repeat tuning, ms |
|---|---:|---:|---|---:|---:|---:|
| matrix16-healthy | 16 | 12 | cuda / HASH / 1 | 20 | 18.111 | 5.913 |
| matrix16-first-candidate-fault | 16 | 11 | cuda / HASH / 4 | 20 | 29.855 | 5.745 |
| matrix64-healthy | 64 | 13 | cuda / HASH / 1 | 17 | 33.519 | 5.608 |
| matrix64-first-candidate-fault | 64 | 12 | cuda / SORTED_RUNS / 4 | 18 | 47.007 | 152.721 |
| matrix64-short-prefix | 64 | 12 | cuda / HASH / 1 | 20 | 31.518 | 5.510 |
| matrix64-admission-reject | 64 | 12 | cuda / HASH / 1 | 17 | 33.024 | 5.536 |

The two fault cases inject CUDA999 at rank 1/layer 2 of the first candidate plus a secondary cleanup error. They require preserved primary fault, durable evidence, fresh surviving-worker PIDs, no retry of the rejected candidate and actual measurement of the survivors. The short-prefix case runs one real probe to depth 10 and requires its exclusion without losing other measured candidates. The negative-admission case first measures CUDA choices, then invokes the actual native planner with an oversized optional GEMM capacity. No BFS runs at that negative capacity. It requires the actual native nonfatal-plan marker, rejection metadata, a complete exact BFS and the same healthy worker PIDs on the second full BFS.

[Raw measured GPU receipt](acceptance/measured-tuning-20261017.json) · [CPU/native CI receipt](acceptance/measured-tuning-ci-20261017.json) · [Native build receipt](acceptance/resident-plan-native-20261017.json) · [Reproduce the GPU checks](../examples/acceptance_measured_tuning.ipynb).

## Verification and limits

- 465 rust-native tests: [39053510108](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/39053510108).
- 817 python-package tests: [39053875888](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/39053875888).
- Red Rust regression [39053234274](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/39053234274): one intended assertion failure, four passing guards; no compilation/setup error.
- Clean wheel and pinned-source installs/imports are checked without GPU or network native preparation in package CI.

The original eight-SM120 server CUDA999 root cause remains **unknown**: the supplied archive records matrix16/i64 and six generators but lacks the graph definition. These physical two-card cases are bounded acceptance of selector/protocol behavior, not a reproduction of that original graph, eight-rank/SM120 acceptance, or proof of a global performance optimum. Timing values are single validation observations. No CUDA kernel or driver fix is claimed.

The new common archive includes the CLI and unchanged runtime libraries, so upgrading an old cache requires one common-component download of 310,801,425 bytes. Existing architecture components are reused unchanged. Prefetch before an offline job; preserve confirmed server environment settings. A stale `MGBFS_EXECUTABLE` override bypasses indexed native selection and must be removed to use the released CLI.
