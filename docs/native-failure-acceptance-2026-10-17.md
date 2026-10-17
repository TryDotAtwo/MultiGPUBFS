# Native failure context acceptance — 2026-10-17

Published package: [0.2.0.dev23](https://github.com/TryDotAtwo/MultiGPUBFS/releases/tag/native-pip-20261017-v23).
Server instructions: [native failure evidence](native-failure-evidence.md).
Exact GPU receipt: [JSON](acceptance/native-failure-context-20261017.json), SHA-256 `4bc1bb67f18e12ec53820d1b65cc6aaed667fbe1f19b2fc7c341017c1ddb9086`.

## Changes and validation

| Requirement | Evidence |
|---|---|
| Label the reporting native call, rank/device, target layer and last completed layer | 3 Rust context regressions; 140 checked generic callsites; real installed CLI test with an injected `cudaMemcpyAsync` return 999 |
| Retain failed-job phase/response/current logs before ephemeral worker cleanup | 5 Python evidence regressions, bounded copies and damaged/missing-file handling; both persistent and one-shot two-rank GPU failures retained files after worker cleanup |
| Keep new JSON evidence in `diagnostics.zip` | A regression first demonstrated missing `phase.json`, then passed with the archive filter fix; job environment files remain excluded |
| Preserve ordinary BFS results | Physical 2×Tesla T4, exact CPU layer and terminal-state oracle on permutation, repeated-symbol and matrix fixtures; two persistent runs per fixture |
| Publish installable source and package | 783 Python tests; clean wheel and GitHub-source installs; 460 workspace Rust tests; public asset hashes/readback; installed public wheel GPU acceptance |

Python/package source: `c7217ee8397bb71aeae811e12723fd7a2efaea73`.
Native CLI source: `cb9a5931d6ee60491ddf8dcd887fe9ff017289c2`.
Wheel SHA-256: `69d6b88b0c431b95d58ecc276aad38796838abf90d6cb4854eb9f85efb6e4b69`.
Native CLI SHA-256: `6c579b686a945305b2ea1a54ed1d50806696c9e4c3cd22ad6224bca594127b01`.
Native index SHA-256: `5aea6723a2867547c5656ec932d7a66cdd52df0e6b98b49a0fd80ad70abc483a`.

The CUDA kernels, all per-SM CUDA libraries, CUDA runtime and NCCL binaries were reused unchanged. Only the Rust CLI and Python diagnostics/evidence handling changed. Obsolete Rust test expectations for a 10% reserve were updated to the existing absolute reserve; no memory planner or buffer geometry was changed.

Physical acceptance covers **2×T4 / SM75**, not the affected eight-GPU server. The 999 failures here were deliberately injected through a test-only `LD_PRELOAD` wrapper outside the shipped library. Their reporting API was captured as `cudaMemcpyAsync`, rank/device 1, target layer 1, last completed layer 0, source `generic_distributed_native.rs:155`, in both process modes. This establishes that the new evidence path works; it does **not** establish or fix the cause of the original production CUDA 999 error. No automatic restart or performance claim is made.

CI: [native build and full Rust suite](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38999857388), [783 Python tests and clean installs](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/39000659699), [focused context/archive regressions](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38999663000).
