# Native component diagnostics acceptance — 2026-10-13

The installed GitHub wheel was tested in a fresh private Kaggle job on two Tesla T4 GPUs. This is single-host component acceptance, not throughput or multi-node acceptance.

Package source commit: `9516759a9cc24fa4326b12ab7515f2e7e320b128`. Native source reused unchanged: `7046577446e546b86317f6e2e24cf319552b29cd`. Wheel version: `0.2.0.dev7`. SHA256: `3f2bb77f724efc2c3121433a30d4f4a0e1f54d26569e7581994d74748771318e`.

Nine CPU framework/dispatch tests passed on Ubuntu 22.04, including retained failure logs, continuation after component failure, bounded timeout, termination of a nested diagnostic child, output-directory protection, metadata-only ZIP, BFS dispatch isolation unknown-stage rejection and captured loader stderr preservation.

Actual installed CLI GPU acceptance:

## doctor-ok

Status: PASS. Wall time: 53.332 s.

| Component | Status | Seconds |
|---|---|---:|
| system-info | PASS | 0.169 |
| filesystem | PASS | 0.168 |
| native-runtime | PASS | 3.279 |
| cuda-discovery | PASS | 1.875 |
| gpu-0-context | PASS | 2.425 |
| gpu-0-memory | PASS | 2.325 |
| gpu-0-kernel | PASS | 2.426 |
| gpu-0-persistent | PASS | 2.927 |
| gpu-0-bfs | PASS | 4.532 |
| gpu-1-context | PASS | 2.425 |
| gpu-1-memory | PASS | 2.375 |
| gpu-1-kernel | PASS | 2.425 |
| gpu-1-persistent | PASS | 3.879 |
| gpu-1-bfs | PASS | 4.632 |
| nccl-init | PASS | 5.537 |
| nccl-exchange | PASS | 5.636 |
| multi-gpu-bfs | PASS | 5.287 |

## doctor-timeout

Status: FAIL. Wall time: 0.266 s.

| Component | Status | Seconds |
|---|---|---:|
| system-info | WARNING | 0.027 |
| filesystem | TIMEOUT | 0.025 |
| native-runtime | TIMEOUT | 0.025 |
| gpu-components | SKIPPED | — |

## doctor-invalid-device

Status: FAIL. Wall time: 5.681 s.

| Component | Status | Seconds |
|---|---|---:|
| system-info | PASS | 0.169 |
| filesystem | PASS | 0.168 |
| native-runtime | PASS | 3.228 |
| cuda-discovery | PASS | 1.922 |
| device-selection | FAIL | — |

## doctor-missing-native

Status: FAIL. Wall time: 0.667 s.

| Component | Status | Seconds |
|---|---|---:|
| system-info | PASS | 0.169 |
| filesystem | PASS | 0.168 |
| native-runtime | FAIL | 0.119 |
| gpu-components | SKIPPED | — |

All metadata checksums were read back, ZIP contents were checked to exclude state payloads/raw job environments, and diagnostic parent PIDs were checked after exit. Controlled timeout, unavailable GPU ID and missing native executable all returned a failed report rather than claiming completion.

Build and CPU tests: [GitHub Actions](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38437674495). Private Kaggle test: `trydotatwo/multigpubfs-component-diagnostic-acceptance`; sanitized stage summaries are below. Raw diagnostic logs stay in the private job output and are not automatically published.

The earlier v5 diagnostic candidate failed dispatch acceptance and was superseded; no failed result was accepted as a successful GPU test.
