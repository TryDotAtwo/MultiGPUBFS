# Driver diagnostic acceptance — 2026-10-14

Package: `0.2.0.dev9`. Package source: `e317c2762744fdde27504b27a2f82961f49a3747`. Native source reused unchanged: `7046577446e546b86317f6e2e24cf319552b29cd`. Wheel SHA256: `eb26ead19851f7f627625b97208707772482a2e605926675e2885aff8c01b30a`.

Eighteen Linux tests passed, including framework cleanup, error classification, actual ctypes shared-library loading against controlled driver/runtime libraries and preserved partial evidence during a deliberately blocked cuInit. Synthetic old-major/stub/mismatch/no-device failures are fixtures, not claims about the affected remote server. A lower minor CUDA version or a directory named stubs alone is not classified as incompatibility or a confirmed stub.

Actual GPU checks below ran in a private fresh Kaggle 2×T4 job, using the installed GitHub wheel. In the separate direct-driver test PATH contained no command tools, so the probe could not rely on nvidia-smi. Diagnostic metadata checksums/ZIP exclusions and test-process termination were checked. Raw environment/library-path reports are private job outputs; only sanitized summaries are published.

CPU loader/timeout tests: [Actions](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38468708411). Package build: [Actions](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38468712093).

## doctor-ok

Status: PASS. Wall: 58.607 s.

| Stage | Status | Seconds |
|---|---|---:|
| system-info | PASS | 0.17 |
| filesystem | PASS | 0.169 |
| driver-api | PASS | 2.226 |
| native-runtime | PASS | 3.479 |
| cuda-discovery | PASS | 1.974 |
| gpu-0-context | PASS | 2.677 |
| gpu-0-memory | PASS | 2.577 |
| gpu-0-kernel | PASS | 2.576 |
| gpu-0-persistent | PASS | 3.227 |
| gpu-0-bfs | PASS | 4.884 |
| gpu-1-context | PASS | 2.577 |
| gpu-1-memory | PASS | 2.528 |
| gpu-1-kernel | PASS | 3.478 |
| gpu-1-persistent | PASS | 3.127 |
| gpu-1-bfs | PASS | 4.783 |
| nccl-init | PASS | 5.886 |
| nccl-exchange | PASS | 6.053 |
| multi-gpu-bfs | PASS | 5.59 |

## driver-without-nvidia-smi

Status: PASS. Wall: 0.516 s.

| Stage | Status | Seconds |
|---|---|---:|
| driver-api-with-empty-PATH | PASS | — |

Sanitized direct-driver evidence: `{"classification": "DRIVER_RUNTIME_READY", "driver_api_version": 13000, "runtime_version": 12090, "gpu_count": 2, "resolved_driver_basename": "libcuda.so.580.178.04"}`.

## doctor-timeout

Status: FAIL. Wall: 0.316 s.

| Stage | Status | Seconds |
|---|---|---:|
| system-info | WARNING | 0.026 |
| filesystem | TIMEOUT | 0.026 |
| driver-api | TIMEOUT | 0.026 |
| native-runtime | TIMEOUT | 0.026 |
| gpu-components | SKIPPED | — |

## doctor-invalid-device

Status: FAIL. Wall: 8.24 s.

| Stage | Status | Seconds |
|---|---|---:|
| system-info | PASS | 0.169 |
| filesystem | PASS | 0.168 |
| driver-api | PASS | 2.175 |
| native-runtime | PASS | 3.478 |
| cuda-discovery | PASS | 2.026 |
| device-selection | FAIL | — |

## doctor-missing-native

Status: FAIL. Wall: 1.42 s.

| Stage | Status | Seconds |
|---|---|---:|
| system-info | PASS | 0.169 |
| filesystem | PASS | 0.169 |
| driver-api | PASS | 0.67 |
| native-runtime | FAIL | 0.169 |
| gpu-components | SKIPPED | — |

The affected server itself was not accessed. Its exact cause still requires rerunning the updated diagnostic there. This acceptance covers a single host, not multi-node, all hardware architectures or maximum graph capacity.
