# Broad user-space diagnostic acceptance — 2026-10-14

Package `0.2.0.dev10`; package source `7d649eb9973d5759b2dc91b8a2f9a622e08bc36e`; unchanged native source `7046577446e546b86317f6e2e24cf319552b29cd`. Wheel SHA256: `f06b9689c0edf4bd5b29782bab4ca1da951bb1a97b05c1cdfb7b30931fe18b49`.

47 Linux diagnostic tests passed. These include actual controlled shared-library failures, partial evidence during blocked Driver API, observed-vs-hypothetical security conclusions, cgroup-v1 rule parsing, GPU-allocation preservation in loader contrasts, baseline recheck, ABI errors, paths containing spaces, parent spawn denial, unavailable optional probes and truthful resident/standalone timeout messages. Synthetic security/driver cases are test fixtures, not claims about the affected server.

GPU evidence below is from the installed wheel in private fresh 2×T4 Kaggle jobs. The regular run checks exact native single/multi-GPU S3 layers and NCCL. The separate restricted-visibility job preserves CUDA_VISIBLE_DEVICES=-1 during contrast probes, exercises a real native loader failure trace, and does not run the GPU BFS/kernel phases after discovery fails. The direct API test with empty PATH requires no nvidia-smi. Analysis files, metadata checksums, ZIP exclusions and process completion were verified. Only sanitized summaries are public; environment/cgroup/identity metadata remains private.

[CPU diagnostic tests](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38473917470), [package build](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38474144435), [tuning regressions](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38475035973), [clean GitHub pip install](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38475039908).

## doctor-ok

Observed diagnostic status: PASS. Wall time: 57.573 s. FAIL is expected in injected failure cases.

| Stage | Status | Seconds |
|---|---|---:|
| system-info | PASS | 0.168 |
| filesystem | PASS | 0.118 |
| environment-evidence | PASS | 0.218 |
| nvml-evidence | PASS | 0.169 |
| torch-evidence | PASS | 7.536 |
| driver-api | PASS | 1.974 |
| native-runtime | PASS | 2.828 |
| cuda-discovery | PASS | 1.824 |
| loader-contrasts | SKIPPED | 0.124 |
| native-loader-trace | SKIPPED | 0.121 |
| gpu-0-context | PASS | 2.076 |
| gpu-0-memory | PASS | 1.973 |
| gpu-0-kernel | PASS | 1.973 |
| gpu-0-persistent | PASS | 2.777 |
| gpu-0-bfs | PASS | 4.436 |
| gpu-1-context | PASS | 2.575 |
| gpu-1-memory | PASS | 2.074 |
| gpu-1-kernel | PASS | 2.074 |
| gpu-1-persistent | PASS | 2.878 |
| gpu-1-bfs | PASS | 4.38 |
| nccl-init | PASS | 4.934 |
| nccl-exchange | PASS | 4.886 |
| multi-gpu-bfs | PASS | 5.037 |

Sanitized analysis: `{"primary_failure": null, "confirmed_codes": ["NVML_MANAGEMENT_AVAILABLE"], "hypothesis_codes": [], "unknown_count": 4}`.

## driver-without-nvidia-smi

Observed diagnostic status: PASS. Wall time: 0.516 s. FAIL is expected in injected failure cases.

| Stage | Status | Seconds |
|---|---|---:|
| driver-api-with-empty-PATH | PASS | — |

## doctor-timeout

Observed diagnostic status: FAIL. Wall time: 0.416 s. FAIL is expected in injected failure cases.

| Stage | Status | Seconds |
|---|---|---:|
| system-info | WARNING | 0.025 |
| filesystem | TIMEOUT | 0.025 |
| environment-evidence | WARNING | 0.025 |
| nvml-evidence | WARNING | 0.025 |
| torch-evidence | WARNING | 0.026 |
| driver-api | TIMEOUT | 0.025 |
| native-runtime | TIMEOUT | 0.025 |
| native-loader-trace | WARNING | 0.025 |
| gpu-components | SKIPPED | — |

Sanitized analysis: `{"primary_failure": "filesystem", "confirmed_codes": [], "hypothesis_codes": [], "unknown_count": 8}`.

## doctor-invalid-device

Observed diagnostic status: FAIL. Wall time: 10.242 s. FAIL is expected in injected failure cases.

| Stage | Status | Seconds |
|---|---|---:|
| system-info | PASS | 0.17 |
| filesystem | PASS | 0.118 |
| environment-evidence | PASS | 0.218 |
| nvml-evidence | PASS | 0.169 |
| torch-evidence | PASS | 2.425 |
| driver-api | PASS | 1.974 |
| native-runtime | PASS | 2.826 |
| cuda-discovery | PASS | 1.824 |
| loader-contrasts | SKIPPED | 0.119 |
| native-loader-trace | SKIPPED | 0.119 |
| device-selection | FAIL | — |

Sanitized analysis: `{"primary_failure": "device-selection", "confirmed_codes": ["NVML_MANAGEMENT_AVAILABLE"], "hypothesis_codes": [], "unknown_count": 4}`.

## doctor-missing-native

Observed diagnostic status: FAIL. Wall time: 4.177 s. FAIL is expected in injected failure cases.

| Stage | Status | Seconds |
|---|---|---:|
| system-info | PASS | 0.168 |
| filesystem | PASS | 0.168 |
| environment-evidence | PASS | 0.168 |
| nvml-evidence | PASS | 0.169 |
| torch-evidence | PASS | 2.376 |
| driver-api | PASS | 0.62 |
| native-runtime | FAIL | 0.12 |
| native-loader-trace | WARNING | 0.119 |
| gpu-components | SKIPPED | — |

Sanitized analysis: `{"primary_failure": "native-runtime", "confirmed_codes": ["NVML_MANAGEMENT_AVAILABLE"], "hypothesis_codes": [], "unknown_count": 4}`.

## masked-gpu

Observed diagnostic status: FAIL. Wall time: 18.82 s. FAIL is expected in injected failure cases.

| Stage | Status | Seconds |
|---|---|---:|
| system-info | PASS | 0.169 |
| filesystem | PASS | 0.119 |
| environment-evidence | PASS | 0.219 |
| nvml-evidence | PASS | 0.169 |
| torch-evidence | WARNING | 8.589 |
| driver-api | FAIL | 1.724 |
| native-runtime | PASS | 2.827 |
| cuda-discovery | FAIL | 1.523 |
| loader-contrasts | PASS | 1.674 |
| native-loader-trace | PASS | 1.523 |
| gpu-components | SKIPPED | — |

Sanitized analysis: `{"primary_failure": "driver-api", "confirmed_codes": ["DRIVER_INITIALIZATION_FAILED", "CUDA_VISIBILITY_DISABLED", "NVML_MANAGEMENT_AVAILABLE"], "hypothesis_codes": ["CUDA_SPECIFIC_FAILURE"], "unknown_count": 4}`.

This is single-host component/diagnostic acceptance, not all-architecture, multi-node or graph-throughput acceptance. The affected remote server was not accessed. Hidden kernel/device policy may remain inaccessible to the user; the report identifies that boundary rather than guessing a cause.
