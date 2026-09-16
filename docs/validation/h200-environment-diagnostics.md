# H200 environment diagnostics (2026-09-17)

Status: **INCOMPLETE**. This is not a clean sanitizer gate or a performance result.

The first eight-H200 rental was destroyed after collecting small diagnostic
artifacts. Environment work moved to two H200s before another eight-GPU rental.
The shared project budget remains USD 100, including a USD 20 reserve.

## Reproduction

- BFS source: `cc6aa3c` (two-GPU build).
- BFS compiler/runtime: CUDA 12.9 (nvcc 12.9.86). Initial diagnostics used the
  host Compute Sanitizer 12.8; this mismatch was discovered by inspecting
  `runtime-env.json` and both nvcc versions. A matching 12.9 sanitizer is required
  for the next validation. Do not present the initial diagnostics as a matched
  toolchain gate.
- Two physical H200 devices connected through NV18.
- Host image NCCL package: `2.31.2-1+cuda13.3`.
- Diagnostic NCCL: upstream `v2.28.9-1`, commit
  `dbc86fd06e8b0c4517b95d8958a09ccacf9520c9`, built with CUDA 12.8 and
  `NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"`.
- Diagnostic processes explicitly select this library using `LD_PRELOAD` and
  `LD_LIBRARY_PATH`; the system package was not replaced.
- Exact Rust test: `library_two_rank_layers_and_archives_match_oracle`.
- Sanitizer arguments: `--print-limit 0 --error-exitcode 97`. No API-error
  suppression or `--report-api-errors no` is used.

## Observed results

| Test | Result |
|---|---|
| Native/CUCO/Rust SM90 build | Completed |
| Uninstrumented two-device full-state/archive oracle | 1 passed, 0 failed, 0 ignored; 19.57 s |
| Same oracle under memcheck | Oracle passed; process exit 97; 99.54 s |
| Raw memcheck diagnostics | 5,758 errors: 384 code 209 and 5,374 code 704 |
| Racecheck / initcheck / synccheck | Pending at this checkpoint |

All 5,758 raw memcheck messages are API diagnostics. The full log contains no
additional diagnostic category. This observation is not a substitute for the
other sanitizer tools or a clean raw gate.

## Source-backed explanation, not suppression

In the pinned NCCL source, `src/enqueue.cc:52` calls
`cudaFuncGetAttributes` while enumerating available kernels. Its explicit error
branch drains the CUDA error and continues. Code 209 diagnostics occur at that
call and its `cudaGetLastError`. An independent NCCL-initialization-only probe
also reproduces these without running BFS kernels. A single-SM90 NCCL build
does not eliminate them.

In `src/transport/p2p.cc:329-331`, same-process ranks enable peer access. NCCL
explicitly handles `cudaErrorPeerAccessAlreadyEnabled` (704) with
`cudaGetLastError`. The two-device fixture uses threads in one process, not the
production one-process-per-rank topology. The raw stack traces identify that
path. A production torchrun validation remains necessary.

Changing NCCL versions did not produce a raw-zero gate. Do not continue random
version changes, disable API reporting globally, or reclassify this run as PASS.
If expected vendor probes are eventually classified separately, that must be a
documented validation-contract change with complete raw logs retained and unknown
API errors, memory errors, races, and synchronization errors still fatal.

## Matched-tool follow-up

The official CUDA 12.9.1 redistribution manifest identifies sanitizer 12.9.79,
SHA256 `e23aad21132ff58b92a22aad372a7048793400b79c625665d325d4ecec6979bf`.
The downloaded archive matched this digest; its executable reports Compute
Sanitizer 2025.2.1.0. The NCCL-only two-device probe still reports 16 API errors
under that tool, with successful NCCL initialization. Thus the initial toolchain
mismatch is real but is not the explanation for these vendor probe diagnostics.

`remote_build.py` now includes the matching sanitizer in its verified downloads
and exported PATH. Unit tests reject compiler/runtime version drift. This change
has not yet been exercised through a full fresh remote build.

`audit_nccl_sanitizer.py` classifies the complete initial memcheck log as 384
kernel-availability and 5,374 peer-already-enabled diagnostics at the exact
NCCL host call sites above. It rejects unknown call sites, extra diagnostic
categories, missing/duplicate summaries, and unaccounted counts. Its output
always has `gate_pass: false`: neither library identity nor oracle completion is
proved by a diagnostic classifier. The existing eight-GPU gate is unchanged.
The raw log SHA256 is
`82d0385d35c7f12762419a376bbbb0597d0ab336e930921fe96468e84ad8c70d`.

## Completed two-H200 follow-up

- Raw racecheck: oracle passed, 753.96 s, 0 hazards/errors/warnings.
- Raw synccheck: oracle passed, 26.75 s, 0 errors.
- Raw initcheck with default NCCL allocation: oracle passed, 166.76 s, but
  **4,633 uninitialized-memory diagnostics**. These are not API probe messages
  and were not allowlisted.
- An independent C++ NCCL-only collective probe fully initializes 4 KiB send
  and receive buffers on both devices, checks 1/1024-element max reductions and
  a 4 KiB peer exchange. Correct results with default NCCL allocation still
  produce 3,589 initcheck diagnostics under matched sanitizer 12.9.
- Changing only `NCCL_CUMEM_ENABLE=0` yields **0 errors** on that probe.
- The complete two-device BFS full-state/archive oracle was then repeated with
  `NCCL_CUMEM_ENABLE=0` and sanitizer 12.9: **PASS, exit 0, 0 initcheck errors**,
  56.31 s test time (57.60 s including tool overhead). This localizes the
  diagnostic to NCCL's virtual-allocation path; it does not prove whether the
  underlying issue is NCCL behavior or sanitizer alias tracking.
- Production configuration will explicitly use `NCCL_CUMEM_ENABLE=0`; this is
  selected before startup, not a runtime fallback or disabled memory check.
- Actual two-process torchrun, DENSE compact permutation states, search-only:
  CUB and cuCollections both completed S4 (24 states, 7 layers) and S8
  (40,320 states, 29 layers), with matching complete layer counts. These are
  smoke tests, not a statistically meaningful performance comparison.
- The first CLI attempt rejected an incorrectly supplied library-pool setting
  for CUB (`REFERENCE_UNUSED_LIBRARY_POOL`). Correcting the harness to set that
  option only for cuCollections resolved the configuration error.

The two-H200 rental was deleted after preserving logs. Estimated base rental
cost is USD 4.67; final invoice/traffic reconciliation is still pending.
The eight-H200 production run has not yet happened at this checkpoint.

## Evidence location

Local, ignored evidence for the retired first rental:
`test_results/vast-51248110/evidence/` and its `termination.json`.
Current two-GPU raw logs are under `/workspace/gate2-raw/` on the setup rental;
they must be copied before rental deletion. No graph-state dataset is downloaded
to the workstation. Rental details and watchdog records are local ignored files,
not credentials in source control.
