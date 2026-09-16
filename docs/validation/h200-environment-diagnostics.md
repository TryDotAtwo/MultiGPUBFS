# H200 environment diagnostics (2026-09-17)

Status: **INCOMPLETE**. This is not a clean sanitizer gate or a performance result.

The first eight-H200 rental was destroyed after collecting small diagnostic
artifacts. Environment work moved to two H200s before another eight-GPU rental.
The shared project budget remains USD 100, including a USD 20 reserve.

## Reproduction

- BFS source: `cc6aa3c` (two-GPU build).
- CUDA toolkit / Compute Sanitizer: 12.8.
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

## Evidence location

Local, ignored evidence for the retired first rental:
`test_results/vast-51248110/evidence/` and its `termination.json`.
Current two-GPU raw logs are under `/workspace/gate2-raw/` on the setup rental;
they must be copied before rental deletion. No graph-state dataset is downloaded
to the workstation. Rental details and watchdog records are local ignored files,
not credentials in source control.
