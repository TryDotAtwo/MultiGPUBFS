# NCCL source-build admission, 2026-10-02

Scope: pinned NCCL v2.29.7-1, b91894bd5b190c874d98a017f93f5daa515b65d0,
stock versus minimum-architecture experimental patch. No production default change.

## Observed failures before GPU acceptance

- Stock sm86 source build with NVTX=0 completed, but the actual CLI under
  unfiltered memcheck exited 1 before its first instrumented CUDA API:
  `undefined symbol: _Z23initNvtxRegisteredEnumsv`.
- `ldd` resolved libnccl.so.2 to the intended stock source-build directory,
  not the wheel. Raw log is in
  `test_results/local_lsa_20261001/stock-source-sm86-memcheck/rank0.log`.
- Upstream src/Makefile excludes init_nvtx.cc with NVTX=0, while init.cc
  calls initNvtxRegisteredEnums. This run is not a sanitizer pass or a BFS result.
- Candidate source copied from Windows initially failed with
  `/usr/bin/env: 'python3\r': No such file or directory`.
  Build-copy tracked files were mechanically normalized to LF. Its git diff
  remained exactly five files and 19 additions matching the experimental patch.

## In progress

Candidate full CUDA build resumed successfully after line-ending normalization.
Stock host sources are being rebuilt with the upstream default NVTX=1.
Device source search found no NVTX use in src/device; existing device objects
remain the full, unchanged sm86 collective families. Candidate must also use
NVTX=1 before the paired test. No sanitizer acceptance is claimed here.

Both builds emit upstream dev_runtime.cc:1042 maybe-uninitialized warning for
outDevCommPreserve. It is retained as a warning, not promoted to a proven bug.
Two-rank T4 acceptance and the registration initcheck failure remain open.

## Stock NVTX=1 actual BFS result

The stock host rebuild completed. The same CLI/config with the verified stock
library path then executed S4 with capture enabled under unfiltered memcheck.
Observed process exit 1, ERROR SUMMARY: 12 errors, all reported as error 209
at cudaFuncGetAttributes/cudaGetLastError in ncclInitKernelsForDevice.
Thus compiling pristine source specifically for sm86 does not remove the
previous wheel's architecture-probe failures.

Raw log and archive:
`test_results/local_lsa_20261001/stock-source-sm86-nvtx1-memcheck/`.
Seven owner DAG launch markers were present. Independent checksummed archive
reader plus canonical CPU oracle passed all 24 unique states, with layer sizes
`[1, 3, 5, 6, 5, 3, 1]`. Command:
`verify_process_archives(case, n=4, world=1)` from the existing replay script.
This is correctness evidence for one rank only; memcheck remains FAILED.
Candidate comparison is still pending its live full CUDA build.

## Stock NVTX=1 initcheck

Same one-rank S4/config, unfiltered initcheck: exit 0, ERROR SUMMARY: 0 errors.
Seven capture launch markers and durable archive are present. Independent
full-state oracle again passed 24 states with the same seven layer sets.
Raw artifacts: `test_results/local_lsa_20261001/stock-source-sm86-nvtx1-initcheck/`.
This does not close the previous two-rank T4 registration initcheck failure.

## Remaining stock local tools

Unfiltered synccheck and racecheck both exited 0. Synccheck reported zero
errors; racecheck reported zero hazards, errors and warnings. Each produced
seven capture launch markers. Each separate archive passed the full-state
CPU oracle, 24 unique states and the same seven layers.
Artifacts use the same prefix with `-synccheck` and `-racecheck` suffixes.
No filtering or suppression was used. Overall four-tool acceptance still
fails because memcheck failed; this is one rank, not T4 group acceptance.

Hardware: RTX 3070 Laptop GPU, 8192 MiB, driver 572.70.
Compute Sanitizer 2025.1.0.0 build 35583870; CUDA build 12.8.93.
SHA256 of actual tested binaries:

- stock libnccl.so.2.29.7: a7b0971d91e22e28a8dcb2a4b0901778e367d4a50b6d53a1d970656a5ef5a49d
- libmgbfs_cuda.so: 5ab7a5ec3d0c2634f73584b79e791c1b84063a61738787103231d6f7ae3a03de
- mgbfs: 438a1a92a66a26b8304a0675537101d4a53fa81136bf59760f35f97d55a8f747

## Candidate result: four local tools PASS

The NVTX=1 candidate host rebuild completed and `ldd -r` showed no unresolved
symbols. Actual candidate NCCL SHA256:
`dc42ed2f993d31908ca248cc9994bc0b46d3e124ecb4a509d7dd14f145a18856`.
Actual runtime resolution selected /linux-build/nccl-archguard-build/lib.
Same CLI binary, native library, S4 config and GPU as stock above.

All four unfiltered tools exited 0: memcheck/initcheck/synccheck report zero
errors; racecheck reports zero hazards, errors and warnings. Each run launched
seven captured owner DAGs. Each independent archive passed the canonical CPU
oracle at every depth, 24 unique states, [1,3,5,6,5,3,1]. Raw logs and archives
are under `test_results/local_lsa_20261001/archguard-sm86-nvtx1-<tool>/`.
No sanitizer API reporting was disabled or errors suppressed.

Stock versus candidate isolates a local minimum-architecture probe fix:
stock memcheck 12 errors versus candidate zero, with both full-state oracles
passing. Generated normal/symmetric tables retain all stock tokens except
added metadata, respectively 40 and 50 aligned entries. Comparison initially
failed byte equality because of one extra blank line; token comparison passed.
This does NOT establish two-rank T4 registration correctness, full mode
coverage, protocol races, production-default admission or performance benefit.

## Profile and failure follow-up

CUB DENSE/HASH_FIRST with pre-dedup ON/OFF each passed unfiltered memcheck
and the full-state S4 archive oracle. Prefixes:
`test_results/local_lsa_20261001/archguard-CUB_SORT_MERGE-<profile>-<pre>/`.
BMMA sm86 was rejected by the explicit sm75-only hardware policy, status 3;
zero sanitizer errors in that failed application are NOT a passing gate.

Runtime constructor message fallback fixed in 38479e5 after a RED test
observed the empty string. Seven CUDA runtime unit tests, full Windows CPU
workspace and all eleven Linux CUDA CLI tests passed. Rebuilt CLI actual
sm86 BMMA refusal now reports NATIVE_PLAN_CREATE_FAILED status=3 (exit 1).

Fresh CLI with candidate NCCL, one rank: injected startup, actual owner
capacity exhaustion, archive admission and archive finish each exited 1
within the 60-second outer limit, with no group-complete.json. Logs:
`test_results/local_lsa_20261001/archguard-single-injected-<fault>/`.
Owner-host fixture is unreachable on this single-rank schedule: it completed
normally, so no owner-host failure coverage is claimed. First startup attempt
also completed normally due to a mistyped environment name without FAULT;
both raw attempts are retained, not counted as fault passes.

Existing private T4 wrapper pins 6e22335540d99f3de2dfd0c0683fad5c52ef1450,
explicit minimum_arch_guard, published in 8702d26. Candidate is built at the
same dependency root used by independent-process replay; original wheel is
preserved separately. Local Python suite: 202 tests, eight skips, exit 0.
Kaggle push was rejected with Maximum weekly GPU quota of 30.00 hours reached.
No new worker was admitted. All two-rank candidate gates remain open.

## Larger local full-state gate

Fresh CLI following constructor status fix: S8, batch 1024, capacity 40320,
future 80640, CUB with candidate NCCL, both DENSE and HASH_FIRST, pre-dedup OFF.
Both unfiltered memcheck runs exited 0 with zero errors. Each checksummed
archive independently matched every CPU layer set: 40320 unique states,
29 layers. Artifacts: `test_results/local_lsa_20261001/archguard-s8-<profile>-OFF/`.

Initial full-BFS timeline attempt failed with PROFILE_NVTX_NOT_BUILT before
search. Its trace is NOT a BFS timeline or an overlap result. Separate native
NVTX build was started under /linux-build/lsa-cuda-nvtx, preserving the tested
non-NVTX library. Trace ranges alone do not enable diagnostic TRACE_ROUTE waits.

## Full local S8 timelines

NVTX include root initially shadowed installed NCCL headers and compilation
failed on missing doca_gpunetio_device.h. Isolated nvtx3 headers resolved it;
the separate profiling native build completed. Nsight Systems 2025.6.3,
trace=cuda,nvtx,osrt, sample=none, cpuctxsw=none; debug capture and TRACE_ROUTE
were unset. DENSE attempt requested CUDA backtraces but profiler warned that
they were not collected with sampling disabled. No callchain claim is made.

Both actual full BFS runs exited 0, exported SQLite, and each archived all
40320 states at the CPU oracle's 29 layers. Each trace contains 61 mgbfs.batch
ranges. SQL attributed CUDA APIs whose start/end and globalTid fall within a
batch, excluding nested mgbfs.archive_d2h ranges on that same thread.

| Batch-exclusive API/copy | DENSE | HASH_FIRST |
| --- | ---: | ---: |
| host synchronize | 0 | 0 |
| synchronous memcpy | 0 | 0 |
| H2D/D2H copies | 0 | 0 |
| cudaEventQuery | 29 | 126 |
| cudaEventRecord | 244 | 244 |
| cudaStreamWaitEvent | 183 | 244 |
| kernel launches | 3842 | 6099 |
| async copies, all D2D | 122 | 244 |
| D2D bytes | 2903040 | 2910848 |
| async memset | 774 | 1994 |

CopyKind 8 was checked against trace ENUM_CUDA_MEMCPY_OPER (Device-to-Device),
using CUDA runtime/memcpy correlationId joins. Startup/FinalizeDepth/archive/
teardown have additional waits and are not falsely declared absent.
Artifacts: test_results/local_lsa_20261001/archguard-s8-<profile>-nvtx-timeline/.
This establishes bounded one-rank timeline evidence only. EventQuery remains;
no maximum-overlap, two-rank transport or performance acceptance is claimed.

## Larger local four-tool completion

Remaining S8 DENSE/HASH_FIRST OFF initcheck, synccheck and racecheck all
completed with exit 0; unfiltered summaries are zero errors/hazards/warnings.
All six archives independently match 40320 canonical CPU states at 29 depths.
Together with their earlier memchecks this covers all four tools on both
larger local profiles. No capture or TRACE_ROUTE was enabled in these six.
Raw artifacts: test_results/local_lsa_20261001/archguard-s8-<profile>-<tool>/.
Expanded T4 configuration adds CUCO_RANK to the same profile/equivalence/fault/
sanitizer/timeline loops (8896665), wrapper pin published in 08fcb8a.
Fresh complete Python suite: 202 tests, eight skipped, exit 0.

## Local pinned CUCO admission and four-tool gate

CUDA 12.8.93 cannot compile pinned RAPIDS 26.04 headers because
cudaDevAttrHostNumaMemoryPoolsSupported is absent. No vendor-header edits or
library downgrades were made. The existing four CUDA_COMPONENTS archives from
kaggle/library-owner/kernel.py were downloaded and SHA-256 verified.
CUDA 12.9.86 built the library in a separate build directory.
Pinned cuCollections: 532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4.
Library-enabled debug CLI SHA-256:
fce68988462b73837c93b92aa0467c62b0b52ada3b116a8ab00b61de6841b411.
Library owner SHA-256:
b5689f1ff55ee52e9430de4b43f3c1768ed3fadf2205c5da232159f86369eec7.
ldd -r resolves separate CUDA 12.9, pinned RAPIDS and the experimental
minimum-arch NCCL candidate without unresolved symbols. Previous native-only
executables and libraries were not replaced.

Actual one-rank CUCO_RANK + DENSE + LSA, pre-dedup OFF, S4 batch 7 completed
with seven owner-DAG capture launches. Its archive independently matches all
24 canonical CPU states and layer sizes [1,3,5,6,5,3,1]. Artifacts:
test_results/local_lsa_20261001/cuco129-s4-DENSE-OFF/.

All eight additional runs, DENSE/HASH_FIRST OFF times memcheck/initcheck/
synccheck/racecheck, completed with exit 0 and zero errors/hazards/warnings.
No kernel filters, CUDA API error suppression or reporting waivers were used.
Each archive independently matches all 24 canonical states at all seven depths.
Artifacts: test_results/local_lsa_20261001/cuco129-s4-<profile>-<tool>/.
This is local RTX 3070 Laptop evidence, not two-rank T4 acceptance, a fix for
T4 NCCL registration initcheck, an ideal-overlap claim or an A/B speed result.

## Larger CUCO full BFS gate and timelines

S8 CUCO_RANK OFF DENSE/HASH_FIRST each completed all four unfiltered tools,
exit 0 and zero errors/hazards/warnings. All eight archives independently
match 40320 canonical CPU states at the 29 reference depths. Artifacts:
test_results/local_lsa_20261001/cuco129-s8-<profile>-<tool>/.

Separate complete Nsight runs use native NVTX library, no owner-DAG capture,
no TRACE_ROUTE, trace=cuda,nvtx,osrt, sample=none, cpuctxsw=none. Both archives
match the same full-state oracle. Each trace has 61 mgbfs.batch ranges.
Attribution contains API start/end within same-thread batch ranges and excludes
nested same-thread mgbfs.archive_d2h, as in the native comparison above.

| Batch-exclusive API/copy | CUCO DENSE | CUCO HASH_FIRST |
| --- | ---: | ---: |
| host synchronize | 0 | 0 |
| synchronous memcpy | 0 | 0 |
| H2D/D2H copies | 0 | 0 |
| cudaEventQuery | 28 | 22 |
| cudaEventRecord | 244 | 244 |
| cudaStreamWaitEvent | 183 | 244 |
| kernel launches | 3110 | 5367 |
| async copies, all D2D | 122 | 244 |
| D2D bytes | 2903040 | 2910848 |
| async memset | 957 | 2177 |

Whole recorded GPU activity (not batch-exclusive, occupancy or critical path):
DENSE span 475616685 ns, busy 25153490 ns, multi-stream 1272124 ns,
compute/copy overlap 414644 ns. HASH_FIRST span 558475845 ns, busy 46145260 ns,
multi-stream 1010806 ns, compute/copy overlap 262641 ns. Whole trace includes
setup/archive/finalization; no ideal pipeline utilization is claimed.
Raw full-bfs.nsys-rep/SQLite and oracle archives are in
test_results/local_lsa_20261001/cuco129-s8-<profile>-timeline/.
This remains one-rank local evidence, not two-rank T4 or paired A/B acceptance.

## CUCO local failure-path validation

The same library-enabled CLI and pinned dependencies ran DENSE/HASH_FIRST
OFF S4 batch 7 with each of six debug fault hooks: NCCL startup, early
constructor, late constructor, real owner reservation capacity, archive
admission, archive finish. All 12 runs returned exit 1 under a 60-second outer
timeout (none returned 124); each expected fault marker was checked and no
result/group-complete.json exists. Capacity reaches actual sticky device
fatal LIBRARY_RANK_DEPTH_FATAL_16_16; teardown records revoke in-progress 7,
ready 0 and abort 0. Startup/admission/finish preserve their original errors.
Raw logs and exit codes:
test_results/local_lsa_20261001/cuco129-fault-<profile>-<fault-env>/.
The post-owner host hook is intentionally not counted: its scheduled
multi-rank condition is unreachable at world=1. None of these runs proves
asymmetric cancellation, receive-slot reuse or termination on two T4 ranks.
Kaggle status rechecked: trydotatwo/mgbfs-native-rank-owner-t4 COMPLETE;
no new worker was launched and the prior quota rejection is not a gate pass.

## CUCO pre-dedup ON full-state gate

Session 88584 completed exit 0. S8 DENSE/HASH_FIRST pre-dedup ON each passes
memcheck, initcheck, synccheck and racecheck unfiltered, zero errors/hazards/
warnings. All eight canonical archives independently match 40320 states at
29 CPU oracle depths; thus the checked layer sets match the OFF cases too.
Artifacts: test_results/local_lsa_20261001/cuco129-s8-<profile>-ON-<tool>/.
No speed claim is based on instrumented timings. Two-rank ordering, asymmetric
failures, T4 registration and remote paired A/B remain separate open gates.

## sm75 compilation and T4 admission recheck

The sole private native-rank-owner T4 gate was COMPLETE before a fresh push;
Kaggle again rejected admission with Maximum weekly GPU quota of 30.00 hours
reached. CLI exit 0 does not establish a new worker or kernel version.

Separate CUDA 12.9.86 sm75 builds of current native LSA and pinned CUCO owner
both completed exit 0 (session 71947), existing sm86 builds preserved.
Native SHA-256: 72ef1d190454beb2c32e24ed25f60cde4aa3806404c869b350f729b43c4a1d20.
CUCO SHA-256: 06fe8a7dab8ab8fadaa9ca5e79f9b0b353eae23b117cda7c544b7369a67e2938.
cuobjdump confirms sm_75 cubins in both libraries. CMake warns unused
CUDAToolkit_ROOT for the native target; actual compiler is explicitly pinned
/linux-build/cuda-12.9/bin/nvcc. The linked experimental NCCL library in this
local compile environment remains its sm86 build: these are compile artifacts,
not a complete deployable T4 dependency set or hardware acceptance.
Build paths: /linux-build/lsa-cuda-sm75-129 and library-owner-sm75-129 in the
existing isolated Docker volume. No new rental or second notebook was started.
