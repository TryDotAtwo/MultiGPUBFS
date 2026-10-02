# Physical protocol build architecture

Runtime source: bb97d5a7674d1cc315b840e75690d4844b128082.
Own diagnostic lease: 53907608, machine 149819, two RTX A4000.
This is not the mandatory T4 acceptance.

The first seven replay invocations failed their initial healthy case. They
are not completed sanitizer or fault-injection gates. Both independent ranks
exited 1; initial DENSE case took 1.656 seconds, without forced cleanup or
group COMPLETE. A healthy-only replay with NCCL_DEBUG=INFO exposed
enqueue.cc:1703 CUDA failure `named symbol not found`.

The notebook incorrectly built NCCL, native CUDA and library owner with sm75
even for the explicitly admitted A4000 diagnostic (sm86). Build target now
follows the admitted hardware, rejects unknown targets before provisioning,
and is recorded in the report. T4 and RTX2070 remain sm75, A4000 uses sm86.
No runtime fallback, payload-padding or synchronization change is introduced.

Regression was RED for missing hardware selection, then GREEN. Python suites:
scripts discovery 23 tests, 1 skipped; tests discovery 203 tests, 8 skipped.
Existing unrelated worktree changes were preserved.

Physical sm86 rebuild resolves the healthy startup failure. Runtime source
remained clean bb97d5a; original wheel and prior sm75 build were retained.
NCCL variant is explicitly experimental minimum_arch_guard.

| Physical two-process gate | Result |
| --- | --- |
| DENSE full S4 state/layer oracle | PASS |
| HASH_FIRST full S4 state/layer oracle | PASS |
| DENSE 18 asymmetric startup/constructor/owner/archive/capacity errors | PASS |
| HASH_FIRST same 18 asymmetric errors | PASS |
| Owner-DAG capture, DENSE and HASH_FIRST | PASS |
| racecheck | PASS |
| synccheck | PASS |
| memcheck | FAIL: one CUDA API error per rank |
| initcheck | FAIL: target fails during NCCL LSA activation |

All 36 fault cases terminate without forced cleanup, with no group or per-rank
false COMPLETE. Observed process duration 0.654-2.208 seconds; these are bounded
failure checks, not performance measurements. Supplemental in-process two-GPU
U/S state/archive/rank-map/pre-dedup oracle and capacity tests also pass.

memcheck marks CUDA_ERROR_NOT_PERMITTED (800) at NCCL allocator.cc:60:
ncclMemAlloc first attempts a FABRIC handle and then retries POSIX when FABRIC
is unavailable. The application completes but sanitizer exits97. No error
filter or vendor-probe suppression was applied: gate remains open.

initcheck rank1 first reports unspecified launch failure719 at NCCL
dev_runtime.cc:1037 cudaStreamSynchronize during ncclDevCommCreate. Rank0
shadow-pool cleanup errors occur after peer cancellation and are not the
established root cause. Both exit1 without false COMPLETE; ERROR SUMMARY:0
does not make this a pass. The known NCCL registration/initcheck problem is
NOT closed.

Dependency SHA256 before NVTX rebuild:

- NCCL sm86: c8a716d75bc40425bdd0640663f5a7f11334416f667401b187cf82e40f77a9ab
- native: 12c9ffdbc1046e62acd5882eac68c2956c39c5506483114d565a1dc5686a0adc
- library owner: b1d86813d68781e1b1520c4ccd94019549b4afeaffa12c55e7651655fa973f7d

Small protocol evidence package saved locally (not a large graph dataset):
build/a4000-sm86-protocol-20261002/sm86-protocol-results.tar.gz,
SHA256 0400d8c7b5766359ef9300860bbca346af3c03fda02df48538becc7400f6eafc.
No benchmark speed or memory advantage is claimed. Full S8 Nsight Systems
timeline uses checksum-pinned2025.3.2 and its isolated NVTX3
headers. An attempted reuse of NCCL-source
include failed before execution because it shadowed installed NCCL headers;
the isolated profiler include avoids that conflict. Mandatory physical2T4
remains open.

Full S8 timeline completed successfully:40320 canonical states and29 layers,
two independent rank-process archives checked against CPU oracle. Each rank
records178 explicit same-thread mgbfs.batch ranges. Within them:

| API category | Rank0 | Rank1 |
| --- | ---: | ---: |
| cudaStreamSynchronize | 0 | 0 |
| synchronous cudaMemcpy | 0 | 0 |
| cudaEventSynchronize | 0 | 0 |
| async copies, all nested mgbfs.archive_d2h | 352 | 346 |
| unscoped batch copies | 0 | 0 |

Nonblocking event queries and GPU-side StreamWaitEvent remain. No evidence in
these ranges of D2H counts/control required to submit the next healthy batch.
Finalization still contains synchronous controls as designed. API callchain
symbols were not resolved by this capture; attribution above uses explicit
same-thread NVTX containment, not an inference from missing stacks.

Across the entire recorded GPU interval span (including setup/finalization),
rank0/rank1 busy union is384.981/429.010ms, multi-stream union10.606/8.684ms,
compute/copy overlap0.561/0.511ms. These are recorded intervals, not occupancy,
FLOPS, useful work, or benchmark throughput. This small profiled S8 does not
establish optimal overlap or an advantage over CayleyPy.

Trace package7.3MiB excludes graph archives, saved locally at
build/a4000-sm86-protocol-20261002/sm86-timeline-evidence.tar.gz,
SHA256 eef1ce73f6a1b75353349f6d587dd0a7f8c4db6e11aa5cd66a156c6c0bf3691d.
Nsight2025.3.2 exact package digest verified; isolated profiler NVTX3 headers.
The reduced vendor device probe also now uses the same admitted architecture
instead of its separate hardcoded sm75 flag.

Reduced existing experiments/nccl_window_isolation.cu (sm86, same NCCL/CUDA)
also reproduces the initcheck failure without any BFS code. Two independent
plain processes return0 and both reach window_register/device_comm_create.
Under initcheck, both return7 and neither reaches window registration success;
rank1 first reports dev_runtime.cc:1037 unspecified launch failure719, while
rank0 later times out at the probe's unchanged30s progress bound. Both report
zero sanitizer memory errors. This supports independent vendor/tool-path
reproduction, not a proven underlying cause or a completed gate.

Probe evidence saved locally at build/a4000-sm86-protocol-20261002/
sm86-vendor-probe.tar.gz, SHA256
a29cf26ba62ea467b32abec5c21fee178c33741575c9614f4a51de670ca11869.
Final Python tests203 pass,8 skips; supervisor9 pass,1 skip.

## Additional profile and owner coverage

Clean runtime bb97d5a, same two independent A4000 rank processes:

- HASH_FIRST S8 timeline passes all 40320 states and 29 layers. Each rank
  records 178 batch ranges, with zero host StreamSynchronize, synchronous
  memcpy or EventSynchronize inside those ranges. Non-archive copies are
  1068 D2D copies per rank; archive D2H copies are 352/346. This does not
  establish optimal overlap or performance superiority.
- CUCO_RANK and CUB_SORT_MERGE, each DENSE/HASH_FIRST, pass S8 with rank map
  [1,0] and pre-dedup OFF. Every canonical state at every depth matches CPU.
- CUB_SORT_MERGE passes the healthy S4 case plus 18 asymmetric failures per
  profile: 36 additional failures terminate without false group COMPLETE.
- Seed 1 passes both profiles with CUCO_RANK. Correct seed 20260828
  (hex 013527dc) passes both profiles with CUB, map [1,0], pre-dedup OFF.
  Earlier directories labelled seed20260828-cub actually used 20261084
  (013528dc) and are not credited as the required seed.
- Ordinary parallel cargo test --locked passes again, without strace or
  serial override. This does not explain prior intermittent bootstrap timeouts.

Small continuation evidence excludes graph archives:
build/a4000-sm86-protocol-20261002/sm86-mode-continuation.tar.gz,
SHA256 7ef8d38e666c30a81fbe6714a60977bf54397e0a1234ed3eb7abe81fbdfbbe42.
This package covers timeline, map/pre-dedup and CUB failures, not seed runs.
The physical T4 gate, unfiltered memcheck/initcheck, paired A/B, production
CLI and distributed macro-depth remain open. No runtime fallback, vendor
error suppression or paid lease extension was introduced.
