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
timeline is in progress using checksum-pinned2025.3.2 and its isolated NVTX3
headers; timeline still requires inspection. An attempted reuse of NCCL-source
include failed before execution because it shadowed installed NCCL headers;
the isolated profiler include avoids that conflict. Mandatory physical2T4
remains open.
