# Production protocol replay and timeline

Vast 53711193: two RTX A4000, driver 595.71.05, bidirectional P2P.
Not T4 acceptance. Runtime 9c07246, clean candidate diff.

DENSE and HASH_FIRST each pass eleven independent-process cases:
full S4 state/depth oracle, either-rank startup, owner, admission,
archive-finish and owner-capacity errors. No forced cleanup or false
group COMPLETE. Additional U/S oracle and capacity tests return zero.
DENSE fault exits take 0.65-1.61 seconds.

Initial verifier failed because pyarrow was missing. Environment repair
uses pyarrow 19.0.1; 820ff06 adds it to the existing provisioner.
Evidence: test_results/vast-a4000-protocol-20261001.tar.gz, SHA256
159396655f1d81f2a692eed29c1536dfba027be15f2f149e6b317c09416e115c.

Full independent-process DENSE S4 sanitizer results:
- Racecheck and synccheck PASS, including full archive oracle.
- Memcheck FAIL: both ranks exit 97; 13 API errors each. Kernel probes
  in NCCL initialization and cuMemCreate FABRIC probe in ncclMemAlloc.
- Initcheck FAIL: both ranks exit 1, CUDA_STATUS_1, no group-complete.
  Zero tool memory errors does not mean application success.

Sanitizer 2025.2.1.0 build 35969825, NCCL wheel 2.29.7. No suppression.
NCCL upstream v2.29.7-1 is b91894bd5b190c874d98a017f93f5daa515b65d0.
enqueue.cc tolerates failed kernel attribute probes; allocator.cc retries
without FABRIC on NOT_PERMITTED. This does not waive strict gates.

b06bc92 adds archive NVTX ranges and enables batch ranges in Nsight
replay without trace_route waits. NVTX build ON, Nsight 2025.3.1.90.
DENSE S8: all 40320 states at every depth match CPU oracle. Each rank
records 178 batch ranges. Same-host-thread runtime API attribution:
zero Synchronize, zero synchronous memcpy, zero non-archive D2H.
Small 16-byte D2H copies (2/3 per rank) are nested archive copies.
Raw reports and SQLite: test_results/vast-a4000-dense-s8-timeline-20261001.tar.gz.

Earlier same-workload trace: multistream activity 4.38%/3.03% of recorded
busy GPU intervals, not occupancy, critical-path overlap or A/B proof.
FinalizeDepth host checks remain. Covers DENSE/LSA/CUCO_RANK only.
HASH_FIRST S8 timeline also passes the full 40320-state oracle, with 178
batches/rank, zero non-archive D2H, host Synchronize or synchronous memcpy
inside those ranges. Evidence: test_results/vast-a4000-hash-first-s8-timeline-20261001.tar.gz.
T4, memcheck/initcheck, remaining backends and full objective stay open.

## Startup follow-up

NCCL_DEBUG phase logs place the initcheck error inside window registration,
with dev_runtime.cc reporting CUDA launch failure. The existing independent
NCCL-only fixture also reproduces registration failure with two separate
rank processes: plain exits 0/0 in 1.61 seconds, initcheck exits 7/7 in
35.54 seconds, without supervisor timeout. This narrows the failure but
does not prove a universal NCCL or sanitizer defect.

Initializing the entire symmetric window before registration does not
repair full BFS initcheck (both ranks still exit 1). That candidate was
removed from both source trees; no persistent extra payload writes added.

25f1db1 passes the existing startup cancellation token into admission,
LSA setup and final constructor-vote waits, which previously passed None.
DENSE and HASH_FIRST each pass all eleven process cases and the additional
U/S oracle/capacity checks after this change. These regression tests do
not isolate cancellation halfway through a stalled constructor vote;
that specific injection and constructor error notification before cleanup
remain to be verified.

Evidence: test_results/vast-a4000-startup-initcheck-20261001.tar.gz contains
the reduced fixture logs, phase diagnostics, rejected initialization
candidate and DENSE regression. HASH_FIRST regression remains on the
bounded rental pending artifact collection.
