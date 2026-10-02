# LSA logical failure closes bounded epoch admission

Base: a35d734. A completed device epoch previously released its host credit
even after a logical owner/ring failure. Device predicates suppressed semantic
writes, but repeated completed poisoned epochs could admit the rest of the
scheduled depth. K live events alone did not bound that cumulative fatal tail.

The existing owner LSA vote now publishes nonzero into the already allocated
mapped terminal word, after its final rendezvous. Only an owner vote does so;
generic reductions may carry nonzero values without an owner error. Value 1
means host cancellation, 2 device logical failure. Concurrent publishers only
store nonzero; no PCIe atomic RMW, new allocation, count/control memcpy,
callback, maximum payload transfer or second NCCL-calling thread was added.

The real runtime borrows this word from Comm and checks it using host Acquire
loads at existing admission probes. A completed credit is checked again after
cudaEventQuery success, closing the probe/completion race. This is failure-only
mapped signalling, not a successful batch count readback. Outer advance retains
the existing failure report -> serialized abort/retirement protocol. Comm owns
the word for the lifetime of the borrowed LsaView. FinalizeDepth remains.
The existing GROUP_OWNER_OR_PRE_OWNER_FATAL category is retained as the
logical-error prefix for the independent replay and two-GPU capacity gate.
The aligned four-byte mapped load/store choice follows NVIDIA's
[CCCL memory model](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html):
system-scope aligned mapped loads/stores are covered separately from RMW
operations requiring host native atomic support. The alignment is checked.

## Verification

Real runtime regression was RED: `completed fatal epoch reopened batch
admission`. GREEN tests a healthy vote, ring fatal, owner error, and nonzero
generic vote without false cancellation. No mock CUDA/NCCL implementation.

- Final 19 runtime CUDA/library unit tests pass (session 25884).
- Seven macro integration and 12 CLI integration tests pass (session 69040).
- Default-members cargo test --locked passes (session 26781, exit 0).
- Four unfiltered Compute Sanitizer tools pass on the new real one-rank LSA
  regression, error-exitcode 97: memcheck/initcheck/synccheck zero errors;
  racecheck zero errors/warnings/hazards. Session 25884 exits zero.
- Four fresh S8 archives: DENSE/HASH_FIRST x pre-dedup ON/OFF, CUCO_RANK+LSA,
  batch 1024. Independent full-state CPU oracle verifies 40320 unique matrix
  states and all 29 exact layer sets in each archive.
- Four actual S4 capacity failures, same profile/pre panel, batch 1, owner
  capacity injection rank 0: each exits 1 inside external 30-second deadline,
  reports LSA_DEVICE_LOGICAL_FATAL, no rank COMPLETE or group-complete file.
  Existing serialized abort completes locally; archive stays incomplete.
  The final eight-case panel (session 59052, exit 0) was independently
  verified again on the backward-compatible error-prefix binary.

Artifacts: test_results/lsa_logical_fatal_20261002/, including four tool logs,
final eight actual BFS case directories and binaries-final.sha256. Native library SHA256:
03bce45f18d58de34ee33e792c1767db383fd5e2f23f88dbbd3ec16bcc87ca89;
runtime test: 977f99cdcc2af1009bf11c3078b7098504b9c4002981708b94d01d483bc33482;
CLI: 4be24145ba4f0296ed2cefcee7d3b8a8f00e869732ad223f36c017924788637f.

Scope: one local sm86 RTX 3070 Laptop, diagnostic NCCL archguard. This does
not prove peer-visible two-process mapped cancellation/retirement, full BFS
four-tool gates on T4 or NCCL registration initcheck. Fresh two-T4 asymmetric
capacity/owner/archive tests and final timeline/A-B remain required. Earlier
c555339 timeline describes the preceding source, not this candidate's trace.
No distributed macro or full-goal completion claim.

Published runtime: edd45cd712a6b2964f1db8236fb1f7e97b65e35f. The sole
private native-rank-owner T4 gate pins that source (config 17c5bfe). Previous
worker status was COMPLETE; new push was rejected with Maximum weekly GPU
quota of 30.00 hours reached. CLI exit zero is not admission. No new worker
or concurrent notebook was started; target-hardware evidence remains open.
