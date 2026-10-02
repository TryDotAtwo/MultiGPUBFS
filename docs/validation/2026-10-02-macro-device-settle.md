# Macro producer and device-count settlement

The working macro runtime now uses existing `mgbfs_macro_settle_run_frontier`
directly on the sorted future produced by future merge. Identity refs index
the dense future state plane. The redundant target radix sort, D2H future
counts followed by H2D count upload, and producer-ending host drain are gone.
All future fatal/count inspection remains at depth finalization, including
provisional targets later than the layer being settled. History count
publication is device-to-device; state reset is stream-ordered memset. The
extra drain after history publication was removed without changing archive
last-reader events. No new CUDA ABI, device buffer or parallel runtime added.

The actual producer was RED under CUDA capture: CUDA_STATUS_900/end 901 from
its host synchronization. The new test captures and executes three parent
batches, joins/reuses both producer banks, settles the target and compares
full states with the independent matrix CPU oracle. It is GREEN. This is a
producer DAG test, not a full distributed BFS timeline.

Latest candidate verification after device-history publication:
- 18 CUDA/library-owner runtime unit tests pass.
- All seven macro full-state/compact/archive/capacity/preflight tests pass.
- All 12 CUDA/library-owner CLI integration tests pass (session 54226).
- Four actual S8 runs: K2/K3, local pre-dedup OFF/ON, matrix_u8 mandatory archive,
  batch 1024, capacity/future capacity 40320, archive rows 1024, slots 128.
  Each completes; independent full-state archive verification proves all
  40320 unique states and exact canonical layer sets at all 29 depths.
- CLI SHA256 f16cd9321b94a5a6d07cb6f5e5dc7b5a1a9a6a4214e9605537f7b6dcd4c4b9a3.

Local scope: one physical sm86 RTX 3070 Laptop, existing CUDA 12.9 dependency
volume and diagnostic NCCL. Four fresh unfiltered sanitizer tools completed
on the latest test binaries (session 11315, exit 0): memcheck, initcheck,
racecheck and synccheck each pass the producer capture test and all seven
macro integration tests. No suppressions; error-exitcode 97. Racecheck reports
zero hazards, errors and warnings (integration 629.30 seconds). Synccheck
reports zero errors (integration 4.59 seconds). Prior logs under macro_device_settle_20261002
must not be used to prove the later device-history change.

Current logs/binary hashes and S8 archives are under ignored local
`test_results/macro_device_history_20261002`. Runtime unit binary SHA256:
8709ce8ae6e81248061c8fc284f6aaf8d4b0afa546a35b506f108d32e05bf168.
Macro integration binary SHA256:
c4d85d1148b278136a1c979b967f9ae4d3f0054a74885cec93a7295560f69909.

This does not implement distributed macro execution. The existing
MACRO_MULTI_GPU_UNSUPPORTED admission guard remains. Two-T4 acceptance,
full healthy-path timeline, independent-rank fatal coverage, paired tuned
CayleyPy A/B, other profiles, DB/framework analysis and HF remain open.
The local four-tool gate does not certify distributed execution or NCCL registration.

Kaggle preparation: the existing private native-rank-owner wrapper was repinned
locally to published 49e86fd (archive worker I/O replay included), with AST and
metadata validation passing. Its current worker is terminal COMPLETE according
to a fresh CLI status check; this is the prior worker, not a run of the new
pin. No new notebook or GPU run was launched.
