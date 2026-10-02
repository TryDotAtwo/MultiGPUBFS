# Tensor HASH_FIRST hardware admission

Base e2cb67a. The existing experimental Tensor HASH_FIRST function queried
current CUDA device/properties on every parent batch and rejected non-sm75
only on first generation, after the launcher had reserved its archive.

Device capability validation is now an explicit initialization-only C ABI.
Reference preparation agrees its result over the existing control group,
before archive/pinned admission or communicator creation. The direct runtime
constructor retains its own check before runtime allocations/communicator.
The two runtime generation callsites use an admitted enqueue API with no
device/properties queries. Its contract requires the same admitted current
device for the runtime lifetime. No extra device buffer, stream, allocation,
count readback, fallback, hash/state layout or arithmetic kernel was added.

Legacy mgbfs_generate_hash_only_tc retains its per-call hardware and shape
checks for standalone callers and delegates to the same admitted kernel path.
The scalar API is unchanged. This is additive C ABI, not a new GPU pipeline.

## Real RED/GREEN

- Direct constructor regression RED: unsupported Tensor backend was admitted
  until first batch. GREEN: HASH_FIRST_TC_DEVICE_UNSUPPORTED at construction.
- Real CLI regression RED: unsupported backend reserved an archive before
  hardware admission. GREEN: same specific error, no archive reservation and
  no group-complete marker. Initial test linkage failure was corrected before
  observing this behavioral RED; it is not counted as the regression.
- Final 20 runtime CUDA/library unit, 13 CLI integration and seven macro
  integration tests pass (session 42473); final hardware/capture/archive CLI
  fixture run also passes after environment isolation and final test naming.
- Full default-members cargo test --locked passes (session 21730 exit 0).
  The existing unrelated reference_bench unused_mut warning remains.
- hash_first_generate.cu cross-compiles with CUDA 12.9, -arch=sm_75. Initial
  invalid target spelling sm75 was corrected; the actual sm_75 compile passes.
- Four unfiltered Compute Sanitizer tools pass both the unsupported-device
  admission fixture and current real LSA logical-fatal fixture: zero errors,
  racecheck zero warnings/hazards. Session 32201 exit 0, error-exitcode 97.

Artifacts: test_results/tc_admission_20261002/, eight tool logs, exact hashes:
native bc61042eed619230675cd3760c07f78fc88edfb215298a42cc76daccd42866ce;
runtime test bffb4002c133ce0233e3018181e0626a605ddecdca82fcbcb1a39a2418f386ff;
sm75 object a6f243ad3e87cb1b4eb27ea9b8f848a84457e486bcae48feacad35730337e308.

## Scope and required positive gate

All executions above used one local sm86 RTX 3070 Laptop and diagnostic NCCL
archguard. The Tensor arithmetic kernel was NOT executed: hardware admission
correctly rejects this GPU. Cross-compilation and negative sanitizer tests
cannot establish positive Tensor correctness, overlap, performance or T4
registration/initcheck safety. Earlier scalar full-state evidence stays scalar.

The hardware-conditional CLI fixture runs actual HASH_FIRST INT_MMA_SM75 BFS
on sm75, checking literal S4 layer counts [1,3,5,6,5,3,1], selected generation
record, group marker and committed archive verification. On another GPU it
requires early rejection with no archive. This fixture is connected to the
existing native-rank-gate workflow; it is counts/checksum evidence, not a
full-state oracle, and its positive T4 branch remains unexecuted. No second
notebook was started. Fresh two-T4 full-state, asymmetric faults, all four
full-runtime sanitizers, positive Tensor timeline and paired A/B remain open.
