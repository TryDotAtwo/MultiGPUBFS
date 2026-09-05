# Full warmup process gate, Kaggle v26

Kernel `trydotatwo/mgbfs-distributed-sanitizer`, source
`af8c3451f15c5f1231b8b6fce96e9d36c7f8967d`, physical two Tesla T4 devices.

Verified downloaded evidence under
`test_results/distributed-sanitizer-v26/distributed-sanitizer/`:

- All ten runtime archive fixtures passed in plain, memcheck, racecheck,
  initcheck and synccheck runs (five literal `10 passed; 0 failed` results).
- Across 32 sanitizer logs including supporting leaf and macro fixtures:
  24 zero-error summaries and eight zero-hazard/error/warning summaries.
- Twelve two-process profile smoke runs passed. All 24 measured rank outputs
  have `warmup_completed=true`, and each has the same local layer counts as
  its separately retained warmup output. Global S4 layers are
  `[1,3,5,6,5,3,1]` for every profile.
- All 24 measured archives have CLI `VERIFIED` logs. This check covers committed
  checksums and counts, not an independent state oracle on every process run.
- The pinned launcher additionally asserts rank-local warmup archive removal
  before completing each smoke. No assertion about local large archive copies
  is needed: those archives remained on Kaggle.

The profiles are DENSE, scalar HASH_FIRST and integer-MMA HASH_FIRST, each
with CUB/BMMA and pre-dedup OFF/ON. DENSE's SCALAR field is an unused
HASH_FIRST selector, not its actual generation implementation.

These tiny smoke timings are not performance measurements. This gate proves
the full warmup wrapper on the reference runtime, not the still-unfinished
fully overlapped architecture or global memory preflight.
