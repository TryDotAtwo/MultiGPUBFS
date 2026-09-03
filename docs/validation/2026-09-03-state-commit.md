# GPU reservation and materialization integration

Source `881dda5e9cd69efbf77f8eeb4c74ffbd16b332d5`.
Private Kaggle `trydotatwo/mgbfs-state-commit-t4`, version 1.
Downloaded evidence: `test_results/state-commit-v1/state-commit-gate/`.

## Implemented boundary

`cuda/state_commit.cu` connects the bounded owner to GPU reservation and
dense-state materialization. In one stream, without intermediate host count
readback: Compare -> reserve -> hash Commit -> materialize -> StateReady.
The commit grant is now the actual device reservation result, not a fixture
capacity. No device allocations occur in these new entry points.

Reservation operates on a flat 64-byte ring control record with absolute
head/tail and descriptor counters. It checks live capacity, wrap padding,
descriptor capacity and arithmetic overflow before updating counters. An
empty ring can skip wrap padding; zero survivors consume no descriptor.
Materialization validates the entire selected-index mapping before copying
16-byte words into a contiguous reserved range. Fatal errors do not publish
StateReady. Reservation failure does not advance either tail counter.

Concurrency contract: exactly one reservation writer at a time. The caller
must retain the same buffers/descriptors through completion, serialize access
by events and publish reclamation heads only after all corresponding leases
are discharged. `ready` is read after completion, not concurrently polled.
Reclamation, archive credits and multi-lane/rank scheduling are not supplied by
this leaf. A later materialization error makes the run incomplete; it does not
roll back irrevocably committed hashes.

## Tests and evidence

- RED: reservation and materialization tests failed against their stubs.
- GREEN locally on RTX 3070 Laptop, CUDA 12.5.
- Wrap, insufficient free space, exhausted descriptor ring, zero survivors,
  full-capacity empty-ring reservation and absolute-counter overflow fixtures.
- Valid state permutation and out-of-range source-index fixture; invalid input
  leaves output state bytes untouched and StateReady unset.
- Full layer sets of U4(2), 64 states, and U4(3), 729 states, match the host
  all-visited set oracle. Parent batches contain at most 16 states, each with
  six generator applications. Four buckets, I=128, K=1024, state capacity=2048.
- The layer harness feeds CPU-generated canonical 4x4 u8 candidates and
  CPU-prepared sorted descriptors into the real GPU compare/reserve/commit/
  materialize chain. It uses an injective fixture encoding, NOT the production
  GEMM hash. Snapshots and accepted-count metadata are read by the harness.
  This isolates owner/state integration; it is NOT an end-to-end native BFS
  performance run or a GPU generation/hash gate.
- Two independent Tesla T4s: `GPU-01151b96-b668-a5d2-5cd4-b270abf5aec9` and
  `GPU-4d5c3140-fee8-e15a-6904-74b311fe6c35`, sm75 build. Both ran the complete
  fixture set in plain, memcheck, racecheck, initcheck and synccheck modes.
- Independent downloaded-log verifier returned `VERIFIED_STATE_COMMIT_GATE
  10/10`; all eight sanitizer logs report zero errors, both racechecks zero
  hazards and warnings.
- CPU contracts passed locally (62 Windows tests); source CI run 33758026088
  succeeded. Evidence verifier suite passed 14 tests.

Verification command:

`python scripts/verify_bounded_owner_gate.py test_results/state-commit-v1/state-commit-gate --source 881dda5e9cd69efbf77f8eeb4c74ffbd16b332d5 --marker STATE_COMMIT_PASS`

Still pending: production job preparation/source merge, Rust runtime wiring,
lease-aware reclamation, archive/pinned-ring integration, NCCL scheduling and
the full end-to-end correctness/performance gates. No A/B speed claim.
