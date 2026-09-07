# Framed transport v43: full raw gate reconciliation

Kaggle `trydotatwo/mgbfs-distributed-sanitizer`, v43, completed.
Source `a7378e39cc4813524829cfff2dae091e4a090c23`; package `9712d87`.

Two physical Tesla T4 GPUs (15360 MiB each):

- `GPU-6360e4e0-1728-4996-658d-2209069cec00`
- `GPU-35470d6a-7409-05e5-8c6b-c316d8bfcdd2`

Downloaded only logs/JSON into `test_results/distributed-sanitizer-v43/`.
Strict raw reconciliation passed: 40 tool logs, 36 measured rank records,
36 warmup rank records and 36 archive verifiers. All 24 reference profile
selections have global layer counts `[1,3,5,6,5,3,1]` and COMPLETE archives.
The three transport tests passed under plain/memcheck/racecheck/initcheck/
synccheck. All sanitizer summaries are clean.

This verifies the direct frame pack, late-bound prefix, native scatter,
zero-copy self view and receiver prefix validation at the pinned source.
The reference full-runtime suite includes generation5 compact S4 archives.
It does not prove production asynchronous dispatcher integration, later
packed materialization, batched owner completion, or any speedup.

## Next full-runtime gate

Source `66d82d03cb055daa08dae328978208efda7c8ede` includes packed DENSE
materialization, its consumer-lifetime integration fixture, and batched
DENSE owner completion. It reuses only each consumed span's first 64-byte
job descriptor for the captured extent, after all descriptor readers in
the same stream. Later spans use disjoint slots. One batch wait/readback
replaces per-span host waits. Sticky ring fatal prevents publication after
any failed job; a DENSE group-capacity regression complements HASH_FIRST.

GPU allocation totals are unchanged. A fixed host vector of 64-byte records
(`buckets + 1` entries) is allocated at runtime construction. Per-depth
control/transport/archive synchronization outside this owner loop remains.
Local tests cover captured-result selection, ring wrap/coalescing, malformed
metadata, job splitting and unchanged device allocation accounting. CUDA
feature Rust type checks passed; hardware verification of this source is
pending. No performance claim is attached.
