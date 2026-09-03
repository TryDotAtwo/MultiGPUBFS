# Native modes: measured implementation versus planned profiles

Run: private Kaggle `trydotatwo/mgbfs-mode-sweep-t4` version 1.
Native source `5482f3bb9d20db5780bf6b5c915c4d93c8cd321c`, immutable baseline
`f0f2b8e5ee61173039ab9742f3a7756c9b6365e6`, CUTLASS
`ffa119a1255d78998536107466cc7097ecefa393`, Rust 1.75.0, CUDA sm75.
The native source already passed native-runtime v4 10/10 T4 validation suites.
This sweep does not claim sanitizer coverage at every large configuration.

Implemented native path is single-rank DENSE with CUB sort/merge and asynchronous
mandatory archive. HASH_FIRST, BMMA owner and NCCL multi-rank are unavailable,
not zero-speed measurements and not substituted with CPU simulations.

## Panels

1. Generation 0..4 x local pre-dedup OFF/ON at fixed parent batch 262144,
   256 buckets, 16 shards, J=16: ten configurations.
2. Shards 1/4/64 with generation 0, pre-dedup ON, batch 262144: three more.
   J=min(16,256/shards), so H=64 necessarily also changes J to 4. This panel
   compares valid scheduling configurations, not an isolated shard-count effect.
3. Batch 65536/524280 with generation 0 or 4, and batch 1048576 with generation 4,
   all pre-dedup ON, H=16/J=16: five more configurations.
4. CayleyPy batches 65536/262144/1048576: three configurations.

Generation 0 uses original orientation and CTA 64x32x64; 1 is transposed
64x32x32; 2 transposed 128x32x32; 3 transposed 64x32x64; 4 uses variant 1 GEMM
and vectorized U4 output. These are explicit options, not runtime fallbacks.

All eighteen native configurations first compare full U4(8) layer digests with
the baseline. Then all twenty-one native/baseline configurations get five fresh
process repetitions on U4(16), deterministically randomized order. Best native
and baseline by median m16 search time then receive five fresh alternating
U4(24) repetitions. No hindsight tuning of the large graph is performed.

Batch and shard panels are not the full Cartesian product of every parameter.
There are 19 verification workers, 105 m16 timing workers and 10 m24 workers.
Failed verification prevents that configuration's performance qualification;
all failures remain in summary/logs. Successful medians require five successes.

## Comparable timing and storage

All workers warm a complete same-workload BFS. Startup/build/allocation are not
search time. Native search includes D2H submission and dependencies, durable time
also includes archive-worker completion. Baseline has no archive; report that
output-contract difference prominently. Native F=min(m^6,32,000,000), pinned
slots=ceil(m^6/batch)+128, all chosen before search. Preserve 1 GiB VRAM reserve.

Every native run creates and durably commits a real disk archive under its
task-owned /tmp directory. After timing, SHA256 is computed over the entire file.
First archive per config/workload/phase is gzip-compressed at level 1 into output,
decompressed and checked against SHA256. Repetitions must match it byte for byte.
Only then are duplicate temporary files removed. Every successful run links its
archive object in summary; all exact bytes remain recoverable from gzip. A
nondeterministic archive aborts retention rather than silently dropping output.
Compression/checking are outside both search and durable timers, and finish
before the next worker starts. No archive is disabled to improve performance.

Memory metrics: cudaMalloc request sum, cudaMemGetInfo, whole-device nvidia-smi
samples every 50 ms including warmup/startup, and pinned host bytes. Baseline also
reports Torch allocated/reserved bytes. Sampling is not a byte-exact peak.

Raw results remain private. Local downloads exclude `*.archive.gz`; do not
confuse local absence of large archives with remote retention failure.

```
python scripts/mode_sweep_report.py test_results/mode-sweep-v1/mode-sweep --csv docs/validation/2026-09-03-mode-sweep.csv
```

The report recomputes statistics and checks raw logs, full-state digest equality,
cardinality/layer counts, repeated archive identities, generation/pre-dedup/batch
fields, and the expected comparison matrix. It does not infer per-stage kernel
times, hardware counters or multi-GPU scaling from these end-to-end trials.
