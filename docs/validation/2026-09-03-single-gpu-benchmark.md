# Single-GPU native/CayleyPy experiment

## Scope

Compare the current development single-bucket `DenseDeviceStepper::new_pipelined`
against ordinary `CayleyGraph.bfs(num_gpus=1)` at immutable CayleyPy commit
`f0f2b8e5ee61173039ab9742f3a7756c9b6365e6`. The native measurement source is
`791c1d647a7f8f63d3ee87904590ee5b47c8c641`; CUDA algorithms are unchanged from
the passed T4 primitive/full-depth gate. The only runtime addition is a getter
for the already published frontier count.

This is **not** the production archived runtime or the planned two-rank A/B gate.
Neither implementation writes an archive. No archive has been disabled in a
production runtime to obtain a timing. Native state/hash storage is u8/128-bit;
CayleyPy retains its unmodified int64/64-bit representation. Both are hash-based
probabilistic searches. CayleyPy retains the first/last and singleton layers;
native retains only rolling frontier buffers.

## Protocol

- Private Kaggle kernel `trydotatwo/mgbfs-single-gpu-cayleypy-benchmark`, v1.
- One physical T4 selected by `CUDA_VISIBLE_DEVICES=0`; second T4 unused.
- U4 over Z/mZ, m=5,8,12; identity start; six adjacent elementary +/- generators
  in identical order; seed integer 20260828 (different hash families).
- Native release Rust 1.75.0; CUDA/CUTLASS sm75 Release. CUTLASS pinned to
  `ffa119a1255d78998536107466cc7097ecefa393`.
- Separate verification workers compare SHA-256 of lexicographically sorted
  canonical 16-byte states for every full BFS layer, plus total m**6 states.
  Verification timings/memory are not performance samples.
- Calibration: native batches 4096/16384/65536, pre-dedup OFF/ON; CayleyPy
  batches 65536/262144/1048576. One calibration per configuration. Pick minimum
  successful calibration time, then five NEW repetitions of that configuration.
- Every worker is a fresh process and warms one complete same-workload BFS.
  CUDA synchronization brackets measured work. Alternate backend order each
  repetition. Record all calibration, failed, and measured rows, not only minima.
- Native setup/allocation time is separate; no allocation inside native search.
  CayleyPy dynamic allocation inside BFS stays inside its search time. Imports,
  compilation, process startup, warmup, digests and readbacks are excluded.
- Native fixed frontier/bucket capacity is m**6, not a hindsight oracle peak.
  This is deliberately the existing executor's conservative allocation scheme.
- Native per-depth times are recorded. Ordinary CayleyPy's unmodified timed
  run has no per-depth synchronization hook; only total synchronized BFS time.
- Per-worker external timeout 600 s includes warmup/setup, not only search.
  A timeout is a failed row, not a truncated successful BFS.

## Memory interpretation

- Native `cuda_fixed_allocation_delta_bytes` is the measured cudaMemGetInfo
  difference after warm context/before plans versus after all plan/buffer setup.
  It includes allocator rounding and is **not** a byte-exact allocation ledger.
  No byte-exact queried native planner exists yet.
- Native `cuda_observed_used_bytes` includes the CUDA context and other
  device-wide consumption at setup/end boundaries; fixed allocations live
  throughout search. It is not a high-frequency per-process peak measurement.
- CayleyPy records `torch_peak_allocated_bytes` and `torch_peak_reserved_bytes`,
  reset after warmup and cache clearing. Driver/context consumption is separate.
- Both have raw `nvidia-smi` samples every 50 ms. The process-lifetime maximum
  includes warmup/setup and search; it can miss a sub-50ms transient. It is
  device-wide (idle device overhead included), not a precise allocator counter.
- The output keeps these fields distinct; do not compare native allocation delta
  to PyTorch reserved as though either were full-device consumption.

## Local preflight

- New benchmark contract tests: RED (missing harness), then 2 PASS. They reject
  mismatched layers/digests, incomplete runs and silently filtered failures.
- Existing CPU Rust contracts: 25 PASS; existing Kaggle guard tests: 2 PASS.
- Release native benchmark built; local RTX 3070 Laptop m5 full-depth digest
  worker completed with 15,625 states. This is not a T4 performance result.
- Local toolchain image has no NumPy/PyTorch, so the baseline worker's first
  execution is on Kaggle's Python environment. No dependency installation or
  modification of the frozen CayleyPy checkout was performed locally.

## Result

Kaggle v1 completed: 63/63 worker rows COMPLETE (6 verification, 27 calibration,
30 measured). Downloaded and independently checked all 63 raw worker JSON logs;
all three full-layer SHA-256 pairs match. PyTorch 2.10.0+cu128, CUDA 12.8;
CayleyPy used `_make_hashes_older_gpu`.

| m | States | Native median +/- MAD ms | CayleyPy median +/- MAD ms | Native allocation delta MiB | PyTorch peak allocated / reserved MiB | External max native / CayleyPy MiB |
|---|---:|---:|---:|---:|---:|---:|
| 5 | 15,625 | 5.189 +/- 0.057 | 16.991 +/- 0.224 | 50 | 11.887 / 24 | 157 / 145 |
| 8 | 262,144 | 31.590 +/- 0.795 | 45.195 +/- 2.693 | 130 | 180.436 / 378 | 237 / 499 |
| 12 | 2,985,984 | 331.042 +/- 0.609 | 288.547 +/- 3.604 | 1106 | 1584.243 / 3072 | 1213 / 3193 |

Best calibrated native batches: 16384,16384,65536, pre-dedup ON in all cases.
CayleyPy batches: 1048576,262144,1048576. Five new repetitions for each selected
configuration. Native speed ratios: 3.27x and 1.43x faster at m5/m8, 14.73% slower
at m12. At m12 the external sampled peak is 62.0% lower, while allocation delta
versus Torch peak allocated gives a smaller difference (different counter semantics).

The m5 native search is shorter than the sampler interval: some individual
native rows miss the allocation plateau (107 versus actual 156.875 MiB). Table
uses the maximum over all five processes; the cudaMemGetInfo plateau independently
confirms the native allocation. Do not present this sampler as exact peak capture.

Evidence: `test_results/kaggle_single_gpu_benchmark_v1/single-gpu-bench/`.
Summary SHA-256:
`3a63bd2176333e6091616481bfe77ca4c295f04db1372c5e51a878f030cb494b`.

These are short-run results only. The user requested larger, minute-scale graphs
before drawing sustained-load conclusions. A separate size escalation follows;
these small timings do not establish sustained performance or a production win.
