# GPU benchmark contract

Date: 2026-08-27

## Stack boundary

- Rust owns CLI parsing, configuration, workload construction, correctness
  validation, artifact schemas, failure reporting, and experiment orchestration.
- C++ appears only in CUDA translation units (`.cu`) for kernels and the minimum
  CUDA Runtime/C ABI needed to expose them safely to Rust.
- No Python, C++, or host-side per-element loops belong in the measured data
  plane. Existing Python references remain independent correctness oracles.
- Build, test, run, and profiling execute inside Docker. Host commands only
  operate Docker and inventory the device/runtime.

## First benchmark: exact visited/dedup hot path

The first performance comparison isolates the component already shown by
REF-003/004 to dominate exhaustive implicit BFS: exact candidate convergence
against same-batch duplicates and prior visited state.

### Common input and output

- Input: a device-resident array of compact integer candidate keys, a prior
  exact visited structure, and a fixed-capacity output frontier.
- Output: exactly one occurrence of every candidate not previously visited,
  plus updated visited membership, accepted count, overflow status, and a
  deterministic validation digest. Output order may differ by backend and is
  not part of correctness unless explicitly requested.
- Every backend receives identical candidate and prior-visited sets.
- A capacity overflow is a failed row, never silent truncation.

### Initial backends

1. Fused dense-rank bitmap: `atomicOr` membership/claim followed by compact
   accepted output.
2. CUB radix sort + unique, then exact prior-visited filtering and compaction.
3. Fixed-capacity open-addressed 64-bit hash set for non-rankable state spaces.

The bitmap is a specialization for practical bijective ranks, not a universal
BFS implementation. Sort and hash comparisons must include their temporary and
table memory.

### Workload families

- controlled uniform keys with configurable accepted fraction;
- controlled duplicate multiplicity and clustered keys;
- recorded/reconstructed Cayley candidate batches matching measured frontier
  sizes and rejection decomposition;
- sizes increasing geometrically until stable throughput, capacity failure, or
  the explicit 8 GiB memory guard.

Synthetic generators are useful for crossover surfaces, but architectural
selection requires at least one replayed real candidate distribution.

## Timing rules

- Allocate all device buffers, CUB temporary storage, events, and contexts
  before warmup.
- Keep input, visited, intermediate, and output data device-resident during the
  timed steady state.
- Warm up until module loading, context creation, and one-time library setup are
  excluded.
- Record isolated CUDA-event duration and end-to-end Rust wall time separately.
- Synchronize only at declared measurement boundaries.
- Record every repetition, plus median and a dispersion statistic; do not
  report only the fastest sample.
- Fix GPU UUID/model, compute capability, driver, container digest, CUDA
  toolkit, native architecture, build type/flags, workload seed, capacities,
  and power/clock state when observable.
- Record thermal/power state before and after longer sweeps. Laptop GPU clocks
  are not assumed stable.

## Metrics

- generated candidates and accepted unique states;
- same-batch duplicates and prior-visited rejects;
- kernel and end-to-end milliseconds;
- candidates/s and accepted states/s;
- device bytes allocated, peak capacity, and backend temporary bytes;
- hash probes/load factor or sort passes/temp storage where applicable;
- output overflow, CUDA errors, OOM, timeout, and validation status;
- for later end-to-end BFS: per-level frontier, transitions, accepted count,
  phase timings, and total solution/path validation.

## Correctness gates

- Reduced cases compare the full accepted set with the independent CPU oracle.
- Large cases compare count and an order-independent digest, with sampled full
  replays; any digest mismatch is reduced to a full comparable case.
- Re-running a backend from a cleared visited structure must be deterministic in
  set semantics even if output order is not.
- Pre-seeded visited keys, duplicate keys, boundary keys, zero candidates,
  exact capacity, one-over-capacity, and deliberate hash-collision fixtures are
  mandatory before performance claims.
- Compute Sanitizer runs on reduced cases before Nsight profiling.

## Progression after the primitive benchmark

1. Fuse implicit generator application with key/rank computation and the best
   exact visited path; compare against materialized candidates.
2. Implement level-synchronous single-GPU BFS with fixed capacities and parent
   metadata optional by declared query semantics.
3. Validate end-to-end on small symmetric groups, then scale to a state space
   large enough to occupy the GPU without exceeding capacity.
4. Profile timelines and the dominant kernel before changing architecture.
5. Only after a stable one-GPU baseline, introduce owner partitioning and GPU-
   resident multi-GPU exchange using the REF-010/011 accounting contracts.
