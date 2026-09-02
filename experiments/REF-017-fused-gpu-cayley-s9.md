# REF-017: fused exact GPU Cayley BFS on S9

Date: 2026-08-28 (Europe/Moscow)  
Status: pass; first end-to-end device-resident frontier traversal, performance
conclusions provisional because S9 levels are small

## Question

When successor generation, Lehmer ranking, exact visited claims and next-frontier
output are fused into one CUDA kernel, do the locality effects from REF-016
translate into traversal speed? How much time is outside the kernels when every
BFS level synchronizes with the Rust host?

## Implementation

Rust owns context lifetime, Mahonian oracle construction, exhaustive frontier
validation, benchmark repetitions and JSONL output. CUDA C++ is limited to the
GPU translation unit and narrow C ABI.

Each frontier state is one packed `u64` containing nine four-bit permutation
values. One CUDA thread represents one `(parent, generator)` transition and:

1. swaps adjacent packed values;
2. computes the exact Lehmer rank;
3. optionally elects one claimant per equal key in a warp;
4. performs an atomic bitmap visited claim;
5. atomically appends the first accepted packed state to the next frontier.

Candidate arrays are never materialized. Two fixed-capacity device frontier
buffers alternate between levels. The measured kernel path performs no dynamic
allocation. The explicit allocations for S9 are approximately 5.58 MiB: two
full-state-capacity packed frontier buffers, a 44.3 KiB bitmap and scalars.

## Exactness gates

For every one of 37 level expansions in all four configurations, Rust copies
the resulting frontier after the timed kernel and checks:

- every packed value is a valid permutation of `0..8`;
- inversion count equals the expected BFS depth;
- Lehmer ranks are unique within the frontier;
- frontier count equals the corresponding Mahonian coefficient;
- order-independent rank fingerprints agree across all configurations;
- the final level produces an empty frontier.

Thus each traversal visits all `9! = 362,880` states, has diameter 36, peak
frontier 29,228 at depth 18, and generates exactly 2,903,040 transitions.

The complete fused S8 traversal passed baseline/warp and both layouts under all
four Compute Sanitizer tools with zero errors or warnings. The retained S9
artifact validator reports:

```json
{"status":"pass","validator":"rust-ref017-artifact-v1","level_rows":148,"traversal_rows":40,"exact_depth_groups":37}
```

## Benchmark protocol

For each baseline/warp × parent/generator-major configuration:

1. one full level-wise oracle traversal records actual frontier locality;
2. two full traversal warmups are discarded;
3. ten traversal-only repetitions run without full frontier copies or CPU
   locality analysis;
4. every step still returns and verifies the next Mahonian count.

The host loop is oracle-bounded, not a generic `while frontier is nonempty`
driver: both traversal functions iterate over `0..expected.len()`. For S9,
this means expanding the 37 nonempty frontiers `F_0` through `F_36` and
checking that the last output `F_37` is empty. An unexpectedly early empty
frontier fails the count check instead of ending the loop successfully.
This validates exhaustion on this known graph, but does not demonstrate an
unknown-depth stopping driver or device-side termination detection.

`kernel_ms_sum` is the sum of 37 CUDA-event intervals. `traversal_ms` is the
Rust-observed interval around the 37 steps and includes event synchronization,
scalar count/overflow copies, launches and FFI/host overhead, but excludes
context allocation/reset and the full correctness copies.

## Results

Ten-repetition medians:

| variant | layout | kernel sum ms | kernel Gtransition/s | traversal ms | traversal Gtransition/s |
|---|---|---:|---:|---:|---:|
| baseline | parent-major | 0.541 | 5.37 | 3.909 | 0.743 |
| warp aggregate | parent-major | 0.515 | 5.64 | 3.462 | 0.839 |
| baseline | generator-major | 0.519 | 5.60 | 3.812 | 0.762 |
| warp aggregate | generator-major | 0.606 | 4.79 | 4.361 | 0.666 |

Observed median changes:

- parent-major warp aggregation: 4.8% lower kernel sum and 11.4% lower traversal
  time than parent-major baseline;
- generator-major warp aggregation: 16.8% higher kernel sum and 14.4% higher
  traversal time than generator-major baseline.

The ten-run ranges are broad: kernel sums span roughly 0.43-0.90 ms and traversal
times 3.11-6.70 ms across configurations. The direction matches the locality
hypothesis, but S10 and profiler evidence are required before treating the small
parent-major difference as a stable speedup.

## Actual GPU frontier locality

The correctness traversals inspect the candidate streams produced from the
actual atomic-output frontier order:

| variant | layout | generated | equal-key warp savings | fraction | bitmap-word collisions |
|---|---|---:|---:|---:|---:|
| baseline | parent-major | 2,903,040 | 333,938 | 11.50% | 1,122,248 |
| warp | parent-major | 2,903,040 | 334,269 | 11.51% | 1,128,029 |
| baseline | generator-major | 2,903,040 | 40 | 0.001% | 312,933 |
| warp | generator-major | 2,903,040 | 35 | 0.001% | 331,383 |

Concurrent atomic append did not fully randomize parent-major locality: about
one ninth of all candidate claims remain warp-removable. Generator-major again
places almost every cross-generator duplicate outside its warp.

## Findings

1. The REF-016 locality effect survives a real fused GPU traversal. Warp voting
   is conditionally useful for parent-major and pure overhead for generator-major.
2. Layout affects more than equal-key aggregation. Parent-major has roughly
   1.12 million within-warp bitmap-word collisions versus 0.31-0.33 million for
   generator-major. The baseline generator-major median is slightly better,
   plausibly because it spreads bitmap words, but the gap is within noisy S9
   distributions and remains an inference.
3. Time outside the measured kernel intervals dominates this small graph. Host-observed traversal
   is 6.7-7.3x the sum of kernels. Only about 14% of elapsed traversal time is
   inside the fused expansion kernels.
   This does not isolate synchronization as the dominant individual cost.
   Source inspection confirms that per-step counter/overflow resets precede
   the start event, while scalar copies follow the stop-event synchronization.
   The Rust interval also includes count validation and loop bookkeeping.
   No retained timeline decomposition here attributes the residual to these
   individual operations. The ratio is a boundary comparison, not a profiler.
4. Eliminating candidate materialization is effective for memory capacity and
   traffic, but launch/synchronization overhead becomes visible once each level
   kernel is only tens of microseconds.
5. A persistent device loop or CUDA Graph cannot be adopted blindly. The next
   grid size depends on the just-produced frontier count, and exact termination
   depends on the count reaching zero. Device-side work queues, fixed maximum
   launches with early exit, or graph conditionals need controlled comparison.

## Corrected benchmark failure

The first successful S9 artifact used three traversals without warmups. Its
first kernel included CUDA cold-start and inflated totals, making medians depend
on configuration order. That artifact was rejected and overwritten. The final
protocol separates one correctness run, two discarded warmups and ten timed
traversals, and separates kernel sum from host-observed traversal latency.

Artifact-retention correction, 2026-08-31: the failed first sweep is not retained
as a separately identifiable raw artifact in this record. The failure and its
interpretation survive in prose, but its original sample values cannot be
reconstructed from this report. A new no-warmup run would be a new experiment,
not recovery of that historical artifact. The final retained sweep is separate
evidence and does not remove this provenance gap.

## Boundaries

- This is end-to-end frontier evolution on one GPU, but not an application-sized
  Cayley puzzle and not multi-GPU BFS.
- Parent metadata and path reconstruction are not stored.
- GPU clocks were not locked; the laptop host shows material run-to-run jitter.
- Full device allocation uses `n!` capacity rather than the smaller peak
  frontier because overflow must remain impossible in this first exact run.

## Reproduction

```powershell
docker build -f docker/Dockerfile.gpu --target runtime -t multigpubfs-gpu:dev .
docker run --rm --gpus all multigpubfs-gpu:dev cayley-gpu-self-test
docker run --rm --gpus all -v "${PWD}\experiments:/output" `
  -e MGBFS_OUTPUT_PATH=/output/REF-017-gpu-cayley-s9-levels.jsonl `
  multigpubfs-gpu:dev cayley-gpu-s9-sweep
docker run --rm -v "${PWD}\experiments:/input:ro" `
  multigpubfs-gpu:dev validate-cayley-gpu-s9-artifact
```

## Next experiment

Repeat on S10 (`3,628,800` states, `32,659,200` transitions, diameter 45) so
large middle levels dominate launch noise. Then profile the best/worst layouts
and compare launch-per-level with a bounded device-driven or graph-based loop.
