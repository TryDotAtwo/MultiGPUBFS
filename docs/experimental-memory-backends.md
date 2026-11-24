# Experimental BFS memory backends

The existing `legacy` backend remains the default. The additional backends
below are experimental. The current 32-bit-row ABI passed exact small-oracle
cases and complete S11 and S12 on 2x RTX 3060, plus complete S13 in all three
backends on 8x RTX 3090. These checks do not
establish Blackwell/B200 acceptance, full S14 execution, a BFS with more than
2^32 states on one rank, out-of-core execution, or universal performance gains.

## Native build requirement

These modes require a matching native executable **and** CUDA library.
The dev40 release requires `segmented_memory_presets_v1`, `segmented_strategies_v1` and native ABI 3. Changes documented after dev40 require a freshly built matching executable and CUDA library; they are not part of the published dev40 wheel.
The executable and CUDA component must come from the same build. The dev38
binary and CUDA component implement ABI 2 and cannot be combined with this
source. The launcher checks the capability before BFS and the backend in the
receipt; a mismatched runtime is rejected instead of silently ignoring options.
See [installation and offline prefetch](install-native-package.md).

## Select a backend

```python
import multigpubfs
from multigpubfs.cayleypy_backend import NativeOptions
from multigpubfs.generic_session import persistent_graph_workers
from cayleypy import CayleyGraph, PermutationGroups

options = NativeOptions(
    devices=(0, 1),
    memory_backend="segmented",  # or "permutation_bitmap", or "legacy"
    memory_preset="balanced",  # or "max_capacity" on the two experimental backends
    segment_rows=2**32,  # maximum rows per bank per segment; change if needed
    segment_strategy="atomic",  # alternatively "warp" or "batch"; all are experimental
    autotune=True,
)
multigpubfs.enable_native(options)
try:
    with persistent_graph_workers():
        for n in (4, 5, 6):
            graph = CayleyGraph(PermutationGroups.coxeter(n), device="cpu")
            result = graph.bfs()
            if result.bfs_completed:
                print(n, "COMPLETE", result.diameter(), result.layer_sizes)
            else:
                print(n, "INCOMPLETE at completed depth", result.completed_depth,
                      result.report.get("reason"), result.layer_sizes)
finally:
    multigpubfs.disable_native()
```

`device="cpu"` constructs the CayleyPy graph description; native execution
uses the GPUs named in `NativeOptions.devices`. Inspect `result.report["backend"]`
to see what actually ran. The result contains layer sizes and compact samples;
it is not an archive of every state. `INCOMPLETE` is not a completed diameter.

## Strategy within the selected backend

`segment_strategy="atomic"` is the default. `"warp"` groups reservations
and `"batch"` uses bounded GPU batches. The strategy is independent of
`memory_backend`, `memory_preset` and `segment_rows`; a matching native
executable and CUDA component must support the requested strategy. You can
also set `MGBFS_SEGMENT_STRATEGY=atomic|warp|batch` before activation.
Unknown or unsupported strategies fail rather than silently changing the run.
For segmented hash with `warp`, the CUDA library must expose
`mgbfs_segmented_warp_safe_abi_version=1`. Earlier hash-warp libraries can
reuse a pending slot during the same accept phase and may record a duplicate
when a segment and rank limit interact. The current executable rejects that
old combination with `SEGMENT_WARP_SAFE_NATIVE_ABI_MISSING`; it does not
silently fall back. Use the executable and CUDA library from the same build.
Results are workload and topology dependent: see the
[2-GPU comparison](https://github.com/TryDotAtwo/MultiGPUBFS/blob/72a3e2a0b0521985229823cc6c4c959ec22d81fd/docs/acceptance/2026-11-20-segmented-2gpu.md) and
[8-GPU comparison](https://github.com/TryDotAtwo/MultiGPUBFS/blob/72a3e2a0b0521985229823cc6c4c959ec22d81fd/docs/acceptance/2026-11-20-segmented-8gpu.md). No strategy is
claimed fastest on B200 without a B200 measurement.

## Memory preset

`memory_preset="balanced"` selects for speed among eligible measured profiles.
`memory_preset="max_capacity"` prioritizes estimated admitted rows on the
limiting GPU, then measured speed for equal estimates. It defaults to one owner
lane and a queue ceiling of 2048; bitmap uses two state banks, while hash uses
32-bit slot positions where the number of slots permits. The score describes
the tested allocation geometry, not a guarantee of a global optimum or future
NCCL/driver allocation. Max-capacity can be slower.

The policy and backend are independent. Explicit compatible environment
settings retain priority, and changing backend or policy changes the tuning
cache identity. The legacy backend currently rejects `max_capacity` instead
of pretending to apply it. Small jobs can use a conservative profile without
measured tuning. The worker reuses an exact matching profile; a new graph or
configuration can require tuning again.

The two-bank bitmap keeps global visited bits and retains a bounded sample of
the previous complete layer before reusing its state bank. A resource stop
reports only complete layers. See [validation and limits](https://github.com/TryDotAtwo/MultiGPUBFS/blob/72a3e2a0b0521985229823cc6c4c959ec22d81fd/docs/acceptance/segment-vram-20261120.md).

## `segmented`

An independent segment stores up to `segment_rows` states **per bank**.
The default is `2**32`; valid values are integers from 1 through `2**32`.
Local rows are indexed from 0 through `segment_rows - 1` with 32-bit row
indices. A segment number and bank number are tracked separately. Aggregate
counts, strides, and layer offsets are 64-bit; full states are compared for
hash collisions. The former name `segmented24` remains a compatibility
alias; it does not select a 24-bit index. Receipts may still say
`EXPERIMENTAL_SEGMENTED24` for compatibility.

For an inverse-closed action the implementation uses three state banks per
segment; directed actions retain the visited history in a single bank.
`segment_rows` is a maximum, **not** an immediate memory allocation. The
actual rows per bank can be smaller according to workload and available
memory. On growth, a logical shard can receive additional independent
segments. See `configured_segment_rows`, `rows_per_bank` and
`segment_count` in each entry of `result.report["segment_memory_by_rank"]`.

Segments are allocated lazily, and their tables are looked up sequentially
within a logical shard. Both the number of shards and the number of segments
can affect performance. Neither segmentation nor the configured maximum
guarantees that a graph will fit in VRAM. On a resource stop, only completed
layers are reported as complete.

The CLI accepts the same configuration, for example:

```bash
multigpubfs graph.json result --memory-backend segmented --memory-preset max_capacity --segment-rows 16777216
```

## `permutation_bitmap`

This backend is restricted to permutation actions and ranks each distinct
permutation in a closed, exactly ranked domain of the starting state. It
supports repeated symbols; it rejects matrix actions, domains outside its
checked 64-bit representation and domains whose partitioned bitmap cannot be
admitted into GPU memory. With a compatible native library, the visited bitmap
uses the product of the multiset-permutation domains of the invariant position
orbits. This exact superdomain may still include unreachable states; older
compatible libraries use the full multiset domain. Active frontier states
require separate storage. Ranking and bit operations have a runtime cost, so
the bitmap is not automatically faster.

Do not use this experimental implementation as evidence that a large graph
has completed until its receipt reports `COMPLETE` and the exact hardware and
problem have been validated.

## Measured scope and tuning

In a later exact-exchange comparison on the same 8x RTX 3090, S13 took 393.32 s for hash and 593.55 s for bitmap. The initial batch was the same, but bitmap retried and reduced its batch ceiling from 8192 to 2048 while hash retained 8192. This was one paired run with the earlier three-bank bitmap; it does not isolate the cost of the bitmap. See the [audit](https://github.com/TryDotAtwo/MultiGPUBFS/blob/72a3e2a0b0521985229823cc6c4c959ec22d81fd/docs/acceptance/bitmap-audit-20261120.md) and [follow-up](https://github.com/TryDotAtwo/MultiGPUBFS/blob/72a3e2a0b0521985229823cc6c4c959ec22d81fd/docs/acceptance/segment-vram-followup-20261120.md).

For context, an earlier padded-exchange run measured the following. On 8x RTX 3090 all three backends completed S13 with 6,227,020,800 states;
all 79 layer sizes matched the independent Mahonian oracle. Single-run BFS
times with 8 logical shards, 8 segmented owner lanes, batch 8192 and physical
segments of 1,048,576 rows per bank were:

| Backend | BFS time | Relative to segmented |
|---|---:|---:|
| Segmented | 390.628 s | 1.00x |
| Legacy | 587.988 s | 1.51x slower |
| Permutation bitmap | 1652.309 s | 4.23x slower |

Per-rank segment allocations at completion were 5.327 GB for segmented and
1.636 GB for bitmap plus 0.097 GB for its bitmap; these figures exclude other
allocations and are not total or peak VRAM measurements. See [the full comparison](https://github.com/TryDotAtwo/MultiGPUBFS/blob/72a3e2a0b0521985229823cc6c4c959ec22d81fd/docs/acceptance/segmented32-20261120/rtx3090-complete-s13.json)
and [receipts](acceptance/segmented32-20261120/rtx3090-complete-receipts.tar.gz).
This is not a test of more than 2^32 states on one rank or evidence of B200 speed.

The current 32-bit-row ABI was compared on 2x RTX 3060 with identical fixed
settings (8 logical shards, 8 owner lanes, batch 8192, physical segments of
1,048,576 rows per bank). Each time below is the mean of two runs, and every
layer size matched the independent Mahonian oracle:

| Graph | Segmented | Bitmap | Bitmap relative to segmented |
|---|---:|---:|---:|
| S11 | 6.639 s | 7.531 s | 13.4% slower |
| S12 | 86.975 s | 104.215 s | 19.8% slower |

A separate matched comparison with the legacy backend, under the same fixed
8-shard settings, averaged 5.411 s legacy versus 6.126 s segmented on S11,
and 70.867 s legacy versus 76.053 s segmented on S12. Both used complete
oracles. With `autotune=False` and no manual shard count, the capacity-only
planner selected one shard: S11 averaged 2.796 s legacy versus 17.834 s
segmented, and S12 averaged 36.156 s legacy versus 207.768 s segmented.
This static-planner result does not represent measured autotuning; the shard
count and owner-lane count strongly affect these results.

On a measured-autotuning run, both S11 and S12 selected 16 shards: tuning
took 55.366 s and 58.806 s respectively, then a second graph with the same
profile reused the cached result in 22.8 ms and 28.1 ms. Complete BFS times
were 5.661/5.702 s for S11 and 75.621/75.352 s for S12. These are two GPU
measurements, not evidence of a universally optimal configuration.

These measurements do not establish the relative performance on B200. A
specific S14 adjacent-transposition (Coxeter) memory model estimates around
72.8 GB/GPU for segmented and 25.8 GB/GPU for bitmap on 8x B200, allowing
25% for distribution imbalance but excluding several runtime allocations.
It is not a measured fit or a bound for other generators. Current
uncompressed frontier storage alone exceeds 8x180 GB for Coxeter S15. See
[the calculation and test results](acceptance/segmented32-20261120/summary.json).

The following historical dev36 24-bit-row measurements used eight RTX 3090
GPUs, fixed 8 logical shards, 8 owner lanes, 8192 batch, and 1,048,576 rows
per segment. They do not measure the new 32-bit-row ABI. One complete S12
run gave:

| Backend | BFS time | All 67 layer sizes |
|---|---:|---|
| legacy | 29.273 s | match |
| segmented24 | 24.092 s | match |
| permutation_bitmap | 91.131 s | match |

S12 has 479,001,600 states. These are single measurements on one machine,
not a universal ranking. On 2x RTX 3060, S12 times were 46.495 s, 59.348 s,
and 84.412 s respectively for the same modes. The segmented backend completed
S13 on 8x RTX 3090 in 316.891 BFS seconds versus 410.519 seconds for
legacy (one run each): 6,227,020,800 states, all 79 layer sizes matching
the independent Mahonian polynomial, and 32-40 segments per GPU. Hash-segment scanning remains sequential within a logical
shard; the measured result is not evidence of speed or capacity on 8xB200.

For substantial graphs, first-time segmented/bitmap autotuning measures
several profiles at a common completed prefix; on 2x RTX 3060 S11 it took
about 55-58 s. Repeating the *same graph and matching runtime/device/geometry*
inside a persistent context reused the exact profile in about 16-17 ms.
Changing graph or hardware may require tuning again. Small conservative
profiles are not claimed to be measured optima.

The older dev36 installed SM86 candidate and selective offline-cache layout
passed CayleyPy persistent S3-S5 on 2x RTX 3060 with both modes. Compiling and
inventory-checking CUDA components is not B200 or B300 runtime acceptance.

[Reports, hashes and acceptance boundaries](acceptance/segmented-memory-20261119/manifest.json).

The released dev37 wheel was installed and the SM86 runtime was prefetched
without GPU access. With `MGBFS_NATIVE_OFFLINE=1`, the installed package then
passed the persistent 2-GPU API test for both modes and segment maximum values
64 and 2^32.

### Experimental queue-batch recovery and layer diagnostics

An overflowing destination queue causes a collective retry with a smaller
source batch. By default that reduced ceiling remains in effect for the rest
of the BFS. Set `MGBFS_SEGMENT_BATCH_RECOVERY=1` on **all ranks** before the
worker starts to permit recovery toward the admitted ceiling: after two fully
drained layers with work, no retries, and peak destination-queue occupancy at
most one half, the ceiling doubles (never beyond its admitted value). Invalid
or inconsistent rank settings are rejected. This is an opt-in policy, not a
claim that larger batches are always faster. Reused persistent workers receive
the setting from the job environment; changing it between jobs is supported.

For diagnostic runs, set `MGBFS_SEGMENT_LAYER_STATS=1`; optionally restrict
completed depths with `MGBFS_SEGMENT_LAYER_STATS_MIN_DEPTH` and
`MGBFS_SEGMENT_LAYER_STATS_MAX_DEPTH`. `SEGMENT_LAYER_STATS` JSON lines in each
rank log include source parents, generated edge counts, sent/received counts
per peer and shard, accepted rows, queue peaks, retries, spill traffic and
selected storage/reserve byte counters. Spill counts include replay traffic,
not unique states. Collection adds host work once per wave and is unsuitable
for the final timing comparison. Its storage counters do not include every
CUDA/NCCL transient; separately sampled `nvidia-smi` can miss short peaks.

### Optional bitmap ownership page

`NativeOptions(memory_backend="permutation_bitmap", bitmap_owner_page=64)`
selects an experimental exact-ID ownership permutation; `0` (the default)
keeps cyclic ownership. `run_graph(..., bitmap_owner_page=64)` and
`MGBFS_SEGMENT_BITMAP_OWNER_PAGE=64` provide the same choice. Use a power of
2 up to `2**20`, or zero for cyclic. All ranks must agree, and requesting a
page on a native library lacking the feature fails explicitly. Profiles include
this setting. In the current native ABI the mapping is passed to each job's
kernels and remains immutable for that job. The default cyclic layout can
still run with a compatible older CUDA library; a nonzero page requires the
new immutable-layout ABI and otherwise fails explicitly.

The layout rotates full groups of `world * page` exact IDs and leaves an
incomplete final group cyclic. It permutes the range of IDs bijectively,
retains the same bitmap size and per-owner address range, and does not change
state values, external archive format, or the meaning of `segment_rows`. The
page controls distribution between GPUs; it is not a shard-size cap. It can
reduce skew for some graphs while increasing remote traffic for others, so it
remains opt-in until measured for a workload. No migration of live layers
between ownership layouts takes place.

### Product-domain bitmap

For permutation actions, positions connected by the supplied generators are
grouped into orbits. Each orbit retains the multiset of symbols from the
start, and the bitmap uses a mixed-radix rank across these orbit domains.
This is an exact closed superdomain; it need not be the set of all reachable
states. For example, fixed positions no longer multiply the bitmap domain.
The state payload and output archive keep the original symbols and positions.
If a compatible native library does not advertise this representation, the
full multiset domain remains the compatibility path. Capacity and performance
still depend on frontier size, allocation, communication, and workload.
