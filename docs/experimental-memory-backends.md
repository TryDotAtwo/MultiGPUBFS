# Experimental BFS memory backends

The existing `legacy` backend remains the default. The additional backends
below are experimental. Checks cover exact small-oracle cases on 2x RTX 3060 and 8x RTX 3090,
and complete S12 and segmented S13 on those devices. They do not establish
Blackwell/B200 acceptance, full S14 execution, a BFS with more than 2^32
states on one rank, out-of-core execution, or universal performance gains.

## Native build requirement

These modes require a native executable **and** CUDA library built from this
source. The existing dev35 GitHub native assets predate these modes. Installing
the source while continuing to use those old assets is not enough. The
launcher queries key-info and raises NATIVE_MEMORY_BACKEND_UNAVAILABLE
before starting BFS if the matching executable lacks the requested mode. It
also checks the actual backend in the receipt. A new cross-architecture
published native package is not part of this source change.

## Select a backend

```python
import multigpubfs
from multigpubfs.cayleypy_backend import NativeOptions
from multigpubfs.generic_session import persistent_graph_workers
from cayleypy import CayleyGraph, PermutationGroups

options = NativeOptions(
    devices=(0, 1),
    memory_backend="segmented24",  # or "permutation_bitmap", or "legacy"
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

## `segmented24`

Each independent hash segment has at most 2^24 rows per bank. A logical GPU
owner can have many segments: the limit is local to one segment, not a
32-bit limit on all states held by one GPU. Global and per-rank layer counts
use checked 64-bit arithmetic. Collision resolution compares full state values.
For an inverse-closed action, only three distance banks are needed; for a
directed action the visited history is retained.
The 24-bit local row is a field inside a 64-bit hash slot; each slot itself
uses 8 bytes, in addition to state storage and other metadata. It is not a
three-byte-per-entry hash table.

The current implementation allocates segments lazily and consults the existing
segment tables for a shard sequentially. The number and layout of segments
therefore affect throughput. Segmented storage does not guarantee that a graph
fits in VRAM. A memory stop preserves the last completed layer rather than
reporting a partially processed one as complete.

## `permutation_bitmap`

This backend is restricted to permutation actions and ranks each distinct
permutation of the starting multiset exactly. It supports repeated symbols; it
rejects matrix actions, domains outside its checked 64-bit representation and
domains whose partitioned bitmap cannot be admitted into GPU memory. Visited
bits cover the complete multiset-permutation domain, even if the chosen
generators reach only a small subset. Active frontier states still require
separate storage. Ranking and bit operations have a runtime cost, so the
bitmap is not automatically faster.

Do not use this experimental implementation as evidence that a large graph
has completed until its receipt reports `COMPLETE` and the exact hardware and
problem have been validated.

## Measured scope and tuning

With eight RTX 3090 GPUs, fixed 8 logical shards, 8 owner lanes, 8192 batch,
and 1,048,576 rows per segment, one complete S12 run gave:

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

A separately built SM86 wheel was installed offline in a clean environment;
CayleyPy persistent S3 and S4 passed with both modes on 2x RTX 3060. The
wheel is an unpublished test artifact. The new CUDA modules have also
compiled to SM100 objects, but a complete SM100 library and any B200 runtime
acceptance remain separate checks.

[Reports, hashes and acceptance boundaries](acceptance/segmented-memory-20261119/manifest.json).
