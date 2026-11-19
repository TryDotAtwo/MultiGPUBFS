# Experimental BFS memory backends

The existing `legacy` backend remains the default. The additional backends
below are experimental. Checks cover exact small-oracle cases on 2x RTX 3060 and 8x RTX 3090,
and complete S12 and segmented S13 on those devices. They do not establish
Blackwell/B200 acceptance, full S14 execution, a BFS with more than 2^32
states on one rank, out-of-core execution, or universal performance gains.

## Native build requirement

These modes require a matching native executable **and** CUDA library.
The dev36 GitHub release provides independently checksummed components for
SM75/80/86/89/90/100/103/120 and a universal build. Existing dev35 native
assets do not contain the new kernels. The launcher checks the requested
native capability before BFS and checks the actual backend in the receipt.
See [installation and offline prefetch](install-native-package.md).

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

An installed SM86 candidate and the selective offline-cache layout passed
CayleyPy persistent S3-S5 on 2x RTX 3060 with both modes. All advertised CUDA
components, including SM100 and SM103, were compiled and inventory-checked;
this is not a B200 or B300 runtime acceptance.

[Reports, hashes and acceptance boundaries](acceptance/segmented-memory-20261119/manifest.json).
