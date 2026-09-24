# GPU database and framework stage for MultiGPUBFS

Status: evidence-based selection, **not** completion of all candidate probes.
This is the separate stage requested after the library/DB discussion. Its
question is whether a ready-made component can replace owner deduplication,
scheduling, or archive I/O without weakening bounded GPU-resident BFS or
increasing peak total VRAM and end-to-end time beyond the accepted gate in
`library-first-bfs.md`. The main architecture remains `ARCHITECTURE_NEED.md`.

## Boundary being replaced

Each owner receives sorted Hash128/state or Hash128/origin batches, checks
history and same-batch duplicates, reserves every affected capacity before
publishing survivors, and returns dense source indices/extents. Persistent
state stays GPU-resident through the required layer window. The owner must
support deterministic shard routing, fixed physical reservation, fatal
exhaustion, explicit CUDA streams, and no mandatory host-sized count/readback
between successive batches. Generic SQL ingestion or a hash-set `contains`
call alone is not this boundary.

## Components and decisions

| Component | What it provides | Fit in this BFS | Decision / missing evidence |
|---|---|---|---|
| CCCL/CUB | GPU radix sort, scan, select, compaction | Ordered route and dense output | Keep integrated. S13 CUB timeout is unresolved; diagnose stage before claiming relative speed. |
| CUTLASS | Tiled GEMM kernels | Matrix successor and affine hash computation | Keep integrated; Tensor Core utilization and shape sweep remain measured gates. |
| NCCL | GPU rank-to-rank transfers | Owner exchange | Keep integrated; it does not deduplicate. |
| RMM | GPU memory resources/pools | Fixed-budget library allocation | Keep the project's physically reserved, non-growing pool; ordinary growable pool configuration is insufficient. |
| cuCollections (`static_set`/device refs) | Fixed-size GPU hash table and device-side lookup/insert | Best ready-made primitive for owner membership | Retain `CUCO_INDEXED` and the newer `CUCO_RANK` rank-batch backend. The latter runs compare/reserve/commit/materialize on GPU, but route sizing, collective control and retirement still have host dependencies. S10 4-shard search wins in the older indexed comparison, with higher total VRAM; do not transfer that result to `CUCO_RANK`. |
| libcudf | GPU tables, joins, relational transforms | Frozen-history membership probe, not an incremental owner by itself | Keep explicit experimental `CUDF_RELATIONAL`; `distinct_hash_join` and `filtered_join` can build once and probe many batches, but neither documented probe API inserts newly committed next-layer keys. Rebuild cost, same-batch dedup, control synchronization, and peak VRAM remain gates. |
| Taskflow / CUDA Graphs | Cached GPU task DAG / lower repeated launch overhead | Fixed-shape owner jobs and pipeline submission | Conditional graph nodes can branch/loop on a device value, but their body is restricted to one device and allowed node types; this is not a multi-rank NCCL failure/ordering protocol. Probe **after** the device-count and collective-control protocol is defined. No integrated A/B result. |
| Apache Arrow / Parquet | Portable columnar output and interchange | Durable catalog/HF artifacts, not owner | Keep archive format; CPU/Arrow and libcudf writer comparison remains open. An Arrow CUDA buffer alone does not make generic Arrow algorithms GPU-aware. |
| KvikIO / cuFile | GPU-storage I/O, including registered reusable buffers | Optional archive consumer/producer path | Bounded experiment only. `CompatMode::AUTO` may fall back to POSIX I/O and `OFF` errors if GDS is unavailable; neither mode alone proves overlap. Existing D2H pinned ring is functional; disk geometry, explicit ordering and end-to-end benefit must be measured. |
| nvCOMP | GPU compression | Optional archive size/bandwidth trade | No end-to-end profile yet; cannot assume compression pays for its GPU cycles or memory. |
| cuGraph / Gunrock / GraphBLAST | BFS over materialized adjacency/sparse graph | Mismatch for implicit Cayley successor generation at target graph sizes | Do not replace owner/runtime with an explicit edge list. A reduced small-graph oracle is possible, not a production replacement. |
| Sirius | GPU-resident SQL/scan engine with internal pinned-table registration | Candidate full DB owner | Source-level GPU ingestion candidate exists; no bounded closed-loop GPU query, fail-fast reservation or speed proof. Not rejected, not accepted. |
| HeavyDB | GPU analytical database, in-process D2D buffer copy | Candidate full DB owner | D2D copy is not query-visible table registration. CPU retry and slab-growth paths require explicit exclusion; no closed-loop adapter passed. Not rejected, not accepted. |
| Kinetica, SQreamDB, PG-Strom, historical BlazingSQL | GPU SQL systems | Possible external comparison | No demonstrated fixed-reservation, device-pointer, stream-ordered owner API in this project. Not a dependency or measured result. |

The cuCollections API documents `static_set` as fixed-size and exposes
device-reference operations, which is why it is a plausible **primitive**.
libcudf's public functions commonly return owning columns/tables; that
ownership boundary matters for recyclable BFS slots. CUDA Graphs reduce CPU
launch overhead but do not change BFS data dependencies. cuGraph BFS accepts
a graph/sparse adjacency representation rather than our implicit generator
function. These are API facts, not performance claims. Primary references:

- https://github.com/NVIDIA/cuCollections#static_set
- https://docs.nvidia.com/cudf/26.10/libcudf/developer_guide/DEVELOPER_GUIDE/
- https://docs.nvidia.com/cudf/26.08/libcudf/api_docs/column_join/
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- https://taskflow.github.io/taskflow/GPUTasking.html
- https://docs.nvidia.com/cugraph/26.08/api_docs/api/cugraph/cugraph.bfs/
- https://arrow.apache.org/docs/cpp/api/cuda.html
- https://docs.nvidia.com/kvikio/latest/cpp/index.html

The libcudf distinction matters for the owner contract: a frozen right-side
table can filter duplicates against older layers across many candidate
batches, but accepted keys from the current layer must also become visible
to later batches before `FinalizeDepth`. The documented join objects expose
probe methods, not an incremental commit API. Rebuilding the right side per
batch or pairing a join with another mutable set is a different memory/time
contract and must be benchmarked as such; this is an API-based inference, not
an executed rejection of libcudf. Likewise, CUDA conditional graph nodes
evaluate their condition on-device, yet their body is single-device and
limited to kernel/copy/memset/child/conditional nodes; a graph does not supply
the cross-rank fatal and NCCL issue-order semantics by itself.

The Sirius/HeavyDB API and implementation evidence is pinned by commit and
file URL in `db-interface-audit.md`; do not replace it with a generic claim
that GPU-native ingestion is impossible.

## Executed comparison: S10, 4 shards

Within each of the Kaggle v10 sessions, same S10 full BFS, 256 buckets,
96 MiB library pool/rank, five repeats and verified rank archives:

| GPUs | Owner | Median search s | Median durable s | Sampled total MiB/rank |
|---:|---|---:|---:|---:|
| 1×T4 | native | 1.029791 | 4.771606 | 835 |
| 1×T4 | cuCollections | 0.421478 | 4.771085 | 901 |
| 2×T4 | native | 0.897227 | 3.981189 | 457, 457 |
| 2×T4 | cuCollections | 0.479693 | 3.818518 | 529, 529 |

Thus cuCollections search is about 2.44×/1.87× faster for 1/2 ranks in
this particular configuration, with about 66/72 MiB more sampled full-device
VRAM per rank. Durable 1-rank time is effectively tied. These are 50 ms
samples, not byte-exact peak proof; other shard counts are separate sessions,
not a paired causal experiment. Source and archived logs:
`library-first-bfs.md` (Shard calibration: 16 and 4),
`test_results/library-capacity-v10/`. This is a Pareto trade, **not**
evidence that cuCollections is faster or smaller on every graph.

The later device-count `CUCO_RANK` owner has its own paired physical 2×T4
S10 archive screen (five repeats, source `4b68552`, preserved native
`013ed5c`): search medians 0.380726 s versus CUB's 0.823757 s, durable
medians 3.774029 s versus 3.899147 s, and sampled peak VRAM 945 versus
457 MiB **per rank**. This run reserved a 512 MiB fixed cuCO pool/rank;
its highest requested suballocations were about 59 MB, which is not by
itself a proof that a smaller pool fits. A 96 MiB fixed-pool paired gate is
now complete: all ten S10 runs have verified rank archives and matching
46-layer histograms. With 96 MiB fixed pool/rank, CUCO_RANK versus CUB
search medians are 0.407692 versus 0.840257 s, durable medians 3.635942
versus 3.775333 s, and sampled full-device peaks 529 versus 457 MiB/rank.
The smaller pool therefore gives a measured S10 search-speed/VRAM Pareto
point, not a universal owner choice or an end-to-end CPU-free path. See
`docs/validation/cuco-rank-paired-s10-2xt4.md` for all samples and limits.

On eight H200, the cuCollections S13 search-only run completed in
11.716894 seconds (five-run median) at 195.55 GiB externally sampled total
VRAM. The CUB counterpart used 76.10 GiB sampled total VRAM but timed out at
180 seconds while GPUs were active. Without the CUB stage trace and a
completed matched run, no S13 native-vs-cuco speed ratio is established.
Search-only S13 does not cover durable archive cost.
Evidence: `docs/validation/lrx13-eight-h200.md`.

## Remaining executable gates

1. Finish the **end-to-end** device-driven path. The `CUCO_RANK` rank-batch
   owner transaction exists and has bounded full-state/capacity/sanitizer
   evidence (`docs/validation/cuco-rank-runtime-2xt4.md`), but it is not an
   end-to-end GPU-driven pipeline. Route counts, payload sizes, collective
   control, and retirement still cause host dependencies; the whole-run S10
   Nsight diagnostic records aggregate host operations but cannot attribute
   them by stage or isolate the timed BFS interval
   (`docs/validation/cuco-rank-nsight-2xt4.md`). Remove or justify each
   dependency without changing NCCL issue order or fail-fast semantics, then
   rerun the full-state 1/2-rank and four-sanitizer gates. A later scoped
   two-T4 search trace at `c73b637` still recorded 1,462 host stream drains
   and 1,740 synchronous copies for the LSA/cuCO-rank path; this is not a
   callsite or critical-path attribution. See
   `docs/validation/poststore-timeline-v36.md`. The v61 four-tool T4 pass
   covered the default library BFS tests, not the ignored LSA-specific full
   BFS/failure fixtures; see `docs/validation/poststore-sanitizers-v61.md`.
2. Pair native/cuCollections/cuDF on identical S10/S11/S12, U4 workloads,
   batch, rank count, shard count and archive contract. Record five runs,
   search and durable medians/MAD, external VRAM peak, allocation ledger,
   routed bytes, host drains, build/dependency and adapter code size. Check
   the accepted <=20% time regression and minimal peak-VRAM frontier. For
   libcudf, separately count one frozen-history index build per depth,
   candidate probes, within-batch unique, and any rebuild or mutable-set
   insertion needed to make accepted keys visible to the next batch. A
   frozen-history anti-join alone is not a correct full owner comparison.
3. For Sirius and HeavyDB separately, execute bounded GPU generate -> ingest
   -> query/dedup -> consume with fixed physical reserve, fatal exhaustion,
   stream/lifetime proof and no hidden CPU fallback. If the adapter cannot
   satisfy this with acceptable code and measured overhead, record the exact
   failing gate and reject it; source inspection alone cannot settle it.
4. Only then test CUDA Graphs/Taskflow on the repeated device-count DAG and
   KvikIO/nvCOMP/libcudf writer on the archive lane. They address different
   bottlenecks and cannot substitute for owner correctness.

Current ruling: keep native and cuCollections as selectable backends, do not
silently switch at runtime, and do not replace the owner with a DB until a
DB adapter passes the same fixed-memory, GPU-resident end-to-end gate.

The first matched three-owner S10 DENSE screen on 2×T4 (Kaggle v63,
`docs/validation/db-owner-s10-v63.md`) completed one archive-verified run
each for CUB, CUCO_RANK and CUDF_RELATIONAL. All reached 3,628,800 states
and 46 layers. CUDF_RELATIONAL took 1.920906 s search and 1134 MiB sampled
total VRAM, versus CUCO_RANK's 0.491433 s and 1058 MiB and CUB's 0.871420 s
and 914 MiB. A one-sample result is not a stable Pareto ruling; the
five-repeat same-source v65 screen completed: CUCO_RANK's search median
was 0.376393 s, CUB's 0.829891 s, and CUDF_RELATIONAL's 1.745511 s;
all 15 runs had verified rank archives and matching 46-layer S10 counts.
See `docs/validation/db-owner-s10-v65.md` for MAD, durable time and VRAM.
This does not settle other
workloads, DB APIs or the CPU-free pipeline gate.
