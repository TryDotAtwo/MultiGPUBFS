# MultiGPUBFS completion ledger

Status: active, 2026-09-23. This file separates accepted requirements from
implemented paths and measured evidence. `ARCHITECTURE_NEED.md` remains the
architecture contract; `library-first-bfs.md` is the library experiment log.

## Decisions recovered from the conversation

1. V1 is single-source exhaustive BFS of matrix Cayley graphs. LRX repeated-tail
   words add a Schreier-graph workload with the same position moves. No target,
   bidirectional search, paths or runtime profile switch is implied.
2. Owner is a deterministic function of a seeded 128-bit hash. A user-supplied
   logical-owner-to-rank map is fixed before allocation. Deduplication at the
   owner is mandatory. Source-local pre-dedup is selectable ON/OFF.
3. DENSE sends states and hashes. HASH_FIRST sends hashes plus origin references
   and materializes accepted states later. Both are fixed launch profiles.
4. Parents are processed in bounded batches, while the committed frontier and
   the adjacent layer window remain GPU-resident. Slots, rings and local shards
   enable overlap; depth finalization is the semantic barrier.
5. Generation and the specified hash use tensor-core GEMMs when their verified
   shape allows it. Sorting, routing, scans and compaction use CUDA/CCCL; a
   tensor-core owner equality backend is an explicit experimental choice.
6. Capacities, GPU/pinned-memory/disk extents and at least the selected hardware
   reserve are admitted before depth zero. Exhaustion is fatal; no hidden
   fallback, unbounded queue, or host data-plane substitution.
7. Macro depth is selected by the user, may exceed two, and must preserve exact
   original-distance layers through settlement/postprocessing. It is not an
   automatic switch based on observed survival.
8. The complete output profile archives unique states and layer statistics;
   the separately approved Vast experiments used search-only counts. Full graph
   publication to Hugging Face in Parquet is a separate output gate.
9. Benchmark claims require same graph and output contract, full-state small
   oracles, target GPUs, external VRAM, per-depth data and all repetitions.

## Implementation and evidence inventory

| Boundary | Current code | Executed evidence | Remaining gate |
|---|---|---|---|
| One-rank DENSE/CUB matrix BFS | `dense_device.rs`, CUDA generation/hash/owner | T4 U4 sweeps, full small-state checks | Keep regression gate |
| Multi-rank DENSE owner exchange | `distributed_native.rs`, NCCL, `reference_bench.rs` | 2xT4 small oracle; 8xH200 S13 and LRX15r4 cuCollections | Diagnose CUB S13 timeout; repeat large comparison |
| Local pre-dedup and rank map | Runtime config and CUDA stages | 8xH200 small full-state oracle OFF/ON, reversed maps | Large asymmetric replay |
| HASH_FIRST | Runtime request/response/materialization path | Small 8xH200 matrix oracle | Large LRX unsupported; profile/VRAM gate |
| BMMA_BUCKET | CUDA backend and runtime selection | Isolated and small tests | Large end-to-end win or explicit experimental label |
| Macro depth | `macro_native.rs`, `macro_owner.rs` | Single-device macro fixtures | Connect to distributed runtime and CLI; full-state 1/2/8-rank oracle |
| Async state archive | `pinned_archive.rs`, `advance_archived` | Bounded T4 archive tests | Complete large-run throughput and durable verification |
| Search-only LRX15r4 | `lrx_multiset.rs`, CLI reference | 8xH200 54,486,432,000 states, 93 layers, 90.093 s | Independent large histogram; no state archive exists for this run |
| HF Parquet catalog | Upload and codec experiments under `scripts/` | Earlier smaller published experiments | Large graph end-to-end archive/upload/replay certificate |
| Device-driven library owner | Proposed in `device-driven-library-owner.md` | Synchronous cuCollections winner on S13 | Remove per-shard host readbacks and prove event-driven execution |

The LRX15 run's `ORBIT_TOTAL_AND_CONFIG_ONLY` certificate verifies count and
configuration, not full state equality or independent per-layer distances.
Its 8xH200 rental is deleted; evidence is in
`test_results/vast-51254782/multiset-evidence.tgz` (ignored local artifact).
The single-rank legacy map must be `[0,0]`; the corrected GPU test fixture is
still uncommitted as of this ledger.

## Independent DB and framework stage

The user asked for a *separate* analysis of finished GPU DBs/frameworks to
simplify code without sacrificing the fixed-memory, GPU-resident BFS hot path.
The recovered source audit is `docs/plans/db-interface-audit.md`; the broader
library experiment is `docs/plans/library-first-bfs.md`.

| Candidate | Intended role | Evidence level / decision gate |
|---|---|---|
| CCCL/CUB | radix sort, scan, compact | integrated; S13 CUB timeout unresolved |
| CUTLASS | generation/hash GEMM | integrated; inspect utilization and fallback conditions |
| NCCL | native rank exchange | integrated; physical 2/8-GPU runs |
| RMM/cuCollections | fixed pool and GPU key membership | integrated reference owner; fast S13, high reserved VRAM |
| libcudf | relational dedup/join | isolated prototype; compare full owner semantics and host sync |
| Taskflow/CUDA Graphs | scheduling repeated jobs | experiment only; require timeline and no CPU count dependency |
| Arrow/Parquet, KvikIO, nvCOMP | durable archive | bounded codec experiments; require end-to-end drain rate |
| Sirius | GPU DB owner candidate | source audit only; pin-table adapter and fail-fast proof pending |
| HeavyDB | GPU DB owner candidate | source audit only; D2D ingress exists but closed-loop table adapter pending |

Sirius and HeavyDB must each pass a bounded generate -> ingest -> dedup/query
-> consume cycle using GPU-resident data, prove fixed physical reservation and
fatal exhaustion, and report host sync, copies, code size, speed and peak VRAM.
If an adapter cannot satisfy those contracts, document the specific failure and
reject it; do not infer rejection from its public SQL/RPC surface alone.

## Next dependency order

1. Commit the already-passing one-rank LRX oracle fixture and update the stale
   LRX status document with the observed n=15 result.
2. Diagnose CUB S13 using preserved logs and scoped reproducer. No new paid GPU
   run until an actionable failure hypothesis and capacity estimate exist.
3. Specify and connect macro depth to multi-rank DENSE, with exact-distance
   settlement tests before large hardware measurement.
4. Complete profile/backend matrix, archive/HF path and device-driven library
   owner; test each vertical slice before performance claims.
5. Execute the independent DB/framework probes and final Pareto comparison.

User-owned dirty files are not implicitly part of this ledger's implementation.

### CUB S13 diagnostic, preserved trace

The 180-second CUB run emitted only torchrun startup text before the harness
timeout. Its last 800 external nvidia-smi samples show GPU utilization
min=99%, mean=100%, while memory-controller utilization averaged 3%; all eight
devices were resident at about 9.5 GiB each. This is evidence of sustained GPU
work, not evidence of an idle NCCL deadlock or a capacity failure. The exact
kernel/stage is still unknown because there is no per-rank progress artifact or
Nsight trace for that attempt. Next reproduce first on a bounded S11/S12
configuration with per-stage GPU event timings and a timeline, then compare the
same fixed configuration with cuCollections. Do not call the timeout a CUB
correctness failure or claim that this trace identifies the hot kernel.
