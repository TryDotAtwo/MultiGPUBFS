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
| Macro depth | `macro_native.rs`, `macro_owner.rs`, single-rank reference CLI | T4 full-state/sanitizer macro tests and archived K=1/2/3 CLI | Connect to distributed runtime; full-state 2/8-rank oracle |
| Weighted DENSE exchange leaf | `exchange_pack.cu`, schema2 macro frame and `MacroExchangeMemoryPlan` | sm75 compile, RTX 3070 word-for-word GPU fixture; CPU memory-bound tests | Integrate NCCL/owner, T4 execution and sanitizer |
| Async state archive | `pinned_archive.rs`, `advance_archived` | Bounded T4 archive tests | Complete large-run throughput and durable verification |
| Search-only LRX15r4 | `lrx_multiset.rs`, CLI reference | 8xH200 54,486,432,000 states, 93 layers, 90.093 s | Independent large histogram; no state archive exists for this run |
| HF Parquet catalog | `stream_hf_archive.py`, `promote_hf_stream.py`, verifiers | Full S13: 6,227,020,800 archived states and 6,228 Parquet objects verified by size/SHA on HF | Full-state remote replay and later LRX15 archive/publication |
| Device-driven library owner | Proposed in `device-driven-library-owner.md` | Synchronous cuCollections winner on S13 | Remove per-shard host readbacks and prove event-driven execution |
| Production CLI `run`, hardware `preflight`, `calibrate` | `mgbfs-cli/src/main.rs` explicitly reports unavailable | `bench --reference` and offline config preflight only | Wire versioned RunConfigV1 to production dispatcher; test admission and output commits |

The LRX15 run's `ORBIT_TOTAL_AND_CONFIG_ONLY` certificate verifies count and
configuration, not full state equality or independent per-layer distances.
Its 8xH200 rental is deleted; evidence is in
`test_results/vast-51254782/multiset-evidence.tgz` (ignored local artifact).
The single-rank legacy map must be `[0,0]`; the corrected GPU test fixture was
committed earlier on this branch.

## Independent DB and framework stage

The user asked for a *separate* analysis of finished GPU DBs/frameworks to
simplify code without sacrificing the fixed-memory, GPU-resident BFS hot path.
The recovered source audit is `docs/plans/db-interface-audit.md`; the broader
library experiment is `docs/plans/library-first-bfs.md`. The separate
component-role decision and executable gate are now
`docs/plans/db-framework-stage.md`. It records the S10 cuCollections speed/VRAM
Pareto trade without equating it to a universal improvement or a DB acceptance.

| Candidate | Intended role | Evidence level / decision gate |
|---|---|---|
| CCCL/CUB | radix sort, scan, compact | integrated; S13 CUB timeout unresolved |
| CUTLASS | generation/hash GEMM | integrated; inspect utilization and fallback conditions |
| NCCL | native rank exchange | integrated; physical 2/8-GPU runs |
| RMM/cuCollections | fixed pool and GPU key membership | integrated reference owner; fast S13, high reserved VRAM |
| libcudf | relational dedup/join | isolated prototype; compare full owner semantics and host sync |
| Taskflow/CUDA Graphs | scheduling repeated jobs | experiment only; require timeline and no CPU count dependency |
| Arrow/Parquet, KvikIO, nvCOMP | durable archive | S13 Parquet stream published; comparative codec/KvikIO/nvCOMP gate remains |
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
3. Wire production `run`/hardware `preflight`/`calibrate` to a versioned config;
   the current executable exposes only the reference benchmark. Add a fail-fast
   guard for unsupported macro depth until the next item is implemented.
4. Specify and connect macro depth to multi-rank DENSE, with exact-distance
   settlement tests before large hardware measurement.
5. Complete profile/backend matrix, archive/HF path and device-driven library
   owner; test each vertical slice before performance claims.
6. Execute the independent DB/framework probes and final Pareto comparison.

### 2026-09-23 implementation slice

The `bench --reference` CLI previously ignored `MGBFS_MACRO_DEPTH`; requesting
depth 2 or 10 therefore silently ran ordinary depth-one BFS. It now accepts
only absent/`1` and returns `CLI_BENCH_MACRO_DEPTH_UNAVAILABLE` for every other
value until distributed macro settlement is connected. The new CLI regression
failed before the guard and passed afterward (5/5 bench tests). This is a
correctness guard, **not** implementation of distributed macro depth.

The follow-up connects `MGBFS_MACRO_DEPTH>1` for a **single rank** to the
existing `MacroNativeBfs` weighted CUDA backend, with the same reference CLI
archive/search-only contract, seed and timing JSON. The CLI still rejects
multi-rank K>1 before launch; the runtime independently rejects it. This
exposes a real macro runtime rather than quietly substituting depth-one BFS.
The CLI acceptance test was RED on the old guard and GREEN after wiring; Linux
CUDA cross-target `cargo check` passes. Kaggle T4 version 3 then completed six
CLI archive runs over U4(2)/S5 and K=1/2/3, all with matching layer counts and
verified archives. The existing six GPU full-state macro tests passed plain and
under memcheck/racecheck/initcheck/synccheck (racecheck: zero hazards/warnings).
Evidence and limitations: `docs/validation/2026-09-23-macro-reference-cli-t4.md`.
The broader requirement for distributed weighted settlement is **not**
discharged by the single-rank path.

The next distributed-macro contract slice is CPU-only: transport now has an
explicit lookahead window, so unit-cost K=1 rejects offers beyond the next
depth, while `new_macro(..., K)` admits only depths `current+1..current+K`.
Schema2 adds `MacroDense` and `MacroHashFirst` frames with separate 16-byte
`MacroCandidateRef` planes; `decode_at` requires the metadata's
`source_depth + weight` to equal the frame's target depth. Wire and transport
tests pass. This does **not** yet route those frames through NCCL or settle
distributed weighted depths on GPU.
The follow-up payload validator checks each macro reference against its frame
target before owner offer. An end-to-end CPU protocol fixture drives a distant
offer and a later shorter offer through ticket issue/ACK/consume and depth
finalization; only the shorter one commits. The GPU receive/owner path does not
yet use these new frame kinds, so this fixture is a protocol oracle, not a
claim of native multi-rank K>1 correctness.
The next leaf writes weighted DENSE frames directly on GPU from a sorted owner
range: hash16, MacroCandidateRef16 and padded state bytes, with zero padding
and sticky invalid-reference failure. A checked preallocation plan bounds send
and receive frame regions per route slot; it is not yet wired to runtime
admission. Local RTX 3070 execution passed byte-for-byte; `sm75` compilation
passed, but this is not T4 execution or a multi-GPU run. Compute Sanitizer on
Windows could not launch the test executable (exit 13). Full evidence:
`docs/validation/2026-09-23-macro-pack-local-gpu.md`.
The framed NCCL adapter now prepares MacroDense rank ranges, writes ticket-bound
headers, and decodes source depth separately from target depth. The existing
ControlPump keeps its unit-depth ticket at the **source** layer; macro header
byte 12..15 stores that source depth and header.depth stores the weighted
target. `AdmittedBuffers::macro_dense_consumer` returns a leased, validated
hash/ref/state view. CPU control and Linux CUDA cross-target checks pass; the
modified two-rank `native_scatter` gate is prepared but has not yet executed
on 2xT4. The GPU owner still lacks device-side per-row ref validation and
future-slot merge wiring, so distributed K>1 is still unavailable.

User-owned dirty files are not implicitly part of this ledger's implementation.

The distributed reference benchmark now accepts `MGBFS_HASH_SEED_HEX`: exactly
32 hexadecimal digits interpreted as a numeric u128 and serialized little-endian
for `GEMM_U8_P32X4_V1`. If unset, its historical seed is 20260828
(`000000000000000000000000013527dc`). The canonical seed is included in the
cluster/archive config digest, so ranks with different seeds reject one another
at bootstrap. The result JSON records `hash_seed_hex`. Parsing and byte order
have CPU tests. A new-seed GPU end-to-end
run remains to be done; changing seed only changes collision sampling, not
graph semantics or the need for exact-state validation.

The S13 HF publication is recorded in `docs/validation/2026-09-05-s13-hf-complete.md`:
native search 4012.695 s, durable archive 4090.478 s on 2xT4; all 6,228
objects matched manifest size and LFS SHA. The successful server commit initially
returned HTTP 504 to its client. The later `promote_hf_stream.py`
reconciliation path (commit `420b8a7`) and `tests/test_promotion_reconcile.py`
already address ambiguous responses; do not reimplement or relaunch S13 for
this issue. Remote object checks do not prove a fresh decode of every row.

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

`MGBFS_TRACE_ROUTE=1` now emits per-rank/depth/batch markers for generation,
route, pack, exchange and owner completion. It forces diagnostic stream drains
around generation/route, so use it only to locate a stalled stage on a bounded
reproducer; its wall times are **not** benchmark numbers. On Windows the CUDA
feature check cannot run without a built `MGBFS_CUDA_LIB_DIR`; the added Linux
CUDA path still needs compile and physical GPU execution before it is relied on.
