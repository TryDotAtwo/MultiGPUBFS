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

The subsequent private Kaggle gate on 2xT4 passed the weighted DENSE scatter
fixture plain and under memcheck/racecheck/initcheck/synccheck, all with zero
errors (`docs/validation/kaggle-weighted-scatter-2xt4-v1.md`). The received
`MacroCandidateRef` plane now has a no-allocation CUDA validator for source
depth, target depth, weight and source-state index. The adapter exposes its
enqueue through the receive lease, and the two-rank fixture calls it before
materialization. This closes the isolated transport/metadata gate, **not**
distributed weighted owner settlement or production BFS. A second physical
2xT4 gate using source `787a9fe` passed plain and all four sanitizer tools.
The subsequent invalid-weight rejection fixture at source `4e08b44` also
passed on 2xT4 under all four sanitizers, before owner materialization. The
weighted future-slot/settlement path remains unconnected to the distributed
runtime, so full multi-rank macro BFS is not yet verified.

Next, `mgbfs_future_merge_run_bounded_checked` accepts the GPU metadata-fatal
word and refuses to publish a future slot when it is nonzero. Its failure
preserves the previously committed slot count as well as hashes/states; a
local RTX 3070 regression found and fixed the old count-reset behavior.
`MacroDenseRead::enqueue_future_merge` validates then merges a received sorted
frame using local dense identity refs (the sender's state_ref is not a receive
row). The two-rank fixture now tests that same-key offers from two sources
retain the first future state. Private Kaggle 2xT4 version 4 reached the plain
fixture but failed at an incorrect test expectation: the two owners correctly
receive different rank-sorted keys (31 and 41). No sanitizer result is
claimed for v4. The expectation is corrected at `a7a9d5f`. Version 5 then
passed both the two-rank provisional merge and standalone failure-atomic merge
fixtures plain and under all four Compute Sanitizer tools on 2xT4; details are
in `docs/validation/kaggle-weighted-scatter-2xt4-v1.md`.

The next two-rank gate feeds the provisional target-depth slot through the
existing GPU macro settlement primitive. Its depth-2 history is deliberately
synthetic: owner 0 contains the same key and must discard the depth-3 offer;
owner 1 has no such key and must retain its offer. This checks the receive ->
future -> history-membership boundary, not complete distributed BFS generation
at depth 2. Private Kaggle version 6 pinned to `a4cfe56` passed the two-rank
fixture and independent guarded-merge fixture plain and under all four
sanitizer tools on physical 2xT4. See the validation record. The missing
boundary is now the production scheduler that supplies genuinely generated
history and finalizes each depth in globally agreed order.

The next fixture exercises a narrower scheduling invariant: after a source
depth with no parents, the existing control pump still emits `FinalizeDepth`,
and the pending depth-3 future is settled inside that event, before rank
`Finalized` acknowledgement and `Publish`. It does not replace the synthetic
depth-2 history with generated states. Private Kaggle version 7 is pinned to
`278d439` passed on physical 2xT4 plain and under all four sanitizers (v7).
The device-driven settlement API consumes the future slot's count/fatal
directly, removing that D2H/re-upload dependency. Local RTX 3070 test and
private Kaggle v8 on physical 2xT4 passed plain and all four sanitizers for
both fixtures; see the validation record. This remains a whole-layer
reference boundary with synthetic shorter-depth history, not a production
distributed macro BFS.

Architectural correction before extending this path: section 3 of
`docs/matrix-runtime-architecture-v2.md` explicitly rejects whole-layer
per-batch owner merge and scratch. The production path must use existing
`mgbfs_bounded_owner_compare/commit`-style per-microbucket jobs, with bounded
`I,J,K` scratch, sole shard writer, provisional target-depth membership,
and a `2*Km` committed-history window. Neither `macro_native.rs` nor the
current two-rank full-layer fixture meets that memory contract. Do not
present their positive tests as production acceptance. Next implementation
gate: define/prove bounded future bucket ranges and owner reservation before
integrating the distributed weighted scheduler.

A first CPU preflight slice now exists as `FutureBucketLayout` in
`mgbfs-core::macro_memory`. It assigns each `(target_depth mod Km, bucket)` a
fixed contiguous hash extent, with configured per-bucket capacities whose
sum is exactly the physically reserved `Qfuture`. This avoids an implicit
`Km*B*K` future allocation. Its target-depth window and checked byte/count
arithmetic have standalone tests. This is **not yet** the production GPU
future owner: the capacity vector must be frozen in RunConfig, GPU compare/
commit must accept per-bucket offsets/caps, slot reuse must wait for settlement
and archive leases, and the full memory query must include the new metadata.
The per-bucket capacity distribution is a proposed preflight policy, not a
measured choice; skew overflow remains fatal rather than dynamically growing.
The CUDA bounded owner now has a compatible `compare_layout/commit_layout`
ABI taking device-side prefix offsets and capacities. Its CUB leaf passed
physical 2xT4 plain + four sanitizers per GPU at `e085f0a`; the BMMA leaf
passed the stricter compact-layout fixture at `e085f0a` (15/15 checks).
See `docs/validation/bounded-owner-compact-layout-2xt4.md`.
The new ABI has **not** been wired into `RunConfigV1`, provisional slot
lifecycles, the `2*Km` history-window compare, or the distributed scheduler.
An additional `compare_history_layout` leaf now accepts a fixed sequence of
bounded read-only history ranges and the same compact accepted directory. Its
three-slot CUB fixture passed physical 2xT4 plain + all four sanitizers per
GPU at source `80be3aa`; BMMA passed 15/15 checks at the same source. This covers GPU
membership across more than two old layers but **not** the host/device slot
rotation, full 2*Km residency, or weighted distributed finalization.

`MacroHistoryWindow` now supplies a fixed `2*Km` host control-plane slot
generation and lease contract. A layer at `d-2*Km` remains readable while
depth `d` settles; only after settlement, all owner reader events and its
archive D2H event may physical slot `d mod (2*Km)` be rebound to depth `d`.
Focused RED/green tests cover blocked reuse, stale depth, order and lease
underflow; the full default `mgbfs-runtime` CPU suite passed locally. This
module is **not yet wired** to the actual GPU completion events or global
`FinalizeDepth` pump, so it is a CPU scheduling contract, not an end-to-end
macro-runtime proof.
After commit `0155190`, `cargo test -p mgbfs-core -p mgbfs-runtime -p
mgbfs-cli --quiet` passed locally. The broader `cargo test --workspace`
could not start its CUDA package because this Windows checkout lacks
`MULTIGPUBFS_CUDA_LIB_DIR`; this is an environment/build prerequisite, not a
passing or failing GPU-runtime result.

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

## 2026-09-23 macro control gate

Commit `edc738b` added an `AdmittedBuffers` gate: `FinalizeDepth` ACK requires settled macro history and a free replacement slot; admitted `Publish` advances the history window. CPU tests pass. Kaggle private version 9 on two Tesla T4s completed from exact source `edc738be1498fe2b1b1bd5dca881a88f80da5ceb`: the two-rank scatter/rollover fixture and compact-future-bucket fixture each passed plain, memcheck, racecheck, initcheck and synccheck. The Kaggle `summary.json` and logs are retained locally under `test_results/macro-scatter-gate-v9/`; the digest-free summary is transcribed in `docs/validation/macro-scatter-gate-2xt4-v9.md`. This is control-plane and leaf integration, **not** a complete weighted BFS. Full CUDA event binding and end-to-end weighted scheduler remain open.

Commit `0efee98` adds nonblocking completion polling for settlement, reader release and archive D2H lease release; a not-ready poll leaves all control/lease state unchanged. The two-rank fixture now records and queries a real owner-stream CUDA event before settlement. Kaggle private version 10 passed plain and all four Compute Sanitizer modes for both the two-rank fixture and the future-bucket leaf, at exact source `0efee986ac59fd195ef45e5e656d782cec3239b9`; see `docs/validation/macro-event-gate-2xt4-v10.md`. **The production multi-GPU weighted scheduler still does not call this API.** Its event ownership, archive D2H notification, full weighted owner exchange and graph oracle remain outstanding.
