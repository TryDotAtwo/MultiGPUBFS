# MultiGPUBFS completion ledger

Status: active, updated 2026-09-26. This file separates accepted requirements from
implemented paths and measured evidence. `ARCHITECTURE_NEED.md` remains the
architecture contract; `library-first-bfs.md` is the library experiment log.
The source-level owner/transport/retirement audit and connected change set are
in `owner-transport-retirement-batch-audit.md`; it does not establish a measured
bottleneck or completed asynchronous pipeline.

## 2026-09-26 constructor admission gate

At source `8f308ef284fb12d6d6fad2204028205076d12c43`, the private Kaggle
v77 two-P2P-T4 gate passed an asymmetric post-NCCL constructor fault: rank 0
reported `CUDA_STATUS_-1` for an intentionally nested RMM pool, rank 1 reported
`REMOTE_CONSTRUCTOR_FATAL`, and the fixture completed in 1.41 s. The same run
passed HostSized and LSA S4 BFS, archive/group-marker checks, small-layer
archive capacity and independent full-state oracle. v73/v75 had timed out;
v76 exposed a CUDA-stream lifetime error before the v77 correction. Exact
logs and limits are in `docs/validation/archive-admission-t4-v69.md`.

This is a post-NCCL allocation failure gate only. At `ed19401`, the depth-one
reference launch was changed to rendezvous by launch identity, then agree on
all 256 config-digest bits and rank-local config/device-admission failure
before archive admission or communicator creation. A two-rank CPU control
test, the full available runtime CPU suite, and Linux/CUDA Rust typecheck
passed. Private Kaggle v78 is queued for its actual two-P2P-T4 fault and
normal-BFS gate; no GPU result for this change exists yet. The macro launch
does not use this path. Other pre-NCCL constructor failures, control-buffer
allocation failures after NCCL init, and errors inside LSA activation remain
open. Neither v77 nor the config change removes hot-path CPU readbacks or
per-round collectives. See `docs/reviews/2026-09-26-connected-runtime-audit.md`.

At `a931f93`, multi-rank macro-depth parsing moved inside the configuration
agreement, eliminating an earlier one-rank return before rendezvous. This has
CPU tests and Linux/CUDA typecheck, not a physical two-T4 gate. A subsequent
working change adds a fourth `GroupPublished` boundary after the rank-zero
group marker write: a rank-zero publication failure is then reported to all
ranks. The two-rank CPU boundary test and full available runtime CPU suite
pass; late filesystem failure in separate GPU processes remains unverified.

The HF stream publisher now has a scoped local change that carries per-file
row counts from Parquet production through rank promotion, rejects incomplete
inventories, pins staging branch revisions to SHA, checks remote size/LFS SHA,
and verifies the resulting commit before returning success. The full local
Python test suite passes (171 tests, 6 skipped). Live HF publication, actual
remote Parquet footer row counts and legacy staged manifests without `rows`
remain separate gates.

A scoped archive change retains `O_NONBLOCK` after FIFO admission and gives
each FIFO write a bounded absolute deadline. The stalled-writer CPU test and
the available runtime suite pass; a real Linux FIFO test typechecks but still
needs execution. This only bounds FIFO write stalls, not every archive worker
wait or a full two-rank stream-archive run.

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
| Device-driven library owner | `CUCO_RANK` GPU rank-batch owner transaction in `distributed_native.rs`/`cuco_rank_batch.cu` | Physical 2xT4 full-state and bounded capacity/sanitizer gates; whole-run S10 Nsight trace; scalar device-store full BFS v26 and leaf four-sanitizer v11 gates | Remove remaining host-sized exchange, per-batch fatal-vote, HASH_FIRST and archive-control dependencies; retain the semantic FinalizeDepth barrier; capture/trace the full BFS DAG, then repeat correctness and full-app sanitizer gates |
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

## 2026-09-23 CPU synchronization audit, verified against e18859b

The transferred static audit is preserved as the tracked file
`docs/validation/2026-09-23-hot-path-sync-audit.md`. The following source
dependencies remain on current HEAD; their performance cost is **unmeasured**.

| Boundary | Current code dependency | Required end-state gate |
|---|---|---|
| Buffer upload/readback | `distributed_native.rs` `Buffer::put/read/one` synchronizes uploaded host slices and reads device words; `native.rs` has the same synchronous upload pattern | Stable pinned lifetime and device-resident control; no unsafe removal of the wait |
| cuCollections owner | `commit_library_batch` reads the device directory, loops over shards on CPU, receives a host survivor count, snapshots reserve/materialization, then appends host extents; `cuco_owner.cuh` and `control_transfer.cpp` also drain streams | One bounded device-driven owner transaction across compare, all-capacity reservation, commit and descriptor publication |
| Native owner/HASH_FIRST | `commit_owner_batch` reads directory and extents; HASH_FIRST reads extra controls/counts and `materialize_hash_first` has its own host waits | Device descriptors and source/target identity preserved through materialization; full profile parity |
| Route/transport | `advance_inner` reads routed/owner counts; each peer round exchanges host-sized count before variable-size NCCL payload | Explicit device-count transport protocol with bounded bytes, identical collective order and zero-peer participation; fixed-size padding only after a measured traffic/VRAM decision |
| Parent retirement | `advance_inner` drains producer/owner streams and reads the ring before recycling parent storage | Event- and consumer-owned retirement after archive, transport and owner readers drain |
| Fatal consensus | `all_max` performs a blocking upload, NCCL reduction, stream drain and readback several times per depth, including batch rounds | Preserve fail-fast and rank-wide fatal propagation while removing per-batch host decision chains or proving their cost acceptable |

This invalidates any claim that the current BFS is fully asynchronous or has
only a `FinalizeDepth` barrier. The next implementation slice must cover the
owner-to-transport-to-retirement DAG, not a standalone cuCollections wait.
Required evidence: exact capacity/traffic formulas, CPU oracle for descriptor
and lifetime ordering, full-owner CUDA Graph capture without host readback,
Nsight Systems timeline of a real BFS, complete DENSE/HASH_FIRST and 1/2-rank
oracles, all four sanitizer modes, and repeated end-to-end time/VRAM panels.

The current NCCL wrapper takes `send_bytes` and `recv_bytes` as host scalars
(`cuda/nccl_transport.cpp`). If a future protocol sent each peer's entire
candidate capacity, let `C = batch * moves`, `W = world`, and
`Q = 16 + packet_stride` bytes per hash+payload pair. Padding would send
`(W-1)*C*Q` bytes per rank per batch, versus at most `C*Q` actual remote
bytes and approximately `(W-1)*C*Q/W` when ownership is balanced. Balanced
traffic inflation is `W` (8x at eight ranks), before retransmission or extra
staging. A fixed-size NCCL scheme is therefore **not accepted by default**;
the count/transport protocol remains open until its full cost and GPU event
ownership are measured.

The first route-to-pack dependency is now removed at source `a272a17`:
`mgbfs_exchange_pack_device_n` consumes the route's device-resident valid
count, launches at admitted capacity and guards the tail/overflow on GPU.
`distributed_native.rs` queues route and pack on the same stream without
reading `route_count` between them. The count and owner directory are still
read after pack to schedule NCCL payloads, so **the batch remains CPU-driven**.
The independent T4 primitive gate passed on both physical T4s with plain,
memcheck, racecheck, initcheck and synccheck; see
`docs/validation/device-count-pack-2xt4.md`. Full `mgbfs-distributed-sanitizer`
Kaggle v47 stopped because the harness expected 12 archive tests while 13
passed. V48 then passed the five-mode two-T4 distributed/macro gate, but its
CLI smoke panel did not start: six passing CLI tests exceeded the harness's
stale expectation of three. The corrected private v49 later completed, pinned
to the same source; see the v49 result below and
`test_results/distributed-sanitizer-v48/REPORT.md`.
This slice is not an owner DAG capture,
transport redesign or latency improvement claim.

## 2026-09-23 completed library calibration audit

The previously described Kaggle `trydotatwo/mgbfs-library-owner-t4` v55 is
**COMPLETE**, not running. Its remote `summary.json` was downloaded again and
matches the preserved `test_results/library-owner-v55/library-owner/summary.json`
(SHA256 `29cb7dbc6537bc7ff9afc814399d854a2e3a24d972b7e41e24d681ffc70a637`).
Source was `f5b52c9f240e89c5b8b30828919ef56c367fdad6`. The five-repeat S10
DENSE screen gives cuCollections/cuCO search medians 0.488662 s (1 T4) and
0.465600 s (2 T4), versus the preserved native CUB medians 1.032990 s and
0.824495 s. Durable medians were 4.837174/3.843546 s for cuCO versus
4.797040/3.995748 s for native. The 50 ms sampled full-device VRAM was
901 versus 835 MiB (one rank), and 529 versus 457 MiB per rank (two ranks).
Thus this 8-shard cuCO setting wins search but *not* peak VRAM, and the
archive-inclusive result is mixed. It does not resolve the CPU-driven owner
hot path; the summary's sanitizer field points to earlier v50 evidence rather
than sanitizers of this exact v55 source/configuration.

## 2026-09-23 transport platform preflight

The private Kaggle two-T4 preflight in
`docs/validation/transport-feasibility-2xt4.md` found bidirectional CUDA P2P
access but NCCL runtime 2.25.1. NCCL's device-initiated LSA API requires
2.28+. Current host-sized NCCL payload submission therefore cannot simply be
replaced by a device-side call on this environment. A newer pinned NCCL LSA
stack or a separate CUDA IPC peer-memory protocol needs a real transfer and
failure/lifetime gate before implementation selection; this is a platform
feasibility result, not yet an asynchronous transport implementation.
The follow-up private notebook v5 installed pinned NCCL 2.29.7 in isolation
and passed a real two-T4 LSA device-kernel round trip with exact results on
both ranks. The newer communicator reported one LSA team and device API
support. Thus the NCCL LSA route is viable on the target pair, subject to a
device-count, capacity/fatal/lifetime protocol and end-to-end BFS gates; see
the updated transport feasibility report. Production still uses NCCL 2.25.1.

The corrected full small-graph regression Kaggle v49 completed at pinned
source `a272a17` with all four sanitizer tools clean and 24/24 one/two-rank
CLI profile smokes passing, each with verified archives and S4 global layers.
See `docs/validation/distributed-sanitizer-2xt4-v49.md`. This removes the
v47/v48 harness uncertainty, not the CPU hot-path or large-graph validation
gaps.

An opt-in NCCL 2.29.7 LSA C ABI candidate is now on branch
`codex/library-first-bfs` at `ae22e29`. The private two-T4 transport probe v15
compiled and linked the production `nccl_transport.cpp`, then passed exact
payload/control checks for `(3,1)` and `(0,5)` and all-rank no-payload fatal
for aggregate capacity overflow `(20,20)` and `(33,1)` at capacity 32.
Kaggle T4 hosts vary: the probe encountered both P2P `OK` with
`deviceApiSupport=1` and P2P `NS` with `deviceApiSupport=0`. LSA is therefore
an explicit preflight-selected backend, not a universal 2×T4 assumption or
silent runtime fallback. The candidate is not connected to the BFS owner DAG;
multi-epoch lifetime, sanitizer, 8-rank ordering, performance and full
owner→transport→retirement integration remain open. Details and raw results:
`docs/validation/transport-feasibility-2xt4.md` and
`test_results/kaggle_transport_probe_v15/`.

The follow-up v16 sanitizer gate was mixed: racecheck/synccheck clean,
unfiltered memcheck reported NCCL setup CUDA API errors, and unfiltered
initcheck timed out. V18 then passed a *kernel-filtered* memcheck for the two
LSA kernels while filtered initcheck again timed out after 600 seconds. The
LSA backend is still experimental and not accepted as fully sanitized.

The default `MGBFS_NCCL_LSA=OFF` build at `3bbb416` repeated the full small
two-T4 v50 gate: plain plus four sanitizers and 24/24 profile smokes PASS.
That does not sanitize or exercise the LSA code. See
`test_results/kaggle_distributed_sanitizer_v50/distributed-sanitizer/summary.json`.
V51 repeated this gate at `ae22e29` and again reports COMPLETE, all five
tool modes PASS, 24/24 profile smokes PASS and identical S4 layers; it also
kept LSA off. See `test_results/kaggle_distributed_sanitizer_v51/`.

The device-side rank-batch reservation leaf was added at `63377c4`. It checks
all shard accepted capacities, aggregate/layer/request capacity and the
StateRing/descriptor capacity before advancing the ring or layer count. One
extent and GPU offsets cover the batch. The private Kaggle state-commit gate
v8 built that exact source on two physical T4s and completed 20/20 checks:
plain plus memcheck, racecheck, initcheck and synccheck for state commit and
archive pack on each GPU. Raw output is in
`test_results/kaggle_state_commit_v8/state-commit-gate/`. This is a tested
leaf only: cuCO compare still returns host counts, and the runtime does not
yet call the new rank-batch ABI. It is not evidence of a CPU-free BFS.

The follow-up distributed regression v52 completed at the same source SHA on
two T4s: plain plus all four Compute Sanitizer modes PASS, 24/24 reference
profile smokes PASS with verified S4 archives and layers `[1,3,5,6,5,3,1]`.
It tests the unchanged runtime path, not integration of the new reservation
leaf. See `docs/validation/distributed-sanitizer-2xt4-v52.md`.

The pinned-cuCO dynamic-ref feasibility probe at `e836a11` passed on both T4s
with all four Compute Sanitizer modes. Per-candidate GPU selection from two
persistent table refs is therefore demonstrated, but the rank-batch cuCO
compare/commit, GPU shard offsets and runtime wiring remain absent. See
`docs/validation/cuco-dynamic-refs-2xt4.md`.

The GPU shard-count/offset leaf at `ec988b7` passed the focused two-T4 v10
gate (20/20 plain/sanitizer checks). It derives counts from device-resident
CUB-selected indices and sorted hash prefixes without a host readback, with
malformed input failing before offset publication. The RED v9 link failure
and GREEN v10 logs are recorded in
`docs/validation/rank-shard-counts-2xt4.md`. The production cuCO rank-batch
compare/commit and runtime wiring remain unfinished.

The subsequent `CucoRankBatch` candidate at `a47ba02` captures cuCO compare,
device shard counting, all-shard StateRing reservation and persistent commit
as one CUDA Graph without reading survivor counts on the host. The private
two-T4 v5 gate passed plain and all four sanitizer modes on both GPUs; it
checked a second batch against the same persistent tables and an accepted-
capacity failure that leaves counts and ring tail unchanged. See
`docs/validation/cuco-rank-batch-2xt4.md`. This is still an isolated C++
library path: the Rust runtime uses its synchronous per-shard V1 owner, and
transport/retirement retain CPU dependencies. No full BFS or performance
claim follows from the captured leaf.

The DENSE materialization leaf `mgbfs_state_materialize_rank_batch` now takes
source and selected counts from device words and checks every source ordinal
before copying into the reserved extent. The two-T4 v11 RED link failure and
v12 GREEN 20/20 plain/sanitizer checks are recorded in
`docs/validation/rank-batch-materialize-2xt4.md`. A Rust FFI declaration is
present, but the runtime still does not call this leaf. HASH_FIRST and the
owner→transport→retirement event/lifetime protocol remain open.

The cuCO rank-batch now has a stable C ABI and matching Rust FFI declarations.
The v6 RED link failure and v7 two-T4 GREEN gate exercise that ABI across
captured compare/commit, reuse and overflow; see
`docs/validation/cuco-rank-c-abi-2xt4.md`. The production Rust scheduler still
does not invoke it, so this closes an interface gate only, not the runtime
CPU-dependency audit.

The C-ABI cuCO owner and device-count DENSE materializer have now been
composed in one captured CUDA Graph. The v8 Rust-toolchain failure and v9
two-T4 pass (actual state bytes, reuse, overflow, four sanitizers) are in
`docs/validation/cuco-rank-dense-state-2xt4.md`. This validates the local
owner→StateRing edge only; it does not remove the runtime's host-sized NCCL
exchange or CPU-driven retirement, nor prove complete BFS layers.

The rank owner now exposes a FinalizeDepth `seal`: it rejects a pending epoch,
releases borrowed cuCO history tables after caller drain, and retains accepted
keys for export. The v10 RED link failure and v11 two-T4 plain/sanitizer gate
are in `docs/validation/cuco-rank-seal-2xt4.md`. Runtime creation, event
ordering and finalization remain to be wired; this is not full BFS evidence.

The Rust DENSE reference runtime now invokes `CucoRankBatch` through the C ABI.
The private two-T4 v12 test failed at the expected `REFERENCE_CUCO_RANK_NOT_WIRED`
marker before integration; v13 passed the one-rank full-state oracle on both
physical T4s. The separate two-T4 v56 run passed the two-rank full-state and
archive oracle with both rank maps and pre-dedup ON/OFF, then completed the
S4 and U4(m=2) torchrun CLI fixtures. Exact commits and downloaded records are
listed in `docs/validation/cuco-rank-runtime-2xt4.md`. This proves small-graph
runtime correctness for DENSE/CUCO_RANK, not HASH_FIRST support, large-graph
capacity, speed, memory superiority, or removal of host dependencies. The
per-batch route count upload and owner control snapshot still synchronize in
`distributed_native.rs`; transport size exchange and parent retirement still
use CPU readbacks. The v57 capacity-failure/recreation fixture passed plain on
both T4s. V15 passed the one-/two-rank full-state runtime oracles under
memcheck, racecheck (zero hazards/warnings), initcheck and synccheck at exact
source `aabbf45`; see `docs/validation/cuco-rank-runtime-2xt4.md`. The v58
capacity-failure/drop/recreation fixture passed memcheck on both physical T4s
at exact source `c16c8c7`, zero errors. V59 subsequently passed racecheck,
initcheck and synccheck for the same fixture on both T4s, with zero reported
hazards/errors/warnings. Neither gate
removes the confirmed CPU-driven route/retirement dependencies.

The rank-owner teardown had an independently reproduced lifetime defect:
private 2xT4 v16 failed `RANK_DESTROY_MUST_DRAIN_IN_FLIGHT_WORK` after a delayed
same-stream callback; v17 passed on both GPUs after destruction synchronized
the creating stream. This is teardown-only and adds no hot-path wait. The
device-count AoS→SoA owner-window bridge then showed the expected v18 RED
link failure and v19 GREEN exact-data/fatal fixture on both T4s. Its begin
and count stay on GPU and source ordinals remain absolute. V20 passed all four
sanitizers on both T4s, with zero racecheck hazards/warnings. See
`docs/validation/device-owner-window-2xt4.md`. The Rust scheduler
does not yet call this bridge or the experimental LSA transport, so no
end-to-end CPU-dependency claim follows.

The next device route-window leaf derives logical-owner packed begin/count
from GPU `owner_counts`. Private 2×T4 v4 failed at the expected undefined
symbol, and v5 passed plain plus all four Compute Sanitizer modes on both
physical T4s; see `docs/validation/device-owner-route-window-2xt4.md`.
The Rust scheduler does not yet consume this output, and transport/retirement
remain CPU-driven. This closes a primitive contract only.
The route-window contract now also rejects `sum(owner_counts) != route_count`
on GPU. A 2×T4 v6 RED signature gate preceded v7 plain/four-sanitizer GREEN;
see the same validation record. The nonzero-window materialization fixture
passed plain on both T4s in v21 and all four sanitizer modes in v22, proving
that absolute source ordinals need the *whole source* row bound rather than
the narrower owner-window count.
The rank-owner candidate-copy guard had a v23 expected RED after sticky fatal,
then v24 plain and v25 four-sanitizer GREEN on both physical T4s at exact
source `540698c`; see the same validation record. This still does not wire
the owner path into Rust or remove transport/retirement host dependencies.
The initially clean `valid_rows > capacity` case then produced a distinct v26
RED: one thread set fatal while others still copied candidate input. A
uniform early return in `copy_candidates` at `62bec61` passed the v27 plain
owner fixture on both T4s. V28 passed all four sanitizer modes on both T4s,
with zero errors and racecheck hazards/warnings. This strengthens
fail-fast but remains an isolated owner leaf, not end-to-end BFS integration.

The next rank-owner cut removes the **per-batch extent/control snapshot**:
`mgbfs_state_publish_next_extent` accumulates at most two next-frontier
physical ranges on GPU, then the Rust runtime reads them once at
`FinalizeDepth`. The C ABI v13 RED/v14 plain and four-sanitizer GREEN on two
T4s are recorded in `docs/validation/device-next-extents-2xt4.md`.
Integration source `b15b74b` passed local Rust type-check with CUDA/library
features and the CPU suite. V29 passed the 1-GPU full-state oracle but stopped
on the old capacity-error assertion; v30 at `5f87e24` passed the full plain
1/2-GPU layer/archive oracle and two-process CLI S4/U4m2 verification. The
v31 at the same source passed the full integrated plain/four-sanitizer
1/2-T4 rank-owner fixture gate with zero reported errors/hazards; see the
validation record. An Nsight timeline is still pending. This does not remove
per-batch route count upload, NCCL host size exchange or parent retirement
readback.

The next retirement cut passes the parent extent by value into CUDA, removing
the `Buffer::put()` H2D extent upload and its protecting host wait on both
DENSE and HASH_FIRST paths. The CUDA leaf had expected v15 RED and v16
plain/four-sanitizer GREEN on both T4s; Rust integration at `0e3d1d6`
passed local typecheck and the CPU suite. A separate full BFS 2xT4 v1 gate
passed plain full-state/layer/archive fixtures; the v2 four-sanitizer gate
also passed the full 1/2-T4 BFS fixtures with zero reported errors/hazards.
The stream drain and ring-fatal D2H after retirement remain at this source,
so this is **not** yet CPU-free retirement. Evidence:
`docs/validation/device-parent-retirement-2xt4.md`.

Static follow-up at `247d7ce`: in the DENSE round-1 path,
`distributed_native.rs` returns immediately on `ring.fatal` after retiring a
parent prefix, before the `process_owner_pair`/`all_max` failure vote. Another
rank may already be waiting in that collective. This is a rank-ordering
correctness risk, not merely a performance cost; the current capacity fixtures
do not inject a retirement FIFO failure and therefore do not close it. A fix
must route the retirement fatal through an identical collective sequence on
all ranks before either rank proceeds to owner commit, including zero-payload
rounds. Verify with a two-rank injected retirement-failure fixture and a
bounded timeout before claiming fail-fast. The HASH_FIRST path already votes
on its retirement fatal, but still reads the ring on CPU; neither path is
device-driven end-to-end.

The first rank-ordering fix (`b71c16b`) makes all DENSE ranks vote on a
retirement error before owner commit, after the current P2P completion event.
Local RED/GREEN helper test, CUDA-feature typecheck and CPU suite passed;
private two-T4 full-BFS v2 passed its normal-path full-state/archive fixtures.
The fixture does not inject the retirement FIFO failure, so the abnormal path
remains unproven. This adds one blocking collective per DENSE round and is
**not** the requested CPU-free owner→transport→retirement implementation.
The next RED test (`1c29487`) asks CUDA to derive the NCCL vote word directly
from sticky ring fatal. Private T4 v17 failed at the expected undefined
symbol, then v18 at `8aacb24` passed 20/20 state-commit/archive-pack plain
and four-sanitizer checks on two physical T4s. Rust now uses the device word
for both DENSE and HASH_FIRST retirement votes; local CUDA-feature typecheck
and the CPU suite pass. The full 2×T4 BFS v3 integration gate passed plain
single-device and rank-owner layer/archive fixtures. The v4 gate at the same
source passed these fixtures on both physical T4s under memcheck, racecheck,
initcheck and synccheck (zero reported errors or hazards). A focused two-rank
FIFO fault injection at `4bd474b` passed on
two physical T4s: one local sticky fatal 17 became group fatal 1 on both
ranks without a hang. Full scheduler-level fault injection remains open.
The same source passed a second 2×T4 v4 gate with the injected fault under
plain, memcheck, racecheck, initcheck and synccheck; all reported zero test
failures and zero sanitizer errors/hazards. This still exercises the focused
rank fixture, not scheduler-level injected failure.
This removes one ring-fatal D2H but the collective remains host-blocking and
owner/transport count decisions remain CPU-driven.
The follow-up `ced41ab` removes a duplicate DENSE stream drain after the
blocking first-round fatal vote. Its plain 1/2-T4 full-BFS gate passed in
private Kaggle notebook v5: 3/3 tests per GPU and 3/3 two-GPU tests,
including the injected FIFO fatal vote. V6 at the same source completed
plain, memcheck, racecheck, initcheck and synccheck on both physical T4s:
3/3 per device and 3/3 two-rank tests in each mode, with zero sanitizer
errors/hazards. The two-rank set includes the injected FIFO fatal vote.
This validates the scoped wait removal, not a CPU-free owner→transport→retirement
path (`docs/validation/dense-wait-full-bfs-2xt4-v6.md`).

The subsequent unprofiled S10 DENSE five-repeat screen on physical 2×T4
compared `CUCO_RANK` (512 MiB fixed pool/rank) with preserved native CUB,
both with verified archives and identical 46-layer histograms. Search medians
were 0.380726 and 0.823757 s, while sampled full-device peaks were 945 and
457 MiB/rank. Durable medians were 3.774029 and 3.899147 s. This is a
search-speed/VRAM trade, not a durable throughput win or proof for larger
graphs. The paired 96 MiB-pool screen subsequently completed on physical
2×T4: ten archive-verified S10 runs, identical 46-layer histograms,
CUCO_RANK/CUB search medians 0.407692/0.840257 s and sampled device peaks
529/457 MiB per rank. This establishes a smaller-pool S10 Pareto point,
not full-state cross-backend equality for S10, larger-graph capacity or a
CPU-free runtime. See `docs/validation/cuco-rank-paired-s10-2xt4.md`.

At `fb8b9f4`, the CUCO_RANK owner now reads local/remote window counts from
device words rather than uploading a host row count again. The physical 2xT4
S10 one-sample gate completed with 3,628,800 states, matching CUB layer
counts and verified archives. The full-state one/two-GPU fixtures and injected
retirement fault passed all four Compute Sanitizer tools with zero reported
errors/hazards. This validates only that scoped owner-count change; host-sized
NCCL payloads, collective control and retirement remain. See
`docs/validation/rank-device-window-2xt4.md`.

At `76c3337`, explicit NCCL 2.29.7 LSA was wired into the actual two-rank
CUCO_RANK/DENSE exchange. The private two-T4 v1 full-state gate passed eight
small-graph oracle/archive fixtures across owner maps and pre-dedup modes.
See `docs/validation/lsa-full-bfs-2xt4.md`. This is an integration correctness
result, not an end-to-end CPU-free, sanitizer-clean, memory-optimal or
large-graph performance result. A subsequent revision removed the
LSA-specific D2H route-count read.
The later P2P-capable two-T4 v5 gate passed the same eight fixtures on
`d95ef21`, which also removes duplicate legacy receive allocations in LSA
mode and replaces the generation-buffer host wait with CUDA-event ordering.
This is still plain correctness; full-path LSA sanitizers, large-graph VRAM
and speed, and remaining owner/retirement/failure host dependencies are open.
The unfiltered LSA full-BFS `memcheck` in Kaggle v6 timed out during NCCL
setup without a test result. V7 narrowed instrumentation to transport kernels
and one fixture: both ranks traversed all depths and the test printed
`1 passed`, but the sanitizer command timed out after the test result without
an error summary. V8 used `--target-processes application-only` on another
P2P-capable two-T4 host; plain BFS passed, but filtered memcheck timed out
after both ranks entered depth 0. Neither run closes a sanitizer gate, and
the child-process-tracking hypothesis was not supported by v8. Details and
raw log paths are in `docs/validation/lsa-full-bfs-2xt4.md`.

The independent two-T4 S10 Nsight Systems diagnostic at the same source
captured a complete archive-verified `CUCO_RANK` run with host-sized NCCL.
Its aggregate, non-search-filtered trace contains 6,266 stream synchronizes,
5,632 D2H copies, and substantial NCCL kernel time. It confirms that the
remaining host dependencies execute, but cannot by itself quantify their
search critical-path cost or prove LSA overlap. See
`docs/validation/cuco-rank-timeline-2xt4-v2.md`.

A same-source paired S10 screen at `34b1c81` completed five unprofiled,
archive-verified runs for each transport on the same physical two-T4 host.
LSA reduced search median from 0.427086 to 0.383635 s (10.17%), but sampled
full-device VRAM rose from 529 to 567 MiB/rank; archive-complete medians were
3.829540 and 3.853357 s. Both variants had the same explicit aligned
allocation plan, so the extra observed VRAM is outside that plan. This is a
small-workload transport screen, not proof of end-to-end CPU independence,
large-graph scaling or sanitizer cleanliness. See
`docs/validation/lsa-paired-s10-2xt4.md`.

At `6c74077`, the parent cursor computes a fixed number of peer epochs from
immutable frontier extents at each depth boundary. A rank with fewer parents
keeps issuing zero-payload rounds; the per-batch host `all_max(more)` is gone.
The P2P-capable two-T4 v10 gate passed three full-state/oracle/archive
integration tests spanning LSA DENSE, host-sized CUCO_RANK DENSE and
host-sized native/CUDF/CUCO DENSE/HASH_FIRST. This validates the scheduling
change on small graphs but leaves per-batch archive/error votes, owner and
retirement readbacks, sanitizer and large-frontier timing open. See
`docs/validation/fixed-depth-rounds-2xt4.md`.
The matching post-change S10 v2 screen also completed five paired runs per
transport with 20 verified rank archives. Within that session, LSA/host-sized
search medians were 0.381001/0.465085 s, durable medians
3.711824/3.759861 s, and sampled peaks 567/529 MiB per rank. The v1/v2
comparison crosses Kaggle sessions, so it does not isolate the fixed-round
change's speed contribution. See `docs/validation/lsa-paired-s10-2xt4.md`.

At `88f0610`, a scheduler-level asymmetric archive failure fixture uses a
two-slot rank-0 pinned ring and a slow disk worker while rank 1 has ample
slots. The private two-T4 Kaggle v12 gate completed: rank 0 returned
`ARCHIVE_PIN_RING_FATAL`, rank 1 returned `REMOTE_ARCHIVE_FATAL`, and both
worker threads exited (`test_results/kaggle_lsa_archive_fault_v12/lsa-bfs-gate/`).
The first v11 attempt never reached the protocol because one slot violated
`ArchiveRingPlan::new`'s minimum of two; it is not a failed protocol run.
This validates the existing per-batch fatal vote for that injected path. It
does **not** justify removing the blocking vote: `archive_range` can fail on
the host before the next peer exchange, and NCCL documents that all active
ranks must participate in communicator abort. A replacement needs an
asymmetric-failure protocol with identical rank participation and an actual
two-rank fault gate before any ordinary-path wait is removed.
At `57953cf`, the same scheduler-level fixture was extended to LSA with the
`CUCO_RANK` owner. Kaggle v13 and v14 returned `UNSUPPORTED_HOST` before
build because each selected two-T4 host reported P2P disabled; neither is a
test failure. V15 selected a P2P-capable two-T4 host and completed both the
host-sized native-owner and LSA/CUCO_RANK one-rank archive-slot-exhaustion
tests. Both returned the expected local/remote fatal errors and exited without
a hang. Raw summaries/logs are under
`test_results/kaggle_lsa_archive_fault_v15/lsa-bfs-gate/`. This extends the
fault gate to both transport backends, not to every asymmetric CUDA/NCCL or
disk-write failure and not to an asynchronous ordinary path.

Static follow-up at `26f25f2`: the `CUCO_RANK` owner already runs compare,
shard counting, reserve, commit, materialization and next-extent publication
from device-held counts in `commit_rank_library_batch`; it reads the resulting
extent list only in `FinalizeDepth`. Do not build another per-shard CPU extent
replacement for this backend. On the LSA DENSE batch path, the remaining
blocking host boundaries are instead the archive-fatal `all_max` before
generation, the ring/transport-fatal `all_max_ring_fatal` after exchange, and
the host-error `vote_group_error` after owner work. The first has now passed
asymmetric failure injection on both transports, so removing its wait requires
a new rank-consistent failure protocol, not deleting a local sync. The latter
two votes must be considered together with device-side commit gating and
identical collective order; the code audit alone does not prove they can be
combined safely.

At `dd679d5`, same-stream scalar control stores moved from host-to-device
copy plus stream drain to a fixed CUDA store kernel. The two-T4 standalone
fixture passed plain execution and all four Compute Sanitizer tools, and the
full LSA DENSE layer/archive oracle passed. The five-repeat paired S10 screen
at `d7c5984` measured HostSized/LSA search medians of 0.420060/0.367077 s
with sampled peaks of 529/567 MiB per rank. This is a scoped control-store
change, not a CPU-free pipeline; see `docs/validation/scalar-control-store-t4.md`.

The attempted nonblocking post-owner device vote at `0ceb7c7` passed the
no-fault and one-rank capacity fixtures but, after host-fault injection at
`5b4bfe8`, hung until the external 300-second timeout on two P2P T4s.
`6077bf8` restored the blocking vote. The corrected v33 gate at `db3e81c`
passed the full DENSE LSA oracle/archive fixture plus one-rank capacity and
host-owner-error tests, both terminating the two-rank group in about 1.3 s.
The host vote remains necessary for this tested protocol and exits the
current batch loop on a nonzero result. The rejected GPU-only candidate could
leave the fixed remaining parent rounds of the depth to execute; bounded
cancellation would be required before removing the blocking vote. The whole
owner-to-retirement DAG remains open.
See `docs/validation/postowner-device-vote-candidate.md`.

At `3fffcbc`, HostSized NCCL peer-count publication moved from blocking
`Buffer::put` on the producer stream to a device scalar store on the consuming
exchange stream. Two independent physical 2xT4 Kaggle gates (LSA v34 and
library-owner v60) passed full-state layer/archive fixtures, 1/2-GPU library
owner regression and tiny two-process CLI cases across DENSE/HASH_FIRST where
supported. The full CPU suite also passed. This removes one host drain, not
the received-count readback, payload-size decision or fatal votes. No new
sanitizer, speed, VRAM or overlap claim is attached to this change; see
`docs/validation/scalar-control-store-t4.md`.

The subsequent same-source two-T4 S10 v35 screen completed five paired
archive-verified runs per transport at `6dd41bb` (runtime unchanged from
`3fffcbc`). HostSized/LSA search medians were 0.435774/0.390792 s and
durable medians 4.008849/4.057025 s; sampled VRAM was 529/567 MiB per
rank. All 20 rank archive logs reported VERIFIED. The older v27 session is
not a causal baseline for the scalar-store edit. See the same validation
record for MAD, samples and limits.

The matched-source Nsight v36 diagnostic completed on two P2P T4s but covers
startup, warmup, search and archive together. HostSized/LSA traces recorded
4,090/3,046 `cudaStreamSynchronize` calls and 4,960/3,528 synchronous
`cudaMemcpy` calls; LSA is still host-dependent. These aggregate counts do
not give per-stage critical-path time. See
`docs/validation/poststore-timeline-v36.md`; search-range instrumentation
and a scoped timeline remain required.

At `c73b637`, the measured pass gained an opt-in CUDA profiler capture range.
Two-T4 Nsight v37 completed with startup/warmup/final archive drain excluded:
HostSized/LSA still show 2,002/1,462 `cudaStreamSynchronize` and 2,460/1,740
synchronous `cudaMemcpy` calls during captured search, while pinned
allocation/free calls are zero. This establishes remaining hot-path host
dependencies, not their individual critical-path cost. Stage/callsite
attribution and full DAG overlap remain unverified; details are in
`docs/validation/poststore-timeline-v36.md`.

Kaggle library-owner v61 completed the four Compute Sanitizer tools on two
physical T4s at `6dd41bb`. Default one-/two-GPU BFS suites reported zero
failures and zero sanitizer errors/hazards; tiny two-process CLI archives
verified. The default suite ignored LSA full-BFS and injected fault tests,
so a four-tool LSA pipeline gate is still open. See
`docs/validation/poststore-sanitizers-v61.md`.

Two scoped two-T4 callchain attempts (v38/v39) completed and reproduced the
1,462/1,740 LSA sync/copy counts. The v39 release-debug build and Nsight
symbol-resolution option still exported raw addresses, not source callsites;
versioned memcpy records lacked callchains. No per-stage wait has been
selected for deletion from this evidence. See
`docs/validation/sync-callsite-v38-v39.md`.

The independent scoped Nsight v41 S10 run at runtime source `b93d2a2` again
completed on P2P-capable 2×T4 with both archives verified. It observed 1,462
host stream drains and 1,740 synchronous copies during search; the summed
durations across both rank processes were 492.282 and 107.014 ms. The
`gpu_gaps` rule's 500 ms default threshold exceeds this run's 0.452 s search
window and therefore does not establish an absence of short idle gaps. This
still lacks source callsite and critical-path attribution. The ignored LSA
four-tool sanitizer gate v62 timed out under its first unfiltered memcheck
after NCCL setup reported API errors and both ranks entered depth 0; no test
result or sanitizer summary exists. The remaining tools did not run. See
`docs/validation/lsa-s10-nsight-v41.md` and
`docs/validation/lsa-sanitizer-v62.md`.

At the same runtime source, private Kaggle v42 repeated the asymmetric
archive-slot-exhaustion gate on P2P-capable 2×T4. Host-sized NCCL and
LSA/CUCO_RANK fixtures both passed, with local/remote fatal propagation and
thread exit. This protects the existing blocking archive vote; it is not
evidence that the vote can be removed. See
`docs/validation/lsa-archive-fatal-v42.md`.

An isolated two-rank LSA one-peer-exchange fixture at `48bc324` passed plain
on P2P-capable 2×T4 in Kaggle v44. Under unfiltered memcheck the test itself
passed and exited, but the sanitizer returned 26 NCCL API reports (209/800)
and nonzero status. Because the same report family coexists with successful
leaf progress, it does not alone explain the v62 full-BFS depth-0 timeout.
The full-app four-tool gate remains open; see
`docs/validation/lsa-leaf-sanitizer-v44.md`.

The next isolated leaf attempt, private Kaggle v47, passed plain,
memcheck and racecheck on P2P-capable 2×T4 with NCCL API-error reporting
disabled. Initcheck failed LSA activation on one rank and the old fixture's
barrier hid that assertion as a timeout; synccheck was not run. The fixture
now aborts on any rank-thread panic (`ae3dce9`). Private Kaggle v48
confirmed that an intentionally corrupted peer hash yields an assertion
and SIGABRT rather than a timeout on 2×T4. The independent initcheck /
synccheck v49 attempt selected a no-P2P host and stopped at preflight;
v50 used P2P-capable 2×T4. Synccheck passed with zero errors; initcheck
returned code 6 on NCCL window registration despite a zero-error sanitizer
summary. Thus the four-tool leaf gate and the full-BFS sanitizer gate remain
open. See
`docs/validation/lsa-leaf-sanitizer-v47.md`.

Private Kaggle library-owner v63 completed one S10 DENSE archive-verified
run per CUB, CUCO_RANK and CUDF_RELATIONAL backend on 2×T4. All produced
46 layers and 3,628,800 states. The single-sample search seconds were
0.871420, 0.491433 and 1.920906 respectively; sampled total VRAM was
914, 1058 and 1134 MiB. Five-repeat v64 failed on a CUDA SDK download
timeout before BFS; v65 completed 15 archive-verified S10 DENSE runs on
2×T4. Search medians for CUB/CUCO_RANK/CUDF_RELATIONAL were
0.829891/0.376393/1.745511 s and sampled total VRAM was
914/1058/1134 MiB. This is one configuration and leaves the pipeline and
cross-graph gates open. See `docs/validation/db-owner-s10-v63.md` and
`docs/validation/db-owner-s10-v65.md`.

The later one-T4 v66 and two-T4 v67 S10 diagnostic Nsight captures completed
with verified archives. They show substantial aggregate host waits/copies;
v67 also shows NCCL send/receive and all-reduce activity on the
HostSizedNccl path. Neither gives an LSA critical-path attribution or proves
the runtime CPU-free. See `docs/validation/library-owner-nsys-s10-v66.md`
and `docs/validation/library-owner-nsys-s10-v67.md`.

The subsequent two-T4 v58 S10 LSA full-BFS diagnostic completed with both
archives verified. The captured range contains 1,462 host stream waits and
1,740 synchronous CUDA copies across both ranks. Callchain symbols were not
resolved, so this is aggregate evidence of remaining host dependence, not
per-callsite critical-path proof. See `docs/validation/lsa-nsys-s10-v58.md`.

Private Kaggle LSA leaf v52 repeated the `ncclCommWindowRegister` failure
under initcheck on P2P-capable 2×T4. The standalone NCCL-only v57 fixture
then passed plain registration on both ranks but failed registration under
initcheck (rank 1 reported an unspecified CUDA launch failure; its peer
timed out). Because that fixture contains no BFS runtime code, the observed
registration failure does not require a BFS bug. Its cause within the
NCCL/sanitizer/driver stack remains undetermined; no sanitizer gate is
waived. See `docs/validation/lsa-leaf-sanitizer-v47.md` and
`docs/validation/nccl-window-isolation-v57.md`.

An independent two-T4 NCCL 2.29.7 nonblocking-abort fixture at Kaggle v59
completed: one rank issued an unpaired all-reduce, both rank threads aborted
their communicators and exited. This is not the BFS runtime and does not
replace its blocking communicator or establish socket/capacity/archive
failure handling. See `docs/validation/nccl-nonblocking-abort-v59.md`.

The private two-P2P-T4 NCCL-only v61 fixture also registered and deregistered
the LSA symmetric window on both ranks using either blocking or nonblocking
communicators. This clears a leaf compatibility question, not the production
runtime protocol (`docs/validation/nccl-window-nonblocking-v61.md`). A
launch-to-commit source audit at `cd83c90` additionally found that rank-local
archive admission can fail after bootstrap but before communicator creation,
and that local RunCommit/`COMPLETE` publication has no final rank-group
agreement. These failure windows join the owner/transport/retirement work
packet; neither has an injected two-process gate yet. See
`docs/plans/owner-transport-retirement-batch-audit.md`.

On 2026-09-26 the reference launcher admission was tightened locally: the
first rendezvous path is now shared even if ranks disagree on warmup; warmup,
stream-archive and archive-disable settings are checked inside the post-
bootstrap configuration vote; and warmup archive removal participates in the
output boundary before any group marker. CPU contract tests and Linux/CUDA
cross-target typecheck passed. The private Kaggle v79 two-T4 run at `5b04b2a`
then passed four asymmetric configuration/fatal cases plus a normal warmup
with verified measured archives. Warmup archive *removal* fault injection and
the broader owner/transport/retirement CPU round trips remain open. See
`docs/validation/warmup-admission-2xt4-v79.md`.
