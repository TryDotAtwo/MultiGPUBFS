# MultiGPUBFS completion ledger

## 2026-10-05 independent HOST count handshake v7

Runtime25bdf95981eda723550f0045660cff2b709429e1 moves the previous owner's
last-reader wait from before the independent scalar count exchange to before
payload overwrite. Dedicated scalar buffers do not alias the one payload slot.
No new payload storage, max-padding, fatal guard removal or CPU-free claim.

Physical2xT4 v7 passed54/54 selected DENSE HOST cases; raw reconciliation
covers14pages1330entries. Evidence970829cc1f0048c4abf086bb4085be36b8918571,
`test_results/kaggle-dense-packet-v7/raw-small-evidence.tar.gz`.
Verified162nestedcases,54full-state/depth oracles,48zero-error/hazard
sanitizer rank logs, no false COMPLETE or forced cleanup. Not HASH_FIRST,
LSA registration initcheck closure, large-graph or general asynchronous acceptance.

v8 paired_measure running against accepted b8bee, five ordinary repetitions,
S10 batch32768 HOST/noarchive/full process warmup. Performance unproven.
Earlier b8bee v5 ordinary medians: old0.480817267s/2817MiB per rank,
CUB0.711930507s/913MiB,CUCO0.54754036s/1361MiB; five-repeat raw409695d.
Scoped v6 timeline analysisv15 saved05019d06a5b2bfdc92b0431cd2006489a3c11362:
44healthy intra-depth windows per new rank; host wait CUB146/178ms,
CUCO125/212ms; multi-stream interval union7-15ms. Profiler timings are not
causal proof or ordinary A/B. HOST payload sizes still require CPU visibility.
Full goal remains OPEN.


## 2026-10-05 DENSE packet-prefetch gate v4 (verified)

Runtime `b8bee990f4083aad0dc8d825a49b7a43d9046748`; exact launcher
`089143e22d83ace391363dfd0143c4665af8438f`. The integrated DENSE producer
now queues generation/hash/route/pack before packet-ready, and retains the
bank until transport and owner readers complete. A bank-local preallocated
count word avoids aliasing the active archive-fatal collective control.
HOST_SIZED_NCCL still needs host-visible payload sizes: not CPU-free completion.

Physical Kaggle 2xT4 v4: 54/54 outer records passed, 162 nested cases,
54 full canonical state/depth archive oracles. No forced cleanup or false
COMPLETE in asymmetric startup, admission, constructor/late-constructor,
owner, capacity, worker-write/sync, or archive-finish cases. Maximum observed
case duration 24.074823199 seconds (individual configured deadlines retained).
All 48 instrumented rank logs (24 runs, memcheck/racecheck/initcheck/synccheck)
contain zero-error/hazard summaries. This covers the selected DENSE HOST
fixtures, not LSA registration initcheck, HASH_FIRST, large graphs or all seeds.

All 14 output pages, 1330 entries reconciled; 856 small outputs retained in
`test_results/kaggle-dense-packet-v4/raw-small-evidence.tar.gz`, evidence commit
`81e6f6a51f04594cabc18167430e0f0cf73e25bb`. Large graph archives remain cloud-only.
NCCL required-cache restore worked; no unchanged NCCL rebuild was performed.

v5 paired_measure is running: S10 batch32768, HOST_SIZED_NCCL, archive off,
full per-process untimed warmup, five unprofiled repetitions of old4ef9ce1,
before2c839a4 CUB/CUCO and afterb8bee CUB/CUCO. No new speed claim yet.
Full goal remains OPEN: no-readback timeline, remaining profiles/macro depth,
HF and separate DB/framework analysis are not closed by this fixture gate.


## 2026-10-04 remote-only T4 evidence

Kaggle matrix v106, source d3d1e2aa350c8dc3d34cad9367fc4f8e90a7ea64:
360/360 full-state two-rank U4 cases passed, moduli 2..6, both profiles,
three rank owners, pre-dedup ON/OFF, both maps and three seeds.
This is not performance, macro-depth or full sanitizer acceptance.
Root SHA256: 2f193f13c18f36b8b7e5033248b9b0abf14ef5eb478ee3251ed1c58dc2be595c.

Kaggle typed v11, source c221b93d739a7f0cafbc6cf79757c035dcd24677:
13/16 attested sanitizer runs passed. Host initcheck failed in both profiles;
CUDA12.9 initcheck passed DENSE once but failed HASH_FIRST before BFS in
NCCL activation. No suppression was used. The initcheck gate remains OPEN.
Root SHA256: 1e7a6ba1c49bf1e925a9778673f522d8b1b2c99776789f4d3ea1a6f15ebfc502.

All 3840 small log/JSON outputs, 9184157 bytes, are retained under
`test_results/kaggle_t4_20261004/` with per-file SHA256 receipt.
Receipt SHA256: f5cedaa3e62feb18f07de69bb38de0c079a78bf0ea4bb3afda55d42002d04e5b.
No graph archives were downloaded to the user computer.
Vast 54179932 replay uses exact source 52bb77fd8e4cf7ed4187ceeddce598ae5880af43,
two physical T4s, P2P enabled both directions and driver 595.71.05.
Its compilation is in progress; no new GPU acceptance result is claimed.

Status: active, updated 2026-09-28. This file separates accepted requirements from
implemented paths and measured evidence. `ARCHITECTURE_NEED.md` remains the
architecture contract; `library-first-bfs.md` is the library experiment log.
The source-level owner/transport/retirement audit and connected change set are
in `owner-transport-retirement-batch-audit.md`; it does not establish a measured
bottleneck or completed asynchronous pipeline.

## 2026-09-28 owner/transport continuation

`faead68` closes the local HF manifest/reconcile patch: new manifests retain
per-file `rows`, while retry accepts an intact legacy manifest/verification
pair without rewriting remote objects. The local Python suite passed (172
tests, 6 skipped). Kaggle HF footer audit v29 ended in a 7200-second timeout
after 5,448/6,228 objects; it did not produce a complete remote-footer proof.

`66b1822` connected a two-rank TCP search sideband to the reference runtime,
added a rank-local cancellation token, changed communicator initialization
to nonblocking NCCL and polled `ncclInProgress` on transport calls. The local
bootstrap suite passed 18/18; Windows Rust CUDA/library-owner typecheck passed
from a safe temporary copy. The private Kaggle boundary v1 compiled the real
CUDA/NCCL C++ library on two P2P-capable T4s and passed bootstrap CPU tests,
but stopped on a Rust timer-scope error before GPU BFS. That error is fixed by
`f17d286`; `7ef8b19` adds cancellable CUDA/NCCL stream polling and explicit
communicator abort on remote search failure. `2632691` extends nonblocking
progress handling to LSA activation and communicator finalization. No full
two-rank GPU result for those later commits is accepted yet.

The separate private sanitizer v1 reached two real T4s with P2P unavailable
and reported `UNSUPPORTED_HOST` before GPU checks; this is not a sanitizer
pass or BFS failure. A host-sized-only gate was added at `3fee878` so such
hosts can still validate two-rank full-state/archive semantics without an
LSA claim. Its v1 run stopped at the same pre-fix timer compile error.
Host-sized-only v2 on `7ef8b19` completed on two physical T4s: both the
Native/cuDF/cuCollections profile matrix and CUCO_RANK DENSE full-state/archive
oracle tests passed. This validates those small HostSizedNccl fixtures, not
LSA or a performance claim. Boundary v2 on `7ef8b19` compiled CUDA/NCCL and
passed CPU tests but failed the constructor-fault fixture with
`CUDA_STATUS_6` before its full BFS section. `2632691` addresses the likely
nonblocking LSA registration/finalization transition; a physical replay is
now available: private boundary v3 on `a6a2d43` completed on two P2P-capable
T4s. It passed the asymmetric post-NCCL constructor fault, HostSized and LSA
S4 group markers/archives, one-rank archive-admission and configuration
failures, small-layer capacity, and the independent full-state oracle. This
supports the corrected NCCL lifecycle for these bounded fixtures, not a
no-readback timeline or large-graph performance claim. Logs are retained in
`test_results/kaggle_nonblocking_boundary_v3/lsa-bfs-gate/`.
The private LSA leaf sanitizer v2 on the same source passed plain exchange,
memcheck and racecheck. Initcheck failed during NCCL device-communicator
registration on both ranks (`ncclUnhandledCudaError`), while reporting
`ERROR SUMMARY: 0 errors`; this known registration interaction is **not** an
initcheck pass. Synccheck was not reached by that script and remains open.
Sanitizer v3 on `a6a2d43` passed the LSA leaf `synccheck` with zero errors;
`initcheck` still failed in NCCL registration with zero reported memory errors.
The v4 Nsight diagnostic on `ef4657b` completed a verified two-T4 S10 archive
run: 3.574 s search, 16.411 s durable file commit. The earlier `a6a2d43`
diagnostic measured 5.600 s search and 15.725 s durable commit on another
two-T4 allocation. These are single profiled runs, not paired A/B evidence.
CUDA runtime timeline contains 550 stream synchronizations in both runs,
consistent with six per rank per depth (46 depths), rather than growth with
the 180 rank-batches. Runtime synchronous memcpy calls dropped from 1,740 to
1,380: exactly two calls per rank-batch. This is strong evidence that the
removed hot-path readbacks disappeared, but a call-stack/NVTX attribution of
all remaining calls and overlap proof are still required.

`ef4657b` removed the healthy DENSE+LSA+CUCO_RANK post-owner host vote and
bounded two outstanding completion events. The v5 two-T4 gate passed LSA
full-state and archive, HostSized profile checks, and the one-rank owner
capacity failure. It timed out on the injected one-rank **host owner error**.
This contradicts a complete failure-protocol claim. Root cause in the test
path: `advance()` aborts NCCL before returning, while the search sideband was
not told about the local failure until after `advance()` returned. The test
also lacked a sideband/cancellation relay. `fa5ee1f` now publishes failure
before abort, tests cancellation relay, and extends the device fatal gate to
all peer rounds. Private Kaggle v6 ended before compilation because the package
index did not supply locked `libkvikio-cu12==26.4.0`; v7 reached two physical
T4s but both P2P directions were disabled (`UNSUPPORTED_HOST`), so it never
entered LSA tests. v8 on a P2P-capable two-T4 host repeated the host-owner
fault timeout after full-state/archive and capacity-failure gates passed. Thus
pre-abort notification plus the test relay are insufficient; do not ascribe
the hang to the earlier missing signal alone. `a9cebb2` adds stage markers
and an isolated 45-second host-fault gate. Private Kaggle v9 passed on P2P
two-T4 hardware in 1.58 s, with both ranks reporting bounded abort. However
v9 enabled `MGBFS_TRACE_ROUTE`, whose CUDA synchronizations change scheduling;
it does **not** refute the untraced v8 hang. `de85ba7` removes that trace for
an otherwise identical isolated test. Kaggle rejected its first push because
the account had reached the two batch-GPU-session limit, so no v10 result
exists yet. Until an untraced fault gate passes, the fix is a candidate, not
accepted correctness evidence.

The latest user instruction allows **one** active Kaggle notebook at a time.
Safe LSA receive-slot reuse beyond the existing single payload slot,
HASH_FIRST integration, full-app sanitizer gates, production Nsight call
attribution and paired A/B remain open. Do not call this an asynchronous
end-to-end BFS yet.

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
passed. Private Kaggle v78 passed its actual two-P2P-T4 configuration-fault
and normal S4 BFS gates, including both transports and verified archives;
see `docs/validation/lsa-full-bfs-gate-v78.md`. The macro launch
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
Python test suite passed (171 tests, 6 skipped). A subsequent local change
reads each newly encoded Parquet footer before upload and checks its row count
against the source Arrow table; the existing pinned remote LFS SHA/size check
then binds publication to those exact bytes. Its 172-test Python suite passes
(6 skipped). An independent remote footer read, live publication with this
version, and legacy staged manifests without `rows` remain separate gates.

At `9f53edc`, an independent remote Parquet-footer auditor and local tests
passed (176 Python tests, 6 skipped). Its anonymous S13 scan verified only
5,601/6,228 footers before HTTP 429. The private Kaggle v28 token-backed run
failed in Kaggle's Secrets service at `get_secret("HF_TOKEN")` with HTTP 400,
before the auditor started; see `docs/validation/s13-hf-footer-kaggle-v28.md`.
Remote footer completeness remains open. The current user instruction permits
only one Kaggle notebook at a time.

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

Private Kaggle v81 repeated the S10 LSA callsite diagnostic on a P2P-capable
two-T4 host at `8f6bcc2`. Both archives verified and the global layer counts
sum to 3,628,800. The captured range again has 1,462 stream synchronizations
and 1,740 synchronous copies; the versioned API rows still lack resolved
symbols. The diagnostic neither locates the critical-path waits nor closes
the owner→transport→retirement CPU-independence gate. See
`docs/validation/lsa-callsite-v81.md`.

The rank-result aggregator now rejects mixed or partially present hash seed,
bootstrap digest, and owner-rank map, and the distributed reference result
publishes its owner-rank map. A RED fixture previously accepted all six
mixed/missing cases; after the change the Python suite passed (173 tests,
6 skipped), the host Rust suite passed, and Linux/CUDA+library-owner
cross-target `cargo check` passed without linking. A subsequent real
two-rank result must confirm the new serialized field; this is a
result-integrity fix, not a GPU hot-path change.

Private Kaggle v82 exercised that new rank-map field in a full two-T4 S10
run at `868c118`: both ranks published `[0, 1]`, both archives verified,
and the layer counts remained exact. An explicit Nsight debug-symbol path
did not resolve the CUDA callchains; the 1,462 stream waits and 1,740
synchronous copies remain aggregate evidence. See
`docs/validation/lsa-callsite-v82.md`. The connected CPU-free path is still
unimplemented.

Private Kaggle v83 captured process mappings for both rank binaries while
repeating the same full S10 diagnostic. The 1,462 stream waits now group
into 14 executable-relative address patterns on each rank, but exact
function/line attribution remains open because the same-build ELF was not
exported. See `docs/validation/lsa-rank-maps-v83.md`. This profiling does
not change the production pipeline or close the owner-slot/failure gate.

Private Kaggle v84 resolved the address-only callchains against the exact
rank ELF before cleanup. `all_max_ring_or_host_fatal()` is the dominant
observed host wait: 90 calls/rank and 165.97/192.97 ms CUDA API duration
on ranks 0/1 in one profiled S10 run. See
`docs/validation/lsa-resolved-callsites-v84.md`. This makes the connected
owner-slot/failure replacement the first measured target, without implying
that its aggregate API duration is directly recoverable search time.

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

The LSA receive-slot `owner_consumed` event was added and exercised on
physical 2×T4 in private Kaggle v85/v86. The v85 S10 run completed 46 layers,
3,628,800 states and verified both archives. The v86 two-GPU integration
fixtures passed full-state/archives and asymmetric owner capacity/host-error
propagation. These are one-process GPU-worker fixtures, not independent
process cancellation tests. Source-resolved v85 profiling still found 90
post-owner `all_max_ring_or_host_fatal` stream waits per rank; the event
protects receive-slot lifetime but the connected CPU-free pipeline is **not**
complete. `all_max()` also synchronizes a four-byte host upload before its
NCCL result readback; replacing only that upload cannot remove the host
dependency. The next change must address the post-owner result/fatal protocol
and archive error path together, preserving NCCL issue order and bounded
asymmetric failure termination. See
`docs/validation/lsa-owner-consumed-v85-v86.md`. Kaggle validation now uses
only one 2×T4 notebook at a time.

On 2026-09-28, private Kaggle `mgbfs-nonblocking-boundary-t4` v12
ran source `a85bc59` on a P2P-capable physical 2×T4 host. The isolated
LSA peer-payload fixture passed. The asymmetric host-owner fault fixture
terminated both rank threads in 1.65 s, removing the earlier observed
`ncclCommAbort` hang, but failed its expected-error assertion: both ranks
reported `LIBRARY_RANK_DEPTH_FATAL_22_22` at depth 0 instead of reaching
the injected owner error. The cause was reuse of `collective_recv` after the
host-sized depth-round-count collective; its nonzero count was interpreted
as the initial LSA group-fatal predicate. Commit `bcad079` clears that word
on the compute stream before the first packed/exchange event of each device
epoch, adds an archive-failure device vote, and exposes a debug-only per-rank
host-owner fault switch for independent process testing. Local CUDA-feature
Rust typecheck passed; a new 2×T4 run and both process-level/error and
full-state gates remain required. The `cefb42c` Kaggle script can run the
thread and two-process fault fixtures sequentially within one notebook;
its push is currently rejected by Kaggle's account GPU-session quota, so
that fixture has **not** run. Raw v12 logs are in
`test_results/kaggle_nonblocking_boundary_v12/lsa-bfs-gate/`.

Private Kaggle v13 at `cefb42c` confirmed the depth-start reset on two
P2P-capable T4s: the LSA peer-payload fixture and the one-rank host-owner
failure test both passed without a hang. The new independent-process fault
fixture then reported a false `COMPLETE`, but its input used S4 with
`batch=7`; the debug injection intentionally requires more than one
scheduled batch and was never reached. This was a test-configuration miss,
not evidence that the runtime ignored an injected failure. Commit `ae99680`
sets `batch=1` for that fixture. The corrected two-process test has not yet
run, and the v13 result does not establish full-state correctness for this
source. Raw evidence is under
`test_results/kaggle_nonblocking_boundary_v13/lsa-bfs-gate/`.

Private Kaggle v14 at `e2a702d` completed on two physical P2P-capable T4s.
Three independent two-process torchrun fixtures injected an error on rank 0:
owner processing during search, archive admission before NCCL communicator
creation, and an archive-finish result after search. In each case both rank
processes exited within the 60 s external timeout, the originating and peer
errors appeared in the logs, and no `group-complete.json` was published.
These are debug-only fault injections; the admission/finish cases prove the
group-boundary protocol but are not physical pinned-allocation or disk-failure
tests. The v14 notebook did not run a healthy full-state oracle, sanitizer,
or performance measurement. See
`test_results/kaggle_nonblocking_boundary_v14/lsa-bfs-gate/`.

Private Kaggle v15 at `c093d35` completed a healthy two-rank S4 reference
BFS on two physical P2P-capable T4s. Independent rank processes produced
the exact global layer counts `[1, 3, 5, 6, 5, 3, 1]` under both
`HOST_SIZED_NCCL` and `NCCL_LSA`; both rank archives passed committed
checksum/count verification. A separate full-state CUCO_RANK/LSA GPU
oracle fixture passed. The notebook also passed one-rank archive-admission
and asymmetric invalid-config group-fatal cases, plus the small-layer-
capacity archive check. This establishes the healthy S4 output contract
for this source, not larger-graph correctness, all failure paths, or
CPU-free overlap. Raw evidence is in
`test_results/kaggle_nonblocking_boundary_v15/lsa-bfs-gate/`.

Private Kaggle v16 at `e354997` passed the two-T4 one-rank CUCO_RANK
owner-capacity fatal fixture. The next fixture, real archive pinned-slot
exhaustion before LSA exchange, failed: the faulting rank returned
`ARCHIVE_PIN_RING_FATAL`, but its peer waited for an outstanding epoch and
returned `EPOCH_CREDIT_TIMEOUT` after 120 s rather than a prompt remote
fatal. This particular low-level two-thread fixture had not attached the
search cancellation sideband; the production two-process launcher does.
The test is being changed to exercise that required sideband explicitly.
The v16 result does **not** prove archive-slot failure propagation, and the
retirement FIFO test was not reached. Logs are under
`test_results/kaggle_nonblocking_boundary_v16/lsa-bfs-gate/`.

Private Kaggle v17 at `62316e3` completed on two physical P2P-capable T4s.
The CUCO_RANK/LSA owner-capacity, real pinned-archive-slot exhaustion,
and device retirement-FIFO fatal fixtures all passed. The archive fixture
now attaches the same cancellation sideband used by the production launcher;
it finished in 46.21 s rather than the earlier 120 s epoch timeout. This
proves bounded handling for those fixture cases, not all possible CUDA/NCCL
host errors or full application recovery. Raw logs are in
`test_results/kaggle_nonblocking_boundary_v17/lsa-bfs-gate/`.

Private Kaggle v18 at `e46fb3f` attempted all four Compute Sanitizer
tools on the complete S4 CUCO_RANK/LSA BFS fixture on physical 2×T4.
`memcheck`, `racecheck`, and `synccheck` each hit the harness's 600 s
timeout; that harness captured output only after process exit, so their
partial logs were lost. `initcheck` returned 6 before the oracle ran:
NCCL reported an asynchronous unhandled CUDA error during construction,
and the fixture panicked on `CUDA_STATUS_2`; its printed `ERROR SUMMARY:
0 errors` is not a pass. No full-BFS sanitizer gate is closed by v18.
The next harness revision writes logs during execution and terminates the
whole launched process group on timeout, so failure stages can be examined
without orphan contamination. Raw v18 report and initcheck log are in
`test_results/kaggle_nonblocking_boundary_v18/lsa-bfs-gate/`.

## 2026-10-04 Vast 54182144: full BFS sanitizer gate remains incomplete

Runtime source 52bb77fd8e4cf7ed4187ceeddce598ae5880af43, two independent T4 processes. 16 completed full typed BFS replays: 12 PASS (both DENSE/HASH_FIRST under memcheck/racecheck/synccheck), 4 FAIL (all initcheck selections). Host and pinned executable were both actual 2025.2.1.0, not an independent-version comparison. Failure precedes BFS at NCCL activation with CUDA 719; zero reported sanitizer errors does not establish a pass.

Independent existing NCCL probe, same resources and library: all three plain modes pass, all three initcheck modes fail. The same three failures recur with official CUDA 13.2.23 sanitizer component (actual instrumenter 2026.1.0.0). No filters, suppressions or resource-disabling workaround. A diagnostic-only separately linked NCCL build adds host checkpoints to locate the first failure; it is not production runtime or acceptance evidence. Original runtime library remains unchanged.

Raw logs, JSON and per-file SHA256 receipt: test_results/vast_54182144_nccl_activation/. No graph archives or credentials retained here. Four-tool acceptance, paired performance A/B, multi-GPU macro depth and remaining HF/DB stages remain open.

## 2026-10-04: local-first NCCL mapping candidate fixes the observed initcheck failure

Diagnosis narrowed CUDA 719 to rank-1 internal resource-window memset. Driver-API memset still fails; mapping local memory before importing peer handles passes. The patch preserves the rank-indexed VA layout and maps every rank exactly once. No resources disabled, filters, extra production host drains or payload padding. This establishes mapping-order sensitivity, not the vendor-internal cause.

Implemented versioned patch plus digest-checked integration in the existing Kaggle build, not a parallel runtime. Three remote mapping-contract tests and seven existing typed/sanitizer-audit tests pass. Production candidate full typed S4: eight healthy two-rank runs pass (DENSE/HASH_FIRST x four sanitizer tools), full archive/state/depth oracle. Two fault suites: 38/38 including healthy, no false COMPLETE, no forced cleanup, max whole-case 2.511363s. Logs/JSON/checksums: test_results/vast_54182144_local_first/. Initial missing-pyarrow verifier runs remain failed and excluded.

Exact clean pinned patch rebuilt, SHA256 95a042db1698504182c0cf0ee34c8c74dbaf5cb376d1aeeae93d464b0378b5b1, library SHA256 340237ca761410da432e8d36fe01b2824eb86c55cab1c1670129d7d69f76c00c. Repeat full production gates are live, including original 2025.2.1.0 initcheck and all four 2026.1.0.0 tools. Do not promote the pending repeated set to PASS until its retained summaries verify that. Full timeline and A/B remain open, as do macro-depth and the remaining HF/DB requirements. Baseline checkout remains immutable f0f2b8e5ee61173039ab9742f3a7756c9b6365e6; separate remote-only benchmark environment is being prepared.

### Exact pinned repeat gate completed

Exact pinned library 340237ca761410da432e8d36fe01b2824eb86c55cab1c1670129d7d69f76c00c: all 12 panels / 48 cases PASS. Both DENSE and HASH_FIRST pass old 2025.2.1.0 initcheck and all four 2026.1.0.0 sanitizer tools with full-state archive oracle. Two fault suites include 36 asymmetric injected failures: false COMPLETE=0, forced cleanup=0, max failure duration 2.510993s. Evidence: test_results/vast_54182144_local_first/pinned-local-first-gate/. This closes the observed NCCL initcheck activation blocker on this hardware/fixture, not all-owner/macro/performance admission. Full S8 timeline is now running on the same instance; analysis pending.


## 2026-10-04 retained S8 publication gate

- Kaggle publisher v29 completed without GPU allocation. The retained release archive SHA256 matched `589b137c665692bd43c42fd7634b90c6151003c1d17bd3816fb4e3beeb68a9ac`.
- DENSE and HASH_FIRST: each full canonical layer set matched independent CPU S8 oracle, 40,320 states and 29 layers; checksummed rank archives and group RunCommit verified. Existing Parquet verifier reported PASS including all state/hash pairs. Full configurations were retained; earlier suspected missing-config diagnosis was incorrect. Publisher seed parsing was corrected from byte reversal to numeric hex, matching runtime `{:032x}` of the little-endian seed value.
- Parquet packages remain in Kaggle v29 outputs. HF publication is NOT completed: Kaggle Secrets endpoint returned HTTP 400 for HF_TOKEN; no HF upload performed. This is not a demonstrated invalid HF token.
- GPU A/B attempt v107 did NOT run search: both physical T4 P2P queries returned allowed=0. Harness preflight fix independently RED/GREEN with 16/16 Python screen tests; published in 1c08dc8. This is not the full project test suite.
- Vast 54182144 no longer exists; S8 raw trace/archives retained on GitHub release, but final tuned S10 measurements/archive were not published before lease deletion. Do not claim retained durable evidence for that tuned run. No new paid rental admitted until remaining total USD20 budget is bounded.
- Overall objective remains OPEN: tuned paired A/B recovery, broad backend/profile gates, macro-depth integration, complete HF catalog and DB/framework stage remain required.


## 2026-10-05: RTX3060 owner-job regression follow-up (not full acceptance)

- Remote instance 54278405, machine 29756, two RTX3060/sm86, driver 580.126.20. P2P denied both directions; HOST_SIZED_NCCL only, no LSA/T4 acceptance.
- Runtime old 4ef9ce1 versus new 8a43bc8, S10 LRX generators, batch262144, archives disabled. Full untimed BFS inside every rank process before timed pass, five repeats each. Raw warmup/measured rank records reconciled, all46 historical layers match, zero archive pinned/disk allocations. This is not an independent S10 full-state oracle.
- Median/MAD seconds: old .322034983/.001556670; CUB .629792587/.002134299; CUCO .506971360/.003630180. Peak MiB/rank3457,1959,2413 respectively. Regression persists after in-process warmup.
- Three-repeat CUB tuning, fixed256 buckets/rank and four shards: job4 .622921210; job16 .468594302; job64 .456448759 median seconds. Peak MiB1959,1965,1989. Thus job granularity accounts for part, not all, of the regression.
- Production preparation candidate f29a72a chooses min(buckets_per_shard,candidate_capacity), replacing arbitrary min4. Native memory-query small-batch RED→GREEN and bounded dense-directory split test; all7 production preparation tests pass. e80b1c1 without candidate clamp is superseded and is not deployable on all small batches.
- Hardware full-state reference path: two independent ranks, S6, batch32, DENSE+CUB+HOST_SIZED_NCCL, shards4, job4/64, preOFF/ON: all four full archives match independent CPU state/depth oracle. Binary data path remains8a43bc8 with explicit job config; this does not prove the new production CLI path on multi-rank LSA.
- Default-member Rust test run is NOT GREEN: distributed_archive has10passes and4failures: bmma_owner_preserves_dense_and_hash_first_archived_layers; dense_lookahead_preserves_archived_layers_with_one_parent_batches (BMMA plan status3); single_rank_reference_profiles_preserve_full_archive_layers; tensor_hash_first_preserves_archived_layers_with_both_owner_backends (HASH_FIRST_TC_DEVICE_UNSUPPORTED). Both implementations explicitly admitSM75 only. Remaining tests after failing integration executable were not completed. Workspace-wide run also blocked by unbuilt legacy multigpubfs_cuda prototype library.
- Nsight captures exported, but process-scope wrapper reports forced cleanup of profiler descendants. Preserve failure reports; do not call instrumentation gate PASS. Raw full-session kernel counts include both ranks and warmup; measured-phase timeline reconciliation remains open.
- Release https://github.com/TryDotAtwo/MultiGPUBFS/releases/tag/validation-vast-54278405-20261005 retains raw measurements, warmups, configs, launchers, traces and test logs. Followup archive SHA256 a7dc377e61194b416e3dd09d4d294d12085c4acbdff92e89963ce21cdf21f669,31591245bytes.
- OPEN: complete CPU-free owner→transport→retirement matrix, real2xT4 faults/four-sanitizer gates and timelines after candidate change, paired tuned comparison, macro-depth production integration, archive/HF publication and DB/framework analysis. No waits/fatal votes removed by this owner-size change; no complete goal claim.


### Same-day production CUB follow-up and rental closeout

- Built actual mgbfs CLI debug binary with the f29a72a source delta. RunConfigV1 S4, one RTX3060, DENSE+CUB+LSA, four shards64 buckets each, parent_batch1 (candidate capacity3), three route banks/credits. Both plain and full owner-DAG capture runs complete; all24 canonical states/depths match independent CPU oracle, seven route-bank reuses. Capture log contains24 `MGBFS_OWNER_DAG_CAPTURE launched` records. This verifies the candidate clamp and actual single-rank dispatch, not two-rank transport correctness or a timeline absence claim.
- Same complete production run passed unfiltered memcheck/racecheck/initcheck/synccheck. Compute Sanitizer2025.2.1.0/build35969825; all API checks enabled, zero errors/hazards and archived full-state oracle matches. Scope one RTX3060 only; multi-rank NCCL registration/initcheck and actual2xT4 gates remain open.
- Tuned CUB Nsight full-session kernel count22328 versus original CUB201568 (both ranks, warmup+measure+setup/teardown). Profiler process-scope wrapper still reports forced cleanup; do not promote these counts into completed timeline/overlap admission.
- Final evidence asset `vast-54278405-production-cub-gates.tar.gz`,38594313bytes,SHA2568ea690012bde8446f29d6e52ce3f04c58ebca846193a3ca786137b35d4a2ea87 on the existing validation release. All server logs/configs/scripts/traces and oracle evidence saved before deletion.
- Owned instance54278405 identity rechecked against machine29756 and exact label before DELETE; API returned success and subsequent list confirmed absence. Other rentals untouched. Specific rental heartbeat paused after completed cleanup; persistent BFS goal remains active, no new rental or extension.


## 2026-10-05 Kaggle v113 production owner capture reconciliation

Runtime/source: `2fda8d25e84a39ebc3c1420a35e9eb2dc7be4e1f`.
Launcher: `d5b93fcfb5c555563fd3bdad709b6a3350a4fda0`,
`scripts/validation/sm75-production-owner-job-gate.py`.
Evidence commit: `6899c74b31ed95d89685a24aa34f8d5423679fff`,
`test_results/kaggle-v113-production-owner/reconciliation.json`.

- Kaggle worker COMPLETE; both output inventory pages collected (185 entries).
- 24/24 actual RunConfigV1 CLI cases pass: batch 1/32 × CUB_SORT_MERGE,
  BMMA_BUCKET, CUCO_RANK × DENSE/HASH_FIRST × pre-dedup OFF/ON.
- Independent CPU oracle checks every canonical S4 state at every depth;
  all cases contain 24 states with layers `[1,3,5,6,5,3,1]`.
- Every raw rank log contains actual `MGBFS_OWNER_DAG_CAPTURE launched`
  markers (24 launches for batch1; 7 for batch32), not merely the env flag.
- 162 small raw/config/log/reconciliation files preserved in GitHub.
  Preallocated state archives were not downloaded to the user's computer.
- Scope is ONE independent rank on physical SM75 T4. This is NOT proof of
  multi-rank transport, fault propagation, sanitizers, overlap or performance.
- Explicit output request version_label=113 returned HTTP404; latest completed
  output was reconciled against the pinned runtime and actual case provenance.

Next live test is Kaggle v114, launcher `d7f7d55c445eebfb6e8c3bf70de3152d7c305557`,
existing `typed_stress_gate`: two independent physical T4 rank processes,
full-state U4(F3), three owners, both profiles/pre-dedup/maps and route banks
2/3/4 with required within-depth bank reuse. RUNNING is not a pass.
Prepared (not launched) next gate is
`02e516830f3a758792617011b7667550c8c89bf7`:
`scripts/validation/sm75-production-all-owner-fault-sanitizer-gate.py`.
It retains existing asymmetric startup/constructor/owner/archive fault replay,
actual capacity faults, and four unfiltered process instrumenters for all owners.

Full goal remains open, including new-candidate two-rank correctness/fault/
sanitizer/timeline acceptance, paired A/B, macro depth, HF and DB/framework stage.


## 2026-10-05 Kaggle v114 production two-rank full-state stress

Evidence: `616eac1b395d8476736c339e38e643d7c7295b5b`,
`test_results/kaggle-v114-production-two-rank/reconciliation.json` and
`raw-small-evidence.tar.gz` (148755 bytes; bundle SHA256 in reconciliation).
Runtime `2fda8d25e84a39ebc3c1420a35e9eb2dc7be4e1f`; launcher
`d7f7d55c445eebfb6e8c3bf70de3152d7c305557`.

- Worker COMPLETE. All eleven inventory pages collected: 1053 entries.
  838 small raw evidence files preserved; state archives not downloaded to PC.
- 72/72 production RunConfigV1 cases pass on two physical Tesla T4 GPUs,
  with distinct independent rank processes and verified P2P both directions.
- Matrix: three owners CUB_SORT_MERGE/BMMA_BUCKET/CUCO_RANK × two profiles
  DENSE/HASH_FIRST × pre-dedup OFF/ON × rank maps01/10 × route banks2/3/4.
  Epoch window3 and parent batch8; three source banks are NOT completion credits.
- Independent full-state U4(F3) oracle: all729 canonical states at every depth;
  layers `[1,6,20,56,116,208,268,52,2]` in every case.
- Raw144 rank records reconciled with immutable requested config digest,
  owner/profile/pre/map/banks/archive enable and summed per-rank layer counts.
  Runtime source patches empty, both ranks exit0, group-complete true,
  no forced cleanup; every case observes within-depth physical bank reuse.
- This closes this finite two-rank correctness/buffer-reuse matrix only.
  It does NOT establish all graph families/seeds, fault behavior, clean sanitizer,
  CPU-free healthy timeline, overlap or end-to-end speed.

Current v115 tests existing production replay with all owners: asymmetric
startup/constructor/owner/archive admission-worker-finish/capacity errors and
four unfiltered process instrumenters. It is RUNNING, not admitted.
Prepared real full BFS timeline launcher `1d2a2d0be513e8850e091aa25eb3f26423e17ad5`.
Prepared paired warm A/B launcher `019861b343d5a9b5ed591191d4e7bf6a06eb743e`.
Important performance scope: CUB HOST_SIZED_NCCL does not activate native_rank;
new LSA CUB and CUCO must be measured explicitly, beside host-sized control.
The prepared A/B uses S10/batch262144, no archive allocation, full in-process
BFS warmup, five alternating repetitions; it is NOT yet run or performance proof.


## 2026-10-05: v115 reconciled; v116 timeline queued

- Runtime pin: `2fda8d25e84a39ebc3c1420a35e9eb2dc7be4e1f`. Two distinct physical T4 GPUs, independent rank processes.
- v115 terminal COMPLETE; reconciled all 14 output pages (1329 entries). 54 healthy full-state oracle checks; 108 asymmetric injected faults, both ranks nonzero within configured deadlines, no forced cleanup and no group COMPLETE reported.
- 24 sanitizer groups / 48 raw rank logs clean: memcheck, racecheck, initcheck, synccheck; three owners and two profiles. Compute Sanitizer 2025.1.0.0 and pinned patched NCCL; evidence does not establish universal absence of defects.
- 856 small raw evidence files retained in GitHub commit `281921924135a956cf7e29a0115e187019f1d562`, under `test_results/kaggle-v115-production-fault-sanitizer/`; large archives not downloaded.
- v116 accepted by Kaggle, QUEUED: all-owner full-BFS timeline launcher `0696d1e153e972c55b890081c74da8a878532f12`. No timeline admission or performance improvement claimed yet.
- Pending: actual healthy batch/interbatch dependency and overlap analysis, paired warm A/B, macro-depth production integration, broader matrix/seed coverage, HF publication and separate DB/framework stage. Full goal remains ACTIVE.


### 2026-10-05: ordinary NCCL production owner candidate

- Runtime aa0d5ad enables DENSE native rank owner with HostSizedNccl; receive planes remain in the transport-neutral allocation plan and are removed explicitly only for LSA.
- Runtime cb94f45 adds explicit RunConfigV1.transport_backend; absent/default NCCL_LSA is omitted in canonical serialization. Production transport selection no longer needs an environment override.
- Kaggle CPU v3 demonstrated missing receive-plane assertion; v5 demonstrated rejected transport field and revealed an old LSA-specific allocation test using the generic planner. Raw logs retained.
- Kaggle CPU v6: cargo test --workspace --exclude multigpubfs-gpu returned 0, including transport serialization/digest and allocation tests. The excluded root requires an external CUDA library; no CUDA admission follows from this CPU run.
- Ordinary NCCL still has host size handshakes and post-owner votes. No no-readback/performance claim; real two-T4 production correctness/fault/sanitizer and later timeline/A-B gates remain open.


## 2026-10-05: ordinary NCCL DENSE v121

Runtime f8b92f8005cdbff176f579bb475af46f7e2d1269; launcher 7edafee690294f9123d666e2ff7884cf5a105e16. Real two-T4 v121 finished TYPED_GATE_PASS, 27/27 replays with actual HOST_SIZED_NCCL verified. All 24 rank sanitizer logs clean (racecheck uses RACECHECK SUMMARY); six requested capacity injections armed across three owners and both ranks, both processes terminated, no false COMPLETE. Raw all-page output evidence is in test_results/kaggle-v121-host-sized-production, originally preserved by 8a3e5be3821363c63394be9ee45540c76a7addcf. This accepts bounded DENSE correctness/fault/sanitizer fixtures only, NOT no-readback, HASH_FIRST, macro-depth or performance.

Next v122 is a deliberately RED production admission test, runtime a1b193f3fbb59a190b337b78e3b596365f48c149, launcher 188150ae485cae45d3d156b732a7f0909b8c13e0. Its opt-in debug assertion requires the next DENSE generation entry in the active bank before host sizing when unadmitted batches remain. No production scheduling change yet; awaiting actual RED before moving generation-input reuse ahead of sizing. Payload/receive lifetime and finalization remain separate requirements.


## 2026-10-05: actual RED and DENSE scheduling change

v122 completed: 27 actual rank logs contain DENSE_PREFETCH_BEFORE_SIZING_MISSING, not an import/build failure. All-page small evidence (261 files) preserved in test_results/kaggle-v122-dense-prefetch-red by runtime commit 61ed2fa9484f00d98fdd01f940277ee6ed12f37f. That commit moves DENSE generation-input reuse/admission after GPU route/pack and before route-count and peer-size readback. Existing event-generation barrier is reused; generation writes children/child_hashes only, not retained packed payload. HASH_FIRST retains its late origin/materializer retirement. No extra device allocation or maximum padded exchange is added.

v123 launcher855534fc4c06b7b5c016b46cb19dc9779ae755ba tests source61ed2fa with the same opt-in admission assertion plus ordinary HOST_SIZED_NCCL full-state/fault/four-sanitizer gates. RUNNING at observation; not accepted yet. Remaining HOST sizing and post-owner host votes remain explicit, so this is NOT the full no-readback pipeline or a throughput claim. After correctness, real BFS Nsight and paired A/B are still required, alongside the full profile/macro/HF/DB scope.


## 2026-10-05: v123 producer-admission gate and v124 followup

- Actual v123 API terminal COMPLETE; summary TYPED_GATE_PASS, 27/27 replay rows.
- Runtime 61ed2fa9484f00d98fdd01f940277ee6ed12f37f; launcher 855534fc4c06b7b5c016b46cb19dc9779ae755ba. Explicit DENSE/HOST_SIZED_NCCL on real two-rank T4.
- All seven output pages retained: 687 entries, 450 small raw files, evidence commit 890301cb5535aefbbc9d8da8292d7dc98d728aa7. 24 sanitizer rank logs clean (racecheck uses RACECHECK SUMMARY); 42 logs carry actual producer-before-sizing admission markers.
- Acceptance scope: early DENSE generation scheduling plus existing correctness/fault/sanitizer replay matrix. NOT elimination of HOST count readback, NOT full no-readback owner/transport/retirement acceptance, NOT performance evidence.
- v124 launched alone, runtime 5cf932475adbe936364c328e5167cd417263da51, launcher ba08309b8bc8a00d41b256a190c5a70fd3c3bfde: bounded actual HOST device-epoch RED then separate ordinary-NCCL full BFS timelines. Currently live; no acceptance yet. LSA is unavailable on Kaggle, no fallback or padded maximum payload introduced.
- Test-only commit baa0173f1d811cd2649bebb1cf89fe167aaafaad adds ordinary-NCCL owner-only fatal vote/admission test. Prepared execution launcher 5ddef63187bf0955b62d983a4df7e19d56ba9442, not launched/executed. Production post-owner waits remain until common cancellation/fatal and receive-reader protocol is implemented and tested.


## 2026-10-05: HOST device epoch v127 and retained timeline

- Runtime `2c839a414923c419e2b5b3efc2b343918cdd19b6`; Kaggle v127 COMPLETE, typed summary TYPED_GATE_PASS, 30/30 rows on two physical T4s through HOST_SIZED_NCCL.
- Actual owner-only fatal unit executed: 1 passed, 0 failed. v126 was a Rust compile/import failure, not a runtime test failure; fixed by qualified typed ABI import.
- 24 raw rank sanitizer logs have clean summaries across memcheck/racecheck/initcheck/synccheck for CUCO_RANK, CUB_SORT_MERGE, BMMA_BUCKET. These fixture gates do not establish universal correctness or all profiles.
- Raw evidence: `55d9d1d8ca03dad6c67704e3e3601060f5cc5ef1`, all 8 pages/751 entries reconciled, 493 small raw files retained. Large state archives and Nsight SQLite/rep remain on Kaggle.
- Existing cloud extractor v9 analyzed six rank traces; evidence `e09126cc8671f9a4db868af22200437c7cc49095`. Each rank has 27 healthy batch-start windows, 81 nonarchive synchronous cudaMemcpy records with D2H copy kind, and no explicit Synchronize API in those windows. This interval analysis is NOT callsite/critical-path proof and NOT full no-readback admission.
- OPEN: exact variable HOST NCCL payload sizing still depends on CPU readback (three copies per observed healthy batch window). Do not hide it with max padding or CPU mapped reads.
- A/B v128 launched, source2c839a4 versus old4ef9ce1, same S10/batch262144, archive disabled, process-internal full BFS warmup, five repetitions. Launcher `da83a32b75d0d91b06899ee5067c9af5d98786e7`. No performance result yet.
- Full objective remains active: no-readback transport, HASH_FIRST/other profiles, macro depth, broader acceptance, HF and DB/framework work remain unclosed.


## 2026-10-05: retained paired S10 trace and transport feasibility boundary

- v129 API COMPLETE, PROFILE_PASS: old 4ef9ce1, native CUB and CUCO
  2c839a4 each completed S10 with equal 46-layer count vectors (3,628,800
  states) and two exported process traces. Archive-off contract checked.
  All two output pages reconciled: 120 entries, 109 retained small artifacts
  including the exact launcher, commit f9c00d3. Large trace/SQLite remain cloud.
- CPU-only extraction v10 ended ERROR with no API diagnostics or files;
  observation retained d37b34f. Observable retry v11 COMPLETE; all eight
  outputs and aggregate retained fa6942f. These are full-process inventories,
  including setup/warmup, NOT critical-path attribution or performance samples.
  New batch NVTX was not enabled in v129. No absence-of-readback claim.
- v130 is an explicitly separate diagnostic, launched from bc369ea with
  CUDA profiler start/stop around measured BFS and MGBFS_TRACE_RANGES enabled
  for new runtimes. Its result is pending. 84f16ab prepares per-device overlap
  and OSRT wait extraction; preparation is not runtime implementation.
- HOST_SIZED_NCCL still takes exact payload sizes on the host. NCCL 2.29.7
  ncclSend/ncclRecv take scalar size_t counts and require matching peer counts:
  https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2297/user-guide/docs/api/p2p.html
  This API does not accept a device count pointer. Async polling does not
  remove that dependency. Host RMA also has a scalar count.
- Device LSA requires CUDA P2P, unavailable on the selected Kaggle allocation
  per user instruction; do not retry LSA there. GIN is a different backend,
  requires suitable NIC/RDMA capabilities, and has not been established here:
  https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2297/user-guide/docs/usage/deviceapi.html
  No hardware-independent no-readback claim follows from ordinary NCCL tests.
- Sleeps of 1 ms in wait_nccl_stream/wait_epoch_credit remain a source-level
  latency hypothesis, not yet a measured explanation of the A/B regression.
  Do not remove cancellation or retirement guards to improve a timing.

The full objective remains OPEN, including HASH_FIRST/macro-depth/archive/HF
and the DB/framework stage. No maximum-buffer payload padding, mapped CPU count
read, or changed transport may be silently substituted for the required gate.

## 2026-10-05: workflow repair and measured-window scope correction

- v128 independent paired noarchive S10/2xT4/HOST_SIZED_NCCL, batch262144,
  five repetitions/full in-process warmup: old median/MAD 0.406016328/0.007652122 s,
  CUB 0.586019162/0.005233789 s, CUCO 0.549852564/0.016125710 s.
  Peak rank VRAM 3431/1933/2387 MiB respectively. Raw f8ff7ea.
  New runtimes are slower and use less VRAM; cause and remedy not established.
- v130 COMPLETE PROFILE_PASS; all 120 entries reconciled, 109 small artifacts
  retained d991129. Scoped CPU extraction v12/v13 all eight outputs retained
  a141d2e/7104f48. NVTX ranges are valid, not discarded or unterminated.
  Each of the 46 depths has one batch at this capacity. All 45 inter-batch
  windows cross FinalizeDepth: this workload does not test intra-depth admission.
  Profiled batch-thread wait sums 209-315 ms are diagnostic only, not recovery
  estimates or an explanation of the unprofiled regression.
- NCCL bundle helper a71fc9b: six cloud CI tests pass after observed RED.
  Validates pinned archive checksum, recipe, member digests and safe paths.
  Launcher e9508fe supports explicit build_export or strict required restore,
  no cache-miss rebuild. Real export/reuse acceptance is still pending.
- CI path filters exclude evidence/docs-only pushes from unrelated failing
  native formatting checks; code contracts remain enabled. Existing formatting
  failure is not fixed or reclassified as pass.
- v131 RUNNING at this update: launcher913c17e, unchanged runtime2c839a4,
  S10 batch32768, multiple batches/depth, measured-search trace and build_export.
  No second notebook. Extractor b7b7053 requires healthy intra-depth windows
  and aggregates same-thread waits/API durations/per-device overlap.
- OPEN: connected runtime transport dependency and all prior full-goal gates.
  Cache/launcher/extractor changes are not runtime completion or speed fixes.


## 2026-10-05: packet prefetch acceptance and rejected handshake experiment

- v8 paired no-archive S10/2xT4, five repeats: b8 CUB median/MAD 0.712803668/0.011045453 s versus handshake candidate 0.720615709/0.000969802 s; b8 CUCO 0.562885956/0.015705805 s versus candidate 0.570561133/0.005746000 s. No demonstrated improvement. Per-rank VRAM unchanged: CUB 913 MiB, CUCO 1361 MiB. Old baseline 0.483016706/0.003722780 s, 2817 MiB/rank. All 30 records COMPLETE, layers and archive-off verified. Small raw: commit 322fdb333871a6eda957fcda32a74eee46eb8d2f.
- Rejected count-handshake reorder restored in d46dcc10f56986766d5b59d9261bca027b364d91; runtime byte-identical to accepted b8bee990f4083aad0dc8d825a49b7a43d9046748 before next candidate. No claim of statistically established regression.
- Runtime eac7086102ead6eb2a149ddf7f726b366f58361b extends producer packet prefetch to HASH_FIRST, orders empty packet scratch on the producer, and initializes speculative route inputs on generation fatal without reading parents. UNVERIFIED candidate, not accepted runtime.
- Kaggle v9 launched using immutable launcher e1cf6d9a46fab79f8604597f31b3ac044eba8c8b: both profiles, three owners, banks 2/3, both rank maps, pre-dedup ON/OFF, full-state/fault/sanitizer/reuse checks. API confirmed RUNNING; no duplicate notebook.
- HOST_SIZED_NCCL exact payload sizing still uses CPU readback; HASH_FIRST materialization and full CPU-free owner/transport/retirement acceptance remain OPEN. Physical LSA unavailable on Kaggle. Macro-depth, archive/HF and DB/framework scope remains unchanged.


## 2026-10-05: HASH_FIRST HOST producer RED and integrated candidate

- v9 is terminal COMPLETE, diagnostic summary INCOMPLETE: 108 outer checks, 26 failures. Unsupported CUCO_RANK/HASH_FIRST/HOST combinations account for 18 (8 ordinary, 8 sanitizer, 2 reuse). Four native HASH_FIRST fault suites stopped at owner-error injection: both ranks completed because the hook was inside rank-mode only. Four supported HASH_FIRST reuse fixtures completed their BFS but verifier rejected zero producer bank reuse (PROCESS_ORACLE_ROUTE_BANK_REUSE). These are not accepted gates and do not establish corruption of healthy BFS states.
- Complete terminal evidence: 9d156d8a2d26818d0b31b2c2fe0c22a80a3af7fa, ALL 21 pages / 2094 listed entries / 1430 retained small artifacts. Earlier 231eaeb59a2e7f71a7cf6908e93d2a05d37d92d7 was a partial publishing snapshot (13 pages); do not cite it as complete. Large archives remain in Kaggle.
- Runtime candidate b49aa8cd227f694f5ebe87226a77f1a0b0ad9b66: producer admission no longer requires LSA materialization storage; HASH_FIRST HOST uses the existing bank event/last-reader protocol. Generation fatal uses the existing host or device group policy. Existing owner-host-error injection moved to the shared owner-job boundary. This is UNVERIFIED until v10 gates finish.
- Launcher 4371801e66b2776e3191c41c9a6f3cc3ccfc5557 / v10 starts with explicit negative preflight checks for unavailable CUCO/HASH_FIRST/HOST (never runtime PASS), then supported reuse before ordinary full-state/fault/sanitizer suites. Only one notebook is active.
- HOST variable-payload sizing and HASH_FIRST materialization still require host counts. Full CPU-free transport, unavailable CUCO/HASH_FIRST/HOST implementation, LSA physical acceptance, macro-depth/archive/HF/DB-framework requirements remain OPEN.


## 2026-10-05: terminal v11 and exact capacity evidence correction

- v11 runtime b49aa8cd227f694f5ebe87226a77f1a0b0ad9b66, launcher be652969a77512c9efa175ad3b9c6e024986e95c: terminal COMPLETE job, application summary INCOMPLETE (86/90 outer checks). All 10 supported producer bank-reuse fixtures passed. All 80 rank sanitizer logs contain zero-error/hazard summaries across four tools. This is HOST_SIZED_NCCL evidence, not LSA acceptance or full CPU-free proof.
- All four rejected native HASH_FIRST CUB/BMMA suites stopped at capacity rank 0: exact injected layer-capacity fatal NATIVE_OWNER_FATAL_16, peer REMOTE_OWNER_BATCH_FATAL, returncodes [1,1], no forced cleanup, no group COMPLETE, 1.806–1.910 seconds. The matcher omitted this exact native fallback error. Owner injection is no longer bypassed. Remaining cases after first capacity failure were not run; do not claim full fault-suite acceptance.
- Terminal retention 8215d383288240eadfd501c2a7e0969b6ed28cbd: all 22 output pages, 2183 entries; 1405 small files plus exact launcher and reconciliation in test_results/kaggle-dense-packet-v11/raw-small-evidence.tar.gz. Large archives stay cloud-only.
- Matcher correction 0ca18e4c090d0749b0c3c9224bf5d9ea2e36934a requires the armed injection and an anchored runtime fatal with exact code 16 and injected rank. Negative fixtures d4baa58b744b461b9c71a789c8ea9cb9562e61d4 reject wrong code/rank, generic remote abort and diagnostic substrings; prepared for remote execution, not locally tested.
- v12 launched only after complete terminal retention and confirmation all three known kernels terminal. Launcher 24a5340ff2ade3de66a1bdd4a27509ce63cae7c6 pins d4baa58, executes fixtures before full gate, stops on ordinary failure before sanitizers. Pending results; no performance acceptance yet.
- Full scope remains active: HOST payload sizing/readback and HASH_FIRST materialization dependencies, unsupported CUCO/HOST/HASH_FIRST, LSA hardware gates, profiles/macro depth/HF/DB-framework, full compiled-dependency cache and Pareto claims are not closed.


## 2026-10-06: v12 live fault correction and prepared cache-aware A/B

- v12 API RUNNING; live 51 result rows: 10 supported producer reuse and all 40 ordinary configurations passed, plus first sanitizer fixture. The previously rejected typed-24/28/32/36 HASH_FIRST native CUB/BMMA fault suites now report DIAGNOSTIC_CASES_PASS with no failed cases. Pending all four sanitizer tools and terminal raw retention; not complete acceptance.
- Prepared launcher 77e1bf88b58f172d410c3965e526ee03fea3b194 and 3710dbafbe03bda1b4a957e71e9bf62c9e001280 integrate compiled artifact import/export into existing main/old/before CMake targets. Recipes cover actual source files including generated profiler shims, CUDA/RAPIDS header/config fingerprints, toolchain/package versions, exact external commits/archive and normalized configure flags. Cache validates exact members/recipe/ELF/checksum and current dependency resolution; required mode rejects a miss, no implicit compilation. Publish mode is explicit first-producer admission, not proof of reuse. Negative admission fixtures run remotely before builds. No cloud execution yet. Conservative cross-root hashing can over-invalidate dependencies; no optimal-cache claim.
- Prepared default paired_measure pins d4baa58 runtime with unchanged b49 CUDA/Rust algorithm. Main native compile enables NVTX capability also in paired mode so its artifact can be consumed by subsequent scoped timeline without changing compile flags. Trace ranges remain disabled in ordinary timed runs. Existing old/before algorithm sources unchanged except already recorded archive/warmup/profiler launcher guards.
- Do not launch while v12 lives or before terminal all-page retention. Next paired run is also first cache producer; a later sequential consumer must demonstrate hits and zero CMake builds under required mode. Full owner/transport/retirement CPU-free and all original scope remain open.


## 2026-10-06: v12 accepted bounded HOST gate; v13 paired run dispatched

- v12 terminal COMPLETE, application TYPED_GATE_PASS: 90/90 outer checks and 270 nested process cases. Healthy full-state oracle records: 80 S4 runs (24 states, layers [1,3,5,6,5,3,1]) and 10 U4/F2 producer-reuse runs (64 states, layers [1,3,5,8,11,13,13,8,2]). These are separate fixtures; do not report all 90 as S4 or generalized coverage.
- 180 asymmetric fault cases: both independent rank processes return nonzero, no forced cleanup or group COMPLETE, within requested deadlines; longest observed 2.2082834469999852 seconds. All 80 rank logs from memcheck/racecheck/initcheck/synccheck have zero-error/hazard summaries. This covers supported HOST_SIZED_NCCL combinations, not LSA registration/initcheck acceptance.
- Full terminal raw retention 2673cbe4d24a56ab8e99d166517ea79173fe073f: all 23 pages, 2204 entries, 1414 small files plus exact launcher/reconciliation. test_results/kaggle-dense-packet-v12/raw-small-evidence.tar.gz; large outputs remain cloud-only.
- v13 dispatched after retention and confirmation all known kernels terminal: trydotatwo/mgbfs-dense-packet-prefetch-gate, launcher 71a5f129a0b907d567457c56ac02bedda129c3bf, SOURCE d4baa58 (b49 runtime). Paired S10 matrix_u8, old4ef/beforeb8/new CUB/CUCO DENSE plus native CUB/BMMA HASH_FIRST, five repetitions/full internal warmup, no archive. Pending timings and provenance validation.
- Compiled cache first-producer code is integrated but not cloud-tested: actual selected RAPIDS venv metadata, source/header/so fingerprints, streamed binary hashing, deterministic archives. Required consumers must pin whole archive SHA from producer summary via COMPILED_CACHE_EXPECTED_ARCHIVES, then validate members/recipe/ELF/payload hash/current dependencies. Never silently rebuild on a required cache miss.
- Full CPU-free owner→transport→retirement is NOT closed: HOST variable payload sizing and HF host materialization remain CPU-dependent; unsupported CUCO/HOST/HASH_FIRST, physical LSA and wider profile/seed/macro/HF/DB-framework/Pareto gates remain open.


## 2026-10-06: v13 launcher RED, metadata correction

- v13 terminal ERROR before CMake or BFS runs. Exact failure: isolated uv RAPIDS interpreter has no pip; cache recipe subprocess `python -m pip freeze` exits 1. This is launcher/cache admission failure, not BFS performance/correctness evidence. Compiled cache pure admission fixtures passed; NCCL cache imported correctly and was not rebuilt.
- All terminal output retained d608ef1647aa9b48a191a3fbf74f13b7253650b9: 1 page, 29 files plus exact launcher/reconciliation in test_results/kaggle-dense-packet-v13/raw-small-evidence.tar.gz.
- Replace freeze with sorted Name/version pairs from standard-library importlib.metadata, executed by the same isolated RAPIDS interpreter. No new pip dependency or change to runtime algorithm/build flags. Physical retry pending; no compiled cache hit or A/B timings yet.


## 2026-10-06: v14 launcher RED and NVTX capability contract

- v14 ERROR before BFS: main native CMake has MGBFS_NVTX=ON for paired cache compatibility, but SDK component selection used profiling_enabled only and omitted NVTX headers. Exact native-configure error: MGBFS_NVTX requires an explicit NVTX3 include directory. This is a launcher dependency bug, not runtime/BFS data evidence. Metadata correction and cache admission fixtures passed. Library compiled successfully, ldd resolved dependencies, cache exported.
- All output retained dde21e240ddbd62075cc924b6a26e2c45568a8bc: 1 page, 40 entries; 39 small files plus launcher/reconciliation. Library cache archive stays in original Kaggle producer v14, filename/key/SHA in retained summary.
- One nvtx_enabled predicate now controls both SDK component admission and native compile flags; it does not enable profiling of ordinary timed searches. Attach producer outputs and validate pinned cache archive SHA on a matching hit even in explicit publish mode. Do not relax recipe matching to force a hit: changed SDK headers can conservatively invalidate the previous library artifact.
- Preserve original producer v14 while the next sequential run uses the other existing notebook. Still only one live notebook; no new rental/subagent/local build. Acceptance v12 remains valid for unchanged runtime d4baa58, pending A/B/cache proof.


## 2026-10-06: v16 independent paired evidence; v15 queued cache consumer

Full objective remains OPEN. Runtime source remains d4baa58/b49aa8; these
benchmark/cache/validation commits are not a CPU-free runtime completion.

- GPU producer `mgbfs-timeline-window-analysis` v16 COMPLETE. All five output
  pages retained: 496 entries, 492 small source artifacts; four compiled archives
  remain cloud-only. Raw immutable evidence: c887c36ec709d1394e82ac2324a65276f3aa7695.
- Independent checks in e2f548d5a956948413bf82385432f722f59f435b validate
  54 search rows, 108 measured rank records, 108 persisted internal full-BFS
  warmup records, IDs, completion/exit status, identical global layer counts and
  rank sums (S10 3,628,800 states / 46 layers), zero pinned/disk archive allocation,
  new archive-off flags, all five repetitions and recomputed median/MAD.
  The old launcher patch executes warmup then measurement in one process;
  absence of newer archive_enabled fields in old rank JSON is not a failure.
- S10 layer equality is not full-state oracle equality. Physical small-state
  oracle/fault/sanitizer acceptance remains the separate v12 bounded gate.
- All 54 external nvidia-smi CSV files independently checked: sampled maxima
  match reported memory values; maximum observed sampling gap is 153 ms.
  This is whole-process discrete maximum (includes startup/warmup), not an
  exact instantaneous or timed-search-only peak. Native cudaMemGetInfo records
  separately sample setup/final only. Do not conflate these sources.

| Configuration | Search median s | MAD s | External sampled MiB/GPU |
|---|---:|---:|---:|
| old 4ef9ce1 | 0.488407710 | 0.001752197 | 2817 |
| new DENSE CUB | 0.734244864 | 0.005045812 | 913 |
| new DENSE CUCO | 0.558328064 | 0.004158928 | 1361 |
| new HASH_FIRST CUB | 1.882798034 | 0.004763286 | 829 |
| new HASH_FIRST BMMA | 2.538455391 | 0.016688182 | 829 |

This is old/new MultiGPUBFS, not a new CayleyPy comparison. Same S10,
batch 32768, 2xT4, HOST_SIZED_NCCL, archive off, five unprofiled repeats.
Old remains faster; sampled memory decreases. No cause or optimization win
is claimed from the above timing alone.

Existing original notebook `mgbfs-dense-packet-prefetch-gate` v15 is QUEUED
(API verified 2026-10-06). Launcher 2d3ab1045baf6edc03c16795152d0a60c513a28d
uses old_new_pair scoped timeline and REQUIRED compiled caches pinned to all
four v16 archive checksums. Preserve v16 and immutable NCCL v131 producers.
Actual compiled-cache reuse / no CMake build is PENDING physical consumer logs.
No second notebook/rental/local build started.

Remaining primary implementation: HOST exact payload counts still block the
submission thread; HASH_FIRST host owner/materialization remains synchronous.
A transport-worker integration must preserve single NCCL caller/abort ownership,
bounded credits, physical receive-slot lifetime, zero-payload issue order,
cancellation and FinalizeDepth. Do not remove existing protective waits in
isolation or claim CPU models/Async APIs complete this path. LSA hardware
acceptance is unavailable on Kaggle and remains OPEN, not bypassed.


## 2026-10-06: measured-search v17 evidence extracted; runtime still incomplete

- The only CPU analyzer `trydotatwo/mgbfs-timeline-window-analysis` v17 is COMPLETE. All one output page / twelve entries reconciled. Ten large full-window JSON remain on Kaggle; no SQLite or large data downloaded to disk. Small summary/probe/exact source retained in 378d2b78; bounded 64 KiB HTTP range tails with Content-Range and SHA256 plus decoded aggregates retained in 526ba346.
- Consumer GPU v15 passed physical compiled-library REQUIRED cache reuse (three hits), measured-search PROFILE_PASS, all five configurations. Evidence b5db6937. No inference that Rust compilation or environment installation disappeared.
- Eight new-rank captures each contain 44 intra-depth windows; depth-boundary windows excluded. Old has no batch NVTX: zero windows is missing instrumentation, not zero waits.
- DENSE CUB healthy windows wall 2.762/2.779 s, kernel/copy union .881/1.034 s, multi-stream union .0122/.0058 s. DENSE CUCO wall 2.485/2.514 s, busy .781/.842 s, multi-stream .00785/.00549 s. Thus measured multi-stream interval coverage is under .5% of healthy-window wall, not evidence of effective stage overlap.
- HASH_FIRST CUB healthy-window synchronous cudaMemcpy API durations .287/.281 s and cudaStreamSynchronize .329/.489 s; BMMA .310/.303 s and .790/.831 s. Counts/control host dependencies remain source-confirmed. These profiled API sums are NOT recoverable critical-path time, occupancy, or unprofiled A/B.
- Current runtime remains d4baa58/b49aa8. No runtime fix claimed by this extraction. HOST exact-payload count handshake and HASH_FIRST legacy owner/materialization remain open. Next production change must connect dispatcher, exact-count transport, owner publication, cancellation and last-reader retirement; do not spend another run only on profiling scope/cache construction or isolated wait deletion.
- All profiles, macro, archive/HF, LSA physical registration/initcheck and DB/framework stages remain in full goal. Goal NOT COMPLETE. No notebook is live after v17 terminal; preserve v15 cloud SQLite and v17 full JSON as immutable evidence producers.


## Work-control correction and v19 receipt (2026-10-06)

The full goal remains incomplete. Launcher/cache/oracle fixes are supporting work,
not owner-to-retirement runtime implementation. Do not rerun an already verified
360-case matrix only to change result classification. Do not attribute host API
duration sums to recoverable critical-path speedup. No local dirty changes are
owned or overwritten by this remote update.

Verified v19 evidence: raw receipt `ad966f94e2f27f6abcb02bba92406bb228bb1571`,
independent validation `c5ad9872eef8a12b6a072e6e6c0b364e10764002`. All 47 output
pages reconciled (4609 entries); graph archives and compiled payloads remain
cloud-only. There are 300 supported two-rank full-state searches, 600 verified
rank records and 300 group commits. Moduli 2..6, seeds 0/1/20260828, both rank
maps and pre-dedup variants agree with the remote CPU full-state oracle. The
60 HASH_FIRST/CUCO_RANK/HOST requests were unavailable admissions, not successful
GPU searches. Historical launcher INCOMPLETE is retained unchanged. Both strict
compiled caches physically hit after oracle Arrow installation was moved after
cache admission. Runtime source remains d4baa58; this is not a new runtime fix.

Execution priorities, not additional acceptance claims:
1. Implement an integrated runtime change before scheduling another generic
   audit/profile/classification run. Reuse existing cancellation, epoch credits,
   packet banks, owner descriptors and last-reader events.
2. Keep HOST exact-count sizing explicitly CPU-visible. Moving it into a thread
   does not meet the device-driven transport contract. Do not pad maximum payload.
3. Close HASH_FIRST owner/materialization gaps without dropping fatal guards or
   relaxing admission. Preserve FinalizeDepth and irreversible owner commit.
4. Run only gates affected by the actual patch first; then full acceptance and
   paired A/B on an unchanged output contract. LSA acceptance remains unavailable
   on the current permitted Kaggle platform; no LSA retry or new rental is implied.
5. Every progress claim must name runtime commit, relevant physical gate and
   unresolved limitations. Full profiles/macro/archive/HF/DB stage remain in scope.


## Integrated HASH_FIRST HOST candidate launched (2026-10-06)

Runtime candidate b5ba973d16c0a431e40cfa75521b006239e7f2b2 /
e37c7504340f218451a71f57f408ac008d27652a, NOT accepted yet:
- Native rank owner now handles HASH_FIRST on HOST as well as LSA. Fixed
  device counts, controls and extents are admitted before construction; other
  non-rank library owners keep their legacy storage and dispatch.
- Shared device materializer branches only at transport: HOST exchanges exact
  request sizes and payloads using existing allocated buffers, regenerates on
  the consuming stream, returns inverse-sized responses, and publishes device
  extents through the existing ring/fatal protocol. No maximum payload padding.
- HOST request-size waits remain explicit and cancellable. No CPU-free claim.
  Response sizing reuses the request handshake, not a second count readback.
- HOST HF route count send uses immutable per-owner bank counts rather than
  collective_send, which may still be consumed by a GPU fatal vote.
- Owner-consumed event for HOST HF is recorded after materialization, protecting
  single receive-slot reuse; generation packet readiness no longer requires a
  prior owner-stream drain for nonempty prepared batches.
- Public HF/CUCO/HOST admission guard is unchanged until physical acceptance.

Kaggle v20 trydotatwo/mgbfs-timeline-window-analysis launched after API confirmed
all three known notebooks terminal. Exact launcher64f4ea5b669089332a04a3188b3a513415821299
selects affected HF CUB/BMMA variants plus DENSE CUB control, full-state oracle,
slot/bank reuse, bounded asymmetric faults and all four sanitizers. Initial
status QUEUED; no compilation, correctness, sanitizer or speed pass yet.
Native CUDA source closure unchanged, required compiled/NCCL caches retained.
Local dirty worktree untouched; no local builds/tests or new rental.


## v20 accepted bounded native HOST HASH_FIRST gate; CUCO v21 pending

Kaggle v20 is terminal COMPLETE/TYPED_GATE_PASS. All 11 pages / 1091 outputs
reconciled; 700 original small files retained in7b79bdf4598bb6a283559fd126943f92bb2bd315.
Independent raw verification4166ce817768beb29ab477500c533e31412f74a9 confirms:
43 healthy cases,86 rank records,43 exact-byte checksummed group commits;
90 asymmetric injected faults with both ranks nonzero, no forced cleanup or
false group COMPLETE, max2.260890622s;40 rank sanitizer logs (10/tool) clean.
Native CUB/BMMA HASH_FIRST onHOST,2/3packet banks,preON/OFF/maps exercised;
remote full canonical state/depth oracle checked S4 and bounded U4(F2) reuse.
No LSA registration/initcheck acceptance, CPU-free transport or speed claim.

Candidate366a545218e4485d7f10e5cbe802bd03daf2dacc admits CUCO_RANK HASH_FIRST
HOST into the same runtime protocol (the rank ABI was already integrated).
Both reference selector and RunConfig guard removed together; fixed pool
requirement/alignment and explicit transport identity tests retained/added.
CUCO physical support is pending, not inferred from native gates.
One Kaggle v21 launched only after v20 retained: exactlauncher
ad1ddb3fdf1b46598d9461d7ee3c3ce4164eb500, focused core tests/full-state/fault/
reuse/all-four-sanitizer gate, plus native CUB HASH_FIRST control. API QUEUED.
No CUDA/library source changes, strict immutable caches still required.


## 2026-10-06 v22 integrated HASH_FIRST/CUCO HOST gate

Runtime 35e981ec1b1550655805ec983b8a71debabf2ad3 passed bounded physical 2xT4 gates: 23 healthy full-state oracle receipts, 54 asymmetric startup/owner/capacity/archive failure cases, 46 rank records checked against group commit byte hashes; all four sanitizer tools, six rank logs per tool. Raw all seven pages/653 outputs reconciled, 426 original small files plus exact launcher retained in a3ebd20. Independent verification saved alongside raw. No false COMPLETE/forced cleanup; max fault duration 2.257606135000003 seconds. v21 failure was stale constructor guard, fixed in 35e981e. HOST exact payload sizing still requires CPU counts; no CPU-free transport, timeline, speed, macro or LSA registration claim. Next: scoped actual runtime timeline and independent paired A/B including newly admitted CUCO HASH_FIRST, then remaining full goal.


## 2026-10-06 v24 independent paired regression evidence

All six pages/535 outputs reconciled; 532 original small files + exact launcher retained in c9bd9aab6901a428b2dbf18aa138899ea8b36e89. Independent e7893fd3205e0a0ddc438907670ae3b1157fb1f9 verifies 60 rows,120 measured rank records,120 full in-process warmup rank records,54 group markers byte hashes, all S10 layer vectors, archive off and sampled VRAM. HOST batch32768, physical2xT4, five unprofiled repetitions. HASH_FIRST CUB before1.764036939s/current0.857492715s (2.057x); BMMA2.458617018/1.520238481 (1.617x). Newly admitted HF CUCO0.723844684s. DENSE CUCO before0.543007721/current0.549137022: no win established. Old MultiGPUBFS4ef median0.473501248s remains faster; not CayleyPy. Sampled whole-process MiB/GPU old2817,DENSE CUB913,CUCO1361,HF CUB/BMMA829,HF CUCO1277. No archive durable measurement or CPU-free HOST claim. v23 failure retainedd4a76eed: before build recipe mismatch, repaired identical immutable native recipe1554f656. Next scoped actual GPU timeline, not another unprofiled benchmark. Full goal remains incomplete.


## 2026-10-06 measured local materializer vote consolidation candidate

v25 scoped producer rawc55caa35c4df0ef56487c3c88ee1cfd5cde467f3: six configs/twelve traces. Cloud analysis9c0e631 output raw902dccfc4353b6d79f31b43be05162e8f1979e6c: 44 healthy intra-depth windows/rank for all new modes; multiple-stream interval overlap under1% healthy wall; small correlated copyKind2 transfers still present. HOST HF full captures have994 u32 allreduce kernels/rank versus364 DENSE; API and kernel intervals are not unprofiled causal speedup. Confirmed code emits two intermediate global votes inside local-only materialization. Candidate3d5c736f keeps imported local/ring sticky fatal predicates and the final local group vote before any remote work; removes only two intermediate local votes. Remote gates, cancellation, receive/parent last readers, bounded credits, FinalizeDepth unchanged. Debug injections after local sort and regeneration, each independent rank; replay authenticates marker plus logical fatal/noComplete. Pending physical all-owner HASH_FIRST full-state/fault/all4san gate, then A/B/timeline. Not accepted/complete yet.


## 2026-10-06 v26 verified and v27 paired A/B launched

v26 raw e186ddf: all15pages/1471outputs reconciled;923 original small outputs retained,548 cloud-only. Independent a5c99f08 verifies54 healthy full-state oracle receipts and108 rank record byte hashes/group commits;132 asymmetric faults, including24 authenticated local sort/regenerate injections; max2.157802317s, no forced cleanup/false COMPLETE. All4san raw48ranklogs,12/tool clean. Physical2T4 HOST path only; not LSA registration, CPU-free transport, macro or full-goal acceptance. Runtime3d5c736f now passes this bounded gate.

v27 launched only after evidence retention and API confirmation all known notebooks terminal. Exactlauncher ca3b9a0316f16c7a43815e409a4684cd26d06aba: five unprofiled S10 paired repetitions/current3d5 vs accepted35e and old4ef, full in-process BFS warmup, archiveoff, batch32768; includes beforeHF CUCO as well as CUB/BMMA. API QUEUED. No speed claim until independent reconciliation. Next scoped actual timeline/current full matrix; full macro/archiveHF/DB-framework goal remains active.


## 2026-10-06 v28 independently reconciled; paired timeline prepared

v28 raw all6pages/583outputs fully recovered in aa7fd3f; independent73a0ae6 verifies66 runs/132 measured rank records/132 in-process warmup rank records/60 group bytehash markers, all S10 layer vectors, archiveoff and external sampled whole-process VRAM. MedianHF CUB before0.908977748/new0.895603715;BMMA1.605209468/1.582290644;CUCO0.794328548/0.741221757. DENSE CUCO0.578959028/0.567443112;old4ef0.498204536 still fastest. Small changes not general speedup proof. HOST CPU sizing remains. v27 unsupported P2P launcher admission error retained36cc459e; fixed only HOST paired admissiond51dafed.

Paired_timeline reuses the same launcher and exact immutable build recipes, accepted35e vs3d5 HF three owners plus old/DENSE CUCO controls. Eight measured-search-only profiler captures; no profiler time in A/B. Actual capture/critical path not accepted yet. Full goal incomplete.


## Work correction and paired scoped analysis v2 retained (2026-10-06)

Full goal remains ACTIVE, not completed. Existing active goal is reused rather than duplicated. Local HEAD52bb77f dirty worktree was inspected and preserved; no local build, test or source edits.

CPU analyzer trydotatwo/mgbfs-current-scoped-analysis v2 COMPLETE/EVIDENCE_EXTRACTED. All one page/19 outputs reconciled: three small JSON retained in604624d4391385d0307783680dab7a4e1d0c909d;16 detailed per-process JSON remain cloud-only. Exact analyzer217e521 and input GPUv29 runtime3d5 versus35e; GPU producer remains unchanged.

Confirmed paired timeline: HASH_FIRST u32 allreduce kernels fell994 to814 per rank; healthy batch D2H counts remained176 four-byte plus44 eight-byte transfers. Healthy multi-stream interval overlap remains below0.5% wall in all new cases. This is measured profile interval coverage, not hardware utilization or unprofiled critical-path speedup. Old lacks batch NVTX, so zero windows cannot be interpreted as zero waits.

Process mistakes: repeated launcher/cache/profile rounds displaced integrated runtime delivery; a removal of local fatal votes was treated as too central despite unchanged payload sizing and lack of overlap. Correctness receipts and small A/B gains do not close the CPU-free contract. No new runtime fix is claimed by this evidence-retention change.

Execution rule: next substantive change must modify the existing integrated runtime, not add another generic audit/benchmark launcher. Trace/source identify the exact HOST route count handshake on submission thread as still open. Ordinary host NCCL counts remain host-valued; a worker can decouple dispatch but must not be called CPU-free. No maximum-payload padding or isolated reader-event deletion. Reuse bounded packet banks, exact payloads, sole NCCL/abort caller, cancellation sideband, physical receive-slot last-reader events and FinalizeDepth. Kaggle LSA unavailability is a hardware acceptance blocker, not a reason to retry LSA or rent without new authority. Remaining profiles/macro/archiveHF/DB scope unchanged.


## HOST count/payload lifetime separation candidate (2026-10-06)

Remote RED source9344f7 test ran on CPU Kagglev5: expected assertion bank0 no independent handshake word, exit101. Full small outputs retained with candidate. v3 was setup failure; v4 receipt cannot establish a separate executed version because latest-output provenance was insufficient. No correctness claim from those receipts. v5 workspace-index plus expectedRED authenticates the actual test.

Candidate modifies existing distributed runtime and allocation planner, not a parallel implementation. Each source route bank reserves a third u32 control word for incoming exact-size handshake. Previous owner continues reading the separate physical recv_count/payload. Move owner_consumed wait after size handshake/host validation; then D2D-publish4B into actualrecv_count and enqueue exact hash/state payloads. Same NCCL call order/caller/abort/cancel path, same single receive slot, bank last-reader lease and bounded epoch window. Aligned control allocations remain256B; payload budget increases4B per bank. Both DENSE and HASH_FIRST HOST routing use this shared path; LSA unchanged.

This is pending physical acceptance and NOT CPU-free HOST transport: D2H size/routing counts remain, and the communicator may add serialization. Do not infer overlap or speedup from source. Next required existing fullstate/fault/all4san gate and paired timeline; full goal remains active.


## v30 failure investigation — focused replay pending

- v30 terminal COMPLETE is not acceptance: typed44 HASH_FIRST/BMMA/preON/banks3 worker_write rank0 exceeded45s on both ranks; sanitizers not run. a194699 remains rejected.
- All1085 selected small outputs retained in e533c4f; missing18 recovered, no large outputs downloaded.
- Focused diagnostic: unchanged production before3d5c736/a194699, exact typed44 config, three fault repeats and separate full-process Nsight failure capture per revision; healthy full-state control first. Native/library closures must match, dependency caches required.
- Injected worker_write in existing healthy-only harness deliberately yields FAIL, not gate PASS. No A/B until bounded group-failure acceptance.
- nccl_abort_begin does not locate stalled API/drain/retirement. No root-cause fix claimed.


### v31 diagnostic rejected; selector correction pending hardware

- All79 v31 outputs enumerated, selected small files retained7d77624; no missing downloads. Source runtime never changed.
- before healthy full-state passed; attempted fault also completed healthy because replay clears inherited fault vars per case. FOCUSED_FAULT_INVALID correctly rejected it; not a production false COMPLETE because the fault never armed. No claim about before/candidate failure path follows.
- Existing replay now has explicit --case-filter worker_write-0 preserving harness key/rank and case-env sanitization; normal full matrix and healthy-only remain. Unknown/ambiguous/conflicting filter fails closed. Five regression tests and existing replay fault/deadline/cleanup suites must run remotely before physical reproduction.
- Original runtime revisions remain3d5c736/a194699; pinned helper1233fa73 is test tooling only and source.patch records it. Compiled closures compared, cache required. Full goal and v30 rejection unchanged.


### v32 focused fault observations, not closure

- v32 COMPLETE DIAGNOSTIC_COMPLETE; all177 outputs over2pages enumerated, all selected small raw retainedcd7155a, no missing. Four SQLite/NSYS traces remain cloud.
- Both unchanged runtimes3d5c736/a194699 healthy full-state24 states pass. All6 unprofiled genuine worker_write0 injections reached, both ranks exit1 in1.657..2.661s, no false COMPLETE;2profiled faults exit1 in5.626/5.877s. Eighteen selector/fault/deadline/cleanup tests passed for each checkout (sitecustomize missing-wrapt warning remains).
- These debug-enabled repeats do not disprove v30 intermittent hang, do not restore a194 acceptance and are not performance samples.
- Next focused stress removes NCCL debug; per revision25 plain fault repeats,24 API-stage-probed faults and1Nsight fault. Diagnostic LD_PRELOAD only forwards ncclCommAbort/cudaDeviceSynchronize and marks enter/exit, owns no cancellation or NCCL calls. Stub forwarding test preserves return codes before physical use; runtime and cached dependency closure unchanged.


### v33 cancellation stress: actual teardown stall narrowed

- v33 terminalCOMPLETE/DIAGNOSTIC_COMPLETE means collection only, not acceptance. All1101 outputs across12pages enumerated;785small files retaineda93419d0, no missing; large traces staycloud.
-102runs:2healthy exactS4 state oracles pass;100genuineworker_write0 faults,14forced45s cleanups: before8/50,candidate6/50. No groupCOMPLETE during any injection. Both runtimes have intermittentfailure, so size-word patch is not shown to be the cause.
-9failed stage-probed ranks0 all show ncclCommAbort enter/exit code0 approximately0.505s, then neither wrapperabort_end nor further output. The vendor abort itself didreturn.
-DeviceSynchronize LD_PRELOAD instrumentation did not appear in these runtime records; its forwarding stub test doesnotprove itintercepts CUDA runtime usedbythecached native library. DoNOT infer which nextstage hangs merelyfrom missingprobe.
-Next diagnostic adds failure-only marker at existing Rust retirement callback entry to distinguish reaching sideband ACK from earlier CUDA drain. No algorithm/protocol/lifetime changes, no GPU/NCCL dependencyclosure change, no fixclaim. a194 remains unaccepted, A/B and full acceptance blocked on actualfaultfix.

## 2026-10-06: v34 actual retirement boundary, not a runtime fix

- Consumer v34 COMPLETE; all 3 output pages (279 entries), all 210 selected small originals retained without missing files in commit 6207dfb6d4d8bb54f636f8e3d4eea6bfb862d8f3. Large trace databases remain on Kaggle.
- Healthy 2-rank S4 matches all 24 canonical CPU states and layers [1,3,5,6,5,3,1]. All 18 replay helper tests pass.
- worker_write-0: 1/20 failures exceeded the 45-second bound (candidate-fault-rep11), both processes required forced cleanup; no group COMPLETE.
- Failed rank 0: ncclCommAbort returned success in ~0.505 s, then retirement_probe_enter(publish=1). Thus that rank completed its GPU drain and waited for the group retirement ACK. Rank 1 log was empty; its blocking API is not yet identified. Do not remove rank-0 reader protection as a purported fix.
- Next diagnostic wraps existing Rust CUDA synchronous waits/copies with opt-in caller-location entry/exit markers. Same calls/arguments/results; no new CUDA operation, NCCL caller, allocation, protocol change or performance claim. Actual Rust compile and physical fault replay are still required.
- Full objective and a194 acceptance remain incomplete; HOST-sized counts are not CPU-free.

## v35: Rust blocking API exclusion and next native boundary

All five output pages (479 entries), 350 selected small originals retained with no missing files in 297b5415e040d0d4db3609e34ca38b52bbcb9fc5. Seven of forty genuine worker_write-0 faults required 45-second cleanup (reps 8,11,14,34,36,38,39); no false COMPLETE. In the failed rank-1 logs every traced Rust cudaMemcpy/cudaStreamSynchronize had a matching successful exit. Therefore these markers did NOT identify the trapped API; do not claim those waits are its root cause. Rank 0 still reached retirement publish=1 after successful abort/drain. Next probe traces dynamic NCCL AllReduce and GroupStart/End with exact once-only forwarding and both positive/negative argument stub checks. Runtime source and compiled CUDA/NCCL closure unchanged; a diagnostic, not acceptance or a fix.

## v36 and quiet failure snapshot candidate

- v36 COMPLETE: all 3 output pages/279 entries, 210 small originals retained without missing files in 059aad0a6c2f5d49f65f70fc82322509c8d948fb. Healthy full-state control and all 20 genuine fault cases passed; this does not close the previous intermittent failures.
- Per-call logging may perturb timing. Quiet diagnostic records the NCCL owner's outermost active API in a host atomic, with exact forwarding unchanged. Existing CancelMirror publishes sticky cancellation FIRST and then reads that snapshot at cancellation and 250 ms later if still alive. No new observer thread, NCCL caller, CUDA operation, abort owner or reader ACK is introduced. Normal runs do not enter this diagnostic branch.
- Rust blocking-call traces are suppressed only for the explicit quiet probe environment; retirement markers remain. Rust compile, snapshot forwarding fixture and real 2xT4 replay are required; this is not a runtime fix or performance claim.


## 2026-10-06: quiet v37 evidence and bounded stack collection

- v37 COMPLETE diagnostic only: healthy full-state S4 PASS; faults 3/20 FAIL (reps5/12/15), both ranks forced-cleaned after45s; no false COMPLETE. All3outputpages/280entries,211small retained at e0d3d9f.
- All failed rank1 cancellation snapshots are `none` at both samples; cancellation reached rank1. This does not locate a wrapped NCCL submission. Rank0 publishes local retirement and waits for peer. Root cause remains unproven; do not remove reader-drain protection.
- Existing replay gains opt-in post-deadline /proc task and bounded gdb stacks before cleanup. No healthy instrumentation, no changed pass predicate, no new NCCL caller. Each owned live rank attach has8s maximum; diagnostic elapsed time is separate from original failure deadline. Remote unit and real ptrace capability gates required before relying on stack evidence. Runtime unchanged.


## Native quiet checkpoints after v39 ptrace denial

- v39 ERROR before BFS: real gdb attach denied (`ptrace: Inappropriate ioctl for device`), not a runtime/correctness result.21helpertestsPASS; unchanged native/library/NCCL cache admission succeeded.64outputs/62small retained7ea63b3. Do not retry ptrace on this host.
- Existing fatal gate exposes host-only atomic stages read by existing CancelMirror:0outside,1combined kernel submit,2last-error query,3native allreduce,4fatal-import kernel,5cancel-word getter,6terminal-publish kernel,7last-error query. RAII resets stage on every return. Explicit diagnostic env only; no new thread, NCCL caller, device op, or lifetime change. Not a fix.
- This native source change requires a new native build/cache once; unchanged library/NCCL dependencies must remain cached. Physical compilation/full-state/fault evidence pending.


## Native-callsite v2: concrete routing boundary

- Healthy exactS4 PASS,2faultsPASS/thirdfaultFAIL45s,nofalseCOMPLETE. Marker/unit testsPASS. Failed rank1 snapshot rust_native_site=2900 bothsamples on sourcea19e64d: actual mgbfs_route_run; rank0 locally retired, waitingpeer. ALL98outputs84small retained01b599dc.
- route.cu contains CUB radix sort/select, kernel launches and D2D copies, no explicit stream drain. Which internal operation blocks is not yet proven.
- Controlled same-runtime LAZY/EAGER experiment queued next; requested modes recorded separately from actual driver mode (not yet queried). NVIDIA CUDA12.8.1 Lazy Loading documents context synchronization/deadlock risks. Hypothesis, not cause proof or a runtime fix. No algorithm/cache/native-build changes.


## Startup loading admission candidate after actual routing fault

- Same-runtime v3: LAZY worker-write replay stalled at mgbfs_route_run on fifth fault; requested EAGER had20/20 bounded genuine fault exits and exact healthy S4 oracle. Actual driver mode was NOT queried in v3. All252 small files retained6c8fe5a7.
- v4 real CLI RED: LAZY module and LAZY data requests were not rejected before input; help and explicit EAGER controls passed. All53 outputs/51 small originals retained26e59edd. No production change in RED.
- Candidate CLI selects absent loading settings EAGER before worker threads/first CUDA. Explicit invalid settings remain errors. Launched ranks defer rejection to existing configuration agreement to avoid stranding peers; standalone CLI rejects before input. Driver cuModuleGetLoadingMode must report EAGER before archive admission/communicator/persistent runtime allocations. Direct distributed runtime callers also query actual mode. Requested env and actual proof are separate output fields.
- Candidate Rust build, black-box GREEN, asymmetric loading failure, genuine fault replay, all sanitizer and timeline gates PENDING. This does not remove HOST variable-payload CPU sizing or prove full objective complete. Other direct single-GPU/library entry points remain to be admitted/tested.


## v5 disproves in-main EAGER sufficiency

- All94outputs83small originals retainedac08574e. Startup black-box4/4PASS, workspace CPU suitePASS, helper21PASS, healthy full-state S4PASS. Both healthy rank records report actual driver EAGER=1.
- Genuine worker-write fault2 FAIL45.048s, both ranks forced-cleaned, no falseCOMPLETE. Rank0 snapshot4428 is mgbfs_owner_import_transport_fatal (kernel submission); rank1 entered retirement publish=1 after REMOTE_SEARCH_CANCELLED. Thus verified driver mode and setting env in main do NOT prove the CUDART/kernel-loading path is safe. Previous external-EAGER20PASS is bounded evidence, not proof of complete fix.
- Candidate re-execs CLI once when it supplies missing EAGER defaults. This puts settings before CUDA-linked process constructors, preserves rank/PID/arguments, precedes bootstrap/BFS allocation, and never changes backend/profile or adds runtime fallback. Explicit rejected settings still go through group configuration admission. Production/hardware verification pending; no speed or CPU-free claim.
- Asymmetric module/data loading gate for both ranks added7277fd85, pending execution.


## v6 pre-constructor loading candidate: bounded passing evidence

- All3pages288outputs219small originals retained8ad69654. Startup4/4PASS, CPU workspacePASS, helper21PASS, exact healthy S4 all24states/layersPASS; both ranks verified actualdriverEAGER1.
- Module/dataLAZY separately on either rank:4/4PASS both nonzero exits0.403..0.454s, no forced cleanup/no COMPLETE. All20 genuine workerwrite-rank0 faultsPASS1.657..1.759s, no forcedcleanup/no COMPLETE. This is not proof that every failure/race is absent.
- v7 validates20physical healthy profile/pre-dedup/rank-map cases, full asymmetric faults+capacity forDENSE/CUCO_RANK andHASH_FIRST/BMMA, and all4unfiltered sanitizer tools forDENSE/CUCO_RANK. Uses HOST_SIZED_NCCL, not LSA/CPU-free acceptance; sanitizer activation failure remains failure. CUDA closure and native cache unchanged. Full objective remains open.


## v7 direct-library test admission failure (not a skipped gate)

All51outputs49small retained64fa525f. CUDA runtime lib suite34PASS/3FAIL: host_owner_only_fatal_closes_admission_after_the_completed_epoch, lsa_logical_fatal_closes_admission_after_the_completed_epoch, one_rank_lsa_abort_retires_local_readers_without_peer_callback. Each rejected actualdriverLAZY2 before runtime construction; these direct library tests do not execute the CLI re-exec/default contract. They must inherit EAGER before process construction. Launcher v8 supplies both settings for cargo test only; replay children still clear them and test CLI default/re-exec. No test disabled/ignored; runtime unchanged. Profiles/fullfault/sanitizers not reached in v7.


## v8 library suite PASS; profile fixture rejected before BFS

All53outputs51small retainede954d10c. Direct CUDA-runtime lib suite37/37PASS with EAGER inherited before process construction; previous3admission failures resolved without disabling tests. First profile replay rejected RUN_REPLAY_LIBRARY_POOL_REQUIRED_ALIGNED before source summary/BFS: launcher derivedCUCO config fromBMMA fixture with null pool. VALIDATION_SOURCE_MISMATCH masked this pre-replay failure. Next launcher explicitly budgets64MiB CUCO pool/None for nonlibrary modes, runs helper config admission upfront, and retains failing row before provenance check. No runtime/backend/default fallback change. Profiles/fullfault/sanitizer gates remain unproven.


## 2026-10-06: work-cycle correction and v9 bounded gates

v9 COMPLETE HOST_VALIDATION_PASS, retained at6270091: all7outputpages611entries/415smallfiles.20healthy full-state S4 cases across DENSE/HASH_FIRST, owner variants, pre-dedup and rank maps pass;40genuine asymmetric fault cases pass without forced cleanup or false COMPLETE. Four unfiltered DENSE/CUCO sanitizer runs and37CUDA-runtime tests pass. These are HOST-sized small-graph gates, not LSA registration initcheck or CPU-free acceptance.

Work-cycle failures: diagnostics displaced integrated-runtime delivery; unchanged CUDA/NCCL dependencies were repeatedly rebuilt; launcher admission/provenance errors consumed hardware runs; some profiling modes did not collect five unprofiled repeats. Corrective rule: immutable required caches, exact preflight, one existing launcher, retained terminal outputs before replacement, correctness before timing, scoped timeline before any performance fix claim. No additional parallel runtime implementation.

Next combined run pins8205a65 runtime versus4ef9ce1 old, physical2T4 S10 batch32768, DENSE old/CUB/CUCO, archive OFF, full in-process warmup, five alternating-order unprofiled repeats and three separate scoped two-rank traces. Duplicate cache copies are accepted only if every copy matches the pinned archive checksum. Algorithm unchanged by this launcher correction. HOST route count readback remains explicitly open; timeline must identify next-batch dependencies, not merely report aggregate API durations. Full goal remains active, including other profiles/macro/HF/DB stage.


## 2026-10-06: measured launch overhead and integrated CUB candidate

v10 paired S10 physical2T4 HOST-sized archiveOFF, full in-process warmup, five alternating-order repeats retained b86a4d2f (239outputs/3pages/225small including log). Median/MAD seconds: old4ef9ce1 .457345317/.007744962, new8205 CUB .711960044/.002655814, CUCO .5514737/.009104446. Per-rank external peak2817/913/1361MiB. All18rows and layer counts match; this is not full-state S10 archive oracle. Six scoped profiler traces remain cloud, separate from unprofiled timing.

v4 cloud analysis retained ab17a662 (9outputs+log):44healthy intra-depth windows per new rank; correlated small copies include88D2H4B and44D2H8B. CUB ~35k launches versus old~4.9k; compare merge17280 and commit merge5760. GPU interval overlap is very low. Aggregate API duration is not recoverable critical-path time; old has no batch NVTX, so zero old windows is not zero waits. HOST route sizing still reads counts on CPU.

Real RED gate v2 retained24e03b90 (45outputs/1page): captured rank compare with4buckets and plan j=2 executed correct2survivors and left persistent lengths unchanged, then reported11kernel nodes and failed <=8launch budget. This is test evidence, not source-text inspection.

Runtime candidate c38235b changes existing bounded_owner rank_compare only: CUB compare emits3tag launches over allbucket jobs because it writes global-row flags and does not use merged scratch. BMMA chunked refinement storage and commit j*k scratch remain unchanged. No new allocations, ABI, NCCL caller, fatal/retirement event or host readback added. Candidate physical GREEN/performance is unproven.

GREEN launcher v3 refused non-P2P T4 host before compile (UNSUPPORTED_HOST), retained768376. v4 launcher768376 permits only owner-only RED/GREEN unit modes without P2P; it does not weaken multi-GPU transport acceptance. Single current v4 job builds changed native once and must publish checksummed cache; unchanged library-owner forcedrequired/NCCL cached. CUB graph budget and existing full owner/BMMA fixture suites run before end-to-end gates. Full goal remains active, including CPU-free transport, allprofiles/macro/archiveHF/DB.


## 2026-10-06: CUB launch candidate full HOST gate and independent A/B

- Owner-only RED/GREEN: retained24e03b90/7dac96da. Actual CUB capture11→8kernel nodes; existing CUB owner suite and unchanged BMMA suite PASS. Runtimec38235b changes existing rank compare, not NCCL/retirement semantics.
- Full GPU workers with attached outputs refused before Python (old consumerv5, fresh consumerv1/v3), logs[]/outputs[]. Minimal no-input fresh consumerv2 executed on both TeslaT4. Mount/admission root cause not proven. Consumerv4 used no mounts and downloaded exact pinned native/library/NCCL caches in-worker, preserving recipe/archive/payload checks and no build-on-miss. It executed the full gate.
- Consumer trydotatwo/mgbfs-batched-cub-full-gate v4 HOST_VALIDATION_PASS, runtimec38235b, launcher91aca6c. All891outputs/9pages enumerated;562small originals including terminal log retainede3b38fcf51383198fe48f304e383aa9ea97ba31f at test_results/kaggle-batched-cub-full-v4/raw-small-evidence.tar.gz. Large archives/binaries stay cloud.
- 21launcher helper tests and37CUDA-runtime library tests PASS.20physical two-rank full-state S4 cases: DENSE CUB/CUCO andHASH_FIRST CUB/CUCO/BMMA, both pre-dedup settings and rankmaps. Four fault panels contain19/19/23/23cases (84total), all passed without forced cleanup. Eight unfiltered sanitizer cases: allfourtools forDENSE CUB andCUCO. Summary32/32panelsPASS. This is smallS4 HOST evidence, NOT proof of LSA-registration initcheck, >2ranks, CPU-free transport, all profiles' sanitizers or absence of all bugs.
- Before paired measurement, repaired native-artifact directory alias: controls and candidate now restore independent pinned artifacts; no requirement that a CUDA-changed candidate have the same native key as its control. Library cache source-closure admission covers paired modes. Syntax parsed; physical paired gate pending.
- Sole live consumer v5 selects paired_measure launcher9f9303ac: candidatec38235b vs immediatebefore8205a65 and historicalold4ef9ce1, S10 batch32768 HOST noarchive, full untimed BFS in each rank-process, five alternating-order repetitions. DENSE/HASH_FIRST CUB/CUCO/BMMA panels. Allsix cache downloads SHA-verified; RUNNING dependencies stage last observed. No candidate speed claim until every row/source/profile/layer/archiveoff/VRAM record is checked.
- Full goal remains ACTIVE: LSA/CPU-free owner→transport→retirement target, all agreed profiles/macro depth/archiveHF andDB/framework stage. HOST exact payload CPU sizing remains open, no maxpadding fallback. No local source/build/tests, no new rental/subagent; one Kaggle consumer.


## 2026-10-06: complete v7 paired CUB launch reduction evidence

- Consumerv5/v6 stopped before BFS on distinct cache admission errors, not search failures. Raw retained8f6c7272/3706e86d respectively. v5 fixed separate cache-reference versus runtime-control source. v6 verified recipes differ ONLY configure -B output directory; canonical frozen recipe retained, actual separate restore destination recorded. No source/dependency/compiler flag relaxation, no build-on-miss.
- Consumerv7 COMPLETE: all681outputs/7pages,679selectedsmall originals+manifest retained26b4304ac0a705adc4e583396fce9079197fefb7. summary3,116,466bytes exceeds default2MiB small-file guard, explicitly admitted8MiB summary-only bound; no state data/large traces downloaded.
- 78rows verified (13panels×[warmup+5reps]): COMPLETE,46layers sum3,628,800 equal, two rank records, archiveOFF. All native/library/control cache hits; independent candidate/control library directories. Actual sourcec38235b vs8205a65 vsold4ef9ce1, matrix_u8 S10 batch32768, HOST_SIZED_NCCL physical2T4. Search timer excludes process/build/setup and uses full BFS in-process warmup; no profiler in timing.
- DENSE median/MAD seconds, externalMiB/rank: old.482148220/.007771526/2817; beforeCUB.718394259/.011197625/913; candidateCUB.581843852/.006219330/913; beforeCUCO.555732010/.006598398/1361; candidateCUCO.550032416/.009202373/1361. CUB search time reduced19.0% without memory increase, still20.7% slower than old. CUCO difference is comparable to dispersion; no improvement claim.
- HASH_FIRST CUBbefore.839282756→candidate.727804996/829MiB; CUCOcandidate.696419698/1277MiB versusbefore.693552192; BMMAcandidate1.521090330/829MiB versusbefore1.521580994. DENSE BMMAcandidate1.384361839/913MiB versusbefore1.387047972. Do not describe tensor-owner as winning end-to-end.
- Sole current consumerv8 RUN queued launcherb7061f867d1840bb6285cbae626e9bd819f732b8 MODEpaired_timeline: old,DENSECUBbefore/candidate,DENSECUCObefore/candidate,HFCUBbefore/candidate. Profiler capture search-only after full warmup,14ranktraces expected; profiler durations never enter A/B panel. Candidate runtime unchanged.
- This evidence does not close HOST CPU sizing, actual LSA window/cancellation/initcheck acceptance, >2rank acceptance, comprehensive macro/archive/HF/DB-framework stages or large-graph Pareto gate. Full goal ACTIVE.


## 2026-10-06: paired payload production candidate

- Integrated `1f500fb2150a0c87d4b90f0527369f32cf53b614`: actual HOST runtime hash/state payload uses one exact-sized NCCL group through C ABI and Rust FFI. Shared single-/two-lane wrapper closes every opened group, settles in-progress GroupEnd even after operation errors, retains first error/cancellation. Receive-slot last-reader event, payload lifetime, variable lengths and FinalizeDepth unchanged. No maximum padding, allocations or extra NCCL caller.
- RED v8 compiled the actual unchanged production wrapper against deterministic stubs and failed on the expected missing paired API assertion. Retained in `test_results/kaggle-payload-wrapper-red-v8/`.
- Initial GREEN v9 compiled but exposed non-reset second-lane injection counters in the fixture; logs retained. Candidate `5e2b1a6d9d79d83a7208c5c3ba26da8f2625c992` fixes only fixture isolation.
- GREEN v10 `WRAPPER_GREEN`, compile/test return codes 0. Exercises empty lane participation, first/second send/recv failure, GroupEnd errors/in-progress settlement and the existing cancellation/retirement suite against actual production wrapper. Retained in `test_results/kaggle-payload-wrapper-green-v10/`. Stub proof is not GPU communication, Rust integrated build, full-state BFS, sanitizer or performance proof. Those gates remain pending for this candidate.
- Scoped actual 14-trace analysis v6 retained `e4c85e62ef1c2a366f305f1609bc663e0cb580ce`: healthy DENSE windows still contain 176 D2H 4-byte and 88 D2H 8-byte transfers across two ranks; overlap remains low. `paired-delta.json` zero launch count is an analyzer field mismatch and must not be reported as zero actual launches.
- HOST exact-size count readbacks remain. This candidate does NOT close CPU-free owner/transport/retirement or the full goal, including remaining profiles, macro, archive/HF and DB/framework scope. No speed claim before repeated paired hardware measurements.


### Paired payload full hardware gate v10

- `5e2b1a6d9d79d83a7208c5c3ba26da8f2625c992`, actual two-T4 HOST path, `HOST_VALIDATION_PASS`: 37 runtime unit tests; 20 S4 full-state oracle panels (DENSE/HASH_FIRST, CUB/CUCO/HF-BMMA, pre ON/OFF, maps 01/10); 84 fault cases across four profile/backend panels; 8 unfiltered sanitizer panels, memcheck/racecheck/initcheck/synccheck for DENSE CUB and CUCO. All 32 panels passed, zero forced cleanup.
- All 894 outputs/9 pages enumerated, 564 small outputs plus manifest retained in `38988ec54c55fbd6f4c7da98b1949cbcc2503dce`, `test_results/kaggle-paired-payload-full-v10/raw-small-evidence.tar.gz`. Large cache binaries remain at this immutable completed producer.
- New native/public-header closure compiled and cached once; NCCL dependency restored from pinned existing archive. Subsequent paired measurement uses required mode for every compiled target, no build fallback.
- Scope does not establish LSA registration initcheck, >2-rank correctness, macro completion, archival performance, HF publication or CPU-free HOST transport. Performance comparison against c382 and historical old 4ef is next, not a result yet.


## HOST mapped-fatal runtime candidate e18bc108 (2026-10-06)

- v14 actual 2xT4 scoped timeline: 7 configs / 14 rank traces, layers equal, archives off. All 220 output names on 3 pages retained; 189 small evidence entries in bb7c29fd. Cloud analysis v1 retained all 19 files in 80123f44.
- Paired payload: SendRecv 270 to 180 kernels/rank; HOST small D2H unchanged. DENSE has 364 AllReduce kernels/rank. Profile overlap is roughly 0.5% of GPU busy interval; not causal unprofiled attribution.
- Integrated candidate e18bc108 replaces HOST queue_owner_fatal_gate AllReduce with sticky local GPU ring/control + system-scope terminal signal. NCCL poll observes logical fatal; progress waiter returns without abort so runtime reports sideband failure first. LSA device vote and FinalizeDepth are retained.
- CPU production-wrapper RED v2: compiled, assertion poll==5 failed. GREEN v3: compiled and tests passed. These stubs do NOT establish physical multi-rank failure correctness.
- Next mandatory gate: full-state 2xT4 oracle, asymmetric faults, all four unfiltered sanitizers, then independent A/B and scoped timeline. Candidate is NOT accepted or performance-proven. HOST exact payload sizing still requires CPU readback. Full goal remains incomplete.


### HOST local gate v4 failure and updated candidate c64c869f

- v4 SOURCE e18bc108: runtime units 38/38; all 20 full-state S4 oracle profile/backend/pre-dedup/rank-map panels passed; DENSE CUCO full fault panel passed.
- DENSE CUB capacity-0 failed admission evidence: both ranks exited nonzero in 1.807 s, no forced cleanup, no false COMPLETE, but originating runtime error was CUDA_STATUS_14 rather than the mapped logical GPU failure class. This is NOT an accepted fault gate; remaining profiles/sanitizers were not run.
- All 533 output names / 6 pages retained; 365 small archive entries in 8888ec19. Large compiled binaries remain cloud.
- 66a8b9cb additionally refreshes local_fatal GPU output consumed by materialization, including healthy reset of stale nonzero values; real runtime regression injects 17 then checks correct reset/fatal behavior.
- c64c869f annotates failed advance/advance_archived from the existing sticky mapped GPU signal, preserving the original API cause. This failure-only read adds no healthy-batch readback or wait. Added classification regression; fault verifier unchanged.
- Next full two-T4 gate pins c64c869f with all original oracle/fault/sanitizer panels. No performance or full-goal completion claim.


## HOST local-fatal runtime gate v5 — verified 2026-10-06

Runtime c64c869fb403d755682c859eb1c72597a3ba88b9; launcher fc7a6c16735abc05732931b98390b29a09687e75. Kaggle trydotatwo/mgbfs-paired-payload-timeline-analysis v5 is COMPLETE / HOST_VALIDATION_PASS on real 2xT4. All 32 validation panels pass: 20 independent two-rank S4 full-state oracle panels (DENSE/HASH_FIRST, CUB/CUCO/BMMA as admitted, pre-dedup ON/OFF, owner maps 01/10), four fault panels with 84 case rows (including panel healthy controls), eight sanitizer panels covering memcheck/racecheck/initcheck/synccheck for DENSE CUB and CUCO. No failed case or forced cleanup in these panels. Runtime unit suite: 39 passed.

All 894 output names across 9 pages were enumerated; 565 small retained entries, including retention manifest, are published at commit 3121ad01dd8fffbaaf131b954184a57ab29776a0 in test_results/kaggle-host-local-fatal-full-v5/raw-small-evidence.tar.gz. Large compiled artifacts remain on immutable v5 cloud producer. Native cache key 58bf64f9a6e0fb2557b329dd44858cb4df6a52db26be22cb4aee75d578fac84b, archive SHA256 70b0c52b045a70ccc95e6992f30c0998733fe121a69a20116c7e2bca2ad154d8; library key cce3cbd7a81aba6a0d4fb792f7903eb26094f0ec17f5d80b6ae2ca9854717fa1, archive SHA256 61db97c48c4ad0b967a0023b6126afe0f5abab4b7cf53715a9014ceeb179fd70.

This closes the tested local sticky fatal / sideband cancellation change, not the full goal. HOST variable-payload sizing still depends on CPU readback. Scoped timeline and independent paired A/B against immediate-before 5e2b1a6 and old 4ef9ce1 remain required before any speed claim. LSA is unavailable on Kaggle; LSA registration initcheck remains unresolved. Larger workloads, all broader agreed modes, macro-depth/HF and DB/framework stages remain in scope.


## Local-fatal paired A/B v15 and scoped timeline v16 — 2026-10-06

Kaggle mgbfs-paired-s10-window-analysis v15 COMPLETE: candidate c64c869fb403d755682c859eb1c72597a3ba88b9 versus immediate-before 5e2b1a6d9d79d83a7208c5c3ba26da8f2625c992 and old native 4ef9ce1d16c8cef62fc610cd6d36e32e673e623b. Real 2xT4, S10 cycle/inverse/swap, HOST exact payload, matrix_u8, batch32768, archive off. All 78 records admitted: 13 configurations each warmup +5 alternating unprofiled repetitions. All layers/rank sums/provenance/cache isolation/archive-off/warmup/VRAM admitted. Original ad-hoc checker wrongly searched warmup filenames; actual *-ranks.warmup/rank-N.json directory records were retained and checked (all 12 old per-rank warmups COMPLETE, sums match expected, pinned/disk zero). Corrected admission is PASS without rerun or weakened requirements.

All 683 outputs/7pages enumerated; 680 retained entries at ffe2766acaca51464e8671eb5c2a9db4729129b3. Full 13-configuration report and corrected independent admission at 97bd14dab03be5129b14400ed4ffe6469b0f05f1, test_results/kaggle-local-fatal-paired-ab-v15/. DENSE CUB before median/MAD .582366329/.004555936s -> candidate .530967293/.003216350s,913MiB/rank unchanged (8.83% less median time). DENSE CUCO before .530986802/.003721381s -> candidate .517115535/.001820807s,1361MiB/rank unchanged (2.61% less). Old native .486461470/.001611729s,2817MiB/rank. Best candidate still takes 6.30% more search time than old on this workload; no universal speedup/significance claim. No durable archive timing in this panel; S10 verifies layer counts, not full archived states.

After v15 retention, one scoped Nsight job launched on same consumer as v16, launcher7db6f30cc92815613ae79b86c3f9b926df58d6e7 MODEpaired_timeline. Same runtime pair with independently cached native AND library owner artifacts; 7 configurations/14 traces expected, CUDA profiler boundaries measured search only. Analyzer11c687a7dce7d42e15eb0e99c744b76e0c2a9f7a pins both sources and distinct owner archives. This is pending; profiler times must never enter A/B samples. Full objective and HOST CPU-sized variable transport limitation remain open.


## Extended HOST owner acceptance v7 — 2026-10-07

Runtime c64c869f, launcher92bf69dd, actual two independent T4 ranks, COMPLETE/HOST_EXTENDED_VALIDATION_PASS. All46panels and86case rows pass:4DENSE-BMMA full-state oracle panels,24actual owner DAG capture panels (all6profile/owner pairs, both pre-dedup settings/maps),2missing fault panels19+23case rows,16unfiltered sanitizer panels (all4tools for DENSE-BMMA and HASH_FIRST CUB/CUCO/BMMA). Each capture has positive launch evidence for exactly2ranks.39runtime units and4pure selection-contract tests pass. No forced cleanup. Owner capture is NOT whole owner→transport→retirement capture.

All892output names/9pages enumerated,591small archive entries retained bf2f2bfd26bacb9ae3d98d9c3e04468ad9ce91ad in test_results/kaggle-extended-owner-validation-v7/raw-small-evidence.tar.gz; cache binaries/state archives staycloud. Four frozen compiled/NCCL cache inputs verified, no unchanged dependency rebuild.

Unlike prior v16 host (P2P0), actual v7 reports CUDA peer access allowed1 in both directions. This materially changes LSA eligibility, but does not prove NCCL VMM/window registration or initcheck. Next single bounded LSA gate selects NCCL_LSA explicitly, requires both-direction P2P before fixtures, has no HOST fallback, and keeps original32baseline oracle/fault/sanitizer panels. A subsequent host may differ; unsupported admission is terminal evidence, not a reason for automatic duplicate retries. Known NCCL-registration initcheck issue remains open until actual full unsuppressed result. Full objective, macro/archiveHF/DB stages remain open.


## Physical NCCL LSA baseline acceptance v8 — 2026-10-07

Runtimec64c869f, launcher48463cc3, actual2T4 P2Pallowed1bothdirections: COMPLETE/LSA_VALIDATION_PASS.32panels112case rows pass:20S4fullstateoracle panels(DENSE CUB/CUCO;HASH_FIRST CUB/CUCO/BMMA;preON/OFF;maps01/10),84faultrows4panels,8unfilteredDENSECUB/CUCOsanitizer panels(all4tools). Initcheck PASS on this exact driver580.178.04/CUDA12.9/NCCLpatchedcache configuration; this does not retroactively change older registration failures or prove arbitrary hosts.

All892output names/9pages retained411ff97bb6a602d3ee49fb8c262179bbf1ee50df in test_results/kaggle-lsa-full-validation-v8/raw-small-evidence.tar.gz,532small archive entries. IndependentadmissionPASS:64outer/replayconfiguration copies match32SHA digests, allselectNCCL_LSA, casesnoforcedcleanup. NoHOST fallback/no native rebuild. First collector incorrectlyexpected32configfilesinstead64pairs; no runtimefailure, correctedbeforepublication.

Scopedprofile command RED CPUv9 COMPLETE/EXPECTED_RED: actualhelper70f4 rejects nsys-search UNKNOWN_PROCESS_INSTRUMENTATION, test5017c90. Rawretained with implementingcommit. Existing helper now has explicit nsys-search selecting cudaProfilerApi capture and MGBFS_PROFILE_SEARCH=1; legacy nsys remains full-process, sanitizers unaffected. RemoteGREEN/scopedLSAfullBFS timeline pending; commandtest/sourcealone are notpipelineproof. Fullgoal remainsactive including macro/archiveHF/DB stage.


## Scoped LSA timeline v10 stopped at capability admission

Actual Kagglev10 COMPLETE/LSA_UNSUPPORTED_HOST, launcher c9969de3. Both CUDA peer-access queries returned allowed0, unlike v7/v8 allowed1. NoBFS/noownerpanels/noNsighttraces/noHOSTfallback.47output names retained b850d683e74cb53446f43b1c1bfc95ce287dcac1,46small archive entries at test_results/kaggle-lsa-scoped-preflight-v10/. Do NOT repeat an identical LSA allocation automatically or count source/capture leaf proof as whole pipeline acceptance. Full LSA scoped timeline has a current hardware capability blocker under no-new-rental constraint; this does not block other implementation work.

Existing frozen SQLite analyzer core868050ed is reused by source/digest-bound LSA adapter1a7bdb0a, not a parallel metric implementation. It requires actual12scopedtraces,6pairs, archiveON config/provenance and healthywindows. No actualLSAanalysis yet. The scoped commandGREEN CPU job uses existinghelper82042335 and test5017c90; it is notGPUproof. Fullobjective remainsactive.


## 2026-10-07: macro reference full-layer gate (not production macro completion)

- Kaggle `trydotatwo/mgbfs-current-scoped-analysis` v13 COMPLETE, runtime `14c9d833f3ca80fcf5d8c13d6825f3778beb6b42`. Two physical T4, each tested independently with one visible GPU. Both actual tests pass on each device: existing producer CUDA Graph capture and exhaustive S4 full-state layers/termination for K=2,3,10, pre-dedup OFF/ON, seeds 0,1,20260828. Batch=1 only in this admitted source.
- Native/library caches REQUIRED HIT; unchanged CUDA libraries were not rebuilt. Raw evidence `test_results/kaggle-macro-reference-v13/raw-small-evidence.tar.gz`, commit `b11a9d6fcef5e92b3c775870574e7fad68fb4dfd`: all44 output names retained,42 small files; large ELF/cache remain cloud. P2P=0 both directions on this allocation, not LSA timeline admission.
- v14 RUNNING at last CLI check, runtime `8b4209320a23c0056c48fb8474f1438398ce1a53`, launcher `37e8c3a8ce79f4a92926d015e9ce807e122e23f6`: adds S4/U4m2 and batch1/3/8, followed by four sanitizer tools on the exact test ELF. No result admitted yet.
- Launcher candidate `01ece25c679ba769aded8acafdf62ef8591c680a` reuses existing `instrumentation_clean` tool-specific parser. v14 was already running the prior launcher; a racecheck parser rejection must be distinguished from a genuine sanitizer finding, retaining its raw log. No warnings/errors may be suppressed to admit success.
- Production dispatch still rejects macro_depth!=1; weighted multi-GPU settlement, common state-ring allocation, group finalization/archive and scoped integrated transport timeline remain incomplete. Reference tests do not close these requirements.

## 2026-10-07: admitted v16/v17 evidence and remaining integrated gate

- Macro reference v16 COMPLETE: runtime7a9f748f, two physical T4 independently (NOT distributed macro),108 exhaustive S4/U4m2 full-BFS fixtures each plus producer capture; four sanitizer tools each device run four bounded full-BFS fixtures and capture, zero findings. All53 output names/51 small retained12c6723e. Production weighted macro dispatch remains incomplete.
- LSA scoped producer v17 COMPLETE on actual2T4, peer access1both: all6 DENSE/HASH_FIRST x CUB/CUCO/BMMA archived S8 oracle cases pass,12 search-only rank traces. Runtime7a9f748f. All194 output names/2 pages and138 small files retained8445b2a8; large traces remain in this preserved producer. Profiler elapsed times are NOT paired A/B.
- Separate private CPU analyzer trydotatwo/multigpubfs-lsa-v17-scoped-analysis v1 RUNNING at CLI check, adapter6c848e22/frozen metric core868050ed. All19 source/config/SQLite inputs verified SHA256 in logs. No aggregate result admitted yet.
- Actual runtime candidate4493303c captures prepared-packet transport-owner-retirement for first two depth3 batches. Explicitly excludes generation and archive publication; archive submission occurs before capture. GPU UNTESTED. Reuses OwnerCaptureProbe; does not introduce a second runtime.
- Remote metadata/type-check CI37541678498 SUCCESS for CUDA/library runtime and tests at75a4a31f, without CUDA linking or execution. Found and fixed preexisting ambiguous native fatal-gate import in library_multi_gpu test. Main contracts job still fails formatting, so its later CPU tests are NOT admitted.
- Next integrated launcher9304490e reuses replay with four pre-dedup/map archived full-state healthy cases; requires exact two capture markers perrank at depth3, followed by existing bounded fault suite. Prepared only, NOT executed while analyzer occupies single allowed notebook.
- HOST transport still sizes variable payload through CPU readback; no hidden padded max-buffer workaround. Whole producer/archive DAG, full fail-fast/sanitizer after new capture, causal timeline, paired A/B, production macro, HF and DB/framework requirements remain open. Goal ACTIVE.

## 2026-10-07: integrated capture is still rejected, not accepted

- Adjacent batch gate v1 actual2T4 reached CUDA905 at distributed_native.rs:4676 on rank1: empty DENSE route packet producer had no owner-origin fork. Both ranks exited in1.8565s without forced cleanup or group COMPLETE. All64 output names/58 small retained357eceec. Runtime d7a2703d adds GPU-only owner-to-producer fork before empty packet enqueue, then existing join; no host drain/count readback.
- v2 on d7a2703d moved past that failure but rank0 reported CUDA_EVENT_WAIT_905, rank1 cancelled; both exited in1.8070s, no false COMPLETE/forced cleanup. All64 names/58 small retainedeeeb42c1. Generic event error did not identify the exact wait site; do not infer proof of its cause solely from905.
- Probe ce3e9b1f now enters after existing producer ready-event import, consistent with explicitly prepared-packet scope; producer generation/archive publication remain outside capture. f10dab54 adds track_caller and failure-only event-wait diagnostic, preserving error code/protocol.
- v3 on f10dab54, launcher2d04cb2a, RUNNING at API/CLI observation. Native/library/NCCL dependency archives unchanged and REQUIRED. Independent remote CUDA/library metadata typecheck37543134651 PASS; not linking/GPU acceptance. No accepted adjacent batch graph/oracle/fault result yet. Healthy uninstrumented v17 LSA oracle evidence must not be conflated with rejected debug capture.

## 2026-10-07: v3 capability stop and current explicit HOST regression

- Adjacent batch v3 COMPLETE/LSA_UNSUPPORTED_HOST: actual2T4 peer access0both, no BFS/capture executed. All47 output names45small retainedb0c38435. No automatic identical LSA retry. f10 prepared-DAG probe remains GPU unverified.
- Separate explicit HOST_SIZED_NCCL regression v1 trydotatwo/multigpubfs-current-host-full-regression RUNNING, sourcef10dab54, launcher73b646b8.78panels planned across all6profile/owner pairs:24archivedoracle,24owner-onlycapture,6fullfaults,24unsuppressedsanitizer. Observed48healthy panels PASS and first two19casefaultpanels PASS; remaining tests not admitted. HOST sizing/readback intentionally remains, not substitute no-CPU acceptance.
- Remote CI37543702357 at e96ba495: all90 CPU test binaries zero failures; all functional metadata/ABI/memory/launcher guards pass, formatting fails unsuppressed. Full raw166581byte CI archive and perstep summary retained84096df24. Reordered existing formatting step after correctness, not changed test semantics.
- HASH_FIRST device materializer's4byte count snapshot at distributed_native.rs:4067 uses cudaMemcpyAsync kind3 device-to-device. Small correlated copies alone are not proof of CPU readback; exact direction/callsite admission is required. HOST branch4145-4149 still waits and reads exact payload counts to CPU. Full objective stays active.


## 2026-10-07 current HOST regression v1: retained and admitted

- Runtime `f10dab54fef41325d9383c174bb4f67c829288e2`; launcher `73b646b8d308069e21e2cadb5ad8b0b24b0f1ab4`. Actual two Tesla T4, driver 580.178.04/CUDA 12.9; P2P unavailable both directions. Explicit HOST_SIZED_NCCL, not an LSA fallback.
- All 18 output pages enumerated (1,733 names); 1,071 small originals and checksummed collection manifest retained at `a9747cc42e7fd26d17e7345b5eb7b24867c0535e`, `test_results/kaggle-current-host-full-regression-v1/raw-small-evidence.tar.gz`. Large cache/data remain cloud.
- 24 archived S4 full-state oracle panels, 24 owner-only capture panels, six fault suites (126 cases), and 24 sanitizer panels all PASS; no forced cleanup or failed cases in original per-panel summaries. Sanitizer original logs contain 36 zero-error summaries and 12 zero-hazard summaries (two ranks per panel), no nonzero summaries found.
- This proves these HOST correctness/fault/sanitizer fixtures only. Owner-only capture is not transport/retirement capture. HOST variable-payload CPU sizing remains. Latest prepared-packet LSA capture remains unverified after v3 capability stop; no identical hardware-lottery retry authorized. Full goal remains active.
- Prepared paired-measure launcher `d1f7082b71171e7ac149251c1768006a0707927f` pins before=c64c869 and after=f10dab54; NOT launched and no performance result claimed.


## 2026-10-07 adjacent-batch LSA gate v5 PASS

- Runtime `d1c1c405c147a90f1c28c0417b542da887648b6c`, launcher `778f3d34776f021800c56301000e7df3527fa713`; actual independent two-T4 processes, P2P allowed both directions.
- Four archived full-state S4 panels: DENSE+LSA+CUCO_RANK, pre-dedup OFF/ON and owner maps 01/10. Each rank launched both prepared-packet transport/owner/retirement captured batches at depth 3; every full-state oracle and group completion passed.
- Existing full fault suite: 19/19 PASS, no forced cleanup. This is not all profiles/macro acceptance.
- All 211 output names across three pages enumerated; 137 small originals retained at `8fcad773568597df96db7f404424a00858056e4e`, `test_results/kaggle-adjacent-batch-dag-v5/raw-small-evidence.tar.gz`.
- The capture includes prepared-packet transport/owner/retirement, NOT producer generation or host archive publication. Input/previous reader events are imported before capture; ordinary owner completion is published after graph launch. No host count readback introduced. Full BFS scoped timeline and sanitizer gates for this candidate remain pending.
- Formatter commit `345b56634756fcbbc7d4d9d23493cf6a53984529` applies exact Rust 1.75 CI formatter diffs in RAM via GitData; local dirty files untouched. CI run 37546302775 SUCCESS.
- Prepared next opt-in `lsa_batch_pipeline` suite: four capture/oracle panels, existing fault suite, all four sanitizer tools with batch capture enabled, and one full-warmup/search-scoped S8 timeline panel (two ranks). Ten panels, not performance timing. Full goal remains ACTIVE.


## 2026-10-07: six-mode physical LSA adjacent-batch gate v7 retained

- Runtime `78805a3f0a21b2988329d3dc701a83c4ea2bed41`; launcher `06ea5a0ec1233352ef08f59cad552edcc1ac7337`. `trydotatwo/multigpubfs-adjacent-batch-dag-gate` v7 COMPLETE / LSA_SCOPED_TIMELINE_PASS on two independent physical T4 ranks, CUDA peer access allowed in both directions. No HOST fallback. Native/library cache admission REQUIRED/HIT, no unchanged CUDA/C++ dependency rebuild.
- All 60 panels pass: 24 archived S4 full-state/captured prepared-packet DAG panels (DENSE/HASH_FIRST x CUB_SORT_MERGE/CUCO_RANK/BMMA_BUCKET x pre-dedup OFF/ON x maps 01/10); six full fault suites totaling 126 case rows (19 each DENSE, 23 each HASH_FIRST); 24 unsuppressed sanitizer panels (all four tools for all six mode pairs); six full-warmup/search-only scoped S8 panels yielding 12 rank traces/SQLite exports. All 180 case rows pass without forced cleanup.
- All 16 output pages / 1,538 names enumerated; 918 small original files (12,377,457 bytes uncompressed) and checksummed inventory retained in `7db098373ab52812cd63936a42cef484e21328f6`, `test_results/kaggle-adjacent-batch-profiles-v7/raw-small-evidence.tar.gz`. Large archives/cache/Nsight/SQLite remain in preserved producer v7; producer must not be overwritten before durable large-artifact handling.
- Independent retained-original check: all 48 capture/sanitizer panels contain exactly two adjacent depth-3 batch markers on each rank; 96 original sanitizer log files inspected, 36 zero-error summaries and 12 zero-hazard summaries, no nonzero summaries. Initcheck is accepted on this exact environment, not retroactive closure of older NCCL registration failures on other hosts.
- Capture scope remains prepared-packet transport -> owner -> retirement. Producer generation and host archive publication are outside that debug capture. Passing capture is not causal timeline proof of every healthy batch or a performance result.
- After terminal raw retention, existing private CPU consumer `trydotatwo/multigpubfs-lsa-v17-scoped-analysis` v2 submitted with adapter `eff69fcbb17051266b18cfed34f6dae97ed1201b`, 19 freshly signed/digest-bound summary/config/SQLite inputs and frozen metric core `868050ed8786484ff24f8240c469686c453cc528`. CLI confirms RUNNING. One notebook only; no new rental/subagent/local build or source edit. Analysis result pending.
- Full goal remains ACTIVE: actual healthy-batch timeline causality, current paired unprofiled A/B, larger workloads/PyTorch baseline, production weighted distributed macro, HF and DB/framework stage remain open. HOST variable-payload sizing still uses CPU and is not relabeled CPU-free.


## 2026-10-07: current six-mode scoped timeline evidence retained

- CPU consumer v2 COMPLETE/EVIDENCE_EXTRACTED. All19 producer summary/config/SQLite inputs verified SHA256; twelve actual rank traces analyzed with frozen metric core868050ed. All15 consumer output files retained at b84a924167d61bbd205129aa15fff1156f5ff567, test_results/kaggle-lsa-six-mode-scoped-analysis-v2/. Large SQLite/nsys remain preserved producer v7; no local large downloads.
- Each of twelve records contains149 same-thread intra-depth batch-start to next-batch-start healthy windows, including launch gap, excluding28 semantic-boundary windows. In the healthy-window CUDA API inventory no StreamSynchronize/EventSynchronize/synchronous cudaMemcpy appears. DENSE has no correlated small count/control copies. HASH_FIRST has4/64byte kind8 GPU copies; actual source device materializer uses kind3 device-to-device, not CPU count readback. This is measured interval/correlation evidence, not universal absence-of-dependencies or occupancy proof. Bounded-credit event queries and OSRT waits remain; no claim that all CPU execution disappears.
- DENSE CUCO healthy-window wall149.76/152.02ms, GPUbusy123.79/110.86ms, multi-stream38.06/21.95ms for ranks0/1. These include profiler overhead and MUST NOT be used as unprofiled A/B timings. Other five mode pairs retained in bounded-aggregate.json.
- Prepared existing launcher MODEpaired_measure pins old4ef9ce1, beforec64c869, candidate78805a3;13configurations with full in-process untimed BFS warmup and five alternating unprofiled repetitions, S10batch32768 archiveOFF HOST_SIZED_NCCL. Existing generator function is byte-identical across all three sources (SHA2567cd23304ac1db79b9eb15a30efe598cad6115c43df2f140777573b6fd1277143), original guard retained. This compares HOST paths fairly; it is NOT an LSA CPU-free performance claim. No measured outcome yet. Full objective stays ACTIVE.


## 2026-10-07: paired A/B admission and macro paired-hash gate

- Independent admission e86cc81: 78/78 retained S10 HOST rows checked, five measured repeats per configuration. Old native median 0.451284134 s / 2817 MiB per rank; current CUB 0.496243315 / 913; current CUCO 0.492507764 / 1361. No demonstrated speedup over immediate-before. Not CayleyPy or LSA performance.
- Macro runtime 5858789: full current/next hash banks swap with state banks; archive reads accepted hashes without recomputing GEMM. CPU contracts, typecheck and format CI 37551235356 pass. GPU evidence is still pending; pre-change macro v16 is not evidence for this edit.
- Existing launcher switched to macro_capture_gate on immutable 5858789 with required unchanged native/library/NCCL caches. Gate checks existing 108-case exact-layer/state oracle plus row-wise CPU hash alignment and four sanitizer tools on two independent single-rank devices. This is NOT production distributed macro admission or actual archive-row content proof.


## Macro paired hash banks: physical v3 PASS

- Kaggle multigpubfs-current-host-full-regression v3 COMPLETE on source5858789, launcher1e9d58d. Both physical T4 run the existing three-test macro producer/oracle suite, including 108-case layer/state and stored-hash alignment coverage per device. All four unfiltered sanitizer tools on each device passed; original eight logs independently checked for zero-error/zero-hazard summaries.
- All 1 output pages enumerated; 51 small originals retained in test_results/kaggle-macro-paired-hash-v3. Native/library/NCCL caches reused. This is independent single-rank macro acceptance, NOT distributed weightedmacro, HF publication or speed measurement.
- Strengthened actual archive-row oracle0619fc3 CI37552073404 SUCCESS; it was not in v3. Existing gate471ffef prepared to run matrix/compact real archive state/hash checks plain+all4san on both GPUs next. Full goal ACTIVE.


## Actual macro archive rows: physical v4 PASS

- multigpubfs-current-host-full-regression v4 COMPLETE on0619fc3, launcher471ffef. Two physical T4 independently passed the existing macro producer/108case oracle/hash pairing and eight all4san panels.
- Existing macro_native archive test now verifies actual recorded states by depth against independent CPU successor oracle, and each stored hash against CPU hash contract, for matrix variant1 and compact variant5, K3 and non-multiple batch7. No snapshot drain before archive D2H. Both GPUs passed plain+all4san (10 panels);16 original sanitizer logs independently zeroerrors/hazards.
- All 1 pages 64 names, 62 small originals retained under test_results/kaggle-macro-archive-hash-v4. Native/library cache HIT. This closes current single-rank paired-hash archive correctness scope, not distributed weightedmacro or speedup.
- No new notebook needed until next actual implementation. Production weightedmulti-rankmacro/RunConfig dispatch, canonical PyTorch comparison, HF per-run catalog reconciliation and DB/framework stage remain open. Full goal ACTIVE.


## Production weighted dispatch: actual two-rank RED v5

- Actual CLI RunConfigV1 S4 K2 DENSE CUB HOST_SIZED_NCCL on2T4, launcher00650bd/runtime0619fc3. Offline configuration valid; both independent rank stderr logs explicitly report RUN_MACRO_DISPATCH_UNAVAILABLE. Group exit 23.43275958700002 seconds, no timeout or false group COMPLETE. This is EXPECTED_RED, NOT supported macro BFS.
- All 1 output pages 51 names, 49 small originals retained under test_results/kaggle-production-macro-red-v5. No repeat of unchanged RED required.
- Required implementation must retain weighted provisional offers until target-depth settlement, 2K hash history, shared StateRing parent/provisional lifetime and existing sole NCCL issuer/cancellation/FinalizeDepth. Simply replacing generators or dispatching independent single-rank macro searches is invalid. Full goal ACTIVE.


## Production preparation RED v6 and weighted schedule integration candidate

- Actual cached Kaggle v6 reached all nine production preparation tests: seven passed, two new tests failed exactly RUN_MACRO_DISPATCH_UNAVAILABLE and RUN_ROUTE_RECORD_COUNT_UNAVAILABLE. Build/link succeeded; preexisting unused-mut and LsaView.terminal warnings remain. All one page /43 output names/41 small originals retained at9101c97b303685680a675ed02951853e59bcce09. No environment failure relabeled RED.
- Candidate prepare_production now compiles and carries the deterministic weighted MacroGeneratorSet using declared route capacity as its bounded operator budget. Actual candidate count uses compiled transitions, not base moves. Spare route capacity is not sent. Original graph and total shared StateRing capacity remain unchanged.
- Execution admission remains RUN_MACRO_DISPATCH_UNAVAILABLE in the existing coordinated startup path before CUDA/device/archive allocation. No composed operator is passed to the unit-cost owner. This is preparation integration ONLY, not production weighted BFS or a performance result.
- S4 K10 test geometry explicitly raised to32 buckets/shard: job_buckets is min(geometry,candidates), not an operator-count accessor. Independent expected operator ball counts8/23 and weight counts3/5 are also asserted directly on the carried schedule. Candidate GREEN pending.

- CI37555097484 candidate functional CPU/metadata/library checks passed; formatter alone failed. Applied the three exact Rust1.75 runner formatter hunks in RAM via GitData, no local build/source mutation. Full original CI logs retained; formatted CI still pending. Kaggle preparation GREEN v7 running on functionalb470d918, not formatting-only source.

- v7 reached9linked production tests: weighted K2/K10 schedule assertions PASS,8testsPASS,1testFAIL due wrong expectation job_buckets3 for fixture geometry2. Actual fixture has3generators but2buckets/shard; min(geometry,candidates)=2 is correct. Corrected the spare-capacity test to verify admission preserves that geometry, not confuse job width with move count. No production algorithm changed to satisfy a bad expectation. Full v7 smallraw retained here; GREEN still pending.


## Weighted production preparation GREEN v8 retained

- Cached Kaggle v8 COMPLETE runtime16e23ed41d9245cf2d4f2a68143082ed2e987fb7/launcher0c42756f737ed174df9428af749458f702c0a200. All9production preparation tests passed, including independent S4 K2/K10 operator/weight counts and spare route admission with unchanged actual job geometry. Native/library caches verified HIT. This is linked preparation testing, not execution of GPU BFS or production weighted multi-rank acceptance. Preexisting unused-mut/LsaView.terminal warnings remain.
- All1outputpage/43names/41small originals retained here. Full CPU/metadata/formatting CI37555559747 SUCCESS on the same runtime source. Guard still refuses weighted execution before allocations in existing group startup agreement.
- No live notebook remains; do not repeat unchanged preparation or production RED. Next production integration must implement provisional owner offers/shared StateRing leases and target-depth settlement before removing guard. Full goal ACTIVE; macro runtime, canonical PyTorch/larger paired benchmarks, HF catalog and DB/framework requirements remain open.


## Weighted StateRing handle / generation-permission separation

- Actual remote CPU RED CI37556081610 reached the new real MacroOwner+StateRing regression and failed at materialized-future state_ref with STATE_REF_NOT_READABLE; original wrap/stale/origin tests passed. Full original CI archive retained here. No CUDA/Kaggle gate claimed.
- Existing StateRing::state_ref now creates an absolute handle for owned Materialized bytes without publishing Current. StateRing::resolve remains unchanged and denies generation from that provisional handle. Reserved/stale/out-of-range and Enumerated-without-origin-lease checks remain enforced.
- Regression includes a farther offer allocated first and an equal-hash shorter offer arriving later; only the shorter settled state becomes Current. Its extent is one row, so publication cannot accidentally expose unselected siblings. A released near parent cannot reclaim the older live future extent, and shared capacity exhaustion remains explicit.
- This is CPU ownership-contract integration of existing MacroOwner and StateRing, not production GPU weighted runtime. Discarded-provisional reclamation, GPU descriptors/reader events and actual distributed weighted settlement remain pending. GREEN full CPU suite pending; full goal ACTIVE.

- CI37556347015 full CPU contract suite and native/library metadata gates passed, including new weighted-owner/ring regression. Formatter alone failed on one long assertion; exact pinned runner format applied here. Original complete CI archive retained. GPU descriptor/lease implementation and weighted execution remain unverified/unimplemented; CPU handle change does not close them.


## Integrated distributed weighted producer candidate (not weighted owner acceptance)

- 112fba56 CPU contract CI37556573906 SUCCESS. Existing distributed DENSE already uses the macro-capable move-major generation C ABI, with unit weights. Candidate new_profile carries an optional compiled weighted schedule, sizes existing physical route banks/hash/GEMM plans from its operator count, and supplies its real matrices/weights to the same generation plan. No CUDA dependency source changed and no new state arena introduced.
- Existing route/pack path now accepts bounded contiguous move-major row subspans; unit callers retain begin0. Weighted producer generates all bounded-batch children+hashes once, publishes existing generation_done, then routes each weight range using GPU event waits and the same sort/pack buffers. No host count/control readback or state transpose added.
- Actual CLI constructor selection is wired to this optional schedule, but startup ensure_native_dispatch STILL rejects weighted execution before allocation, and advance_inner independently refuses a weighted schedule until target-depth owner/StateRing settlement exists. Not unit-cost macro BFS, not supported production macro, no GPU producer result yet. HASH_FIRST weighted path explicitly remains unsupported, not silently coerced.
- Constructor/typecheck and real producer/route validation pending. Shared payload reader lifetime across weight ranges still belongs to the eventual integrated epoch driver; do not claim it solved from this producer preparation. Full goal ACTIVE.

## 2026-10-07 weighted production producer admission

- Existing DistributedNativeBfs now accepts a compiled shortest-weight operator schedule internally and routes equal-weight move-major spans through the existing CUB sort/pack path. No new CUDA dependency or state arena.
- Added executable GPU oracle coverage for matrix/compact states, K=2/10, pre-dedup OFF/ON and empty/partial batches. UNVERIFIED until remote execution. Test drains are observation, not evidence of production overlap.
- CLI weighted execution remains rejected before allocation; owner settlement, target-depth directories, shared-ring provisional reclamation and transport reader lifetime are not integrated. Do not label this weighted BFS completion.

## 2026-10-07 weighted producer and reader lease evidence

- Kaggle current-host-full-regression v10 COMPLETE: existing DistributedNativeBfs producer/routing independently passed matrix/compact K2/10, pre-dedup ON/OFF and parents 3/0/1/3 on both T4s (separate world=1 processes). Full original 1-page/44-name/42-small output retained df044b6. Not weighted multi-rank BFS admission.
- v9 failed before algorithm execution due missing EAGER in the test launcher, not CUDA correctness; all small raw retained285829f. Native/library/NCCL caches verified hits in v9/v10.
- v11 executable RED admits packet overwrite before reader release: unwrap_err on Ok(0,12). All 1-page/43-name/41-small raw retained9499b0e; worker ERROR, summary INCOMPLETE, no false COMPLETE.
- Added per-bank weighted generation/parent-count/order/packet-live admission and explicit owner-stream reader release. Last weight retires raw generation behind the existing last_reader GPU event; intermediate weights cannot free raw children. Unit-cost production now calls the same existing event retirement helper. No new GPU slots, CUDA library changes, count readbacks or host drains. GREEN execution pending.
- This remains a producer/reader-lifetime candidate: target-depth owner settlement, shared-ring provisional descriptor reclamation, weighted transport driver and production macro execution are still incomplete/guarded.

## 2026-10-07 weighted reader lease GREEN, exact scope

- current-host-full-regression v12 COMPLETE on source bf0f530, launcher a1e3121: 45/45 integrated CUDA/library runtime unit tests on EACH physical T4 in independent world=1 processes. Existing unit suites remain green.
- Weighted producer state/hash/packed-range oracle still passes all matrix/compact K2/10/preOFF/ON/empty/partial fixtures.
- New actual queued-reader test prefetches two raw banks, processes 64 distinct-parent batches and 128 weight packets, copies their packed states/hashes/counts on an independent GPU stream, joins reader events, and reuses banks without GPU count/control readback or host drain inside submission. Single final observation compares every output against the CPU oracle. This is device-copy transport reader simulation, NOT NCCL/multi-rank weighted BFS or a measured overlap/performance claim.
- All original small logs/summary/config/provenance retained before any next run. No native/library/NCCL rebuild; all required verified caches hit. CI formatting-only changes in afc196b; full CI37559845886 on afc196b was verified SUCCESS (CPU contracts, CUDA/library metadata typechecks and formatting; not GPU execution).
- Production macro remains guarded. Next integration is GPU target-depth provisional owner/StateRef metadata, shared-ring descriptor retirement/discard and FinalizeDepth settlement in existing runtime. No K hidden state arenas, no unit-cost reinterpretation, no CPU fallback. Full goal remains active.


## 2026-10-07 weighted shared-ring lifecycle: real RED and integrated candidate

- Runtime RED source `1d290d6f65cc75122cf71198ea01454289c87405`, actual
  T4 consumer v13: full runtime suite **45 passed / 1 failed**. Actual
  reserve/materialize produced far-depth allocation before nearer-depth
  allocations; legacy FIFO release of the latter set `ring.fatal=17`.
  This is missing weighted lifecycle support, not evidence that unit-cost
  FIFO BFS previously returned incorrect layers. All 43 output names / 41
  small original files / one output page retained at `f1ef84ce8a33ae90d3d56d5ed6e15202b2502229`.
- Candidate `ba65ea410b951427e3e8b5cd53b4dcf5466e79e8` integrates a 32-byte
  GPU descriptor registry into `DistributedNativeBfs`'s shared ring, charged
  before admission and allocated during setup. StateReady registration,
  reader-joined current-prefix retirement and provisional-depth discard
  mutate device metadata on the owner stream. Reclamation advances only over
  released allocation prefixes; older live future allocations still pin them.
  No K full-state arenas, device allocations or per-call host count reads.
- The hardware GREEN candidate is consumer **v14**, launcher
  `9ccc9a84e9cab468f52c85422915772d97844f99`, full 46-test runtime suite on each
  physical T4 in independent world1 processes. Status was launched, NOT yet
  accepted. Changed `mgbfs_cuda` gets one explicit cache-producer build;
  unchanged shared-library target inputs are checked separately from probe
  executable inputs, NCCL remains REQUIRED pinned cache.
- CI `37561524294` typechecked CUDA/library metadata without compiler errors;
  formatter emitted eight blocks, applied mechanically from retained original
  runner ZIP. This is not CUDA compilation or GPU execution evidence.
- **Open:** actual weighted owner target directories, 2K settled history,
  survivor compaction and distributed weighted transport/FinalizeDepth driver.
  The new runtime lifecycle methods are staged; their real GPU test is not a
  complete weighted BFS. `WEIGHTED_SETTLEMENT_UNAVAILABLE` guard remains.
  Actual two-rank weighted oracle, asymmetric faults, all four sanitizer modes,
  scoped full timeline and paired A/B remain required. Full profiles/macro,
  archive/HF/catalog and DB/framework scope is unchanged; goal ACTIVE.


2026-10-07 v14 GREEN verified from API and original logs: exact runtime
`ba65ea410b951427e3e8b5cd53b4dcf5466e79e8`, **46/46 runtime unit tests on EACH
actual T4**, independent world1 processes. The real weighted non-FIFO ring
reclamation test that failed with legacy fatal17 in v13 now passes. Changed
native dependency was built once; library cache closure was proven and reused,
NCCL pinned cache reused. All output pages/names and original small artifacts
retained under `test_results/kaggle-weighted-ring-reclamation-green-v14/`.
Newer `6564cdf` capacity/stale-descriptor/independent-reader capture tests were
NOT in v14 and still need GPU execution/all4san. CI37562652342 SUCCESS for that
source is CPU/typecheck/formatter evidence only. Production weighted owner,
transport and settlement driver remains open, guard retained. Full goal ACTIVE.


## 2026-10-07 v15 originals and full-driver acceptance test

- Actual consumer v15 ERROR on runtime `6564cdf`: **48/48 runtime unit tests
  passed on each physical T4**, 18.19/18.07 seconds. Independent reader DAG
  capture marker present on each; native/library verified cache hits.
  Independent world1 tests are NOT two-rank weighted BFS acceptance.
- Error occurred afterward in the launcher: Cargo JSON target name is
  `mgbfs-runtime`, not the assumed `mgbfs_runtime`. Executable existed and was
  freshly reused; no sanitizer gate executed. Launcher fix `67dbcdc` checks
  the actual name plus lib target and test profile, preserving unique identity.
- All one output page / 47 names / 45 small original files retained at
  `4b994da4934f8af3f8e03f5bc29e23d049e0f9fb` under
  `test_results/kaggle-weighted-ring-capacity-capture-v15/`. No binary archive,
  SQLite, or large state data downloaded locally. Local dirty tree unchanged.
- Added real public `DistributedNativeBfs::advance` full-state integration
  test in `4dd59e6`, formatted `5b46459`: original S4 CPU exact layers,
  macro depth 2/10, pre-dedup OFF/ON, current state sets and termination.
  This test is NOT hardware-tested; current weighted settlement guard remains.
  It must fail for missing settlement, unit-cost reinterpretation, or a future
  arrival suppressing an earlier shortest-path discovery. Do not remove the
  guard or weaken the oracle to turn this green.
- CI37565375888 CPU/contracts/CUDA-library metadata typechecks passed;
  formatting alone failed, original runner ZIP retained. Formatter correction
  CI37565784420 verified SUCCESS. Neither is CUDA execution evidence.
- Recovered thin HTTPS control using Node TLS1.2 and explicit gzip/deflate/br
  response decoding after repeated Python TLS EOF failures; verification of
  certificates remains enabled. Signed URLs/credentials remain RAM-only.
- Next accepted work is actual target-specific hash/StateRef owner tables,
  2K settled history, shared-ring survivor settlement and weighted
  owner/transport/retirement driver. Component gates and documentation do
  not close that work. Full objective remains ACTIVE, all profiles/macro,
  archive/HF/DB, physical two-rank faults/sanitizers/timeline/A-B still required.


## 2026-10-07: weighted integration gate dependency correction

- v17 authoritative ERROR, runtime 77d12d8; NO runtime tests executed. Exact failure: required mgbfs_library_owner cache miss. All output pages and 32 small originals retained in ab6b1b1b5784a1cf86f7f18429a33756a6b10282 under test_results/kaggle-weighted-driver-red-v17.
- Confirmed launcher defect: weighted_driver_red omitted nvtx_enabled, unlike cached builds. SDK fingerprint had 1563 files vs 1580 in v16, and native configure omitted NVTX flags. f33f6e316a4d65d5687cfb1c06db5518249f901e adds this mode to existing NVTX configuration, preserves admission and required-only caches.
- API accepted sole consumer v18, confirmed version18 QUEUED. Runtime unchanged 77d12d8; launcher f33f6e3. New driver RED is not yet observed. No rebuild fallback, no additional notebook, no local source changes.
- Next acceptance: cache HIT and exact two expected integration failures on both physical devices, followed by real owner provisional commit and original-depth driver fixes. World1 tests on two devices are not a two-rank weighted BFS gate. Full goal remains incomplete.


## 2026-10-07: actual rank-owner provisional registration validated

- v18 EXPECTED_RED observed both physical T4,50 tests/device:48PASS and two exact missing-driver failures; all44small originals/allpages retained3ad753c0d64bc5d060cf5dc7137762c6415aa801.
- Runtime2684188/fbf740b uses packet-specific target_depth, registers StateReady provisional extents after existing native-rank materialization and excludes them from unit-cost next extents. No host count/control read added, no mutable global target. Ordinary unit-cost dispatch passes None.
- CI37657744167 SUCCESS for formatted runtimefbf740b. Initial CI37657355737 typechecks passed but rustfmt failed; raw logs retained in fbf740b, exact format applied.
- v19 authoritative COMPLETE with summary EXPECTED_RED:49PASS/1FAIL on EACH physical T4; owner registration test passes, full original-depth advance still deliberately fails WEIGHTED_SETTLEMENT_UNAVAILABLE. All small originals retained alongside this ledger. This is world1/device execution, not physical two-rank weighted BFS, all4san of this change, or full weighted production acceptance.
- Next actual integration: preserve Hash128-StateRef association through sorted owner merges, target-depth-specific owner tables, settled-history window and shared-ring settlement. Full goal remains ACTIVE/incomplete; guard stays until true driver correctness.


## 2026-10-07: integrated StateRef owner merge candidate

- v20 EXPECTED_RED confirmed on both physical T4:48PASS/2FAIL; original logs explicitly fail OWNER_STATE_REF_LIVE after real rank-owner commits. All44small originals/all output pages retained in15e2ed480ff1061e3059dc2aa3c7773515634c35 under test_results/kaggle-weighted-state-ref-red-v20.
- Runtime15e2ed4 adds StateRef-carrying template specializations to existing merge_tiles/publish and a bounded additive rank_commit_refs C ABI. Weighted runtime uses it after existing GPU reservation; new references are reserved monotonic sequence + exact output_offset/survivor row, old references follow stable key merge. One shared preallocated reference scratch, charged before admission; no full-state arena, device allocation, D2H count or host drain added. Unit-cost path keeps the non-reference specialization.
- CI37660395914 SUCCESS, meaning Rust/CPU/typechecks/format only; no CUDA execution claim. StateRef live/correspondence oracle currently exercises two small S4 commits, not large multitile stress or all weighted profiles.
- Sole consumer v21 confirmed API RUNNING: source15e2ed4, launcherc8cdde588e3a80e661ae6e808c7e90174c91d7f9. Explicit COMPILED_CACHE_MODE=publish for changed CUDA/library closure; NCCL remains required cached. Old immutable cache producers are untouched. No implicit required-mode build fallback.
- v21 requires50tests/device with49PASS and only the known full-advance guard failure. Then the reused sanitizer runner executes6 weighted tests/device under all4 tools without kernel filtering, skipping only the still-unimplemented original-depth advance test by exact name. Even a clean result is not physical two-rank weighted BFS, full owner/transport timeline, performance, or universal NCCL initcheck acceptance.
- After terminal v21 collect all small originals/pages before overwrite; new caches must be preserved as immutable cloud inputs before dependent launch. Next real implementation remains target-specific owner key/ref/count tables, settled-history window and shared-ring settlement, then full transport/finalization/archive profiles. Do not remove WEIGHTED_SETTLEMENT_UNAVAILABLE until this is truly implemented. Full goal ACTIVE/incomplete.


## 2026-10-07: v21 StateRef evidence; v22 target-depth RED live

- v21 terminal COMPLETE / summary EXPECTED_RED, runtime `15e2ed480ff1061e3059dc2aa3c7773515634c35`: full50 runtime tests on each physical T4, 49 passing / one missing original-depth advance. All eight sanitizer panels passed six weighted tests. This does NOT close production weighted BFS or distributed NCCL initcheck.
- All 57 output names / 55 small originals retained in `c4158a688dfc2865adeb957b92f597d66ed874e6`; all eight original sanitizer logs checked for six passes and zero errors/hazards. Large cache archives stayed cloud-only.
- New immutable CPU mirror `trydotatwo/multigpubfs-weighted-stateref-immutable-cache` v1 id137521199 COMPLETE; both archive SHA256 match v21. Original prior cache producers preserved. Mirror receipt retained by `437daeb21cc4c13cd2da411e9fcfa590b4ce6bf5`.
- `18a3d34` CI failed only the exact rustfmt diff in the target oracle. `165476df9c3c3043de68d1a82c1b2ebf76128460` applies it, retains raw CI logs, and CI completed SUCCESS. Not GPU acceptance.
- v22 consumer `trydotatwo/multigpubfs-current-host-full-regression` id137399530 RUNNING confirmed via CLI; launcher `437daeb21cc4c13cd2da411e9fcfa590b4ce6bf5`, runtime `165476df9c3c3043de68d1a82c1b2ebf76128460`. Both native/library and NCCL caches REQUIRED; no CUDA/header changes. Expected 48/2 failures: missing original-depth advance and `WEIGHTED_TARGET_OWNER_KEYS` in actual owner commit. Sanitizers intentionally not repeated on known RED. Await actual failure before target dispatch implementation.
- Open: target-specific accepted/length/count/StateRef dispatch and depth tags, settled-history and correct original-depth driver, later weighted transport/HASH_FIRST and full objective gates. No goal-completion or speed claim.


## 2026-10-07: observed target-depth RED and integrated owner candidate

- v22 stopped before tests as UNSUPPORTED_HOST (P2P disabled). Originals preserved in `91cd0164750c8c263c02820d97e3ef5f03d80a0b`. P2P remains required for LSA; independent one-rank weighted tests on each physical T4 no longer require it.
- v23 observed EXPECTED_RED: both T4 full50 tests 48 passing, two named failures; target1 owner table empty versus three expected CPU successors. Both CUDA/native and library cache admissions REQUIRED/HIT true; no hidden compilation fallback. All44 names/42 small originals retained in `8045d89aeae2c5da001878fdb4a3c2827bc304ec`.
- Runtime `8045d89aeae2c5da001878fdb4a3c2827bc304ec` connects target-specific flat keys/lengths/layer-count/StateRefs slices to existing compare/reserve/commit; sole scratch/one StateRing, no added readback/allocation. Sticky target tag rejects alias of another live target. Target release deliberately absent until original-depth settlement is implemented; weighted advance guard remains.
- v24 RUNNING, source8045, launcher `c59141e794879c8ea53affc9e8042401d68ef74e`, expected target owner oracle PASS, only full-advance missing, eight weighted sanitizer panels requested. Not yet verified and not two-rank weighted BFS acceptance.
- CI37664582501 failed only exact rustfmt lengths-pointer wrapping. Raw CI archive retained with formatting correction; this does not change the running v24 source/semantics.


## 2026-10-07: v24 target-specific owner GPU acceptance (limited scope)

- v24 terminal COMPLETE / summary EXPECTED_RED, source8045d89; two physical T4, full50 tests per device 49PASS/1FAIL. Target-specific owner keys/StateRefs oracle passed; original-depth public weighted advance still fails explicitly WEIGHTED_SETTLEMENT_UNAVAILABLE.
- Eight original sanitizer panels, six weighted tests each: memcheck/racecheck/initcheck/synccheck PASS on both devices, zero errors/hazards. This scope excludes missing advance and does NOT close full two-rank NCCL initcheck or production weighted BFS.
- All output pages and all small originals retained with collection names/bytes/SHA256 in this commit. Both native/library archives REQUIRED/HIT true; frozen source15e CUDA reused. No local large download or duplicate run.
- Remaining implementation: integrated original-depth weighted driver, compact key+StateRef settlement, 2K settled hash history, shared-ring promotion/discard/retirement, pending-depth termination, transport/profile propagation and broad goal gates. Do not remove guards based on these six tests.


## Integrated weighted driver candidate / v25 (2026-10-07)

- Runtime c9d5bbafcffaad105ddf76a9bde57a802d50b9bb: existing DistributedNativeBfs connects producer, target owner commit, reader-ordered retirement, compact/settle and shared-StateRing promotion. First concrete path is world=1 DENSE native rank owner. Multi-rank, HASH_FIRST and library owner remain explicitly rejected before mutation; full goal remains open.
- GitHub CI 37668162901 completed SUCCESS: Rust formatting, type-checks and CPU contracts only. This is not CUDA execution proof.
- Launcher 6010069543fadf7ce45fd74026fda3d8e7347688 removes the expected-RED acceptance and requires all 50 runtime tests on each physical T4 plus seven weighted tests under memcheck/racecheck/initcheck/synccheck, with no skipped public advance.
- Sole notebook trydotatwo/multigpubfs-current-host-full-regression v25 SAVE_AND_RUN_ALL accepted; authoritative CLI RUNNING. No outputs available at inspection. No second notebook launched.
- CUDA/header closure changed: compiled-cache publish is explicit for this producer. NCCL remains required from unchanged pinned v131 artifacts; no NCCL rebuild fallback.
- Pending: actual CUDA compilation and full-state/all4san results, archive-enabled weighted driver, physical two-rank weighted transport, other profiles, scoped timeline and paired A/B. No COMPLETE or performance claim.


## Weighted integrated driver failure and candidate repair

- v25: CUDA compilation succeeded, full suite 49/50; all small originals retained in 11224ee50e05ac66bdcfce9954a31fcfabc773c5. Sanitizers not reached.
- v26: required native/library compiled caches admitted; full suite 49/50. K=2 depth=1 pre=false gives Ring.fatal=28 at first promoted-current retirement. Full archive fixture was active; durable success not reached.
- Static invariant: weighted retirement validates extent.padding[1] == descriptor; weighted promotion had not populated this generation tag. Candidate 566960c0b9a029241d4a679f0c3b87e86089c88b publishes the tag with GPU D2D copy before registering current, retaining stale-extent checks. CPU-readback is not added to healthy batches.
- Candidate still requires GPU oracle/archive/all4san acceptance; multi-rank weighted transport and complete project remain open.


## v27 scoped generation-tag verification

Runtime 566960c: device0 full50 PASS, weighted public advance K=2/10 pre-dedup OFF/ON completed exact original S4 layer states, recomputed archive hashes, checksums and durable RunCommit. Launcher rejected the successful suite because archive timings split test-name and ok stdout; corrected in 05b4e8. Device1 and all4san were not executed. This is not physical two-rank or full project acceptance. Bounded archive candidate 7b8a547 still awaits GPU verification.


## Verified weighted driver v28 — 2026-10-07

- Source `7b8a547d361b1f270acd76d7a191f5281961dd41`; raw retained in `6dc644f48c4f3603750f657523282638fdcc7374`, all 53 output names, 51 small originals.
- Actual logs: full runtime 50/50 on device 0 and device 1; seven weighted tests under all four sanitizer tools on each device; zero errors/hazards and memcheck zero leaks. Racecheck 259.27/259.70 seconds.
- Scope is independent world1 DENSE/native processes, not weighted two-rank acceptance or closure of the NCCL registration initcheck limitation. CUB/K2,K10/pre OFF,ON archival full-state oracle is covered; later seeds/BMMA/production-dispatch changes are not covered by v28.
- Next immutable launcher pins dispatch-enabled `4b90eb78ad3afc9a0df163e45c3e04e02cddf7b9`; 24-case internal matrix plus 48 real CLI archive-oracle runs. Racecheck timeout is scaled to 3600 seconds for 6x case count; no skipped test or sanitizer suppression.
- Remaining integrated weighted transport must share the unit-cost exchange/owner issue order, retain one physical receive slot, and retire only after all weighted packet readers. World2 remains rejected until implemented and real two-rank oracle/fault checks pass.


## Integrated weighted multi-rank candidate — 2026-10-07

- Runtime candidate `8e9b092e7cfd834c9df953b48d9aa1101b870b43` extracts the existing unit-cost prepared-packet peer exchange/owner DAG into one shared method and calls it with immutable target depth for each weighted packet. No second transport implementation, CUDA closure change, new payload allocation, max-buffer padding, or late device allocation.
- Empty/unequal ranks now follow one agreed depth-boundary schedule, use the existing zero-parent producer, and participate in all weight/peer epochs. Raw-bank release follows all packet readers; parent weighted retirement follows every weight and archive last-reader event. Promotion directory uses the logical owner; current/pending termination is group-wide at FinalizeDepth.
- Prior actual two-rank RED is retained at `test_results/kaggle-production-macro-red-v5`: valid preflight, both processes refused missing weighted dispatch, no group COMPLETE. New code is a candidate, NOT GPU-admitted. Linux integrated CUDA/library type-check in CI 37673617296 passed; remaining CI/format status must be checked.
- Existing replay helper now admits positive weighted depths rather than rejecting them in the harness. The next launcher uses that same helper for actual two-rank archived S4 oracle, K2/K10, both native owners, pre ON/OFF, both maps, shared failure cases, and all four unfiltered sanitizer tools. Initcheck registration errors remain failures, never suppressed.
- Current sole Kaggle v29 remains pinned to prior world1 production runtime `4b90eb7`; it does NOT validate this multi-rank candidate. Next launch must wait for terminal retention of all v29 outputs and successful source checks. Full HASH_FIRST/CUCO weighted support, scoped timeline, performance, HF and framework stages remain open.


## Integrated weighted candidate remote checks — 2026-10-07

- Formatted candidate `2ff94c21d7de8ddedc8c424d35edd559517fe85e`: CI `37674274778` SUCCESS, including CPU workspace contracts, CUDA/library integrated type-check, CLI launch-contract type-check, and Rust formatting. Earlier `37673617296` failed formatting only; its raw ZIP is retained. No GPU or performance claim follows from this CI.
- Next two-rank launcher `3cafbf7679ece626688b6616b3eea5afbd8af4c0` pins that exact runtime and the weighted-admitted existing replay helper. S4 does not falsely require within-depth three-bank reuse; separate S5 exact-oracle case does.
- Sole live notebook `trydotatwo/multigpubfs-current-host-full-regression` v29, prior runtime `4b90eb7`, remains RUNNING by API/CLI. Its live log shows weighted memcheck 7/7, zero errors/leaks, and racecheck progressing beyond 600 seconds for the expanded 24-case matrix. This is live partial evidence, not terminal all4san or production-CLI acceptance. No second launch.


## Weighted world1 v29 retained; integrated two-rank v30 launched — 2026-10-08

- v29 COMPLETE; all four output pages (342 names), all 292 small originals retained in c5a911b. Secret-pattern scan zero hits; large outputs stayed cloud. CLI credential expired, authenticated GitHub connector used for non-force GitData.
- Raw logs confirm full50/50 on each physical T4 (31.88/31.78s), and all four sanitizer tools on each device: seven weighted tests each; zero errors/hazards, no suppression. All 48 production CLI S4 archive full-state oracle cases passed: CUB/BMMA, K2/K10, pre OFF/ON, three seeds, both physical devices. This is independent world1, NOT two-rank acceptance.
- Sole consumer trydotatwo/multigpubfs-current-host-full-regression v30 saved and GetKernel exact-source verified. Launcher 3cafbf7679ece626688b6616b3eea5afbd8af4c0, runtime 2ff94c21d7de8ddedc8c424d35edd559517fe85e, MODE weighted_multi_rank_gate. Required immutable caches refreshed URL only; frozen image unchanged. Actual two-rank oracle/fault/all4san results pending, no performance claim. Full profiles/HF/framework/scoped Nsight/A-B scope remains open.


## Weighted two-rank v30 verifier dependency failure — 2026-10-08

- All 57 names, one output page, 51 small originals retained under test_results/kaggle-weighted-multi-rank-v30; secret-pattern scan zero hits. Required compiled caches HIT.
- First actual HOST two-rank healthy BFS exited [0,0] in 1.812s and published group COMPLETE, but verifier could not execute: ModuleNotFoundError pyarrow. This is NOT full-state correctness acceptance. Later panels were not run.
- Harness dependency inventory omitted weighted_multi_rank_gate from existing pinned pyarrow==19.0.1 installation. Added mode to same dependency function; no runtime/source or CUDA cache closure change. Rerun all panels, do not reinterpret missing oracle as PASS.


## Actual weighted two-rank v31 correctness — 2026-10-08

- v31 COMPLETE, exact runtime 2ff94c21d7de8ddedc8c424d35edd559517fe85e, launcher b6a3736. Both HOST_SIZED_NCCL and real P2P NCCL_LSA ran on two physical T4s, independent rank processes. Summary WEIGHTED_TWO_RANK_CORRECTNESS_PASS; 44 panels / 80 cases, all pass.
- Full canonical S4 state sets at every original depth: CUB/BMMA, K2/K10, pre OFF/ON, both rank maps. Separate S5 exact full-state bank-reuse cases passed on both transports.
- 36 asymmetric injected faults across startup/constructor/late constructor/owner/admission/archive write/sync/finish/capacity completed in 0.703–2.007 seconds, no forced cleanup and no false group COMPLETE. All four unfiltered sanitizer tools passed actual two-rank healthy BFS on both transports. Initcheck result is this exact patched NCCL/environment only, not universal registration closure.
- All nine output pages / 837 names, all 555 small originals retained; secret-pattern scan zero hits, large outputs remain cloud.
- This admits tested DENSE/native weighted correctness, not weighted HASH_FIRST/CUCO, scoped critical-path CPU independence, performance, HF or framework completion. HOST variable-payload CPU sizing remains explicit. Next: scoped full-BFS weighted timeline using existing replay profiler; fix measured integrated dependency before A/B.


## Weighted scoped full-BFS traces v32 — 2026-10-08

- Consumer v32 COMPLETE, runtime 2ff94c21d7de8ddedc8c424d35edd559517fe85e, launcher 488bbed. Six actual two-rank panels passed full-state archive oracle: S6 K2/K10 batch8 and S8 K2 batch128 on HOST and LSA. Original-depth layer sets equal; no forced cleanup. These profiler wall times are NOT A/B benchmark timings.
- Twelve per-rank Nsight reports exported to SQLite, capture scope search_after_full_warmup; full producer/owner/transport/retirement/archive runtime, not just prepared packet fixture. Trace files remain Kaggle cloud.
- All two output pages / 186 names, 130 small originals retained; secret-pattern scan zero hits. No claim of healthy-batch CPU independence or overlap until SQLite critical-path analysis. Next cloud analysis then measured integrated fix; do not rerun unchanged gates.


## Weighted scoped timeline analysis v2 — 2026-10-08

- CPU analyzer v2 COMPLETE; all 16 small originals retained, zero secret-pattern hits. Existing weighted_batch NVTX label normalized to batch in the frozen analyzer SQL only; actual runtime unchanged.
- S8 K2 LSA: 149 healthy intra-depth batch-start windows per rank, 28 boundary windows excluded. No Synchronize or synchronous cudaMemcpy APIs and no correlated small D2H copies in those healthy windows. EventQuery remains (108/rank) for bounded credits. This supports the tested healthy batch readback/drain gate, not absence of all CPU dependencies.
- S8 K2 HOST: 894 synchronous memcpy calls per rank in healthy windows, 596 four-byte and 298 eight-byte D2H counts. HOST remains explicitly host-sized, not CPU-free.
- S8 K2 LSA measured healthy-window GPU union 164.11/175.56ms, multi-stream union 9.23/13.59ms; CPU kernel-launch API 131.67/127.06ms for 18,912/18,922 launch calls. Inclusive profiler API duration is not recoverable critical-path time, and these are not unprofiled A/B timings. Low overlap and launch density are concrete next optimization targets.
- Existing CUDA Graph owner support is debug acceptance probe only, creates/destroys per probe; production does not reuse graphs. Next integrated production owner graph replay must preallocate bounded graph instances keyed by stable target/bank addressing, preserve fatal/cancel/event-reader protocol and dynamic GPU counts, and be GPU-oracle/fault/sanitizer/timeline checked. No claim that the prospective optimization already wins.


## 2026-10-08: production owner graph replay candidate (not GPU-admitted)

- v34 real two-rank S4 K2 / batch 1 / HOST DENSE CUB archives passed full-state oracle: 24 states, layers [1,3,5,6,5,3,1], both exits 0. The new integration assertion then failed as intended: PRODUCTION_OWNER_GRAPH_NOT_USED:rank=0. This is the observed RED for missing graph replay, not a correctness regression. All 59 output names / one page / 53 small originals retained under test_results/kaggle-owner-graph-red-v34; secret scan zero. LSA panel was not reached.
- Candidate extracts metadata/reserve/commit/materialize into the same native owner tail. Explicit MGBFS_OWNER_GRAPH_REPLAY=1 pre-instantiates and uploads a bounded target-slot × input-bank graph set before startup agreement/depth zero. There remains one remote receive slot. Device counts/reservations remain dynamic. Compare's history/count/depth arguments and absolute weighted extent registration stay outside replay to avoid stale captures.
- Pointer-lease mismatch, unsupported native/HASH_FIRST/library shape, capture/upload failures and untouched-VRAM shortage reject the requested mode; no runtime fallback. Existing transport, reader events, cancellation and FinalizeDepth are unchanged. Injected owner capacity is armed before capture.
- Graph driver allocation is opaque: separate observed startup free-VRAM delta is emitted, not falsely included in byte-exact state allocation totals. Candidate opt-in only, direct mode remains available. Source compilation, graph reuse, oracle, asymmetric faults, all four sanitizers, scoped launch-density/overlap and paired A/B remain UNVERIFIED for this candidate. No speed claim or full-goal completion.


## 2026-10-08: production owner graph first real two-rank GREEN

v35 COMPLETE / PRODUCTION_OWNER_GRAPH_REPLAY_PASS on real 2xT4, runtime bf86ae9339e799af49515685dbfa646d6a79f7b4. Both HOST_SIZED_NCCL and NCCL_LSA S4 K2 batch1 DENSE CUB passed full canonical-state archive oracle: 24 states, layer sizes [1,3,5,6,5,3,1], rank exits [0,0], no forced cleanup. Each rank instantiated 8 startup graphs and launched them 80 times; late instantiations 0, observed graph startup free-VRAM delta 2 MiB/rank. This is bounded correctness/reuse proof, NOT performance or broad fault/sanitizer acceptance. Formatted equivalent source 378e073 CI37778558497 SUCCESS (CPU/typechecks only). All 1 output pages / 73 names / 63 small originals retained under test_results/kaggle-owner-graph-green-v35; zero secret hits. Larger profiles, faults/all4san, scoped full BFS launch-density/overlap and paired A/B remain open.


## 2026-10-08: production owner graph full weighted gate v36

Runtime 378e073e2193905f7d5a337bf921e75980eb2caa, launcher e9e8fc5d87cae684f22f34ac06dd434134dfd7d8. Real independent ranks on 2xT4, HOST and LSA: 44 panels / 80 cases PASS, including CUB/BMMA K2/K10 pre OFF/ON and rank maps, S5 bank reuse, 36 asymmetric startup/owner/capacity/archive failures. Both processes terminated without forced cleanup; fault cases 0.653–2.057 seconds, no false group COMPLETE. All four unfiltered Compute Sanitizer tools passed healthy full-state cases on both transports; initcheck proof is only this patched NCCL/driver configuration, not universal. Every healthy panel checked actual startup graph/replay receipts. All nine output pages, 837 names, 555 small originals retained under test_results/kaggle-owner-graph-full-v36; secret scan zero. This is correctness/failure/sanitizer evidence, NOT performance. Next scoped full-BFS trace repeats the six pre-graph v32 workloads with graph replay to measure launch density and overlap; profiler timings are not paired A/B.


## 2026-10-08: owner graph scoped trace and paired measurements retained

v37 produced three HOST full-state oracle panels and six scoped traces; LSA was explicitly unsupported on that host, not retried. All 117 names / two pages / 91 small originals retained in 4c4e374. Cloud analyzer v3 retained all ten originals in a913d596; no large SQLite downloaded locally. Same healthy HOST windows show 46–48% fewer individual kernel launches, but unchanged count readback (S8 894 synchronous copies/rank) and negligible overlap. This does not prove critical-path speedup.

v38 actual two-rank HOST and LSA paired graph OFF/ON, five repeats each for S6 K2/K10 and S8 K2, full process warmup and archive enabled: 60 cases passed canonical full-state oracle; all 120 measured rank records COMPLETE/archive_enabled/warmup_completed. All eleven pages and 823 small originals retained in a9986d2; report 3b801ff. Search maximum-rank median OFF/ON: HOST S6K2 .400793/.414680 s, S6K10 1.567444/1.568580 s, S8K2 1.110735/1.126301 s; LSA .153687/.154665, .322385/.320067, .281029/.275705 s respectively. No broad speedup; graph remains opt-in, not default. Observed setup/final cudaMemGetInfo consumption rises 2–6 MiB/rank, NOT external sampled true peak. Durable measurements/MAD and limitations in report. Do not rerun unchanged graph benchmarks.

Full goal ACTIVE. Weighted HASH_FIRST guard WEIGHTED_HASH_FIRST_NOT_READY and weighted driver non-DENSE/library guard WEIGHTED_DRIVER_MODE_NOT_READY remain actual implementation gaps; allprofiles/HF/CUCOweighted/HF/framework scope not closed.


## 2026-10-08 weighted CUCO integrated candidate (not GPU accepted)

Candidate `90fe1f19f43630bd2e5208178cdc47502ac315b3` connects fixed
per-target CUCO membership and matching accepted Hash128/StateRef records to
the existing packet transport, shared StateRing, original-depth settlement,
promotion and FinalizeDepth reader/reset protocol. GPU export sorts keys and
StateRefs together, bounded by settled layer capacity. No K full-state arenas,
healthy-batch count readback, or maximum-payload transport substitution.

CI run `37811077821`: CPU contracts, native/CLI metadata typechecks and
CUDA-library runtime metadata typecheck passed. Formatting alone failed.
The accompanying change applies the 16 exact rustfmt log hunks, without
algorithm changes. Original contracts job log retained under
`test_results/ci-weighted-cuco-90fe1f1/`.

This does NOT compile the CUDA implementation, link the new C ABI, validate
GPU correctness, faults, sanitizers, performance or memory peaks. The library
CUDA closure changed; old immutable library cache is incompatible with the
new symbols. Next: new immutable library cache build, exact-pinned two-rank
weighted library oracle gate, then asymmetric faults/all four sanitizers,
scoped timeline and paired A/B. Weighted HASH_FIRST and library owner graph
remain explicitly unsupported. Full goal remains ACTIVE.


## 2026-10-08 weighted CUCO first two-rank full-state GREEN

Kaggle consumer v42 COMPLETE, runtime `1ad25817f67c585d2b19ff6a1dd3452c0e16475e`,
launcher `f3a0cee5100226167d61e77b4ae3fddd6bee4d8d`. Actual independent
rank processes on two physical T4s passed S4, macro depth 2, batch 1,
DENSE CUCO_RANK, pre-dedup ON, rank map 0,1, both HOST_SIZED_NCCL and NCCL_LSA.
Canonical state sets match CPU oracle at every original depth: 24 states,
layers [1,3,5,6,5,3,1]. Both ranks exit zero, group COMPLETE, no forced cleanup.
Empty source patch SHA e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.
This is a narrow correctness GREEN, not all-profile/fault/sanitizer/performance
acceptance; process-wall diagnostic durations are not search benchmarks.

v41 compiled both changed CUDA libraries and passed ldd. Its 49/50 runtime
suite failure was the stale CUCO rejection contract assertion, corrected in
1ad25817 while retaining CucoIndexed rejection coverage. New immutable cache
producer `trydotatwo/multigpubfs-weighted-cuco-immutable-cache` v1 copied both
archives cloud-only and verified SHA256. Its receipt is retained under
`test_results/kaggle-weighted-cuco-immutable-cache-v1/`. v42 required both
exact archives (cache HIT), no CUDA rebuild. All v42 pages and small originals
retained under `test_results/kaggle-weighted-library-green-v42/`; large
archives remain cloud-only.

Next: expanded K2/K10/pre-dedup/rank-map/reuse full-state matrix, asymmetric
startup/owner/capacity/archive-finalize faults, all four unfiltered sanitizers,
scoped full-BFS timeline and paired A/B. Weighted HASH_FIRST and library graph
remain unsupported. Full goal ACTIVE.


## 2026-10-08 weighted CUCO capacity fix and full two-rank gate

- Runtime 6f35c433ebd3269da45c8569fba1424f4e20a1d6 fixes logical export capacity versus preallocated physical capacity. v43 rejected reduced capacity on host and returned generic ABI -1; GPU prefix overflow guard now handles the reduced budget. Fault matcher unchanged.
- Kaggle v44 COMPLETE: real two independent rank processes on 2xT4, DENSE CUCO_RANK, HOST_SIZED_NCCL and NCCL_LSA. 28 panels / 62 cases: S4 K2/K10, pre OFF/ON, rank maps 01/10 exact full-state oracle; S5 bank reuse; 36 asymmetric faults; all four unfiltered Compute Sanitizer tools per transport. All pass. Fault exits bounded 0.7034–2.1092 s, no forced cleanup or false group COMPLETE. Capacity on both fault ranks reached expected failure.
- All seven output pages / 617 names / 427 small originals retained at test_results/kaggle-weighted-library-full-v44 (8b0600f). Large compiled artifacts remain cloud-only; distinct immutable capacity cache producer v1 submitted after retention, no old producer overwritten.
- This is correctness/fault/sanitizer evidence, NOT performance or universal NCCL initcheck closure. HOST variable payload still CPU-sized. Weighted HASH_FIRST, library owner graphs, remaining profiles/HF/framework scope and scoped CUCO timeline/paired A/B remain open; full goal ACTIVE.


## 2026-10-09 v48 paired weighted owners

All 60 cases/120 rank records pass full-state oracle, full process warmup and archive-enabled contract. Five alternating repeats per owner/workload/transport, no profiling or graph. Raw all 11 pages/823 small originals retained 827b131 in test_results/kaggle-weighted-owner-paired-v48. Report contains search/durable median/MAD and setup/final observed VRAM. CUCO modest LSA S6 gains (~6.1% K2, ~3.2% K10), LSA S8 ~1.9% slower; HOST near parity or ~1.6% slower. No broad speedup or true-peak claim. Full goal ACTIVE: weighted HASH_FIRST, library graph, remaining profiles/HF/frameworks open. Next implement integrated missing weighted HASH_FIRST path with real RED→GREEN evidence; don't repeat unchanged paired gates.
