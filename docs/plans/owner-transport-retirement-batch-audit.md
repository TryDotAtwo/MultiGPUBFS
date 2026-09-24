# Owner -> transport -> retirement: batch audit

Status: static source audit of `e2d4eec` (2026-09-24), supplemented by the
separate one-T4 v66 diagnostic profile and two-T4 v52 leaf test at `ae3dce9`.
This is a change plan, not a completed asynchronous runtime. Keep source
deductions separate from measured evidence.

## Contract to preserve

`ARCHITECTURE_NEED.md` requires irreversible owner dedup, bounded allocations,
identical cross-rank issue order (including empty epochs), fatal capacity stop,
lossless archive and `FinalizeDepth` as the semantic layer cut. CPU work at
initialization, disk worker and finalization is allowed. The hot-path target is
no GPU-count/control readback needed to enqueue the next healthy batch. Removing
a host wait is not progress if it only postpones an asymmetric failure until
the whole frontier has been submitted.

## Current dependency chain

| Boundary | Current source | Consequence / evidence limit |
|---|---|---|
| Generation -> route | `distributed_native.rs:2344-2375` always calls radix route and exchange pack; `route.cu` sorts and optionally compacts | One-GPU CUCO_RANK also pays route/pack. This may be redundant, but local pre-dedup and owner ordering forbid a blind bypass. Profile and ablate. |
| Pack -> peer | `distributed_native.rs:2387-2395,2462-2500` | LSA keeps counts on device; legacy NCCL drains/readbacks to size payload. LSA has a separate initcheck activation failure at `ncclCommWindowRegister`, not yet a proven memory error. |
| Peer -> owner | `distributed_native.rs:2562-2603,2625-2690` | DENSE LSA+CUCO_RANK has a pre-owner GPU fatal gate, but post-owner `all_max_ring_or_host_fatal` waits each batch. This handles asymmetric host/API errors and is not safe to delete in isolation. |
| Rank owner | `distributed_native.rs:1481-1550`, `experiments/library_owner/cuco_rank_batch.cu` | CUCO_RANK compare/reserve/commit/materialize publishes device counts and extents; no per-shard CPU loop. Its CUB select scans fixed incoming capacity, even for a short batch. This is a cost hypothesis, not an established bottleneck. |
| Other owners | `distributed_native.rs:1553-1970` | CUCO_INDEXED/cuDF and native owner retain host directories, counts, snapshots or batch-end extent readback. Do not attribute CUCO_RANK properties to these paths. |
| HASH_FIRST | `distributed_native.rs:1975-2170,2692-2694` | Materialization and request/response sizing have separate host dependencies; DENSE improvements do not complete this profile. |
| Parent retirement | `distributed_native.rs:2540-2560,2696-2710`; `event_generation.rs` | DENSE LSA generation/pack uses event-ordered retirement; archive leases and transport readers must be closed before reclaim. Existing event order is intentional until tested otherwise. |
| Archive | `distributed_native.rs:2183-2240,2845+`; `pinned_archive.rs` | D2H/pinned writer is bounded and asynchronous, but an archive error vote is made per parent batch before peer exchange. Must preserve bounded failure propagation. |
| Finalize depth | `distributed_native.rs:2700-2820` | Extent/control readbacks and rank-owner export occur at semantic cut. They are not automatically unwanted hot-path waits. |

The reusable `EpochCoordinator`/`ControlPump`/`AdmittedBuffers` modules are
not called by `distributed_native.rs` or the CLI's BFS dispatch. Their CPU
unit tests therefore cannot establish the production NCCL issue-order or
retirement protocol. Production still uses the separate fixed
`scheduled_rounds` loop (`distributed_native.rs:2244-2259`) and per-round
owner vote (`:2680-2690`). Connecting these models, or replacing them with
an equally explicit production protocol, is an integration requirement—not
another isolated coordinator test.

Further source audit of the current DENSE+LSA+CUCO_RANK dispatch:

- The device-only pre-owner gate applies only at `round == 1`. At 4/8 ranks,
  later peer rounds take `all_max_ring_fatal()` and the post-owner `all_max()`
  path, both with host synchronization/readback. A successful two-rank LSA
  timeline therefore cannot establish an asynchronous eight-rank pipeline.
- `advance_inner()` makes an `all_max()` archive-error vote for every parent
  batch whenever archive output is enabled, even for empty local batches.
  This protects asymmetric archive failures; moving it requires the same
  bounded failure protocol as the owner vote, not merely skipping the call.
- CUCO_RANK's `compare()` runs `DeviceSelect::Flagged` over the preallocated
  incoming capacity and launches surrounding kernels over that capacity.
  Its valid count remains on device, but the fixed scan is a candidate
  throughput cost at sparse tail layers. Measure valid/capacity ratio before
  changing this contract; a dynamic host count would reintroduce the wait.
- The ignored eight-rank full-state fixture uses HostSizedNccl with
  CUCO_INDEXED or native owner, not LSA+CUCO_RANK. Existing eight-rank
  evidence for other combinations must not be used as proof of this one.
- CUCO_RANK destroys its hash sets at `FinalizeDepth` and recreates the rank
  owner at the next depth (`distributed_native.rs::finalize_rank_owner`,
  `advance_inner`). The constructor suballocates RMM device buffers from the
  pre-reserved fixed pool and synchronizes its stream
  (`cuco_rank_batch.cu::Impl`). This is outside parent-batch processing and
  cannot be counted as a batch wait, but it is still per-depth setup cost and
  not literal precreation of every device object before depth zero. Include
  it in search timing and test pool high-water/capacity at the layer peak.

The v66 one-rank S10 measurement used 256 pinned archive slots and reported
973,078,528 pinned bytes, 0.412 s to search completion and 11.339 s to the
durable run commit. This is one profiled run, not a throughput benchmark; it
does show that a fast GPU search can leave substantial archive work after
search completion. The no-backpressure contract requires capacity analysis
for the maximum producer-minus-consumer backlog, not a claim that 256 slots
or disk transfer will always be invisible. The source enforces fatal slot
exhaustion (`pinned_archive.rs::acquire`) and does not implement a spill.

`Buffer::put` (`distributed_native.rs:265-278`) uses asynchronous H2D followed
by an unconditional stream synchronize; `read` is blocking D2H. `put_u32`
(`:280`) instead launches a device scalar store, so counting all scalar writes
as blocking H2D would be incorrect. `all_max` (`:1427-1439`) still synchronizes
and reads the collective result. The earlier GPU-only fatal-vote substitution
at `5b4bfe8` timed out on two T4s; the restored per-batch vote has a real
correctness purpose (`device-driven-library-owner.md`).

## One connected change set, in dependency order

1. **Measure the whole epoch before optimizing it.** Preserve current 1/2-T4
   full-state and archive baselines. Use a real BFS Nsight Systems timeline to
   separate route sort, duplicate filtering, pack, NCCL/LSA, owner compare,
   reserve/commit/materialize, host gaps, retirement and D2H/archive. Record
   exact graph, batch, pre-dedup, shards, pool, output contract and all runs.
   The isolated one-GPU CUCO_RANK profile is diagnostic, not multi-GPU proof.
2. **Freeze a device-resident epoch/control ABI.** Define one versioned fixed
   record for input counts, owner windows, committed counts, extent descriptors,
   sticky fatal and completion generation. Specify writers, readers and stream
   events for each field; reserve its storage and graph nodes before depth 0.
   Keep host-only archive/IO errors in a separate monotone control word.
3. **Solve failure and exchange together.** Design a bounded epoch rendezvous
   that has the same collective order on every rank, includes zero-payload
   rounds, stops after a rank-local CUDA/API/archive or capacity error, and
   cannot queue an unbounded remainder of the frontier. Decide explicitly when
   NCCL communicator abort is used for host/API failure. Retain the existing
   post-owner vote until multi-round asymmetric-failure fixtures prove the new
   protocol. Do not silently replace variable payload with maximum padding.
4. **Connect device descriptors through owner, archive and retirement.**
   Owner publication is irreversible only after all capacity guards. Each
   source slot and extent has a tracked last consumer: transport, owner,
   materializer, archive D2H and parent generation. Reclaim only after these
   events/leases close. Keep CPU extent reads at `FinalizeDepth`, not at every
   owner batch. Preserve fail-fast pinned-slot exhaustion and disk errors.
5. **Apply the same protocol to HASH_FIRST.** Device-side request compaction,
   sorted OriginRef/StateRef exchange and dense response application must fit
   the epoch ABI; otherwise mark the profile explicitly synchronous. Do not
   describe DENSE-only work as completion of both profiles.
6. **Then simplify proven redundant work.** Compare one-GPU route+pack bypass,
   source-local pre-dedup ON/OFF, fixed-capacity CUB select versus valid-count
   variants, and 4/8/16 shards under identical output contracts. A bypass must
   still produce deterministic owner-order input and exact layers. Accept only
   end-to-end time/VRAM wins, not isolated kernel improvements.
7. **Size archive independently of GPU speed.** Preserve both search and
   durable timers, measure archive drain and maximum outstanding slot count
   at identical workloads with one/two ranks, and derive the required pinned
   capacity from the observed production-consumption envelope. Do not reduce
   slots merely because the final archive is small, or hide a capacity fatal
   by waiting for a free slot.

This is one protocol-level change set, with explicit checkpoints rather than
one-line wait deletions. A rejected experiment must leave the current correct
path available as a separate named backend, never as a runtime fallback.

## Required acceptance evidence

- CPU model for epoch order, event/extent lifetimes and capacity-before-commit;
  adversarial empty/asymmetric/overflow/archive-error cases.
- Whole owner-to-retirement DAG capture with no required host count readback,
  plus Nsight Systems timeline showing actual overlap and no hidden per-batch
  D2H waits on the selected DENSE+LSA+CUCO_RANK path.
- Full-state equality and verified archives for 1 and 2 ranks, both rank maps,
  pre-dedup ON/OFF, different frontier sizes and repeated multi-round fatal
  injection. HASH_FIRST and non-rank library backends are separate gates.
- All four Compute Sanitizer tools on the selected target host. A sanitizer
  tool crashing in NCCL activation is **not** a pass for the BFS.
- Five-repeat search/durable/peak-VRAM panels on substantial graphs. Keep
  fixed cuCO pool and archive cost in the memory total; report every failed
  capacity/preflight attempt rather than selecting only successful runs.

Current acceptance state: only parts of the DENSE CUCO_RANK and LSA path have
physical T4 correctness/sanitizer evidence, and S10 library-owner speed/VRAM
samples exist. One-T4 v66 and two-T4 v67 Nsight diagnostics confirm many
aggregate host waits and transfers, with NCCL send/receive prominent on the
two-rank HostSizedNccl path (`docs/validation/library-owner-nsys-s10-v66.md`,
`docs/validation/library-owner-nsys-s10-v67.md`). They do not isolate the LSA
critical path or prove a particular removal safe. The v52 initcheck leaf
reproduces NCCL LSA activation failure before full BFS; the independent
NCCL-only v57 fixture reproduces registration failure under initcheck too
(`docs/validation/lsa-leaf-sanitizer-v47.md`,
`docs/validation/nccl-window-isolation-v57.md`). No full epoch overlap or
CPU-free claim follows from this evidence.

The subsequent two-T4 v58 LSA full-BFS Nsight diagnostic completed and
verified both S10 archives. It recorded 1,462 host stream synchronizations
and 1,740 synchronous CUDA copies across ranks inside the capture, but its
callchain export contains unresolved ASLR addresses. See
`docs/validation/lsa-nsys-s10-v58.md`. This confirms the pipeline still has
substantial host synchronization; it does not establish which individual
waits dominate the critical path. CUCO_RANK selection is a lower-priority
hypothesis in this workload than LSA copy, materialization, route sort and
control synchronization.
