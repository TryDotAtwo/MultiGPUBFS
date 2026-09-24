# Device-driven library owner: implementation contract

Status: partial implementation. The CUCO_RANK rank-batch transaction and
two-T4 no-fault BFS are GPU-validated; the complete owner-to-retirement
pipeline, bounded fatal termination, and large-run performance are not.
User requested removing CPU from the hot path and calibrating shards after
S10/16-shard results.
Baseline source remains immutable; the existing synchronous library ABI remains
an explicit reference backend, not a runtime fallback.

## Meaning of removing CPU

No survivor-count, reservation-result, shard-directory or materialization-result
readback may be required to submit the next owner stage. Host submission of fixed
CUDA launches/graphs is allowed. Archive threads and depth finalization remain on
CPU. This is not yet a claim that transport or the whole BFS is host-independent:
current variable-count NCCL submission, parent enumeration and FIFO retirement
must be audited separately. A complete result requires their dependencies to be
overlapped or moved, not relabeled as outside the hot path.

## Current evidence and dependencies

`commit_library_batch` currently reads the bucket directory, loops over shards,
reads each compare count, checks reservation via a control snapshot, then reads
materialization and appends a host `Vec<Extent>`. `CucoOwner::compare` itself drains
its stream to read two control words. Native DENSE already batches extent results
but still drains at a batch boundary. Removing one completion wait did not remove
these dependencies. The capture regression only proves no redundant completion
wait, not asynchronous BFS.

At current `6a693c2`, the regular per-batch path still calls
`mgbfs_route_run` and `mgbfs_exchange_pack_device_n` even for `world=1`
(`distributed_native.rs`, around the route/pack block). Selecting cuCO as
owner does not bypass route radix sort, optional pre-dedup, or pack. The
legacy indexed owner still calls `rank_directory`, reads it on the host and
loops over shards inside `commit_library_batch`; it is a different path
from the device-count `CUCO_RANK` transaction. These are static source
observations, not an attribution of elapsed time. A one-GPU shortcut or
fusion must retain exact owner ordering, local pre-dedup semantics and
full-layer correctness before it can replace the existing route stage.

An external single-RTX-3070-Laptop comparison on pinned source `5fea95c`
reported faster exact S9/S10 search with `CUCO_RANK` than CUB, but a larger
sampled full-device VRAM footprint with a 1 GiB fixed cuCO pool. Its
GPUexplore LRX run measured reachability rather than exact BFS layers and
is not an interchangeable speed baseline. The timings do not identify the
cost of route sorting, copies, owner kernels or host waits individually;
stage-resolved Nsight and controlled ablations remain necessary. This
observation must not be transferred to newer HEAD or multi-GPU as a result.

## Selected direction: one rank-batch transaction

Do not reproduce the CPU shard loop with device polling or a persistent spinlock.
Queue one bounded rank-batch DAG with device counts. Shards retain distinct
persistent tables and capacities; kernels select their table by hash prefix.
The transient table and candidate planes are shared across the whole incoming
batch. Copy candidate keys once, not once per shard. CPU never receives per-shard
survivor counts to choose the next launch shape.

1. Validate input count/directory on GPU; copy valid Hash128 rows into stable SoA.
2. Clear the transient table; insert batch indices, find deterministic minimum
   source-row representatives, probe the matching persistent shard table.
3. Produce survivor flags and per-shard counts. CUB compaction uses fixed batch
   capacity with zero flags on the tail. Authoritative valid counts remain device
   words. No sentinel hash may turn into an observable survivor.
4. A single-writer GPU reservation job validates ALL shard accepted capacities,
   StateRing/layer capacity, extent-descriptor capacity and HASH_FIRST request
   capacity before persistent writes. Failure is sticky and grants zero credit.
5. Append selected keys into disjoint accepted ranges; then insert stable accepted
   indices into shard tables. Never publish recyclable transient indices.
6. Materialize DENSE states or produce HASH_FIRST request/target pairs using
   device offsets. Publish a device extent descriptor only after protected writes.
7. A completion event protects source slots, transient scratch and descriptors.
   Next work on the same lane can be queued by stream order. Other streams must
   wait on that event. No host lease release based merely on enqueue success.

cuDF's existing host-sized relational API is not silently converted to this
interface. Keep its synchronous experimental path explicit unless a separate
asynchronous boundary is demonstrated.

## Data and lifetime contract

Let N be incoming slot capacity, S local shards, A[s] accepted capacity.
All arrays are allocated before depth zero. Counts and indices are u32 with
checked N,A[s] <= INT32_MAX; absolute StateRing refs and epochs remain u64.

| Array | Storage / capacity | Writer and lifetime |
|---|---|---|
| Candidate keys | existing aligned SoA, 16*align64(N) bytes | copy job; until all batch consumers finish |
| Representatives/minima | u32[N] each | dedup kernels; lane-exclusive scratch |
| Flags | u8[N], initialized including tail | membership job through CUB select |
| Selected/source indices | u32[N] each | selection/gather; through materialization |
| Shard counts and offsets | u32[S] each | count/scan; through accepted append |
| Persistent counts | u32[S] | one GPU writer; across batches, reset only at drained depth transition |
| Persistent accepted keys | 16*align64(A[s]) per shard | commit; retained through finalization/export |
| Table refs | S device-accessible cuco refs | constructed before batches; backing tables immutable in address |
| Transaction control | versioned fixed-size POD, layout still to implement | GPU stages; sticky fatal separate from resettable job state |
| Extent descriptors | existing 64-byte Extent ABI in a bounded flat array | GPU materializer; retained through enumeration/archive/retirement |

The table-ref ABI and CUB scratch query are implementation gates, not assumed
byte counts. Pool planning must include them, descriptor slots and graph storage.
No pool growth, spill, or host allocation in repeated nodes. A serialized lane can
reuse transient scratch by stream order; overlap requires another fully budgeted
lane, never aliasing unfinished work.

## Remaining blocking boundaries

- The current blocking post-owner group vote reads device/host failure after
  each CUCO_RANK DENSE LSA owner batch and stops before the next scheduled
  parent round. Its asymmetric capacity and host-error fixtures passed on
  two T4s (Kaggle v33), but large-frontier failure latency is unmeasured.
  Removing this per-batch host readback reopens the cancellation problem:
  the rejected GPU-only vote at `5b4bfe8` could leave every remaining
  scheduled round to issue and hung under a one-rank host/API error.
  A replacement must bound wasted work independently of frontier size,
  preserve identical collective order, and report a group failure without
  stranding a rank. A GPU-only vote is not sufficient by itself.
- Host/API failure after one local owner job must rendezvous with peers or
  abort the communicator without leaving a rank in the next exchange epoch.
  Test this with several scheduled rounds, not just a depth-one fixture.
  The simple asymmetric device-vote/host-vote substitution at `5b4bfe8`
  failed this gate on two P2P T4s (300-second timeout, Kaggle v30) and was
  reverted at `6077bf8`. Do not reintroduce it as a host-wait removal.
- Device-count persistent insertion and export must replace host `count_` without
  weakening capacity or epoch checks. Constructor insertion of history is setup.
- HASH_FIRST request compaction currently uses host offsets and extent vectors;
  new GPU append/response application must preserve OriginRef/StateRef pairing.
- Parent enumeration/retirement and archive D2H ownership must consume GPU extent
  descriptors without releasing states still referenced by transport or archive.
- NCCL variable payloads require a distinct transport protocol decision. Do not
  hide CPU counts behind an event query or send maximum-capacity padding silently.

## Validation before selection

CPU independent descriptor/capacity oracle; capture test covering the WHOLE owner
DAG with no host readback; zero/tail/duplicate/collision-policy fixtures; injected
accepted/state/request/descriptor overflow before persistent mutation; queued
multi-batch scratch reuse and finalization/export; full DENSE/HASH_FIRST layer
and archive equality on one/two T4s; all four sanitizers; Nsight validation of
count transfers and waits; five-repeat unprofiled 4/8/16-shard panels, full VRAM
including pools. Existing synchronous results are controls, not async evidence.

## Calibration evidence

At source `f5b52c9`, the 4-, 8-, and 16-shard S10 screens completed on one/two
T4s with 256 buckets and mandatory archive. The 4-shard screen gave the
lowest measured cuCollections search median among those separate sessions;
it did not give a universal optimal shard count or prove an asynchronous
pipeline. The 8-shard v55 gate also passed the full 1/2-T4 plain and four-tool
sanitizer fixtures at that source. Exact samples and limitations are in
`docs/plans/library-first-bfs.md` and
`docs/validation/library-owner-t4-v55.md`. The later CUCO_RANK owner and
LSA transport have separate evidence; do not transfer these indexed-owner
calibration results to them.
