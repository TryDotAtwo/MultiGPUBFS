# MultiGPU acceptance strategies: architecture and verification

Status: design in progress; none of the new strategies below is claimed implemented
or accepted by this document. Baseline: source 2b97b094af3f94c250bd099004024a520c5238f4
on branch at 2598b53e0d5fcc130841a4e76f9668e420c7e6f8.

## Confirmed semantics
The two backends are exact. Logical GPU owner remains the existing owner mapping.
A logical shard has one active writer; duplicate hash values require full-state
comparison; exact bitmap IDs are collision-free only for the admitted permutation
domain. A segment has at most the configured 2^32 rows and rank totals are u64.
An exhausted segment causes retained input origins to spill and retry after growth.
An exhausted per-rank budget yields INCOMPLETE with only completed layers committed.
Inputs stay alive until all consumers and spill retries finish. Other graph families
must not silently select a permutation bitmap.

## Required variants
A baseline uses the existing route and insertion atomics. B groups route queue
reservations by destination and groups frontier reservations for candidate winners.
Open-address hash CAS remains; bitmap same-word aggregation is a separately measured
optimization. C is an actual bounded batch design, not radix sort followed by
the baseline per-item CAS. Selection is explicit and part of configuration,
persistent-worker identity, tuning cache identity and report. Unsupported variants
or insufficient scratch fail explicitly; no silent fallback.

## Bitmap C dataflow
Partition ownership by (logical owner, local_id / 64) with physical-to-logical
owner mapping preserved; all bits in a word map to one local shard. Check seed
insertion, every routing path and owner mapping use the same function. A lane
exclusively owns one shard while accepting; all words in its input are processed
in that lane, with batches ordered by stream events. Other shards remain parallel.

The input has bounded Q original route records and original source indices, valid
until the lane and spills drain. Preallocated per-lane scratch, accounted against
the rank VRAM budget, holds Q key/index pairs, Q sorted key/index pairs,
flags/scan results and CUB temporary bytes queried before steady state. Scratch
reuse waits for the last consumer and spill retry; concurrent lanes use disjoint
storage. Q is no larger than the admitted per-lane queue bound, not a whole layer.

Sort by exact local id, retaining the original index; collapse identical ids,
group by word, load old word, derive new bits, scan per-word popcounts and reserve
bounded rows and any rank-wide quota. Select only representable winners, send
remaining representative origins to the bounded spill queue. Write each selected
full state once to disjoint assigned rows. Publish the committed mask only after
materialization; order later same-word work after that commit. If CUDA or the quota
fails, abort the partial layer rather than publishing it as complete. Empty inputs,
unfilled tails, duplicate-only input at full capacity and all-visited batches
must not trigger spurious growth. Two banks overlap only when an event/timeline
shows it and the scratch has distinct lifetimes.

## Hash C dataflow
Each shard keeps bounded exact sorted runs for the live history and current
accepted frontier. Runs store hash plus stable segment/bank/row identity.
Full-state comparisons resolve equal hashes. New bounded sorted input is deduped
against relevant immutable runs and itself, then scan assigns disjoint output rows.
Index only affected ranges; do not merge every batch against all history. Merges
and compaction have explicit size levels, bounded simultaneously-live input/output
scratch, and a reclamation event. Directed graphs retain complete history;
inverse-closed graphs may retire only banks proven unnecessary. The sorted-run
format, retirement, allocation and recovery require implementation and tests before
this can be called an operational C strategy.

## Audit
Track packed and nonpacked input, 8-bit and 64-bit state elements, two-/three-bank
bitmap, inverse-closed/directed hash, q tails, empty peers, source retries and
NCCL ordering. Account for 32-byte route records, two u64 key/index arrays,
prefix/flags, CUB temporary size, lane replication, existing queues, segment
storage, bitmap and NCCL reserve. Record exact bytes and peaks. Existing code
uses bitmap shard=mix(id); it does NOT yet provide word ownership.

## Authoritative integration constraints (subsequent architectural handoff)
* One validated strategy SegmentStrategy {Atomic,Warp,Batch};
  MGBFS_SEGMENT_STRATEGY=atomic|warp|batch, unset=atomic. Preserve old ABI,
  include choice in capability query, rank agreement, result, tuning/preflight
  cache and persistent worker identity. No silent fallback.
* Warp route groups (rank,shard); overflow compares counts and bounds, not the
  arbitrary subset written before discarded overflow/retry. Hash retains its
  exact slot CAS; grouped grants use bounded u64 cap and segment reservations.
  Temporary cap contention must be reconsidered after wave drain.
* Owner progress has two distinct cursors: next raw input tile and the spill
  tail of filtered winners. Drain the tail before consuming the next tile.
* Batch route uses bounded CUB sort of (bucket:u32, origin:u32), original
  identity:u64 plane and existing exact/padded NCCL. Its route scratch is
  separate from overlapping owner scratch and accounted per rank.
* Batch bitmap local shard hashes (local_id >> 6), preserving logical owner and
  physical mapping. Sort exact ids, one representative/id, group words, scan
  new masks, bound capacity, materialize then publish. Only admitted bits are
  committed; remaining representative origins spill. Same-word transactions
  are ordered across all streams/batches by shard ownership and event; disjoint
  words remain concurrent.
* Batch hash uses per-segment/bank separate flat keys:u64 and row:u32 arrays,
  12 B per row plus aligned descriptors. Exact sort/lookup is ordered by hash
  and full state; row 0xffffffff is legal. Stable bounded merges only combine
  adjacent same-bank runs if workspace fits; otherwise published runs remain
  independently searchable. Do not rescan/rewrite all history each batch.
  Directed mode retains all runs; inverse-closed retirement follows bank drain.
* Measure max/median live hash runs, lookup/merge time and merge bytes. Report
  sampled NVML separately from exact allocation/temporary peak accounting.
* Gate every backend and strategy using full small-layer CPU-oracle sets,
  unequal/empty peers, cap/spill/rollback, collisions, persistent reuse and
  resource pressure; then six matched S12 two-GPU cells with large-window
  metrics and optional S13 if affordable. Identical configuration and code SHA;
  classify any retry or unequal completed prefix.
