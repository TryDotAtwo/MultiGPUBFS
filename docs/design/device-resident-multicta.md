# Multi-CTA resident BFS: bounded exchange

Status: implementation in progress. The existing single-CTA engine remains the
reference and default. No performance or correctness claim follows from this
design alone.

## Scope and invariants

The first supported production profile is packed permutation bitmap; the
second is exact segmented hash on the same transport. Both use one resident,
cooperatively launched grid per rank. The CUDA occupancy query for the actual
kernel, block size and shared memory bounds the grid. Every CTA participates in
every local grid barrier, including empty ranks, failures and tails. The graph,
owner mapping and physical rank mapping are fixed for a job. The coordinator
only handles bounded per-rank/per-destination metadata, never candidate rows.

A completed layer is the exact set of children of the previous complete layer
minus the applicable visited history. A rank with no local work still publishes
empty windows. Every completed layer is globally drained before it is exposed;
an exhausted pool or cap leaves only the preceding complete layer visible.
Fatal errors override resource stops. Remote readers must retire before window
or job storage is reused. The existing supervisor handles a vanished rank.

## One bounded slot

For each round a rank produces at most B candidates (64-bit edge positions).
Each candidate contains its exact ID/hash, lossless packed state and destination.
Raw candidates, position scratch, block histograms and outgoing payload are
fixed-size job-owned buffers. Each cooperative CTA handles disjoint raw indices;
warp groups use CTA-local shared reservations to obtain distinct offsets. A
grid barrier completes histograms. Per-destination prefix of CTA counts and a
prefix across destinations define disjoint contiguous output ranges; a second
grid barrier precedes scatter, and another follows it. All counts and offsets
are 64-bit. Errors are shared and suppress the entire scatter. The histogram
stores at most world * grid_ctas entries, not world * B payload records.

For each physical source/destination, registered LSA window metadata identifies
job generation, 64-bit round, offsets and counts. After every CTA completes
scatter, the local grid rendezvous makes its writes available before the single
publisher's system-scope release of READY. The receiver acquires READY before
its CTAs read their disjoint ranges. A receive wave holds at most C records;
source ranges are consumed in deterministic pieces, so the number of incoming
records need not fit C simultaneously. After acceptance and local grid drain,
the receiver releases DONE. A source reacquires DONE from all readers before
it overwrites its window. Empty ranges and local/self ranges follow the same
epoch. System-scope atomics require the actual directed peer-atomic support
and registered NCCL LSA mapping agreed in preflight. No D2H count or host
submission occurs between rounds. Overlap and extra slots are separate,
measurement-driven changes.

All ranks derive a common round count from the maximum previously committed
per-rank frontier and the common B. CUDA grid barriers synchronize only local
CTAs, never different GPUs. Mailbox progress is divided into publish-then-read:
no rank waits to receive before publishing its own bounded window. Errors stop
publication in a coordinated epoch and terminal outcomes are gathered after
writers drain; fatal beats capacity.

## Owner storage

Bitmap uses exact rank and fixed owner. The implementation must either reserve
a durable row before committing a new visited bit, or discard the complete
unfinished job generation after a failed tentative commit. Duplicate-at-full
must remain a duplicate. Independent CTAs may share a bitmap word only with
device-scope atomics and one representative per exact bit. The preceding
complete frontier stays leased for the INCOMPLETE result.

Hash routes on hash but compares complete canonical states. A writable bucket
has an exclusive owner, or a proven pending/commit/defer protocol; no scalar
exclusive-shard primitive is called concurrently without a lease. Both the
history and already accepted next states participate in equality. Extent
rollover follows drain. Directed history is not shortened without a proven
graph contract. Local row indices are u32 with 2^32 valid rows per segment;
segment IDs, total counts, lengths and byte offsets use checked 64-bit
arithmetic. The total pool is limited by admitted VRAM, not one segment.

## Space and test boundary

For a 16-byte payload and raw record width R, the initial one-slot route uses
B * (R + 8 + 16) + grid_ctas * world * 8 + (world+1) * 8 bytes plus a
separately admitted receive scratch C * 16, window metadata, control,
frontier/history and the existing CUDA/NCCL reserve. Actual alignment and
additional per-CTA counters are included in admission. Non-overlapping buffers
may be overlaid only after their final reader has drained.

The partition primitive is tested first for empty/tail/skew distributions,
invalid destinations, capacity and sentinels with an independent host
multiset. The actual public BFS must then pass full per-layer state
multisets on physical 2/4 GPU, including forced collisions, rank skew,
duplicate-at-full, resource stop and persistent reuse, before S12/S13 A/B.
A partition test alone is not a BFS acceptance or throughput claim.
