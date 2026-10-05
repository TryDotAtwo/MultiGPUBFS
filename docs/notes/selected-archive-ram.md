# Compact archive: bounded RAM samples and terminal export

The `last_complete_small_1000` launcher now selects the `MGBFSAS2` native
stream. Other retention modes continue using the existing `MGBFSAR1` path.
This change is implemented and CPU/type checked; GPU execution and speed
measurements are still required before a production readiness claim.

## Source lifetime

`retire_dense_prefix_impl` advances StateRing head, allowing subsequently
accepted future records to overwrite consumed source ranges. Previous/current
deduplication planes are not a promise that all previous/current state bytes
remain intact. Incomplete output therefore takes a bounded prefix before this
retirement. It does not read a failed CUDA runtime to recover those bytes.

On successful global exhaustion the next layer has zero accepted records on
every rank. Accepted materialization has consequently not overwritten the
source state arena. `advance_selected` saves its at-most-two physical extents
before advancement and exports their bytes immediately after this existing
global exhaustion result, before BFS destruction or another graph starts.
Logical retirement is not treated as a generally valid read lease: this late
read is allowed only on the zero-next-layer success branch. Live errors never
use the late-read branch. No extra full frontier is retained in VRAM.

The GPU gate must confirm this physical-lifetime argument for CUCO_RANK,
physical wrap, empty ranks, both seeds and supported transports. CUDA
Graph-32 must be included in its own acceptance gate before being enabled.

## Transfer and host ownership

- Four preallocated pinned slots per rank, normally 8 MiB each. Slots are
  rounded to 64 KiB so changing `n` does not force different cached host buffer
  sizes. Eight ranks use approximately 256 MiB of this pinned pool.
- Canonical u8 states copy directly in large sequential blocks. Archive block
  size is independent of compute batch size; no new per-batch synchronization
  or allocation is introduced. Matrix conversion retains its batch limit, but
  the compact launcher explicitly requires the compact state representation.
- Intermediate copies contain at most 1000 states per rank. The host selects
  the first at-most-1000 records globally, in rank order. A small global layer
  with at most 1000 records is retained whole. The temporary per-rank samples
  are not a claim that 8000 records are published for INCOMPLETE.
- The compute stream's existing retirement dependency waits for the prefix
  D2H event. The worker waits for slot completion, hashes the transport frame
  on the CPU and writes it sequentially to the FIFO. No per-state Hash128
  calculation/copy is performed for this stream.
- The host descriptor queue is bounded and cancellable. Intermediate samples
  and statistics remain in host RAM; no rank spool or combined layer file is
  written for a large intermediate layer.

The pinned queue is bounded. Ordinary host metadata and selected small layers
accumulate for the current graph, outside the native batch hot path. This is
not a claim that all host metadata is preallocated or that arbitrarily long
graphs have a constant total host-memory footprint.

## Wire format and publication

`MGBFSAS2` has the existing 48-byte header and checksummed sequential frames,
but Records contain state bytes only, without the unused Hash128 plane.
Intermediate Layer counts describe the transmitted prefix, not the full BFS
layer. Full layer counts/timings come from native depth trace and reports.

After the last sample LayerCommit, kind 4 replaces that layer on success;
full terminal Records and LayerCommit follow, then RunCommit. The reader and
verifier require replacement of only the latest layer, at most once. The
replacement resets transmitted totals instead of counting the prefix twice.
A local terminal layer with <=1000 rows already has a full sample and requires
no replacement. Global terminal layers spanning multiple such ranks use the
complete rank samples, not their truncated global INCOMPLETE prefix.

Only terminal replacement payloads create rank spool files, written by the
reader thread. After search, finalization constructs selected packed files,
hashes them and performs durability sync. This still reads terminal rank
spools once to assemble the final file: it removes intermediate SSD traffic,
not every final-output disk pass. Existing Parquet/HF publication and checksum
verification remain unchanged. Old archive modes are not claimed to have
received this optimization.

COMPLETE stores the final whole layer and all earlier whole layers with
<=1000 rows. INCOMPLETE materializes only the newest committed prefix, at most
1000 records total. Statistics use the existing all-rank completed-advance
boundary; an expansion that fails partway does not invent a completed timing.
Snapshots are sealed after the search, not fsynced after every intermediate
layer. Process/controller failure can lose RAM-only intermediate metadata;
native trace remains the available diagnostic record, not a durability receipt.

## Local verification

- Rust archive codec tests: selected replacement, bounded samples, empty rank,
  corruption/truncation and compatibility with old archive encoding.
- Python selected stream/retention tests: no intermediate state files, terminal
  streaming, corruption cleanup, 8-rank global 1000-record limit, missing final
  payload rejection, full small-layer retention and independent L/R/X oracle.
- Existing launcher failure, two-seed, sweep, archive, Parquet and publication
  tests pass.
- `cargo check -p mgbfs-cli --features cuda,library-owner --target
  x86_64-unknown-linux-gnu` passes. This checks Linux Rust code and types; it is
  not a CUDA link test, a kernel execution test or a hardware benchmark.

No GPU instance was created, no state payload was downloaded, and no HF write
was performed to validate this code change.


## Bounded GPU evidence, 2026-10-05

Production source `6dd0e21` passed bounded tests on two RTX 3060 cards.
See [the measured report](../../BFS_2x3060_REPORT_2026-10-05.md).
Independent word oracles verified retained COMPLETE states. A capacity stop
exported 1000 distinct states from a completed frontier of 5478 states.
A slow RAM consumer retained correct layer counts and produced no intermediate
state payload files at 43 layer callbacks. Eight-bit symbols and aligned words
were checked at n=17,33,128 with intentionally bounded incomplete searches.

The post-retirement terminal export branch was forced on four small graphs by
lowering its threshold in an isolated test-only build. Exported states matched
independent word oracles. This is not natural terminal-frontier coverage over
1000 states or a large-copy throughput measurement.

The automatic test attempted 82 pairs:79 COMPLETE with both seeds matched,
two capacity failures with the second run skipped, and one deadline stop.
It pruned 230 pairs;7943 remained unrun. HF readback verified 162 manifests and
nine shared Parquet payloads. This was not an exhaustive sweep. The rental and
remote credential were removed and API absence was confirmed.

Graph32 was rejected because this rental used ordinary NCCL without admitted
NCCL_LSA device-count owners. LSA/Graph32 and memory-saturated B300 performance
remain separate acceptance gates.


## Sweep I/O follow-up, 2026-10-05 (CPU/type-check evidence only)

Full-export policies now use MGBFSAS3 state-only frames. AS2 remains the bounded
selected protocol and AR1 remains readable. Internal search hashes are unchanged;
only unused archive hash generation and its 16 bytes/state transfer are omitted.
All launcher policies use four ~8 MiB pinned slots/rank, rounded to 64 KiB and
reused by resident sessions. State-only copy geometry is independent of compute
batch size. Python descriptor handoff is bounded and cancellation-aware.

Full layers are kept as closed rank parts. The host writer adopts them by rename
and records their global ordinal ranges instead of reading and rewriting a merged
layer. Retention trims complete depths, including all parts of a depth. Final
Parquet cohort conversion still aggregates them into large shared shards; this
does not introduce more HF payload files. Checksums are computed incrementally by
background FIFO readers while packing, then reused during adoption.

Final-only launches retain layer metadata in RAM and defer intermediate fsync and
snapshot construction until sealing; live publication retains durable snapshots.
This changes recovery boundaries: a final-only process crash before sealing does
not promise a usable intermediate manifest. It does not change successful final
manifest durability or checksum verification before publication.

Validation: 16 Rust archive tests and 111 Python tail tests passed; Linux-target
CUDA/library-owner cargo check passed. New tests cover AS3 layers over 1000 rows,
legacy compatibility, rank-part inode adoption, ordinals, whole-depth retention,
final-only metadata/durability boundaries and final snapshot survival after working
tail cleanup. No GPU timing or Graph32 acceptance claim is made for this patch.
