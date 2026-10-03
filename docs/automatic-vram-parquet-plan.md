# Automatic VRAM, bounded CUDA Graphs, analytical archives

Active goal requested 2026-10-03. Isolated branch `codex/bfs-tail-archive`.

## Implemented checkpoint

New automatic sweeps select Parquet publication. Frozen packed tails are
converted on the GPU host in the background publication worker, never downloaded
to the user's computer. Approximate 512 MB uncompressed shard target, bounded
262144-row groups, Zstd, fixed-width binary states. Columns: n, r, depth,
ordinal, state. Multiple layers share a shard; large layers span shards.
Retention still counts packed bytes. Manifest preserves original retained-layer
metadata and separately records physical Parquet bytes/checksums and layer spans.
Old configurations retain their original packed format for resume compatibility.

13 targeted CPU tests pass. New format has not yet been checked on rented GPU/HF.

Native constructor query mode now initializes the real NCCL collective and
reports the existing allocation ledger and free VRAM before large allocation.
Python startup selector predicts an affine capacity, then confirms exact native
queries and corrects rounding/nonlinear boundaries with bisection. New sweeps
use this path before each BFS. Batch remains 32768 for sufficiently large cases;
Cuco <=50% table occupancy is unchanged. Probe errors are fatal startup errors,
not graph capacity/pruning evidence. Existing 1 GiB native reserve is unchanged.

The pool autosizer now calls a native Cuco/CUB query: workspace allocations,
actual rounded hash-table extents, accepted SoA planes and rank metadata, each
charged with 256-byte allocation alignment. Worst histories assume arbitrary
shard skew; this is an upper bound, not an exact maximum reachable pool usage.
Fragmentation slack is max(64 MiB, 5% of queried bytes); hardware calibration
remains required. The original 512 bytes/row survives only as a parser placeholder,
replaced before native admission. Query mode intentionally exits with
MEMORY_QUERY_DONE; only complete
rank-labelled query output is accepted. Windows CPU tests cannot compile or
validate the Linux CUDA constructor branch. Actual GPU gate remains open.

Added CUDA gate assertions: query makes no RMM allocation, upper bound covers
real rank construction, invalid input clears output. These assertions have not
yet run on CUDA hardware. Public API checked against NVIDIA cuCollections extent
and open-addressing implementations; deployed pinned version still needs a build.

## Required remaining work

1. Extract allocation-only native planning using existing CUDA library query APIs.
   Avoid constructing a full BFS for each sizing iteration. Account for runtime,
   library pool, transport, CUDA/NCCL overhead and graph resources. Search capacity
   before BFS starts; preserve batch size and <=50% Cuco load factors. Do not
   substitute a larger fixed percentage for measured capacity discovery.
2. Resolve startup collective ownership for per-rank probes and heterogeneous
   free memory. Fail closed and record measured plans; do not prune search pairs
   due to a failed startup probe.
3. Inspect device/host dependencies to capture reusable 32-batch windows.
   Current graph capture is a debug owner probe only. Host-dependent counts,
   changing ring offsets, archive callbacks and collective ordering must remain
   correct. Never label partial owner capture as a full BFS window.
4. Add publication integration tests for Parquet and document DuckDB/PyArrow
   analysis. Evaluate merging tiny completed cases into a sweep-level dataset
   rather than retaining one small shard per case.
5. Paired target-GPU baseline/proposed runs, identical complete layer counts,
   allocation telemetry and wall time medians. No zero-overhead claim without
   hardware evidence. Check spending remaining under the previously authorized
   $10 total before provisioning; previous instance is destroyed.

Do not mark the active goal complete until remaining gates are satisfied.

2026-10-03 checkpoint: Parquet integration test joins 50 complete layers into one
payload and recovers every state and all layer metadata. Native epoch completion
events can now be preallocated for 1..32 in-flight device-count batches using
MGBFS_INFLIGHT_BATCHES; default remains 2 until hardware validation. Bootstrap
agreement now includes this value, pool autosize and memory-query mode. This is
a bounded submission queue, not yet CUDA Graph capture. Applies only to
NCCL_LSA + rank-owner device epochs; the automatic default HOST_SIZED_NCCL still
observes per-batch exchange counts on the host. NCCL_LSA requires NCCL >=2.29.

User explicitly added $20 for further GPU testing; total authorization now $30.
Old scoped Vast key works for instances, has no billing_read permission. No own
live instances were present at the checkpoint. New stages must remain within
the added $20, independent of unconfirmed earlier billing; lease watchdogs remain
required. States may never be downloaded to the user's computer.

2026-10-03 two-RTX-3090 hardware checkpoint (source 8941007):

- CUDA 12.9, pinned Cuco and Linux Rust build pass. Native Cuco pool query
  assertions pass: no device allocation, bound covers construction, invalid
  input clears output. The two-rank multiset oracle passes for n=5 and n=7
  with CUB, CucoIndexed and CucoRank, prededup on and off.
- Actual repeated native probes exposed two startup bugs: archive-free CLI
  needs explicit --search-only; query teardown needs a collective rendezvous
  after every rank prints/flushed its record. Both are fixed. These operations
  run only during startup queries.
- Archive-enabled startup used 8 MiB more VRAM than the query subprocess.
  Selection now includes explicit query-to-run headroom (default 64 MiB), in
  addition to the existing native 1 GiB untouched reserve. This uncertainty
  margin is recorded in query evidence and is not a BFS-loop allocation.
- Updated selector completed 29 two-rank queries and selected 71,204,144 rows
  per rank for (15,4). Actual archived BFS completed depths 0..34 in the bounded
  30-second gate. Observed peak was 24,194,842,624 bytes on each 25,296,044,032
  byte GPU, about 95.6%, sampled every 50 ms. Status is INCOMPLETE because of
  the deadline, not graph exhaustion. This establishes admission and useful
  execution for this hardware/profile; it is not a universal maximum proof.
- Complete (9,3): all 60,480 states in 30 layers independently match CPU BFS
  layer sets. One Parquet payload of 336,876 bytes was published before its
  manifest; HF stream readback verified SHA-256, bytes, manifest and ledger
  at revision 4be0a4090e0a1345b9f170f7bab61116787fd383.
- Matched (11,4), archive enabled, same capacities and 497,712,640-byte pool:
  three alternating repeats completed all 44 layers / 1,663,200 states with
  identical layer counts. Median search seconds: fixed pool 0.578487657,
  native autosize 0.583000463, ratio 1.007801041. This short-workload result
  does not prove zero overhead or performance on saturated larger layers.
- These cards report P2P CNS. The default HostSizedNccl route was tested;
  reusable full 32-batch CUDA Graph windows and multi-GPU LSA remain open.
  Cross-case consolidation into shared Parquet shards also remains open.
- The INCOMPLETE (15,4) tail retains 616,586,288 packed bytes from its 35
  completed layers, below the 1 GB limit. Four Parquet shards total 574,119,644
  bytes. HF readback verified every checksum, manifest and ledger at revision
  d54a886b011241ac40ef0adb18dbcd84932fce0d. No GPU states were downloaded to
  the user's computer. Frozen evidence report:
  https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/blob/f601b5272534cb2c52a62a33f9eecdbd9c55f372/evidence/20261003-vram-parquet/gate-report.json
- Own instance 53934868 was deleted after publication. Both immediate provider
  GET and the independent watchdog confirmed ABSENT. Exact billed charges
  remain unavailable with the scoped key; the quote and enforced lease deadline
  are separate from a billed-cost claim.
