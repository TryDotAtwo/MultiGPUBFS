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
