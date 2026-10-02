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

Important: fixed library pool still uses a conservative 512 bytes/capacity row.
The selector maximizes admission for that reserved profile, not optimal live
state capacity. Exact worst-case pool sizing and reserve calibration remain
required. Query mode intentionally exits with MEMORY_QUERY_DONE; only complete
rank-labelled query output is accepted. Windows CPU tests cannot compile or
validate the Linux CUDA constructor branch. Actual GPU gate remains open.

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
