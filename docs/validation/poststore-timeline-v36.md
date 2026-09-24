# S10 two-T4 Nsight diagnostic after scalar count store

Kaggle `trydotatwo/mgbfs-lsa-full-bfs-gate-t4` v36 completed at exact source
`6dd41bb1f669a7008e08ab2c2b0769ad64fe71b7` on two P2P-capable Tesla
T4s. It ran one archive-verified S10 CUCO_RANK DENSE case per transport,
with batch 32,768, four shards/rank, 256 buckets and a fixed 96 MiB cuCO
pool/rank. Nsight Systems 2025.3.2 traced CUDA, NVTX and OS runtime; the
collected summary covers **startup, warmup, search and archive**, not just the
timed BFS interval. These are diagnostic API/kernel sums, not a speed A/B.

| Aggregate API/kernel item | HostSized | LSA |
|---|---:|---:|
| `cudaStreamSynchronize` calls | 4,090 | 3,046 |
| `cudaStreamSynchronize` total API time | 869.335 ms | 805.610 ms |
| synchronous `cudaMemcpy` calls | 4,960 | 3,528 |
| `cudaHostAlloc` calls | 1,044 | 1,044 |
| `cudaFreeHost` calls | 1,044 | 1,044 |
| NCCL SendRecv kernel instances | 1,080 | 0 |
| NCCL all-reduce kernel instances | 2,196 | 2,204 |

The 1,044 extra HostSized stream synchronizations and 1,432 extra synchronous
copies are consistent with its host-sized transport boundary, but the
whole-process aggregate does not identify individual call sites or the
critical-path cost during search. Likewise, the large pinned allocation/free
totals include non-search phases; this report does not attribute them to the
archive hot path. The LSA path still has over 3,000 stream synchronizations
and 3,500 synchronous copies in the whole-process trace, so it is not
host-independent. At v36 a search-range marker and stage-resolved trace were
still needed before selecting the next wait for removal or quantifying overlap.

Raw compact summary and Nsight stats are under
`test_results/kaggle_poststore_timeline_v36/lsa-bfs-gate/`. Full profiler
reports were intentionally not downloaded to the local machine.

## Timed search window: v37

Commit `c73b637b06d3e0290b648f5acfc31d05565e8261` adds an opt-in
`MGBFS_PROFILE_SEARCH=1` CUDA profiler range around the **measured** BFS pass;
warmup, setup and final archive drain remain outside it. The ordinary path
does not call the profiler APIs. Kaggle v37 completed on two P2P-capable T4s,
with an archive-verified S10 run per transport and Nsight Systems capture
limited to that range. Its source differs from v36 only by diagnostic
instrumentation and the profile runner. Counts below cover the captured
two-rank search window; profiling overhead means API times are not unprofiled
BFS speed estimates.

| Captured search-window item | HostSized | LSA |
|---|---:|---:|
| `cudaStreamSynchronize` calls | 2,002 | 1,462 |
| `cudaStreamSynchronize` total API time | 475.062 ms | 501.493 ms |
| synchronous `cudaMemcpy` calls | 2,460 | 1,740 |
| `cudaMemcpyAsync` calls | 1,329 | 1,321 |
| `cudaHostAlloc` / `cudaFreeHost` calls | 0 / 0 | 0 / 0 |
| NCCL SendRecv kernel instances | 540 | 0 |
| NCCL all-reduce kernel instances | 1,092 | 1,092 |

The zero pinned allocation/free calls show that the large v36 totals were
outside the measured search window. The LSA route removes the host-sized
transport handshake, yet still executes 1,462 host stream drains and 1,740
synchronous copies in the scoped trace. The aggregate cannot assign these
calls to archive voting, post-owner voting, finalization or other wrappers;
it is not proof of their individual critical-path shares or compute/transfer
overlap. Stage NVTX markers or a callsite-correlated trace are the next gate.
Compact summary and stats: `test_results/kaggle_scoped_timeline_v37/lsa-bfs-gate/`.
