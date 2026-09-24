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
host-independent. A search-range marker and stage-resolved trace are needed
before selecting the next wait for removal or quantifying overlap.

Raw compact summary and Nsight stats are under
`test_results/kaggle_poststore_timeline_v36/lsa-bfs-gate/`. Full profiler
reports were intentionally not downloaded to the local machine.
