# Scoped LSA S10 timeline, physical 2×T4 (Kaggle v41)

Source `b93d2a2701b9ebfe64674178d979dc544f3eaad0`; private kernel
`trydotatwo/mgbfs-lsa-full-bfs-gate-t4`, version 41. The previous v40
attempt stopped **before build/BFS** because NVIDIA CUDA NVRTC download timed
out (`curl` exit 28). V41 resumed the download and completed. The notebook
source was published separately as `86b9c0b`; the measured runtime is the
pinned source above, not that documentation/notebook commit.

The v41 run was **one profiled LSA sample**, despite the notebook's inherited
`scope` text saying "paired". It was S10, two physical P2P-capable T4s,
CUCO_RANK/DENSE, local pre-dedup ON, batch 32,768, four shards/rank,
96 MiB fixed cuCO pool/rank, and archive enabled. Both rank archives passed
committed-checksum/count verification and the 46-layer histogram totaled
3,628,800 states. The profiler range used `MGBFS_PROFILE_SEARCH=1`: it starts
after setup/warmup and stops at search completion, while archive submissions
during search remain inside the range. Rank-maximum profiled search time was
0.451718 s. External 50 ms `nvidia-smi` sampling found 597 MiB/rank; that is
not an exact allocation peak or a comparable unprofiled speed result.

Nsight Systems 2025.3.2 reported **across both rank processes** within the
search capture:

| Activity | Instances | Sum of reported durations |
|---|---:|---:|
| Host `cudaStreamSynchronize` | 1,462 | 492.282 ms |
| Host synchronous `cudaMemcpy` | 1,740 | 107.014 ms |
| GPU LSA exact-copy kernel | 180 | 180.854 ms |
| GPU NCCL all-reduce kernel | 1,092 | 98.776 ms |
| GPU CUB radix onesweep kernel | 2,352 | 81.091 ms |

The sums mix two ranks and concurrent streams; **do not** add them or divide
them by wall time as stage fractions. The long host-sync calls reached 7.32 ms.
The `cuda_api_sync` rule corroborates actual blocking calls, but does not map
them to Rust source callsites. `gpu_gaps` found no gap over its default 500 ms
threshold; the whole measured BFS is shorter than that threshold, so this is
not evidence of continuous GPU utilization. `gpu_time_util` marked three
roughly 15 ms windows at 29.6–36.4% GPU in-use time, including start/end regions.

This trace confirms hot-path host synchronization remains and makes it a
priority for a stage-attributed, rank-aware DAG measurement. It does **not**
prove which wait is removable, that LSA copy is critical-path bound, or that
the end-to-end owner→transport→retirement protocol is asynchronous. Preserve
archive/fatal and NCCL ordering gates when changing those waits.

Raw compact evidence: `test_results/kaggle_gpu_gap_v41/lsa-bfs-gate/summary.json`,
`s10-nccl_lsa-r0/screen-summary.json`, `s10-nccl_lsa-r0-nsys-stats.log`, and
`s10-nccl_lsa-r0-nsys-analysis.log`. Tool semantics:
https://docs.nvidia.com/nsight-systems/UserGuide/
