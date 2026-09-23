# CUCO_RANK Nsight diagnostic on 2xT4

Source: `4b68552b0c0c865862d896a2ca1913043110e575`, private Kaggle
`trydotatwo/mgbfs-cuco-rank-timeline-t4` v1. Raw outputs are in
`test_results/kaggle_cuco_rank_timeline_v1/library-owner/`, including the
Nsight Systems SQLite export and per-rank archive verification. The notebook
finished with `summary.json: PASS` on two physical Tesla T4 devices.

The profiled S10 DENSE `CUCO_RANK` two-process run completed all 3,628,800
states and verified both rank archives. `measure.json` reports
`search_complete_seconds = 0.490374196` and
`durable_run_commit_seconds = 5.527130712`. These are **diagnostic profiled
times**, not A/B performance measurements.

The trace includes process startup, warmup, search, and archive, and has no
search-only NVTX range. Consequently the following counts are *whole-trace*
evidence of host interaction, not shares of BFS wall time or proof of a
particular bottleneck:

| Operation | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| `cudaStreamSynchronize` | 3,490 | 3,480 |
| synchronous `cudaMemcpy` | 2,476 | 2,474 |
| `cudaMemcpyAsync` | 3,498 | 3,420 |
| `cudaEventSynchronize` | 696 | 690 |

The GPU copy trace has 5,630 D2H copies carrying 841,945,984 bytes in
total. Of those, 5,014 copies are at most 4,096 bytes and carry only 137,024
bytes together; 616 larger copies carry the archive bulk. This separates the
many control-sized reads from the large archive transfers, but still does not
assign either category exclusively to the timed search interval. The
whole-trace GPU kernel table includes 1,080 NCCL SendRecv, 2,188 NCCL
AllReduce, 4,704 CUB radix-sort, and 354 generation GEMM launches. Summed
kernel durations can overlap across devices and are not end-to-end fractions.

Next diagnostic gate: mark each rank's measured search interval and depth
boundaries in the trace, then report per-interval host waits, D2H sizes,
stream idle gaps and NCCL/compute overlap. The existing static owner, route,
and retirement CPU dependencies remain unresolved by this diagnostic run.
