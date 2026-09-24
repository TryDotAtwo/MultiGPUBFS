# CUCO_RANK S10 timeline, 2×T4, 2026-09-24

Private Kaggle notebook `trydotatwo/mgbfs-cuco-rank-timeline-t4` v2 ran
source `d95ef218d321cd35b601c7db6440ec22038e966d` on two physical T4s.
The host-sized NCCL `CUCO_RANK`, DENSE, pre-dedup ON S10 reference run
completed with verified per-rank archives. Its profiled measurement reported
0.54 s search and 5.77–5.78 s through local archive commit per rank. These
are **diagnostic Nsight-instrumented timings**, not comparable to unprofiled
speed medians.

Nsight Systems captured process startup, warmup, BFS and archive together.
Aggregate CUDA API summary across captured processes: 6,266
`cudaStreamSynchronize` calls (0.987 s summed API duration), 4,952
synchronous `cudaMemcpy` calls (0.245 s), 6,214 `cudaMemcpyAsync` calls,
and 1,040 `cudaHostAlloc` calls (2.68 s). The pinned allocations occur at
archive construction; this **does not** prove archive allocation is in the
timed BFS hot path. The GPU memory-operation summary reports 5,632 D2H
copies, but these are not isolated to the search interval.

The aggregate GPU-kernel summary assigns 35.1% of summed kernel time to
`ncclDevKernel_SendRecv`, 10.2% to NCCL all-reduce, 13.1% to CUB one-sweep
radix-sort kernels, 11.1% to `modular_materialize`, and 3.9% to the CUTLASS
generation GEMM. These percentages are **not a critical-path breakdown**:
kernels overlap, both ranks are combined, and startup/warmup are included.
Still, the actual run confirms many blocking host calls and D2H operations,
supporting the source audit's CPU-dependency concern. It does not measure
the new LSA branch, which was selected in a separate full-state gate.

Next profiling gate: annotate/capture the search interval and rank/stage,
then count host waits and D2H, launch gaps and overlap specifically within
that interval. Do not attribute the 2.68 s pinned-allocation total or
aggregate kernel percentages to search latency without that filter.

Evidence: `test_results/kaggle_cuco_rank_timeline_v2/library-owner/summary.json`,
`screen-s10-dense-cuco_rank-w2-r0/screen-summary.json`, and
`screen-s10-dense-cuco_rank-w2-r0-nsys-stats.log`.
