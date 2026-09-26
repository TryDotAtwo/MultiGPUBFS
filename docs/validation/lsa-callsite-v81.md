# Two-T4 LSA callsite diagnostic, Kaggle v81

The private notebook `trydotatwo/mgbfs-lsa-full-bfs-gate-t4` version 81
completed on two P2P-capable Tesla T4s. Source revision:
`8f6bcc266d6fd5f0a18d8b420557a0010d8786bc`. This was the only
Kaggle notebook used for this diagnostic. It ran the S10, DENSE,
CUCO_RANK, NCCL_LSA path with local pre-dedup ON, batch 32768, four
shards/rank, and 1,000,000 state/rank capacity. The global layer sizes
sum to 3,628,800; both committed archives passed checksum/count verification.
Raw compact outputs are under
`test_results/kaggle_lsa_timeline_backtrace_v81/lsa-bfs-gate/`.

The Nsight-instrumented search-completion time was 0.492828 s; durable
run commit was 6.430783 s. Sampled full-device peak was 597 MiB per
rank, 1,194 MiB total. These are one diagnostic sample, not unprofiled
speed or exact VRAM peaks.

In the timed BFS capture, the CUDA API summary recorded 1,462
`cudaStreamSynchronize` calls (416.56 ms aggregate API duration in the
versioned row), 1,740 synchronous `cudaMemcpy` calls (104.15 ms), 1,321
`cudaMemcpyAsync` calls, and 177 `cudaMemcpy2DAsync` calls. The SQLite
callchain extraction has empty symbols for every versioned sync/copy row;
the nonversioned alias rows contain only raw ASLR addresses. The aliases
must not be added to the versioned counts. The aggregate API durations
span both ranks and can overlap; they do not assign critical-path cost to
specific Rust source lines.

Compared with v58, the new profile confirms the same count pattern, but
`--resolve-symbols=true` still did not yield source-level attribution.
The next callsite experiment needs a retained unstripped binary and rank
process mappings, or a symbolized export verified before deleting the
trace. Until then, the source-identified owner, archive-vote, and
post-owner waits remain hypotheses about the measured critical path.

This run does **not** establish a CPU-free owner→transport→retirement
pipeline, full DAG capture, HASH_FIRST correctness, four Compute
Sanitizer gates, or a five-repeat Pareto comparison.
