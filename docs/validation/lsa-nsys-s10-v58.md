# Two-T4 LSA BFS timeline, Kaggle v58

Status: one **profiled diagnostic**, not a performance comparison or a
CPU-free pipeline gate. The private Kaggle notebook
`trydotatwo/mgbfs-lsa-full-bfs-gate-t4` version 58 completed with source
`a5e28a2d18c9f2220c811e6c17a503532df0ec75` on two P2P-capable Tesla
T4s. Configuration: S10, DENSE, CUCO_RANK, NCCL_LSA, local pre-dedup ON,
batch 32768, four shards/rank, 256 buckets, 1,000,000 state/rank capacity,
96 MiB cuCO pool/rank, 256 archive slots/rank. Both archives passed the
committed checksum/count verifier. Raw downloaded summaries are under
`test_results/kaggle_lsa_timeline_backtrace_v58/lsa-bfs-gate/`.

Rank 0/1 search completion: 0.460643/0.468028 s. Durable run commit:
6.722636/4.674530 s. These are Nsight-instrumented values and must not be
compared directly with unprofiled benchmark medians.

Within the captured timed-BFS range, Nsight reported 1,462
`cudaStreamSynchronize` runtime calls across both ranks (379.16 ms aggregate
API time), 1,740 synchronous `cudaMemcpy` calls (105.88 ms aggregate API
time), 1,321 `cudaMemcpyAsync` calls, and 177 `cudaMemcpy2DAsync` calls.
CUDA GPU activity includes 180 LSA exact-copy kernel instances (181.70 ms
aggregate), 1,092 NCCL all-reduce kernel instances (79.09 ms aggregate),
2,352 CUB radix onesweep instances (84.10 ms aggregate), and 177 modular
materialization instances (130.99 ms aggregate). These sums span two devices,
can overlap, and are **not** a critical-path decomposition.
The cuCO rank path's CUB selection kernels sum to roughly 4.5 ms across
both ranks in this trace; the fixed-capacity scan remains a source-level cost
hypothesis but is not the first optimization target on this S10 profile.

The callchain export did not resolve source symbols: it contains absolute
addresses for two ASLR processes, plus duplicate CUDA API name variants.
It confirms many runtime waits but cannot attribute their exact counts to
Rust callsites. Source inspection independently identifies per-parent-batch
archive `all_max`, pre-owner LSA gate, and post-owner
`all_max_ring_or_host_fatal` in `distributed_native.rs`; removing any one
without an asymmetric-failure protocol would be unsafe. A follow-up trace
needs rank process mappings and a retained unstripped binary (or a correctly
symbolized callchain export) to distinguish these waits from finalize and
other CUDA library calls.

The completed run validates this one S10 configuration and its archives.
It does not validate HASH_FIRST, eight-rank LSA, four Compute Sanitizer gates,
or a host-readback-free full owner-to-retirement epoch.
