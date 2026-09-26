# Two-T4 LSA rank-map diagnostic, Kaggle v83

Private `trydotatwo/mgbfs-lsa-full-bfs-gate-t4` version 83 completed on two
P2P-capable T4s. The BFS source was pinned to
`868c118d180ee997d9eb13ff59a8086e1da8c58e`; notebook instrumentation
was committed as `799ee08`. No second Kaggle notebook was active.

The single Nsight-instrumented S10 DENSE/CUCO_RANK/NCCL_LSA run completed
46 layers and 3,628,800 unique states. Both rank archives verified. Search
completion was 0.515503 s and durable completion 10.417678 s; the external
50 ms sampler reported 597 MiB per rank. These are diagnostic observations,
not repeated unprofiled performance measurements.

The scoped CUDA API export again contains 1,462 stream-synchronize calls
and 1,740 synchronous memcpy calls. Versioned API rows have no callchains;
the unversioned `cudaStreamSynchronize` rows contain all 1,462 calls in
28 address-only chains. The `cudaStreamSynchronize` names are aliases of
the same calls, not an additional 1,462 waits. Captured `/proc/<pid>/maps`
for both ranks maps these 28 chains to 14 equal executable-relative callsite
patterns per rank. Two patterns have 90 calls each per rank; the others have
45 or 46 each. The 90-call pattern starting at executable offset `0x6909e`
accounts for about 203 ms and 198 ms of CUDA API duration on ranks 0 and 1,
respectively. API time is not critical-path time.

The maps identify the module and offset, but the exact unstripped ELF was
not exported. Therefore this report does **not** assign the offsets to Rust
functions or lines. Source review independently identifies per-batch archive
and post-owner host votes in `distributed_native.rs`; the diagnostic does
not yet prove which offsets correspond to them. A same-build `addr2line` or
Nsight `ResolveSymbols` pass is required before an exact callsite claim.

Compact outputs are under
`test_results/kaggle_lsa_rank_maps_v83/lsa-bfs-gate/`. The source-level
problem remains unchanged: one LSA receive slot is reused after a blocking
post-owner vote. Removing that vote without an owner-consumed event and
bounded group-failure protocol would risk overwriting live receive data.
