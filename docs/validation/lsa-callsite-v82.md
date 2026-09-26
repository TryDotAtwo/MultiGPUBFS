# Two-T4 LSA callsite diagnostic, Kaggle v82

The private `trydotatwo/mgbfs-lsa-full-bfs-gate-t4` version 82 completed
on P2P-capable 2×T4 with source
`868c118d180ee997d9eb13ff59a8086e1da8c58e`. This is one
Nsight-instrumented S10 DENSE/CUCO_RANK/NCCL_LSA run, not a performance
sample. Only this Kaggle notebook was active.

The run completed all 46 layers and 3,628,800 unique states. Both rank
archives passed committed checksum/count verification; both rank result
files now publish `logical_owner_to_rank: [0, 1]`, exercising the result
identity change in `868c118`. The instrumented search and durable times
were 0.499280 and 5.852692 seconds. Externally sampled device consumption
peaked at 597 MiB per rank (1,194 MiB total); 50 ms sampling is not an
exact peak. Raw compact outputs are in
`test_results/kaggle_lsa_timeline_backtrace_v82/lsa-bfs-gate/`.

The scoped CUDA API rows contain 1,462 `cudaStreamSynchronize` calls
(499.87 ms aggregate API duration for the versioned row) and 1,740
synchronous `cudaMemcpy` calls (99.30 ms). The exported callchains still
have zero resolved symbols for versioned rows and only raw ASLR addresses
for the nonversioned aliases, despite an unstripped release binary and an
explicit Nsight debug-symbol search path. These duplicate API names must
not be summed. The result confirms remaining host dependencies but not
their source-level counts or critical-path cost. It does not justify
deleting the post-owner wait independently of receive-slot ownership and
group failure propagation.

The next profile must retain rank process mappings and resolve the
captured addresses against the exact unstripped executable, or use
NVIDIA's `ResolveSymbols` utility on the report before deleting it.
