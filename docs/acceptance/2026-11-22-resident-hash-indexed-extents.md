# Resident hash: indexed extents

`MGBFS_RESIDENT_HASH_INDEXED_EXTENTS=1` is an experimental opt-in for the
resident hash. Every shard maintains a job-owned list of extents it currently
leases and a bounded cursor for finding a free extent. During a layer the
existing shard lease protects that list. At a drained layer boundary one
control thread per rank rebuilds the lists after bank retirement; newly leased extents
join the list only after a successful rank-cap reservation. A failed capacity
reservation leaves the extent free, and a full-pool retry preserves its packet
under the existing all-shard recheck protocol.

The layout is agreed across ranks before the job, uses checked allocation and
64-bit list entries, is included in admission memory, and adds the index bytes
to the native receipt. Incompatible hash-config ABIs produce a capability error.
Unset/`0` retains the prior full descriptor scan. This does not change the
normal host backend or enable the experimental resident path by default.

## Correctness and scope

The tested executable SHA-256 was
`abf8ec60558364e2e3fae3c7e45476dd7fe5e7b4e4ba88585059e28003a8ae9c`; the
loaded CUDA library SHA-256 was
`c936024ea15ac6b3cd969dc4a5c8d3d5775be35da05ee456aa81d3417d46c94b`.
Source parent: `fbd5a2b1e534a715f31617eca5283212d283a4ea`.

On 4 physical V100-SXM2 GPUs, full layer multisets for S4, repeated symbols
and signed/wide symbols matched a CPU BFS, including two active shards,
tiny segments, hard caps, duplicate-at-full and persistent reuse. A separate
2-GPU all-hashes-zero run matched every layer multiset. That collision fixture
concentrates keys in one shard, and does not prove simultaneous cross-shard
full-hash collision handling. A separate owner-primitive CTest ran indexed and
unindexed lists with forced collisions, two active shards, cap replay,
retirement and a virtual last u32 row; the virtual allocation tests
addressability and is not a physical 2^32-row test.

Actual CayleyPy persistent jobs on four GPUs passed S4, repeated-symbol,
duplicate-at-full, resource prefix and healthy-next cases. A pool-pressure
test confirmed exhaustion and a correct prefix. An S6 contention search on
four GPUs observed two overlapping owners and two actual failed freeze
attempts followed by recheck; its completed layer sizes and last layer
multiset matched the CPU oracle. The contention observation is scheduling
dependent. A 2-GPU rank-kill test invalidated the generation and an explicit
new one completed a healthy job. Two-GPU memcheck with API-error reporting
disabled showed zero memory errors on the pressure/healthy-next fixture. The
[previous report](2026-11-22-resident-hash-parallel.md) records API errors
seen with the default sanitizer setting; suppressing them is not a clean
default sanitizer run.

## S9 A/B on one build

Four V100-SXM2-32GB GPUs, CUDA 12.8.93, NCCL 2.30.7; adjacent transpositions,
362,880 states, 37 layers, atomic hash and parallel shard owners, two shards,
8192 rows/segment, 46 admitted extents per rank, eight lanes, eight records per
packet. Both variants used the same binary, segment bytes and topology. The
index used an additional 400 bytes per rank; the 43,336,784 segment-storage
bytes/rank do not include the runtime or device-wide peak. Instrumentation
counters for the experimental owner were disabled. Ordering was off, on, on,
off. Every layer count matched the Mahonian oracle and the complete final
state matched the expected reversed permutation.

| Extent index | BFS seconds | End-to-end wall seconds |
|---|---:|---:|
| off | 27.824423 | 35.267982 |
| on | 15.392179 | 23.223041 |
| on | 15.377560 | 22.812392 |
| off | 27.720143 | 35.094153 |

Two observations in this order show approximately 45% lower BFS time for
this small graph and configuration. A previous single host-hash observation
on the same node completed BFS in 0.428312 s; its allocated geometry differs.
The resident path therefore remains much slower and memory-heavier here.
The result is not an 8-GPU, B200, large-layer or production-capacity claim.

[Machine-readable comparison and contention receipts](artifacts/2026-11-22-resident-hash-indexed-extents.json)
include the per-rank geometry. Full test output, exact layer snapshots,
binaries and sanitizer logs are retained in the corresponding evidence archive.
The hash path still does not support directed history, matrices, resident
warp/batch, max_capacity or autotuning; exact Explore is not implemented.
