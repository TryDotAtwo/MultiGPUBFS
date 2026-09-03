# T4 mode sweep: native DENSE/CUB versus CayleyPy

Kaggle: private `trydotatwo/mgbfs-mode-sweep-t4`, version 1, successful,
runtime 44m 9s on a 2xT4 host. Every timed process saw one T4; this is a
single-GPU comparison, not a two-GPU scaling result.

Native source: `5482f3bb9d20db5780bf6b5c915c4d93c8cd321c`.
CayleyPy baseline: `f0f2b8e5ee61173039ab9742f3a7756c9b6365e6`.
Each table entry is the median of five fresh processes; `+-` is MAD. All 134
workers completed. The raw-result verifier checked every worker log, five
distinct repetitions, layer counts, full-state U4(8) digests, runtime mode
fields, archive identities, and independently recomputed every statistic.

## Fixed batch: generation variant and local pre-dedup

U4(16), batch 262144, 16 shards, 16 job buckets, 256 owner buckets. VRAM is
the maximum whole-device `nvidia-smi` sample, including warmup.

| Generation | Pre-dedup | Search, s | MAD, s | Durable archive, s | Peak VRAM, MiB |
|---:|:---:|---:|---:|---:|---:|
| 0 | OFF | 1.237 | 0.007 | 5.695 | 2215 |
| 0 | ON  | 1.196 | 0.027 | 5.810 | 2215 |
| 1 | OFF | 1.224 | 0.011 | 5.772 | 2215 |
| 1 | ON  | 1.192 | 0.006 | 5.695 | 2215 |
| 2 | OFF | 1.227 | 0.001 | 5.804 | 2215 |
| 2 | ON  | 1.181 | 0.011 | 5.784 | 2215 |
| 3 | OFF | 1.224 | 0.009 | 5.756 | 2215 |
| 3 | ON  | 1.188 | 0.004 | 5.679 | 2215 |
| 4 | OFF | 1.195 | 0.016 | 5.712 | 2215 |
| 4 | ON  | **1.159** | 0.016 | 5.798 | 2215 |

Generation 4 is the best fixed-batch layout. Pre-dedup ON improves every
generation variant by about 2.6-3.7%; it does not reduce allocated VRAM in the
current implementation because both paths reserve the same scratch space.

## Batch and shard panels

U4(16), pre-dedup ON. These are explicitly selected panels, not a full
Cartesian product. H=64 uses J=4; other shard rows use J=16, so that row does
not isolate shard count alone.

| Generation | Batch | Shards | J | Search, s | MAD, s | Durable, s | VRAM, MiB | Pinned RAM, MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 262144 | 1  | 16 | 1.193 | 0.016 | 5.827 | 2215 | 1536 |
| 0 | 262144 | 4  | 16 | 1.205 | 0.012 | 5.733 | 2215 | 1536 |
| 0 | 262144 | 16 | 16 | 1.196 | 0.027 | 5.810 | 2215 | 1536 |
| 0 | 262144 | 64 | 4  | 1.626 | 0.020 | 5.748 | 2191 | 1536 |
| 0 | 65536  | 16 | 16 | 1.941 | 0.004 | 5.658 | 1819 | 768 |
| 0 | 524280 | 16 | 16 | 1.059 | 0.012 | 5.923 | 2753 | 2576 |
| 4 | 65536  | 16 | 16 | 1.908 | 0.015 | 5.697 | 1819 | 768 |
| 4 | 262144 | 16 | 16 | 1.159 | 0.016 | 5.798 | 2215 | 1536 |
| 4 | 524280 | 16 | 16 | 1.004 | 0.006 | 5.895 | 2753 | 2576 |
| 4 | 1048576| 16 | 16 | **0.978** | 0.014 | 6.257 | 3819 | 4608 |

The large batch wins search time but trades memory for it. H=1/4/16 are
effectively tied at this workload. H=64/J=4 is 36% slower than H=16/J=16,
consistent with excessive small owner jobs; it is not a useful default.

## Best native versus tuned CayleyPy

| Graph | Backend/config | Search, s | MAD, s | Speedup vs best CayleyPy | Peak VRAM, MiB | Durable archive, s |
|---|---|---:|---:|---:|---:|---:|
| U4(16) | Native g4, pre ON, b1048576, H16/J16 | **0.978** | 0.014 | **1.651x** | **3819** | 6.257 |
| U4(16) | CayleyPy b1048576 | 1.616 | 0.004 | 1.000x | 6283 | n/a |
| U4(24) | Native g4, pre ON, b1048576, H16/J16 | **14.662** | 0.001 | **1.574x** | **5255** | 68.276 |
| U4(24) | CayleyPy b1048576 | 23.085 | 0.175 | 1.000x | 11285 | n/a |

Native uses 39.2% less peak VRAM on U4(16) and 53.4% less on U4(24). Its
pinned archive rings are 4608 MiB and 9952 MiB respectively. CayleyPy creates
no archive, so native durable time is reported but is not comparable with the
CayleyPy search-only number. The asynchronous archive does not materially
extend `search_seconds`, but disk completion remains the dominant wall-clock
tail after search.

For completeness, CayleyPy U4(16) batch medians were 2.438 s (65536), 1.838 s
(262144), and 1.616 s (1048576), with peaks 1941, 3563, and 6283 MiB.

## Availability boundary

Only the implemented single-rank `DENSE + CUB_SORT_MERGE` runtime was measured.
`HASH_FIRST`, `BMMA_BUCKET`, and native multi-rank NCCL are not implemented and
therefore have no honest timing row. Generation 0-4 are kernel layouts, not five
different BFS semantics. All measured native rows still execute exact DENSE
owner deduplication and mandatory durable archive.

Machine-readable results are in `2026-09-03-mode-sweep.csv`; downloaded raw
logs and the Kaggle summary are under `test_results/mode-sweep-v1/`.
