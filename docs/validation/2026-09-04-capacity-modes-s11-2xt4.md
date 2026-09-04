# Full S11 capacity-mode A/B on 2xT4 (2026-09-04)

Private Kaggle notebook `trydotatwo/mgbfs-capacity-modes-t4`, version 2,
completed the full 39,916,800-state S11 graph under both immutable capacity
policies.  Both runs produced the same 56 BFS layers and mandatory per-rank
archives.  Source under test was
`3378317bfce34ed38c06a886d692c9f8f6a91769`.

| mode | records/rank | records/cluster | search s | durable s | CUDA peak/rank | nvidia-smi peak/rank |
|---|---:|---:|---:|---:|---:|---:|
| `EQUAL_GLOBAL` | 4,000,000 | 8,000,000 | 4.013698 | 47.780198 | 4,279,107,584 B | 4,081 MiB |
| `MAX_PER_RANK` | 8,000,000 | 16,000,000 | 4.136870 | 47.354197 | 7,452,098,560 B | 7,107 MiB |

`EQUAL_GLOBAL` therefore saved 3,172,990,976 device bytes (3,026 MiB) on
each T4 while slightly improving search time in this single A/B run.  Durable
time is effectively unchanged and is dominated by archival I/O.  Each rank
also used a fixed 3,142,451,200-byte pinned archive ring; that host allocation
is identical between modes and is not included in the CUDA figures.

This confirms that the previous roughly 7 GiB per-GPU result was the
`MAX_PER_RANK` capacity policy, not an inherent requirement of two-rank BFS.
For equal total graph capacity, 2xT4 now uses about 4 GiB per GPU.  It does not
halve aggregate device memory exactly because CUDA/NCCL context, route slots,
scratch and other fixed per-rank structures are replicated.

The search comparison is one run per mode, not a five-repeat timing claim.
Raw per-rank records, logs and 50 ms memory samples are retained under
`artifacts/kaggle/capacity-modes-v2/capacity-modes/`.
