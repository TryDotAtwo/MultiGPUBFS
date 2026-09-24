# CUCO_RANK device-count window: 2xT4 validation

Source `fb8b9f4669199d0b95c766f051b4ad6d1d7e4892`. The private Kaggle
notebook `trydotatwo/mgbfs-cuco-rank-paired-s10-pool96-t4` v4 completed on two
physical Tesla T4 GPUs. Downloaded raw output is in the ignored
`test_results/kaggle_rank_device_window_v4/library-owner/` directory.

The plain full-state two-GPU oracle passed (3 tests, one eight-GPU test
correctly ignored). The full S10 screening run completed with `CUCO_RANK`,
96 MiB pool/rank, DENSE, pre-dedup ON and 32,768 parent batch. Its 46 layer
counts match the CUB control exactly and sum to 3,628,800 states. Both
per-rank archives passed checksum/count verification. One unprofiled sample:

| Backend | Search, s | Durable commit, s | Sampled VRAM, MiB/rank |
|---|---:|---:|---:|
| CUCO_RANK | 0.400799 | 3.750525 | 529 |
| CUB_SORT_MERGE | 0.847862 | 3.871365 | 457 |

This is a correctness gate and one screening sample, not a repeated
performance claim. The 50 ms `nvidia-smi` samples are not exact peaks. The
second private notebook, `trydotatwo/mgbfs-rank-retirement-gate-t4` v5,
used the same source commit and finished with `summary.json: PASS`. The
one-GPU BFS suites passed on each physical T4, and the two-GPU suite passed
three tests (the eight-GPU test was correctly ignored) in plain mode and under
`memcheck`, `racecheck`, `initcheck`, and `synccheck`. The two-GPU sanitizer
summaries report zero errors; racecheck reports zero hazards and warnings.
Raw output is in ignored
`test_results/kaggle_rank_device_window_sanitizers_v5/library-owner/`.

These are bounded full-state and fault fixtures, not an S10 sanitizer run or
proof of every asynchronous lifetime. The new owner window removes a
redundant CPU-to-GPU count upload, but transport payload sizing, group votes
and retirement still have host dependencies; this result does not prove a
CPU-free BFS pipeline.
