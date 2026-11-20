# Segmented BFS: 8×RTX 3060 S12, 2026-11-20

One physical host with eight GeForce RTX 3060 12 GiB (SM86), driver 580.178.04. The topology has two four-GPU NUMA groups: PHB paths within groups and SYS paths between them; there is no NVLink. These measurements do not characterize an 8×B200 system.

All six cases use the same GPU source commit `d067734f3e9f012e3e394ccf712f123867c00e0c`, executable SHA-256 `82c40088a24bb8620a9fdc3e8272646546a41d49773739433bb6faa19c1198e3`, CUDA library SHA-256 `c567135bde4bdf014b6137c90956f6ad44fd7f7fea8a2c075e3425213fdf970e`. Reproducibility metadata, scripts, all per-rank receipts, layer sizes/times and one-second GPU memory samples are included in the attached evidence archive. Graph states are not included. The matching modified source files were published at `17e3cfc11e269b77b9da796cc5c02c3c8d1fd1a1`.

Each cell is one uninstrumented run: S12, adjacent transpositions, identity start, 479,001,600 states, 67 layers, 8 devices, 8 shards, 4 owner lanes, balanced preset, configured 2^32 rows/segment with actual allocation in 1,048,576-row chunks, batch 8,192, exact exchange, autotune off. All six completed. All layer counts match Mahonian numbers; the full final two layers match the independent exact sets with no cross-rank duplicates. Earlier layers are checked by counts, not complete sets.

| backend | strategy | BFS s | depths 28–38 s | max live rows/rank | tracked storage peak bytes/rank |
|---|---|---:|---:|---:|---:|
| permutation_bitmap | atomic | 106.472 | 56.657 | 9,640,704 | 301,990,080 |
| permutation_bitmap | warp | 110.822 | 59.088 | 9,640,704 | 301,990,080 |
| permutation_bitmap | batch | 276.451 | 141.347 | 9,640,704 | 301,990,080 |
| segmented24 | atomic | 182.160 | 96.902 | 9,548,478 | 1,040,187,584 |
| segmented24 | warp | 164.708 | 86.573 | 9,548,098 | 1,040,187,584 |
| segmented24 | batch | 193.370 | 104.006 | 9,547,589 | 604,002,084 |

The common window contains 259,128,390 newly discovered states and 2,829,700,863 generated edges. All devices had the same one-second-sampled peak of 1,497 MiB in each case; preflight and short-lived allocations can dominate this measure, and short peaks can be missed. Tracked storage peak counts segment state, index, counters and catalog, excluding queues, scratch, bitmap and CUDA/NCCL internals. It is not total or peak VRAM. Results do not establish optimal profiles or maximum capacity.

The small-graph eight-rank exact/padded test individually passed all 12 CPU exact layer-set checks (both backends, three graphs, both exchange modes) including empty self and middle peer sources and persistent reuse. Its aggregate exit was a failed test assertion because these workloads generated no spill; that is not a passing spill check. A separate forced-spill test passed on all eight GPUs for both backends and both transport modes: hash replayed 57 spill tiles and bitmap 36 per mode. Each run also forced a cap-based resource stop, checked the completed prefix, and completed another graph with the same persistent workers. Two-rank forced spill and resource-stop checks are documented in the companion report.

Bitmap A/B and C use different local shard mapping because C assigns one writer to each bitmap word; the timings compare whole strategies and do not isolate atomic instruction cost. Batch still has bounded grant and error atomics. Batch remains opt-in.

Evidence archive: `2026-11-20-segmented-8gpu.json.gz`, SHA-256 `d523f38f27cb88baa2ade93b213e563cd12d5e602fc7a3e06a9bc13041e607ee`.
