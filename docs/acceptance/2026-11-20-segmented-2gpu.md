# Segmented BFS: 2×RTX 3060 acceptance, 2026-11-20

Source on GPU: `d067734f3e9f012e3e394ccf712f123867c00e0c` (tree `d4a5e50ac0292b1d13e414429284d5adc8032d2c`); the modified source files were overlaid onto published commit `17e3cfc11e269b77b9da796cc5c02c3c8d1fd1a1` (the root tree SHA differs). Native executable SHA-256 `82c40088a24bb8620a9fdc3e8272646546a41d49773739433bb6faa19c1198e3`; CUDA library `c567135bde4bdf014b6137c90956f6ad44fd7f7fea8a2c075e3425213fdf970e`. See archive for exact invocation, per-rank memory receipts, sizes, times, sampled telemetry and test logs. No graph states are included.

All results below are one uninstrumented run for Coxeter S12, 479,001,600 states, depths 0–66; 2×RTX3060, 8 shards, four owner lanes, balanced preset, 1,048,576 allocated rows/segment, configured max 2^32, batch 8192, exact transport, autotune off. Full graph COMPLETE in each cell; all layer sizes match independent Mahonian numbers. The final two layers' complete exact state sets match the oracle, including no cross-rank duplicates. Earlier layers are checked by count, not by full state sets. The window is the same completed depths 28–38 in every cell.

| backend | strategy | BFS s | window s | sampled peak MiB (each GPU) | maximum live rows/rank | tracked storage peak bytes/rank |
|---|---|---:|---:|---:|---:|---:|
| permutation_bitmap | atomic | 75.215 | 39.836 | 935 | 38,179,213 | 603,980,160 |
| permutation_bitmap | warp | 28.041 | 14.693 | 935 | 38,179,213 | 603,980,160 |
| permutation_bitmap | batch | 83.350 | 43.655 | 955 | 38,179,213 | 603,980,160 |
| segmented24 | atomic | 88.940 | 47.561 | 2,313 | 38,186,177 | 2,080,375,936 |
| segmented24 | warp | 41.770 | 22.772 | 2,313 | 38,186,177 | 2,080,375,936 |
| segmented24 | batch | 117.343 | 68.385 | 1,491 | 38,179,331 | 1,208,004,132 |

NVML/`nvidia-smi` memory is sampled every second, so short transients may be missed. Tracked storage peak covers segment states, index, counters, run directory and descriptors; it excludes queues, scratch and CUDA/NCCL internals and is not total VRAM. Per-rank current breakdown and detailed telemetry are in the archive; this is not a measured maximum capacity at equal VRAM. Results apply only to this hardware and graph, not B200/B300. These runs do not establish an optimal default. `batch` is opt-in.

For S13, one earlier bitmap/warp run on the same 2×3060 fully completed 6,227,020,800 states in 575.274 BFS seconds; depths 34–44 took 283.486 seconds. All layer counts match Mahonian, final two exact sets match, and the 1 s sampled peak is 6,461 MiB per GPU. Its distinct source and binary SHAs are in its invocation.

## Dense incoming measurement

Instrumented before/after single runs on the common completed S12 window 28–38 (timings are not wall-clock decomposition): bitmap batch 42.967 → 41.485 s, raw sort/scan items 14,035,048,790 → 2,829,700,863; hash batch 69.098 → 69.242 s, raw items 4,299,258,624 → 2,829,700,863. Both versions saw the same valid input count. This removes padding, but the measured effect is small in this experiment. Hash window inspection: max observed live runs/shard 83–91, median 72–76; unique+probe and merge were about 69–71% of the sum of recorded lane event intervals. Those intervals may overlap and include scheduling; they are not a percentage of wall time. Runs remain valid when compaction is deferred.

## Physical correctness checks

See logs inside the archive for two-GPU exact/padded transport, CPU exact layer sets for small directed and repeated-symbol graphs, persistent reuse, forced hash collisions, constrained rows, spills, resource stops, previous-library compatibility and Compute Sanitizer for bounded incoming mapping. The `2^32−1` last-row test uses virtual mapping of small physical pages; it validates indexing and spill, not a physical allocation of 2^32 rows. Independent small-case overlap instrumentation recorded overlapping route/owner intervals; no end-to-end S12 overlap percentage is claimed.

Atomic strategy uses per-item queue reservations and bitmap atomic OR / hash slot CAS. Warp groups queue and owner reservations; bitmap groups word updates while hash slot CAS remains. Batch routes through bounded sort/scan and uses word-owned bitmap commit or exact sorted hash runs; bounded cap reservation and error/status atomics remain. No strategy buffers all children of a layer.

Artifact: `2026-11-20-segmented-2gpu.json.gz`, SHA-256 `49a84a9addc1e8ae3434f21fb333974debaff90a65434eafed65c8632e620971`. Eight-GPU tests are tracked separately.
