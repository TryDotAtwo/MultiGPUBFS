# Bucket-owned resident hash extents: experimental, not a default

This branch preserves an experiment that assigns each hash bucket its own segment extents. It improves one S12 timing but loses severe usable capacity under pressure and is slower on the observed S13 run. It must not be selected automatically or used as evidence of the maximum capacity.

## Evidence and scope

Four physical Tesla V100-SXM2-32GB, driver 580.178.04; one resident worker per GPU. Segmented24 packed-state resident batch engine; 32 shards; 4 virtual buckets; rehash enabled. The two sides in each row use the same tested executable SHA-256 `043c22dc45928590a36b9e9441578c619f6d377f2044c8e49011c9aba54381b9` and CUDA library SHA-256 `3cb83c0a0a03043218197f4a205e6267df31e46545808b1c216051c5aad78ee7`, differing in `MGBFS_RESIDENT_HASH_BUCKET_EXTENTS=0/1`. Tested source tree `bd1c50524da5cc491283be5a8d4ca5156e238bd5` (original experiment commit `e1fc91286cc6cc57148ed0be5fccbeb179d9c55c`); the source changes in this branch are the same eight files applied over the subsequently published baseline documentation. GPU metrics enabled. One observation per case, not a statistical performance result.

With no equal-budget pressure, S12 baseline/bucket completed in 19.638/15.503 s, sampled worst GPU 5,848/8,536 MiB; S13 completed in 353.475/433.464 s, sampled worst GPU 30,448/30,448 MiB. Geometry and complete receipts for those cases are in the preserved experiment archive. Timings with different effective geometry do not isolate atomic or copy cost.

## Equal segment-storage-budget test

Both cases below use compact position indexes, GPU stage telemetry, B=C=262144 records and the same user cap in each pair. Budgets below limit **segment storage per rank only**, not total VRAM: CUDA/NCCL reserve, queues, transport and scratch are additional. The observed NVML maxima are 0.5-second samples and can miss short peaks.

| Graph | Segment budget/rank | Layout | Outcome | Completed states | BFS s | Worst sampled MiB |
|---|---:|---|---|---:|---:|---:|
| S12 | 3,750,000,000 B | shared extents | COMPLETE, depth 66 | 479,001,600 | 19.527 | 5,592 |
| S12 | 3,750,000,000 B | bucket-owned | COMPLETE, depth 66 | 479,001,600 | 15.491 | 6,046 |
| S13 | 9,000,000,000 B | shared extents | INCOMPLETE, depth 28 | 636,298,069 | 26.968 | 10,848 |
| S13 | 9,000,000,000 B | bucket-owned | INCOMPLETE, depth 3 | 441 | 0.010 | 2,254* |

*The short S13 bucket run ended between samples; this is not its allocation peak. Rank-0 segment storage receipt is 8,999,995,852 B in both S13 cases. Requiring every bucket to own extents causes substantial fragmentation at this geometry. Segment-pool allocation is not the same as the number of simultaneously live states; the latter was not measured.

S12 requests 262144 rows/segment and a cap of 8,000,000 rows/rank; S13 requests 1048576 and 80,000,000 respectively. All published layer counts were compared with an independently computed Mahonian oracle; terminal reverse state was verified for complete S12/S13 runs. This is not a full-state-set check of intermediate large layers. Small 4-GPU S4 layer multisets, normal and forced collisions, and S6 tests passed. Resource stop/persistent recovery were checked on the small tests. There is no 8-GPU or B200 claim.

Per-case JSON receipts are compressed beside this report. Full logs and source/driver/topology evidence are archived as `multicta-hash-group-pressure-evidence.tgz`, SHA-256 `47d2807f8fbfdfc842b7f2af96d46567962240886b523b111497190d8a9e05c4`, and `multicta-hash-group-evidence.tgz`, SHA-256 `8e346a2ca68d044432b5958dd5a39fa689ca621617c16816c58068cdeb47845b`.

To reproduce the equal-budget decision, use the published build/test harness `tests/device_resident_multicta_bench_gpu.py`, configure 4 GPUs, `RESIDENT_AB_BACKEND=segmented24`, `RESIDENT_AB_WORLD=4`, `RESIDENT_AB_SHARDS=32`, `MGBFS_SEGMENT_TEST_BATCH=1`, `MGBFS_RESIDENT_HASH_REHASH=1`, `MGBFS_RESIDENT_HASH_VBUCKETS=4`, `MGBFS_SEGMENT_COMPACT_POS=1`, `MGBFS_RESIDENT_HASH_METRICS=1`, and set the budgets/row limits shown above. Run each graph with bucket extents 0 then 1. Keep the allocation receipts and 0.5 s samples separate.

## S13: reducing segment rows at the same storage budget

A follow-up on the same four V100 GPUs and the same executable/CUDA library changed the configured segment rows while keeping the **9,000,000,000 B segment-storage budget per rank**, 32 shards, four virtual buckets, 80M target-layer cap, transport and instrumentation. Each cell is one observation. The budget excludes queues, scratch and CUDA/NCCL reserve.

| Rows per segment | Extent ownership | Outcome | Completed states | BFS s | Worst sampled MiB |
|---:|---|---|---:|---:|---:|
| 1,048,576 | shared | INCOMPLETE, depth 28 | 636,298,069 | 26.968 | 10,848 |
| 1,048,576 | bucket-owned | INCOMPLETE, depth 3 | 441 | 0.010 | 2,254* |
| 262,144 | shared | COMPLETE, depth 78 | 6,227,020,800 | 550.692 | 11,190 |
| 262,144 | bucket-owned | INCOMPLETE, depth 35 | 2,102,968,914 | 127.453 | 11,190 |

*The 1,048,576-row bucket case ended between NVML samples. Rank 0 allocated 8,999,983,340 bytes of segment storage in each 262,144-row case. Smaller extents substantially reduce stranded capacity for this workload, but the separately owned buckets still stop earlier than the shared pool. Elapsed times from different completed depths are not comparable as whole-graph speed.

Every completed layer count was checked against Mahonian coefficients, and the complete run's terminal reverse permutation was checked. This is not a full-state-set check for the large intermediate layers. The two 262,144-row machine-readable receipts are `artifacts/2026-11-23-hash-fragmentation-262144-shared.json.gz` and `artifacts/2026-11-23-hash-fragmentation-262144-bucket.json.gz`. The interim archive `multicta-hash-fragmentation-midpoint.tgz` has SHA-256 `f19519521cdbec134b201a55d7924bb647d71b7f0b1c4322792b04d75c28a8ea`. No 8-GPU or B200 inference follows from these cases.
