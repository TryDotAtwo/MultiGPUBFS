# Current-source owner/profile timelines

Runtime source edd45cd; inspected public HEAD 5ee2486. Existing Linux CLI
SHA256 4be24145ba4f0296ed2cefcee7d3b8a8f00e869732ad223f36c017924788637f.
NVTX native rebuilt from current source, SHA256
53ab8c92bc55dbefd6ba7f0609004b3e893aefd73727e169f01c62c011bed0fe.
Single sm86 RTX 3070 Laptop, diagnostic NCCL archguard, not two T4s.

Full S8 mandatory archives, batch 1024, layer 40320/ring 80640, buckets 8,
shards 4, job buckets 2, bucket capacity 40320, pre-dedup ON, matrix_u8,
archive rows 512/slots 128. CUCO fixed pool 64 MiB. No warmup; NVTX ranges
enabled, TRACE_ROUTE disabled. Nsight 2025.6.3 cuda/nvtx/osrt, sampling off.

## Full-state and host-dependency evidence

| Owner | Profile | Full canonical CPU oracle | Batch ranges | Batch sync APIs | Non-archive D2H in batch |
|---|---|---|---:|---:|---:|
| CUCO_RANK | DENSE | 40320 states, all 29 layers | 61 | 0 | 0 |
| CUCO_RANK | HASH_FIRST scalar | 40320 states, all 29 layers | 61 | 0 | 0 |
| CUB_SORT_MERGE | DENSE | 40320 states, all 29 layers | 61 | 0 | 0 |
| CUB_SORT_MERGE | HASH_FIRST scalar | 40320 states, all 29 layers | 61 | 0 | 0 |

Batch API classification checks cudaMemcpy (synchronous), stream/event/device
Synchronize APIs, with globalTid and full interval containment in mgbfs.batch.
GPU D2H events are joined to their CUDA API correlation, then attributed to
the nested mgbfs.archive_d2h range. Every batch D2H event belongs to archive;
there is no recorded batch counts/control D2H. Device credit/event querying
and GPU dependencies still exist; this does not mean all CPU work is absent.
FinalizeDepth readbacks/drains remain intentional. This diagnostic cannot
exercise peer traffic, asymmetric cancellation or receive-slot reuse.

CUCO DENSE profiled search/durable: 0.542440/0.855379 s; HASH_FIRST scalar:
0.772934/0.807321 s. Explicit aligned device plans 75813632/75190528 bytes;
archive pinned 5242880 bytes each. Setup/final cudaMemGetInfo observation
1750597632 bytes is not a continuously sampled peak. These are single
instrumented diagnostics, not paired timing or a speedup claim. Recorded
GPU busy interval unions 22.368/41.680 ms, multi-stream unions 0.841/0.770 ms;
small graph, substantial host/finalization/profiling overhead, not saturation.

## Explicit architecture rejections

CUCO HASH_FIRST INT_MMA_SM75 rejected on first generation with CUDA_STATUS_3;
hash_first_generate.cu explicitly requires major=7/minor=5. BMMA_BUCKET DENSE
rejected at owner plan creation with status=3; bounded_owner.cu has the same
sm75-only guard. Neither failed case publishes rank COMPLETE/group-complete.
The sequential panel stopped on rejection; BMMA HASH_FIRST was not run.
Guards were not removed, no scalar/CUB fallback was enabled, and no sm75 pass
is inferred from sm86 behavior. Both Tensor variants need actual T4 evidence.

Artifacts: test_results/owner_timeline_edd45cd_20261002/, per-case full.nsys-rep,
full.sqlite, logs, archives and result/rank-0.json; failed cases retain trace
and rejection logs too. Sessions 33279 exited 0, 82067/46583 exited 1 for the
documented architecture rejections. All four successful archives independently
verified with verify_process_archives(n=8,world=1).

The sole private T4 gate remains pinned to edd45cd. Weekly GPU quota rejected
its last admission; no duplicate notebook launched. Two independent T4 ranks,
four full-runtime sanitizer gates including NCCL registration, asymmetric
failures, current two-rank timeline and five-repeat A/B remain open. Distributed
macro, production CLI and broader DB/HF completion are not established here.
