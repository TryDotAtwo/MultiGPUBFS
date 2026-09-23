# Paired S10 owner benchmark on 2xT4

Private Kaggle `trydotatwo/mgbfs-cuco-rank-paired-s10-t4` v1 completed on two
physical Tesla T4 GPUs. The CUCO_RANK source was
`4b68552b0c0c865862d896a2ca1913043110e575`; the preserved native
baseline was `013ed5c979f4225db273e0015fa9ed72fd230c90`. Raw output is in
`test_results/kaggle_cuco_rank_paired_s10_v1/library-owner/`.

All ten runs are `COMPLETE`: S10 DENSE, batch 32,768, pre-dedup ON, two
torchrun ranks, 1,000,000 declared states/rank and 1,000,000 StateRing
records/rank. Every run has the same 46-layer histogram totaling 3,628,800
states, warmup complete, archive enabled, and both rank archives verified.
The old native record does not report a seed field, but its pinned source
defaults to the same `20260828` seed recorded by CUCO_RANK. No sanitizer or
Nsight profiler was enabled in this benchmark.

| Owner | Search median / MAD | Durable median / MAD | Sampled peak MiB per rank |
| --- | ---: | ---: | ---: |
| CUCO_RANK, fixed 512 MiB pool | 0.380726 / 0.014471 s | 3.774029 / 0.053340 s | 945, 945 |
| Native CUB_SORT_MERGE | 0.823757 / 0.030627 s | 3.899147 / 0.028049 s | 457, 457 |

In this paired configuration CUCO_RANK is about 2.16x faster for the
*search* interval but uses about 2.07x the sampled full-device VRAM per
rank. Durable archive-complete time differs by only about 3.3% and is not a
meaningful end-to-end speed win without a larger workload. The VRAM figures
are 50 ms `nvidia-smi` samples, not exact peaks. The explicit CUCO allocation
plan reserves 536,870,912 bytes for its pool, while its maximum requested
suballocations are 58,623,603 bytes on rank 0 and 58,613,139 on rank 1.
Pool fragmentation/allocator requirements are not captured by that requested
counter. A second paired run with a 96 MiB fixed pool is needed before
interpreting the memory Pareto point.

The two owner builds come from pinned different commits, so this is a
matched-workload comparison, not an otherwise byte-identical binary A/B.
It does not prove performance on S13/LRX or that removing remaining CPU
dependencies will preserve these timings.
