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

The first 96 MiB attempt (`trydotatwo/mgbfs-cuco-rank-paired-s10-pool96-t4`
v1) failed before build or BFS: installation of the pinned 678 MB
`libcudf-cu12==26.4.0` wheel reached the notebook's 900 s task timeout.
Its `summary.json` reports `FAILED` / `TIMEOUT: install`; the downloaded
`install.log` ends at the wheel download. No 96 MiB capacity or speed
conclusion follows. Raw logs are in
`test_results/kaggle_cuco_rank_paired_s10_pool96_v1/`. Version 2 then failed
before cloning because its timeout override referenced a nonexistent
module-level `run` (the function is local to `main`). Version 3 sets an
environment override consumed by the library notebook's installation step,
raising only that task's timeout to 2700 s. Its BFS code, pool size, workload
and paired baseline remain unchanged; the source pin advances to `8b5fb3f`
only for this notebook configuration change.

Version 3 **completed** on two physical T4s at source
`8b5fb3fa0acfc93b93a02e5911f511547f37ead9` (native baseline still
`013ed5c`). All ten measured runs are `COMPLETE`, unprofiled, with two
`VERIFIED` rank archives apiece; all have the same 46-layer histogram totaling
3,628,800 states. The library runs report the same 128-bit seed and a fixed
100,663,296-byte pool on each rank. No four-sanitizer run was included in
this timing screen. Raw output:
`test_results/kaggle_cuco_rank_paired_s10_pool96_v3/library-owner/`.

| Owner, 2xT4 | Search median / MAD | Durable median / MAD | Sampled peak MiB/rank |
| --- | ---: | ---: | ---: |
| CUCO_RANK, fixed 96 MiB pool | 0.407692 / 0.008896 s | 3.635942 / 0.042948 s | 529, 529 |
| Native CUB_SORT_MERGE | 0.840257 / 0.017192 s | 3.775333 / 0.052005 s | 457, 457 |

The five search samples were 0.406767, 0.416588, 0.407692, 0.421317,
0.395751 s for CUCO_RANK and 0.840257, 0.851486, 0.823065, 0.911582,
0.821769 s for CUB. Thus CUCO_RANK is about 2.06x faster for search here,
with 72 MiB (15.8%) more sampled device VRAM per rank. The fixed pool fits
this S10 workload; its peak requested suballocations are 58,623,603 and
58,613,139 bytes, not a fragmentation bound. Relative to the earlier 512 MiB
pool run, the 96 MiB version reduces sampled peak by 416 MiB/rank while its
search median is about 7.1% slower; these are separate Kaggle sessions, not
a controlled pool-only timing experiment. The archive-inclusive median is
about 3.7% faster than CUB in v3, but the graph is too short for a durable
throughput or larger-graph claim. Archive verification checks committed
checksums/counts; equal histograms are not a cross-backend full-state set
comparison. No result here removes the remaining CPU route, collective or
retirement dependencies.

The two owner builds come from pinned different commits, so this is a
matched-workload comparison, not an otherwise byte-identical binary A/B.
It does not prove performance on S13/LRX or that removing remaining CPU
dependencies will preserve these timings.
