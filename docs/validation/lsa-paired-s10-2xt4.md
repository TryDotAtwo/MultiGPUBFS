# LSA versus host-sized NCCL, paired S10 on 2×T4

Private Kaggle notebook
`trydotatwo/mgbfs-lsa-versus-nccl-s10-paired-t4` v1 completed on two
physical P2P-capable Tesla T4s. Both transports used identical source
`34b1c81fa9433bdcc0d9f338076b145429764ec8`, pinned NCCL 2.29.7,
CUDA 12.9, cuCollections `532795b`, `CUCO_RANK`, DENSE, pre-dedup ON,
four shards/rank, 256 buckets, batch 32,768, 1,000,000 state and ring
records/rank, a fixed 96 MiB library pool/rank and the same 128-bit seed.
The run order alternated by repeat. Both variants produced the same 46-layer
S10 histogram totaling 3,628,800 states. All ten runs reported `COMPLETE`,
both rank archives in each run returned `VERIFIED`, and the full-device
50 ms memory sampler completed in every run. Histograms plus archive
checksums are not an independent full-state set comparison for S10.

| Transport, two T4 | Search median / MAD | Archive-complete median / MAD | Sampled peak MiB/rank |
| --- | ---: | ---: | ---: |
| Host-sized NCCL | 0.427086 / 0.009262 s | 3.829540 / 0.059919 s | 529, 529 |
| NCCL LSA | 0.383635 / 0.008364 s | 3.853357 / 0.012396 s | 567, 567 |

LSA was faster in all five paired search samples. Its search median is
10.17% lower (1.113× speedup), but archive-complete medians differ by only
0.62% in LSA's disfavor. This small workload does not establish durable
throughput superiority. LSA costs an extra 38 MiB of sampled full-device
VRAM per rank. Both variants report the same explicit aligned allocation
plan, 348,094,976 bytes/rank: the 12,583,168-byte LSA symmetric slot
replaces exactly 12,583,168 bytes of legacy receive planes. The observed
38 MiB difference is outside this explicit plan; attributing it to a
specific NCCL/driver allocation requires a separate allocation trace.

The source predates the fixed-depth parent-round scheduling change. The
benchmark does not validate that change, the unresolved full-path LSA
sanitizer gate, or elimination of owner/retirement host waits.

Raw evidence: `test_results/kaggle_lsa_paired_s10_v1/lsa-bfs-gate/summary.json`,
ten `measure.json` and ten `screen-summary.json` files in their corresponding
case directories. Large archives remained on Kaggle under `/tmp`; only the
small summaries were downloaded locally.
