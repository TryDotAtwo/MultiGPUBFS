# Two-T4 S10 paired LSA/HostSized screen

Kaggle `trydotatwo/mgbfs-lsa-versus-nccl-s10-paired-t4` v4 ran source
`0ef6ceb6f3c3cf18e09d5a422516d79e4ae088ec` on two Tesla T4s with P2P
available both ways. The v3 allocation had no P2P and was `UNSUPPORTED_HOST`,
not a performance sample. Both transports use CUCO_RANK DENSE, local pre-dedup
ON, 4 shards/rank, 256 buckets, batch 32,768, 96 MiB fixed library pool/rank,
and enabled archives. The order alternated across five repeats per transport.

| Transport | Search median s (MAD) | Durable median s (MAD) | Sampled full-device MiB/rank |
|---|---:|---:|---:|
| HostSized NCCL | 0.420099 (0.003036) | 3.720190 (0.051406) | 529, 529 |
| NCCL LSA | 0.366300 (0.003642) | 3.717922 (0.083506) | 567, 567 |

All ten runs exited 0, completed 46 layers totaling 3,628,800 states, and
verified two committed rank archives per run. LSA search median is 12.8%
lower; durable medians are effectively tied in this screen. LSA uses 38 MiB
more sampled full-device VRAM per rank. These VRAM samples are from a 50 ms
`nvidia-smi` sampler, not an exact peak. This does not establish a universal
speedup or complete the sanitizer gate.

Raw per-run summaries and the aggregate report are under
`test_results/kaggle_lsa_paired_s10_v4/lsa-bfs-gate/`.
