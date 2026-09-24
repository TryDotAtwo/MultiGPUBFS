# Matched three-owner S10 panel on two T4s (v65)

Private Kaggle `trydotatwo/mgbfs-library-owner-t4`, version 65; pinned
runtime source `ae3dce9433508faae6730132b43b072eea84dd78`, native CUB
baseline source `013ed5c979f4225db273e0015fa9ed72fd230c90`.
Five S10 DENSE runs per owner, alternated order, 8 shards, batch 32,768,
pre-dedup ON, 96 MiB fixed library pool/rank, 256 pinned archive slots.
Two physical Tesla T4s. All 15 runs completed 46 layers and 3,628,800
states; both rank archives of every run were checksum/count VERIFIED.

| Owner | Search median ± MAD s | Durable median ± MAD s | Sampled peak MiB total (per rank) |
|---|---:|---:|---:|
| CUB_SORT_MERGE | 0.829891 ± 0.032798 | 3.890360 ± 0.047674 | 914 (457, 457) |
| CUCO_RANK | 0.376393 ± 0.003642 | 3.747809 ± 0.039022 | 1058 (529, 529) |
| CUDF_RELATIONAL | 1.745511 ± 0.021784 | 4.341090 ± 0.167362 | 1134 (567, 567) |

This fixed S10 configuration gives CUCO_RANK the fastest search, CUB the
lowest sampled VRAM, and CUDF_RELATIONAL neither. CUB's search median is
2.20× CUCO_RANK's; CUDF's is 4.64× CUCO_RANK's. The durable medians are
much closer because all three include the archive contract. The external
`nvidia-smi` sampler runs every 50 ms, so these values are not exact peak
VRAM. No sanitizers ran in this panel; separate gates are required. This
does not rank all graphs, profiles, batch/shard counts or GPU generations,
nor establish an asynchronous owner-to-retirement pipeline.

Raw local evidence: `test_results/kaggle_db_screen_v65/library-owner/summary.json`,
15 per-case `screen-summary.json` and 30 `verify-rank-*.log` files.
