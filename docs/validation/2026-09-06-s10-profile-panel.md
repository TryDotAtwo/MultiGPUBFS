# S10 twelve-profile panel on 2xT4

Kaggle `trydotatwo/mgbfs-distributed-bench` v9; source `40715dbc716addff1e44f855c7b41b6c1de8627f`, baseline `f0f2b8e5ee61173039ab9742f3a7756c9b6365e6`.

Verified summary: 68 successful rows (3 baseline calibrations, 65 measurements), 136 rank warmup flags, identical layer counts totaling 3,628,800, and 120 native archive checksum/count verifications. This is not independent full-state equality for every measurement.

Five independent process runs per configuration. Full graph warmup precedes measurement. Native batch 65,536 is fixed; baseline batch 1,048,576 selected from three candidates. Native archive mandatory, baseline no archive. External nvidia-smi memory includes warmup and process lifetime, is sampled, and sums per-rank peaks, not necessarily simultaneous. This is NOT final tuned Pareto acceptance or a long-running workload.

Baseline median 1.3950 s, total sampled peak 18,430 MiB.

| Profile | HF generation selector | Owner | Pre-dedup | Search median s | MAD s | Sum peak MiB |
|---|---|---|---|---:|---:|---:|
| DENSE | SCALAR | CUB_SORT_MERGE | OFF | 0.8464 | 0.0084 | 1910 |
| DENSE | SCALAR | CUB_SORT_MERGE | ON | 0.8399 | 0.0246 | 1910 |
| DENSE | SCALAR | BMMA_BUCKET | OFF | 1.4260 | 0.0240 | 1910 |
| DENSE | SCALAR | BMMA_BUCKET | ON | 1.4002 | 0.1012 | 1910 |
| HASH_FIRST | SCALAR | CUB_SORT_MERGE | OFF | 1.0756 | 0.0406 | 1750 |
| HASH_FIRST | SCALAR | CUB_SORT_MERGE | ON | 1.1046 | 0.0393 | 1750 |
| HASH_FIRST | SCALAR | BMMA_BUCKET | OFF | 1.6050 | 0.0828 | 1750 |
| HASH_FIRST | SCALAR | BMMA_BUCKET | ON | 1.6068 | 0.0748 | 1750 |
| HASH_FIRST | INT_MMA_SM75 | CUB_SORT_MERGE | OFF | 1.3100 | 0.0094 | 1750 |
| HASH_FIRST | INT_MMA_SM75 | CUB_SORT_MERGE | ON | 1.3247 | 0.0216 | 1750 |
| HASH_FIRST | INT_MMA_SM75 | BMMA_BUCKET | OFF | 1.8800 | 0.0174 | 1750 |
| HASH_FIRST | INT_MMA_SM75 | BMMA_BUCKET | ON | 1.8478 | 0.0177 | 1750 |

DENSE uses CUTLASS generation; its SCALAR selector labels only the unused HASH_FIRST path. INT_MMA_SM75 accelerates child generation, not hash projection. CUB is faster than BMMA in this panel; integer MMA HASH_FIRST loses to scalar HASH_FIRST here. No default backend promotion follows from isolated Tensor Core throughput.

All comparison samples are retained in `data/2026-09-06-s10-profile-panel.json`. Full downloaded raw records remain under `test_results/distributed-bench-v9-summary/distributed-bench/summary.json`; per-rank logs and samples under `test_results/distributed-bench-v9/` (download inventory must be checked separately).

