# S11 one-active-T4 profile panel

Kaggle `trydotatwo/mgbfs-distributed-bench` v11, script 347595711, COMPLETE.
Source 195275bf41fc37db00c15b2096068563bf26755c; immutable baseline
f0f2b8e5ee61173039ab9742f3a7756c9b6365e6; CUTLASS
ffa119a1255d78998536107466cc7097ecefa393. Host has two T4s; launcher masks to
GPU 0 and starts exactly one process. Memory statistics below cover that rank.

S11 cycle/inverse-cycle/swap(0,1): 39,916,800 states, 56 layers. Search uses
128-byte matrix stride; mandatory native archives use 11-byte permutations
plus hashes on `/kaggle/working`. Baseline has no archive.

Five independent process measurements per mode after full-graph warmup.
Native fixed batch 65,536; baseline selected 65,536 from a three-batch sweep.
Batch 1,048,576 failed allocating 8.96 GiB with 1.40 GiB free and is retained
as failed calibration. This is not final tuned Pareto acceptance.

Baseline search median **21.703334 s**, MAD **0.027657 s**, external peak
**12,989 MiB**. All five samples are retained in the summary.

| Profile | HF selector | Owner | Pre-dedup | Search median s | MAD s | Durable median s | Peak MiB |
|---|---|---|---|---:|---:|---:|---:|
| DENSE | SCALAR | CUB | OFF | 8.390143 | 0.021095 | 13.442684 | 8141 |
| DENSE | SCALAR | CUB | ON | 8.400908 | 0.029705 | 13.432900 | 8141 |
| DENSE | SCALAR | BMMA | OFF | 69.947593 | 0.012149 | 70.756601 | 8141 |
| DENSE | SCALAR | BMMA | ON | 69.781324 | 0.035069 | 70.638919 | 8141 |
| HASH_FIRST | SCALAR | CUB | OFF | 9.493787 | 0.017403 | 14.586462 | 8037 |
| HASH_FIRST | SCALAR | CUB | ON | 9.540405 | 0.006203 | 14.579282 | 8037 |
| HASH_FIRST | SCALAR | BMMA | OFF | 71.048170 | 0.003457 | 72.012870 | 8037 |
| HASH_FIRST | SCALAR | BMMA | ON | 70.789213 | 0.040836 | 71.805397 | 8037 |
| HASH_FIRST | INT_MMA_SM75 | CUB | OFF | 10.401493 | 0.005175 | 15.449455 | 8037 |
| HASH_FIRST | INT_MMA_SM75 | CUB | ON | 10.340928 | 0.031878 | 15.397959 | 8037 |
| HASH_FIRST | INT_MMA_SM75 | BMMA | OFF | 71.761758 | 0.050782 | 72.825779 | 8037 |
| HASH_FIRST | INT_MMA_SM75 | BMMA | ON | 71.591284 | 0.020405 | 72.597757 | 8037 |

DENSE uses CUTLASS generation despite its unused HF selector being SCALAR.
HF integer MMA accelerates generation only, not hash projection. CUB wins
this panel; activating tensor instructions alone does not establish speedup.

The reference reserves 39,916,800 state-ring records, 5,109,350,400 state
bytes. Its historical hardcoded bucket divisor of 128 unnecessarily doubles
the average-occupancy component for 256 one-rank buckets: accepted hashes
reserve 1,294,114,816 bytes. Fix 68e2a866e302125828f90d50df4bcebd5820dd79
changes future runs, not these measurements. Compare with the equal-global
two-rank report with this qualification; equal state-ring capacity is not
equal total bucket reservation. These are measurements before that fix.

SMI peaks include startup/warmup. Search excludes final archive drain but
can be affected by concurrent archive work; durable includes final completion.
Baseline has no comparable durable output contract. DENSE+CUB OFF search
scales from 8.390143 to 6.034028 s in the separate two-rank v4 panel (1.39x),
not 2x; this is not a paired hardware run or a fully overlapped-runtime claim.

## Evidence

Raw audit reconciled all 68 SMI files, 67 complete measured rank JSON files,
60 COMPLETE warmup files, 60 VERIFIED archive logs, and all 12 five-repeat
statistics against the summary. Native layer counts match baseline and 11!;
explicit device plane sums and rank/global capacities agree. Source and
baseline checkout logs match the pins above. Archive verification checks
committed checksums/counts; this is not an independent full-state oracle for
every large measurement. Small-group sanitizer/oracle gates are separate.

Local metadata/logs only:
`test_results/distributed-bench-v11/distributed-bench/` and
`test_results/distributed-bench-v11-summary/distributed-bench/summary.json`.
No state archive downloaded locally. Published S13 was not rerun.
