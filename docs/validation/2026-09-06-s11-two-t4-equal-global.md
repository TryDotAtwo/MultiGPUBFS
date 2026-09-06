# S11 two-T4 equal-global profile panel

Kaggle `trydotatwo/mgbfs-distributed-bench-s11`, v4, script version
347598670, COMPLETE. Source 8be72b9867f284b437129fb63ef169044336fa9e;
baseline f0f2b8e5ee61173039ab9742f3a7756c9b6365e6;
CUTLASS ffa119a1255d78998536107466cc7097ecefa393.

S11 cycle/inverse-cycle/swap(0,1), 39,916,800 states and 56 layers.
Search stores matrices with 128-byte stride; mandatory native archives store
11-byte permutations plus hashes. Baseline does not archive. Extents are on
`/kaggle/working`, unlike v3's `/tmp`.

Five fresh-process measurements per mode after full-graph warmup. Native batch
65,536 is fixed, not tuned. Baseline calibration selected 262,144 from
65,536 / 262,144 / 1,048,576. The last failed with rank-1 CUDA OOM requesting
5.54 GiB with 4.71 GiB free; that failure is retained, not timed as success.

Baseline median 19.354052 s, MAD 0.052442 s; external peaks 13,175 / 13,193 MiB,
sum 26,368 MiB. This is a profile panel, not final tuned Pareto acceptance.

| Profile | HF generation selector | Owner | Pre-dedup | Search median s | MAD s | Durable median s | Sum peak MiB |
|---|---|---|---|---:|---:|---:|---:|
| DENSE | SCALAR | CUB | OFF | 6.034028 | 0.039518 | 11.117054 | 7586 |
| DENSE | SCALAR | CUB | ON | 5.950696 | 0.061194 | 11.027233 | 7586 |
| DENSE | SCALAR | BMMA | OFF | 37.142452 | 0.031548 | 38.387287 | 7586 |
| DENSE | SCALAR | BMMA | ON | 37.078618 | 0.040639 | 38.376678 | 7586 |
| HASH_FIRST | SCALAR | CUB | OFF | 6.685133 | 0.044299 | 11.736980 | 7378 |
| HASH_FIRST | SCALAR | CUB | ON | 6.575466 | 0.023881 | 11.741675 | 7378 |
| HASH_FIRST | SCALAR | BMMA | OFF | 37.694243 | 0.038149 | 39.020739 | 7378 |
| HASH_FIRST | SCALAR | BMMA | ON | 37.662970 | 0.020272 | 38.782832 | 7378 |
| HASH_FIRST | INT_MMA_SM75 | CUB | OFF | 7.357610 | 0.048586 | 12.370218 | 7378 |
| HASH_FIRST | INT_MMA_SM75 | CUB | ON | 7.447708 | 0.020878 | 12.425584 | 7378 |
| HASH_FIRST | INT_MMA_SM75 | BMMA | OFF | 38.301092 | 0.032599 | 41.788725 | 7378 |
| HASH_FIRST | INT_MMA_SM75 | BMMA | ON | 38.292153 | 0.035611 | 39.650872 | 7378 |

DENSE's SCALAR selector names only the unused HASH_FIRST path: DENSE uses
CUTLASS generation. HASH_FIRST MMA accelerates generation, not hash projection.
BMMA is substantially slower here. These timings do not establish a fully
overlapped dispatcher or identify the bottleneck without stage profiling.

## Capacity and memory qualification

EqualGlobal reserves 19,958,400 state-ring records per rank, 39,916,800 total.
Each rank's states occupy 2,554,675,200 bytes. DENSE sampled peak is 3793 MiB
per rank; HASH_FIRST is 3689 MiB. Sums are sums of per-rank maxima, not necessarily
simultaneous peaks, and include startup/warmup. Explicit device planes exclude
NCCL/driver overhead and pinned RAM. DENSE explicit aligned allocation is
3,766,448,896 bytes per rank. Pinned archive is 1,106,362,368 bytes and reserved
disk is 1,144,862,464 bytes per rank; VRAM reserve is 1 GiB per rank.

Do not call this equal **total allocated memory** versus one-rank v11: although
global state-ring capacity matches, that older one-rank default divides local
capacity by a hardcoded 128 instead of its actual 256 local buckets. Its accepted
hash table reserves 1,294,114,816 bytes versus 655,446,016 bytes summed here.
Commit 68e2a866e302125828f90d50df4bcebd5820dd79 corrects the default for future
runs; these historical measurements are not rewritten. No GPU result for that
new default is claimed by this report.

Search excludes the final archive drain; durable includes it. Concurrent archive
work can affect search. Baseline has no equivalent durable output contract.

## Raw evidence reconciliation

All 68 raw SMI CSV files reproduce reported peaks. All 67 complete row
aggregations match 134 standalone measured-rank JSON files. All 120 native
warmup JSON files are COMPLETE; 120 standalone archive verifier logs report
VERIFIED. All 12 five-repeat median/MAD/sample/peak statistics reproduce the
summary. Native layer counts match baseline and sum to 11!; named device plane
payload/aligned totals and local/global record sums match reported totals.
Downloaded source and baseline SHA logs match the pins above.

This verifies counts, reported measurements and archive checksum/count checks,
not independent full-state equality of every large run. Small full-state oracle
and sanitizer gates remain separate evidence. No archives were downloaded to
the local computer; S13 was not rerun.

Local raw evidence: `test_results/distributed-bench-s11-v4/distributed-bench/`;
separate summary: `test_results/distributed-bench-s11-v4-summary/distributed-bench/summary.json`.
