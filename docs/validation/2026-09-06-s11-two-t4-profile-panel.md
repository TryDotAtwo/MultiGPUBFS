# S11 twelve-profile panel, two physical T4 GPUs

Kaggle `trydotatwo/mgbfs-distributed-bench-s11`, version 3, script version
`347588746`, completed. Source `c75f803cc80fd9d639972733dca1cc4ab4b17872`;
CayleyPy baseline `f0f2b8e5ee61173039ab9742f3a7756c9b6365e6`;
CUTLASS `ffa119a1255d78998536107466cc7097ecefa393`.

Workload: S11, cycle/inverse-cycle/swap(0,1) generators, matrix search states,
39,916,800 unique states, 56 layers (depth 0 through 55). Native output includes
mandatory permutation-u8 archives; CayleyPy produces counts without an archive.
This is a fixed-native-batch profile comparison, not final tuned Pareto acceptance.

## Measurements

Five independent process runs per configuration; full graph warmup precedes
measurement. Native batch is 65,536. Baseline selected batch 262,144 after
calibration over 65,536 / 262,144 / 1,048,576. The largest calibration failed:
rank 1 raised `torch.OutOfMemoryError` trying to allocate 5.54 GiB with 4.71 GiB
free. That failed calibration is retained, not counted as a timing sample.

Baseline search median: **18.860294 s**, MAD **0.003736 s**;
sampled device peaks **13,175 / 13,193 MiB**, sum **26,368 MiB**.

| Profile | HF generation selector | Owner | Pre-dedup | Search median s | MAD s | Durable median s | Sum peak MiB |
|---|---|---|---|---:|---:|---:|---:|
| DENSE | SCALAR | CUB_SORT_MERGE | OFF | 5.523760 | 0.046360 | 10.853452 | 14302 |
| DENSE | SCALAR | CUB_SORT_MERGE | ON | 5.549085 | 0.123402 | 11.210157 | 14302 |
| DENSE | SCALAR | BMMA_BUCKET | OFF | 36.719784 | 0.006089 | 38.813200 | 14302 |
| DENSE | SCALAR | BMMA_BUCKET | ON | 36.658456 | 0.045351 | 38.692899 | 14302 |
| HASH_FIRST | SCALAR | CUB_SORT_MERGE | OFF | 6.486052 | 0.027714 | 12.192008 | 14094 |
| HASH_FIRST | SCALAR | CUB_SORT_MERGE | ON | 6.555376 | 0.009958 | 12.259492 | 14094 |
| HASH_FIRST | SCALAR | BMMA_BUCKET | OFF | 37.569765 | 0.010437 | 38.861839 | 14094 |
| HASH_FIRST | SCALAR | BMMA_BUCKET | ON | 37.484729 | 0.050381 | 38.846664 | 14094 |
| HASH_FIRST | INT_MMA_SM75 | CUB_SORT_MERGE | OFF | 7.182910 | 0.021518 | 12.957992 | 14094 |
| HASH_FIRST | INT_MMA_SM75 | CUB_SORT_MERGE | ON | 7.107206 | 0.025271 | 12.713934 | 14094 |
| HASH_FIRST | INT_MMA_SM75 | BMMA_BUCKET | OFF | 38.176146 | 0.039614 | 39.398087 | 14094 |
| HASH_FIRST | INT_MMA_SM75 | BMMA_BUCKET | ON | 38.129722 | 0.028500 | 39.640938 | 14094 |

DENSE uses CUTLASS generation: its SCALAR selector only names the unused
HASH_FIRST path. INT_MMA_SM75 accelerates child generation, not the hash
projection. CUB beats BMMA in this panel. HASH_FIRST integer MMA does not beat
its scalar path here. These measurements do not establish the limiting stage
or demonstrate fully overlapped execution.

## Memory and output boundaries

Capacity mode is **MaxPerRank**: 39,916,800 state-ring records on EACH rank,
79,833,600 globally. This is not an equal-global-memory 1-versus-2-GPU test.
The state ring alone reserves 5,109,350,400 bytes per rank (128-byte matrix
stride). Compact 11-byte archive rows do not make search states compact.

Each rank declares 1,106,362,368 pinned archive bytes and 1,144,862,464 disk
extent bytes, with a 1 GiB untouched VRAM reserve. Named explicit allocation
planes exclude NCCL/driver overhead and host memory. External nvidia-smi peaks
include process startup and warmup; the sum of rank peaks need not be a
simultaneous peak. Archive extents reside on `/tmp` in this run.

The durable timer includes final archive completion; the search timer does not
include that final drain, but concurrent archive work can still affect search.
Do not compare durable native time to a fictitious durable baseline time.

## Evidence audit

Summary checks passed: 68 rows, 67 complete and one failed calibration;
60 complete native measurements, 134 measured-rank warmup flags, matching
layer counts, 120 archive checksum/count verification entries, and exact sums
of named device allocation planes. All 68 downloaded raw nvidia-smi files
reproduce summary peaks. All 12 five-repeat search statistics and all 67
complete rank aggregations were recomputed from the summary records.

Standalone inventory audit passed: exactly 134 measured rank files match the
summary objects; all 120 native warmup rank files are COMPLETE with matching
local layer counts; all 120 independent verifier logs match the summary and
report VERIFIED. Downloaded source/baseline/CUTLASS SHA logs match the pins above.
Counts and archive checksums are not independent full-state equality proofs for
every measurement; the separate small-group full-state sanitizer gates cover
that narrower correctness scope.

Local evidence (small metadata/logs only, no state archives downloaded):

- `test_results/distributed-bench-s11-v3-summary/distributed-bench/summary.json`
- `test_results/distributed-bench-s11-v3-environment/distributed-bench/environment.json`
- `test_results/distributed-bench-s11-v3/distributed-bench/`

The one-T4 S11 run is separate and not yet incorporated here. The previously
published S13 dataset was not recomputed.
