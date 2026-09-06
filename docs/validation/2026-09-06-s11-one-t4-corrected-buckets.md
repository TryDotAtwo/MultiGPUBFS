# S11 one-T4 panel v12: corrected bucket capacity

Kaggle `trydotatwo/mgbfs-distributed-bench` v12 completed. Source
`68e2a866e302125828f90d50df4bcebd5820dd79`, immutable CayleyPy baseline
`f0f2b8e5ee61173039ab9742f3a7756c9b6365e6`.
S11 has 39,916,800 states and 56 BFS layers for cycle/inverse-cycle/swap01.
One GPU is used on a two-T4 host. Native batch 65,536 is fixed; baseline
262,144 was selected by this run's calibration. Five fresh measured runs per
profile follow separate warmups. Native archives are mandatory under
`/kaggle/working`; baseline produces no archive.

Raw reconciliation: 68 rows, 67 complete rank results, 60 native warmups,
60 archive verifiers and all 12 comparisons. The non-complete baseline
calibration row remains in the evidence. All complete layer counts match.
Archive verification covers committed checksums/counts, not a new full-state
oracle for all S11 records.

| Profile | Generation selector | Owner | Pre-dedup | Search median s | MAD s | Durable median s | Peak MiB |
|---|---|---|---|---:|---:|---:|---:|
| DENSE | SCALAR | CUB | OFF | 8.448760 | .018876 | 13.460637 | 7521 |
| DENSE | SCALAR | CUB | ON | 8.389498 | .006373 | 13.352978 | 7521 |
| DENSE | SCALAR | BMMA | OFF | 69.874361 | .017364 | 70.717888 | 7521 |
| DENSE | SCALAR | BMMA | ON | 69.669592 | .013790 | 70.564422 | 7521 |
| HASH_FIRST | SCALAR | CUB | OFF | 9.589799 | .033858 | 14.723911 | 7417 |
| HASH_FIRST | SCALAR | CUB | ON | 9.528263 | .037044 | 14.621274 | 7417 |
| HASH_FIRST | SCALAR | BMMA | OFF | 71.025277 | .025380 | 72.084197 | 7417 |
| HASH_FIRST | SCALAR | BMMA | ON | 70.890539 | .021618 | 71.762573 | 7417 |
| HASH_FIRST | INT_MMA_SM75 | CUB | OFF | 10.342702 | .042054 | 15.444523 | 7417 |
| HASH_FIRST | INT_MMA_SM75 | CUB | ON | 10.329721 | .033359 | 15.430914 | 7417 |
| HASH_FIRST | INT_MMA_SM75 | BMMA | OFF | 71.938898 | .074687 | 73.047960 | 7417 |
| HASH_FIRST | INT_MMA_SM75 | BMMA | ON | 71.686835 | .037185 | 72.727432 | 7417 |

CayleyPy median 22.288524 s, MAD .011541 s, peak 14,713 MiB.
SCALAR is the HASH_FIRST selector and does not mean DENSE lacks CUTLASS
generation. HASH_FIRST INT_MMA accelerates generation, not its scalar hash.
BMMA remains much slower for this workload. This is not a final tuned Pareto gate.

## One versus two cards with matched global state and accepted capacities

Compare DENSE/CUB/pre-dedup OFF with the previously reconciled S11 two-T4 v4:

| Quantity | One T4 v12 | Two T4 v4, total |
|---|---:|---:|
| Search median s | 8.448760 | 6.034028 |
| Durable median s | 13.460637 | 11.117054 |
| SMI peak MiB | 7521 | 7586 |
| State capacity records | 39,916,800 | 39,916,800 |
| State allocation bytes | 5,109,350,400 | 5,109,350,400 |
| Accepted allocation bytes | 655,446,016 | 655,446,016 |
| Explicit aligned device bytes | 7,287,526,656 | 7,532,897,792 |

This is about 1.40x search scaling, not 2x. Fixed per-rank buffers account for
extra explicit allocation on two ranks. SMI includes runtime/library effects
and is not identical to the explicit ledger. The runs have separate source pins
and are not an interleaved paired experiment. V11's one-rank bucket over-allocation
is corrected here: DENSE peak falls from 8141 to 7521 MiB; historical results
are retained rather than rewritten. Baseline calibration selected a different
batch in v11, so its memory number must not be carried into this table.

Evidence: `test_results/distributed-bench-v12/distributed-bench/` and
`test_results/distributed-bench-v12-summary/distributed-bench/summary.json`.
No large state data was downloaded. Continuous GPU overlap, CLI completion,
long-workload final tuning and S14 capacity investigation remain outstanding.
