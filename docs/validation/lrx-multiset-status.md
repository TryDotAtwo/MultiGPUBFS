# LRX repeated-tail support: pre-hardware checkpoint

User-requested workload: `lrx15r4`, start
`[0,1,2,3,4,5,6,7,8,9,10,11,11,11,11]`, L/R cyclic position shifts and X
swapping positions 0/1. This is a Schreier graph of multiset words, **not S15**.
Orbit order is 15!/4! = **54,486,432,000** distinct words. Graphs are undirected
because L/R are inverses and X is an involution; repeated symbols can give loops.
The previous/current/next BFS window remains applicable.

The CPU contract owns this start and its order. Ordinary MatrixGroup starts
remain invertible; no singular-matrix acceptance was introduced. Native runtime
uses the existing compact position-action GEMM and hash pipeline with a separate
validated multiset start. Supported initial policy is DENSE + compact states,
CUB or cuCollections; unsupported combinations fail rather than falling back.

CLI reference label: `lrx15r4`. Required explicit environment includes
`MGBFS_PROFILE=DENSE`, `MGBFS_STATE_CODEC=permutation_u8`,
`MGBFS_ARCHIVE_CODEC=permutation_u8`, chosen owner and preallocated capacities.
Output has the exact start, graph kind, generators, and expected orbit order.
Archive is not implicitly disabled: use the user-approved `--search-only` mode.

Completed locally:

- Five CPU tests: requested starts/orders; actual L/R/X word action; C5 layer
  fixture; 30-state orbit; capacity failure, malformed state/label and overflow.
- Generator-matrix multiplication agrees with independent position moves.
- Linux-target CUDA/library-owner type-check for CLI/runtime and new GPU test.
- Full core/runtime/CLI CPU test command exited zero.
- Runner cleanup regressions exercised actual Linux independent-session ranks.

Pending before a large-run claim:

- Build and execute `lrx_multiset_one_two_eight_rank_full_state_oracle` on real
  hardware. It compares complete word sets at every depth, both owner backends,
  both pre-dedup settings, reversed owner maps, and worlds 1/2/8 on n=5,7.
- Actual per-rank capacity admission and full n=15 run. Peak window is unknown;
  15!/4! does not predict its memory footprint by itself.
- Preserve every rank's histogram and verify total orbit size, depth coverage,
  common configuration and successful termination. Total-count agreement alone
  is not an independent proof of the large graph's distance histogram.

No multiset hardware success is claimed at this checkpoint. Global budget
remains USD 100, including prior project spending.
