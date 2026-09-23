# LRX repeated-tail support

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

Hardware validation on 2026-09-17, 8 physical H200 with NV18 all-pairs:

- `lrx_multiset_one_two_eight_rank_full_state_oracle` passed all 24
  full-state cases in 83.84 seconds: n=5,7, worlds 1/2/8, CUB/CUCO,
  pre-dedup OFF/ON, reversed owner maps. The initial test failed at setup
  because world=1 requires the legacy map `[0,0]`; only this test fixture
  was corrected. Production runtime code was unchanged.
- The search-only CLI run for `lrx8r4` completed in 1.123975639 seconds;
  all layers matched an independent Python word-action BFS (1680 states).
- Source is a7d267b496e971f2493d7760c9126347d97af6cc plus the test-only map fix.
- CUDA 12.9 and pinned NCCL 2.28.9, SM90, `NCCL_CUMEM_ENABLE=0`.
  The host lacked system NCCL headers; compilation resumed with explicit
  CPATH and LIBRARY_PATH pointing at the separately built pinned NCCL.

Large search-only run completed on the same eight H200s. `lrx15r4` reported
`COMPLETE` on all ranks: 54,486,432,000 states across 93 layers (depths 0..92),
90.093109197 seconds of BFS time. The maximum layer is depth 68 with
2,643,980,566 states. External nvidia-smi samples show up to 119,391 MiB on
one rank and 954,840 MiB as the sum of per-rank peaks. The 800M per-rank layer
capacity and 1.6B per-rank ring were admitted; no capacity failure occurred.
All rank reports and raw layer counts are saved in the ignored local evidence
archive `test_results/vast-51254782/multiset-evidence.tgz` (SHA256
`44d48cf4ee0a1a574928a64982af1358a71913a04a8d86512af3c88393a2e5eb`).
The provider instance was deleted and its absence verified. Estimated base
rental cost was USD 18.83; final invoice and traffic are not verified.

The large-run certificate is `ORBIT_TOTAL_AND_CONFIG_ONLY`: total order and
configuration agree, but there is no independent full-state or per-layer oracle
for n=15. Search-only produced no archived states or HF dataset. Timer includes
first-use work during BFS (`warmup_completed=false`).
