# LSA full-path sanitizer v62: instrumentation timeout, not a pass

Private Kaggle `trydotatwo/mgbfs-library-owner-t4` v62 ran source
`b93d2a2701b9ebfe64674178d979dc544f3eaad0` on two physical,
P2P-capable T4s. The notebook requested unfiltered memcheck, racecheck,
initcheck and synccheck for three ignored LSA/CUCO_RANK tests: full DENSE
layer/archive oracle, one-rank owner capacity failure, and one-rank host
owner error. Each command had a 1,800-second limit.

The **first memcheck command timed out**. Neither the test result nor an
`ERROR SUMMARY` was emitted; the remaining eleven commands did not run.
Both ranks logged construction and entry into `advance depth=0`. During NCCL
initialization the sanitizer reported repeated `cudaErrorNoKernelImageForDevice`
(209) in `ncclInitKernelsForDevice`, then `ncclMemAlloc`/`cuMemCreate`
`CUDA_ERROR_NOT_PERMITTED` (800). This is an instrumentation/NCCL interaction
candidate, not proof that these API reports alone caused the depth-0 stall.
The plain, archive-verified S10 LSA run at the same runtime source completed
in separate Kaggle v41; it does not turn v62 into a sanitizer pass.

The evidence is consistent with prior filtered and unfiltered LSA sanitizer
timeouts (see `lsa-full-bfs-2xt4.md` and `lsa-filtered-memcheck-v21.md`). Do
not repeat the same full-BFS sanitizer command unchanged. A useful next gate
is an isolated NCCL LSA allocation/one-exchange fixture under the same tool,
then an alternate supported instrumented environment or split application-
kernel checks. The four-tool full LSA path remains **unverified**.

Raw evidence:
`test_results/kaggle_lsa_sanitizers_v62/library-owner/summary.json` and
`lsa-cuco_rank_lsa_two_gpu_dense_layers_and_archives_match_oracle-memcheck.log`.
