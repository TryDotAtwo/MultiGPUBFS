# Archive admission and capacity evidence, 2×T4

The evidence is source-pinned; the runs are not performance samples.

| Kaggle version | Source | Result | Scope |
|---|---|---|---|
| v67 | `022f0997016281b608162c50f6cad1ab1674fa6a` | PASS S4 | HostSized/LSA torchrun, verified archives, group markers, asymmetric archive-admission failure, separate two-device full-state oracle. Rank-local file reservation 23,712 bytes. This source incorrectly used one-layer capacity as a bound on the *full-run* archive and is not accepted for larger graphs. |
| v68 | `b1d65a746c9d6b0659ee6bab3c012b21dcdc1e3a` | UNSUPPORTED_HOST | Two T4s but `cudaDeviceCanAccessPeer=0` in both directions; skipped before Linux build/tests. |
| v69 | `70a848e7363a8bcf3bf76738edc2219da779cc08` | PASS S4 | Distinct P2P-capable T4s. Linux `archive` suite passed 13 tests, including missing FIFO reader timeout and normal reader handoff; bootstrap 14/14 and group-commit 2/2. Both torchrun transports completed S4 with archive verification/group markers and the asymmetric admission fault reached both ranks. Separate two-device full-state oracle passed. |
| v70 | `70a848e7363a8bcf3bf76738edc2219da779cc08` | PASS S4 | Same Linux and two-rank gates plus a run with per-rank layer capacity 6, archive rows 1, and more than 6 total states archived by rank 0. Both rank archives verified; rank 0 reports 6,304 reserved disk bytes. |
| v72 | `ff314f68a579662a625e61a0ad0d08e7a48183e9` | PASS S4 | Linux unit test confirms malformed optional numeric configuration returns `ENV_MGBFS_SHARDS` instead of panicking. Both two-rank transports, archives, group markers, asymmetric archive fault, small-layer-capacity run and independent full-state oracle passed again. This does not test asymmetric pre-bootstrap configuration failure. |
| v73 | `26502bbd0966f6554bb850da5113da4f5b802549` | RED: TIMEOUT | A rank-local nested RMM pool forces constructor failure after NCCL initialization. The peer did not finish within the 75-second gate timeout. |
| v75 | `85672f6ee1f567d24191f01c653df14edfe529f3` | RED: TIMEOUT | `ncclCommAbort` on constructor unwind let rank 0 return `CUDA_STATUS_-1`, but rank 1 still hung. v74 used T4s without P2P and did not run this GPU fixture. |
| v76 | `93951e9369efbd804733611e4004fb13de2e8e5a` | RED: SIGSEGV | Group vote after local setup exposed a released CUDA stream on the rank-local failure path. |
| v77 | `8f308ef284fb12d6d6fad2204028205076d12c43` | PASS S4 | The post-NCCL constructor fault finishes on both ranks: rank 0 `CUDA_STATUS_-1`, rank 1 `REMOTE_CONSTRUCTOR_FATAL`, fixture 1.41 s. Linux CPU tests, HostSized and LSA S4 runs, archive verification/group markers, asymmetric archive fault, small-layer-capacity archive, and separate full-state oracle all pass. |

Raw outputs: `test_results/kaggle_group_boundary_v67/`,
`test_results/kaggle_group_boundary_v68/`, and
`test_results/kaggle_group_boundary_v69/`, and
`test_results/kaggle_group_boundary_v70/`, and
`test_results/kaggle_group_boundary_v72/` through
`test_results/kaggle_group_boundary_v77/`. The v70 test confirms compilation,
small-graph behavior and FIFO admission on Linux. Its targeted S4 run has
`layer_capacity=6` and `archive_rows=1`; rank 0's layer counts
`[1,0,2,6,2,0,1]` total 12, so the archive cannot be bounded by a single
layer. This does not validate large-graph capacity, end-to-end throughput, or
remote durability of a streaming archive.

The v77 fault fixture covers one post-NCCL local library-pool error. It does
not prove group fail-fast for malformed config before bootstrap, errors before
`ncclCommInitRank`, failures in the NCCL LSA setup calls, or all later device
and host faults. The normal S4 gates are not throughput evidence.

The reference benchmark now preallocates a conservative per-rank archive
extent from the declared graph-order bound, including record/layer/commit
frame overhead. A file run may fail preflight on insufficient disk rather
than dynamically grow. Stream mode reserves this as a logical bound, not
physical HF staging capacity; that capacity is preflighted separately by
the consumer launcher.
