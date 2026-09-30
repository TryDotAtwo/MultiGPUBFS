# Two-T4 direct control, Kaggle v30

2026-09-30. Terminal Kaggle COMPLETE; validation status VALIDATION_FAILED.
Source: 0303f37c3b86119e286c9d311a00e624c35ecfc7. Runtime 928c25e
is not covered. Evidence: test_results/kaggle_nonblocking_boundary_v30/lsa-bfs-gate.

Explicit HOST_SIZED_NCCL, LSA OFF. Two physical T4, two in-process rank
threads, not independent rank processes. Full-state CPU oracle and archive
fixture covers U3(3), S4, rank-map and pre-dedup variants.

| Stage | Result | Wall seconds |
|---|---|---:|
| Plain | PASS | 6.015 |
| memcheck | FAIL, exit97, 192 API errors | 42.848 |
| racecheck | PASS | 122.616 |
| initcheck | PASS | 25.097 |
| synccheck | PASS | 18.640 |

Memcheck's oracle test itself passed in 41.69 seconds. Printed errors are
cudaErrorNoKernelImageForDevice (209) from cudaFuncGetAttributes and
cudaGetLastError in ncclInitKernelsForDevice. Only 100 of 192 findings were
printed, so the distribution of all findings is not established.
NVIDIA NCCL v2.29.7-1 src/enqueue.cc lines31-79 probes kernel attributes
and deliberately continues on failed probes:
https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L31-L79
This supports the probe hypothesis, not a clean memcheck gate. No error
suppression or tool filtering was applied. Exact packaged-source/build
correspondence and all findings require further validation.

No LSA initcheck-registration gate, CPU-free timeline, independent-process
fault gate, new archive-abort fix acceptance, or performance claim follows.
