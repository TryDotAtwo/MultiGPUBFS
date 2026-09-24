# NCCL window registration under initcheck, independent of BFS

Private Kaggle `trydotatwo/mgbfs-lsa-full-bfs-gate-t4`, version 57, source
`a5e28a2d18c9f2220c811e6c17a503532df0ec75`. The standalone
`experiments/nccl_window_isolation.cu` was compiled as host C++ and linked
to pinned NCCL 2.29.7. It uses two physical P2P-capable Tesla T4s,
`ncclCommInitRank`, `ncclMemAlloc`, `cudaMemset` and collective
`ncclCommWindowRegister(NCCL_WIN_COLL_SYMMETRIC)`. It does not load or call
MultiGPUBFS runtime, cuCollections, CUTLASS or the BFS kernels.

| Run | Result |
|---|---|
| Plain | Exit 0; both ranks printed `stage=window_register result=PASS`. |
| Compute Sanitizer `initcheck` | Rank 1 printed `window_register nccl=unhandled cuda error`, with NCCL last error `Cuda failure 'unspecified launch failure'`; the other rank did not exit, so the notebook's 180-second subprocess timeout ended the test. |

This proves that the observed initcheck/registration failure is reproducible
without BFS code on this host/software combination. It does **not** prove
whether the cause is NCCL, Compute Sanitizer, driver, or their interaction;
nor does it make the LSA full-BFS four-tool gate green. The independent
fixture does not yet have the fail-fast peer-exit behavior of the Rust leaf
fixture, so its timeout is not evidence of a production BFS deadlock.

Raw manifest and logs are under
`test_results/kaggle_nccl_window_isolation_v57/lsa-bfs-gate/`. Versions 53,
54 and 56 stopped at diagnostic fixture compilation/linking; version 55
stopped at a no-P2P host preflight. None of these earlier versions reached
the standalone window test.
