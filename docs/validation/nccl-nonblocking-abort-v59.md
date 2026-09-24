# Independent nonblocking NCCL abort, two T4s

Private Kaggle `trydotatwo/mgbfs-lsa-full-bfs-gate-t4` version 59 ran
`experiments/nccl_nonblocking_abort_isolation.cpp` from exact source
`f49deebe37e8f57a5c294785c3c6a92e9b45009b` with NCCL 2.29.7 on
two P2P-capable Tesla T4s. The notebook status was `COMPLETE`; the binary
returned 0 and both rank threads printed `stage=abort result=PASS`. Raw
summary and logs: `test_results/kaggle_nccl_nonblocking_abort_v59/lsa-bfs-gate/`.

The fixture creates a nonblocking communicator with
`ncclCommInitRankConfig(blocking=0)`. Rank 1 issues an all-reduce that rank
0 deliberately does not issue; the two rank threads then both call
`ncclCommAbort`. This demonstrates one asymmetric host-fault/abort scenario
without the 300-second hang seen in the earlier GPU-only BFS candidate.

Scope is narrow: two threads in one process, one collective, no LSA window,
no BFS owner transaction, no socket failure propagation, no rank-process
isolation, no repeated epoch and no sanitizer. It does **not** authorize
removing the production post-owner host vote. Production still uses a
blocking communicator and wrappers that reject `ncclInProgress`; the
nonblocking configuration needs a complete API/progress contract, a bounded
cross-process failure protocol, receive-slot lifetime events and physical
2×T4 full-BFS failure gates before selection.
