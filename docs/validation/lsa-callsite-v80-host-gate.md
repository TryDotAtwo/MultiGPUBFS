# LSA callsite profile attempt v80: host gate

Private Kaggle `trydotatwo/mgbfs-lsa-full-bfs-gate-t4` v80 reached
`UNSUPPORTED_HOST` before dependency setup or BFS. Source SHA was
`8f6bcc266d6fd5f0a18d8b420557a0010d8786bc`; both physical Tesla T4s
reported `cudaDeviceCanAccessPeer(...)=0` in both directions. The LSA profile
requires P2P, so this run supplies no Nsight timeline, correctness result or
performance measurement. Its compact output is under
`test_results/kaggle_lsa_timeline_backtrace_v80/lsa-bfs-gate/` (ignored by Git).

The same private notebook was re-pushed as v81 only after v80 completed. No
second Kaggle notebook was started.
