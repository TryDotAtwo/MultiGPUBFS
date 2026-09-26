# Reference warmup admission, 2×T4 Kaggle v79

Private kernel `trydotatwo/mgbfs-lsa-full-bfs-gate-t4` v79 completed on
2026-09-26 with source SHA `5b04b2a190a56094f2858f3135bfbcc439a48e5d`.
Both physical Tesla T4s allowed P2P in both directions. Logs and rank JSON
were downloaded to `test_results/kaggle_lsa_full_bfs_gate_v79/lsa-bfs-gate/`
(locally ignored by Git).

The two-process `torchrun` checks completed as follows:

| Case | Observed result |
|---|---|
| Rank 0 warmup=1, rank 1 warmup=0 | Both report `REMOTE_CONFIGURATION_FATAL`; no group marker. |
| Rank 0 malformed warmup | Local `BENCH_WARMUP_CONFIG`, peer `REMOTE_CONFIGURATION_FATAL`; no group marker. |
| Rank 0 unsupported multi-rank macro depth | Local `MACRO_MULTI_GPU_UNSUPPORTED`, peer `REMOTE_CONFIGURATION_FATAL`; no group marker. |
| Rank 0 disables archive without search-only | Local `CLI_BENCH_ARCHIVE_REQUIRED`, peer `REMOTE_CONFIGURATION_FATAL`; no group marker. |
| Both ranks warmup=1 | Warmup rank results have `warmup_ephemeral` scope and no warmup group marker; warmup archive files were removed; measured group marker exists and both measured archives pass `mgbfs verify`. |

The successful measured S4 run produced global layers
`[1, 3, 5, 6, 5, 3, 1]`, totaling 24 states. This validates this launch
boundary and cleanup on real 2×T4. It does **not** establish a GPU-driven
owner→transport→retirement loop, larger-graph correctness, performance, or
the four-tool Compute Sanitizer gate. A deliberately injected warmup archive
*removal* failure was not exercised; its cross-rank handling remains a source
property pending fault injection.
