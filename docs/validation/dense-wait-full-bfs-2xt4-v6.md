# DENSE wait-removal full BFS gate on physical 2xT4

Private Kaggle `trydotatwo/mgbfs-rank-retire-fatal-gate-t4` v6 completed
with `summary.json: PASS` on two physical Tesla T4 GPUs. The source is
`ced41ab9ac731a3c7f7a03d6e1249830e595c2ea`; raw downloaded output is
`test_results/kaggle_dense_wait_full_bfs_sanitizers_v6/library-owner/`.

| Scope | Plain | memcheck | racecheck | initcheck | synccheck |
| --- | --- | --- | --- | --- | --- |
| GPU 0 | 3/3 | 3/3, 0 errors | 3/3, 0 hazards/errors/warnings | 3/3, 0 errors | 3/3, 0 errors |
| GPU 1 | 3/3 | 3/3, 0 errors | 3/3, 0 hazards/errors/warnings | 3/3, 0 errors | 3/3, 0 errors |
| Two GPUs | 3/3 | 3/3, 0 errors | 3/3, 0 hazards/errors/warnings | 3/3, 0 errors | 3/3, 0 errors |

The two-GPU set includes full-state/layer/archive oracles for both the
`CUCO_RANK` and older library owners, plus the injected retirement FIFO
failure requiring a group-wide fatal vote. The eight-GPU case is correctly
ignored on this host. The Kaggle CLI returned a Windows encoding error after
the output files were downloaded; the conclusions above use the saved
`summary.json` and each per-tool `bfs-*.log`, not the CLI exit code.

This validates the scoped removal of a redundant DENSE stream drain at the
tested commit. It does not remove the remaining route-count, NCCL-size,
collective-control, owner-publishing, or retirement host dependencies and
does not establish end-to-end asynchronous execution or performance benefit.
