# Independent T4 NCCL window probe: v12

Notebook trydotatwo/mgbfs-native-rank-owner-t4 v12, exact source
2b2b62054a301f9c6829db208bd6d196e2dab80c. Source patch is committed;
two independent rank processes, two actual Tesla T4s, P2P allowed both ways.
NCCL 2.29.7, same pinned CUDA SDK as the full BFS gate. No API filters or
sanitizer suppressions. Each pair has a bounded process-tree supervisor.
Raw logs: test_results/kaggle_window_process_v12_20261001/lsa-bfs-gate/.

| Tool | Rank exit codes | Registration | Result |
|---|---|---|---|
| plain | 0, 0 | both | diagnostic pass |
| memcheck | 97, 97 | both | fail: 13 errors each |
| racecheck | 0, 0 | both | diagnostic pass |
| initcheck | 0, 0 | both | diagnostic pass |
| synccheck | 0, 0 | both | diagnostic pass |

Memcheck reproduces the full BFS's 12 unsupported-kernel availability API
errors (cudaFuncGetAttributes/cudaGetLastError, code 209, host stack
ncclInitKernelsForDevice) plus one cuMemCreate permission error 800 during
ncclMemAlloc. Both ranks nevertheless reach registration and teardown.
This isolates these 13 reports without BFS; it does not waive the full gate.

Initcheck's independent-process registration succeeds here with zero errors.
Therefore the v11 full-BFS initcheck failure is not proved to originate in
window registration, nor is it a universal failure of this API. Next reduction
must include ncclDevCommCreate and then actual device window operations.
The probe has no BFS owner, transport kernel, retirement, archive or oracle.
None of the full BFS sanitizer failures is closed by this diagnostic.

The summary's inherited t4_acceptance_eligible=true denotes target hardware
only and is misleading for this reduced test; scope/status remain explicitly
diagnostic. Future reduced-mode summaries must set this flag false.
