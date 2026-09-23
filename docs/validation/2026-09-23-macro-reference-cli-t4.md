# Single-T4 macro reference CLI gate (2026-09-23)

Private Kaggle notebook: `trydotatwo/mgbfs-macro-reference-cli-t4`, version 2,
status `KernelWorkerStatus.COMPLETE`. Source commit
`2d7e480f770c472b5e8774a81a55373b6fe13e3f`; CUDA architecture 75;
GPU 0 Tesla T4. Logs and rank JSON:
`test_results/kaggle_macro_reference_cli_v2/macro-reference-cli/`.

The new public CLI path executed `torchrun --nproc-per-node=1 --no-python
mgbfs bench --reference` with archive enabled. For U4(2) (64 states) and S5
(120 states), K=1,2,3 produced identical per-depth counts within each graph.
Each archive passed `mgbfs verify` (checksums/counts). The custom 128-bit seed
`0000000000000000000000000000002a` appeared in every result JSON. These
small timed runs are correctness smoke only, not an end-to-end performance
comparison.

| Graph | K | States | Search s | Durable s | Archive verifier |
|---|---:|---:|---:|---:|---|
| U4(2) | 1 | 64 | 0.02 | 0.06 | PASS |
| U4(2) | 2 | 64 | 0.01 | 0.02 | PASS |
| U4(2) | 3 | 64 | 0.02 | 0.02 | PASS |
| S5 | 1 | 120 | 0.04 | 0.06 | PASS |
| S5 | 2 | 120 | 0.02 | 0.04 | PASS |
| S5 | 3 | 120 | 0.02 | 0.03 | PASS |

The displayed times are rounded for readability; exact values remain in
`summary.json`. This gate compares counts, not complete sorted state sets.
Version 2 alone did not run a full-state oracle or sanitizers. K>1 multi-rank
is not implemented.

## Follow-up: version 3

Version 3 also completed on the same private Kaggle slug, pinned to the same
source commit. Downloaded evidence is under
`test_results/kaggle_macro_reference_cli_v3/macro-reference-cli/`.
All six `macro_native` GPU tests passed in a plain run and under each of
`memcheck`, `racecheck`, `initcheck`, and `synccheck`. Memcheck, initcheck and
synccheck reported zero errors; racecheck reported **0 hazards, 0 errors,
0 warnings**. The full-state fixtures compare exact sorted states for
U3(2), U3(3), U4(2) at K=1,2,3, plus a nonidentity source at K=1,2,3,10.
The archive test covers matrix and compact permutation representations.
The CLI matrix above passed again, with six verified archives and identical
layer counts across K.

This upgrades the small single-device macro backend gate, but does not prove
distributed K>1, large-graph capacity, or a performance win. The CLI test
still compares layer counts rather than complete S5/U4 archive row sets.
