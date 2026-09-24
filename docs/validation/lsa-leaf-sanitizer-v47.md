# Two-T4 LSA leaf sanitizer v47

Private Kaggle `trydotatwo/mgbfs-lsa-full-bfs-gate-t4`, version 47,
source `48bc324c771128e327015c32f4ea4a8f8fab2b3f`. Both Tesla T4s
reported peer access in both directions. This is a single two-rank LSA
exchange fixture, **not** full BFS.

Plain passed. With NCCL API-error reporting disabled, memcheck passed the
test with `ERROR SUMMARY: 0 errors` and racecheck passed with
`RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`.

Initcheck timed out. Its log contains a rank-thread assertion at LSA
activation: `window_register: unhandled cuda error`, return code 7. The
other thread remained at the fixture's unconditional barrier, so the
notebook's timeout is not evidence that the LSA transport or the BFS itself
deadlocked. `synccheck` was not run. The fixture now aborts the process on
any rank-thread panic in `ae3dce9`; a deliberate bad-payload failure gate
is running as private Kaggle v48 to confirm fail-fast behavior.

The leaf results do not close the full-BFS four-tool sanitizer gate. Raw
evidence is under `test_results/kaggle_lsa_leaf_v47/lsa-bfs-gate/`.
