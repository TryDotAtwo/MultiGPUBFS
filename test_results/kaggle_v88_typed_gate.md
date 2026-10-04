# Kaggle v88: typed owner/transport/retirement gate

Observed 2026-10-04. Notebook `trydotatwo/mgbfs-lsa-full-bfs-gate-t4`,
scriptVersionId `355126914`, source `582c06565aff0c9325c29004769fe702fd4178b1`.
Worker COMPLETE; acceptance report INCOMPLETE. Physical hardware: two Tesla T4,
sm75, peer access allowed both directions. Experimental NCCL registration build;
this is not an upstream NCCL acceptance claim.

## Evidence

- 48 healthy S4 configurations match CPU canonical full-state layer sets.
- 108 asymmetric injected failures pass: both independent ranks exit, no forced
  cleanup, no false COMPLETE. Maximum observed case duration: 2.457091805 seconds.
- Six U4/F2 source-bank reuse cases match all 64 CPU states and layer sets.
- memcheck 6/6, racecheck 6/6, synccheck 6/6 pass.
- initcheck 1/6 passes. Five cases fail during LSA activation/window registration
  before BFS (`LSA_ACTIVATE_GROUP: CUDA_STATUS_7`). Zero sanitizer error count
  does not make an unsuccessful process/search a pass. Root cause remains open.
- No complete two-rank BFS timeline or paired performance A/B is proved here.

Local evidence: `build/kaggle-v88-observation/lsa-bfs-gate/summary.json` and
12 initcheck rank logs below its typed profile/bank directories. Top summary
SHA256: `1544842962dd730d78ab50a99d95bdddf2b42c0b4725baaf74db401d23b58ff29`.
Only logs/JSON were downloaded; sparse state archive outputs were not copied.

## Follow-up

Repeat unfiltered initcheck three times per profile/bank configuration with NCCL
registration diagnostics. Collect full two-rank U4/F2 timelines for DENSE and
HASH_FIRST without verbose vendor logging. Keep the initcheck gate open until
the failures are explained and the required clean executions are demonstrated.
