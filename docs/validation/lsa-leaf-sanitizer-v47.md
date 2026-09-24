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
completed as private Kaggle v48 to confirm fail-fast behavior. At
`ae3dce9` the plain peer exchange passed, then an intentionally corrupted
rank-0 hash caused an assertion and process exit `-6` (SIGABRT), not a
timeout. This validates the fixture's rank-failure path on 2×T4; it does
not validate the BFS runtime's fault propagation. Raw evidence is under
`test_results/kaggle_lsa_leaf_fault_v48/lsa-bfs-gate/`.

Private Kaggle v49 selected two T4s without peer access and stopped at
preflight (`UNSUPPORTED_HOST`), before either sanitizer. The same
independent initcheck/synccheck configuration ran on P2P-capable 2×T4 in
v50 at `ae3dce9`. Plain and synccheck both passed; synccheck reported
`ERROR SUMMARY: 0 errors`. Initcheck returned code 6 after one rank's
`ncclCommWindowRegister` failed with `window_register: unhandled cuda
error`; the corrected fixture exited rather than hanging. Initcheck's own
summary was `ERROR SUMMARY: 0 errors`, but its test did **not** pass, so
this is not a green initcheck result and not a proven BFS memory defect.
Raw v50 summary and logs are in
`test_results/kaggle_lsa_leaf_remaining_v50/lsa-bfs-gate/`.

The NCCL-INFO initcheck diagnostic v51 selected a no-P2P T4 host and
stopped at preflight. The same diagnostic was resubmitted as v52.

Private Kaggle v52 completed on P2P-capable 2×T4 at `ae3dce9` with
`NCCL_DEBUG=INFO`. The plain leaf passed. Under `initcheck`, one rank again
failed during `ncclCommWindowRegister` (`window_register: unhandled cuda
error`); the fixture exited with code 6 and `ERROR SUMMARY: 0 errors`.
This independently reproduces v50's activation failure, but neither proves
a Compute Sanitizer false positive nor identifies a BFS defect. The
notebook status is `INCOMPLETE`; full-BFS and four-tool gates remain open.
Raw v52 evidence is in
`test_results/kaggle_lsa_leaf_initdebug_v52/lsa-bfs-gate/`.

The slow `cuda_nvrtc` download seen while v52 was running was transient:
the archive finished downloading and the notebook reached the leaf test.
It was not the terminal failure.

The independent two-T4 NCCL window fixture in Kaggle v57 reproduces a
registration error under `initcheck` without loading MultiGPUBFS; plain
registration passes on both ranks. See
`docs/validation/nccl-window-isolation-v57.md`. This narrows the failure
boundary to NCCL/Compute Sanitizer/driver interaction on the tested host,
but does not classify it as a sanitizer false positive or waive the BFS
four-tool gate.

The leaf results do not close the full-BFS four-tool sanitizer gate. Raw
evidence is under `test_results/kaggle_lsa_leaf_v47/lsa-bfs-gate/`.
