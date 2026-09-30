# v35 full owner DAG: capture passes, fixture readback initcheck fails

Source248434d, two actual T4/P2P1. Expanded owner probe built and ran.
Plain full-DAG marker and final assertions PASS; memcheck PASS; racecheck PASS.
Initcheck exit97: two host-API uninitialized reads in a128-byte cudaMemcpy.
The new fixture copied both64-byte directory entries although only the first
entry was published. No device-kernel uninitialized read reported in this log.
The first entry/count/state assertions passed, but this is still a red gate.
Synccheck and LSA faults NOT_RUN after the failure.

Artifacts: test_results/kaggle_nonblocking_boundary_v35/lsa-bfs-gate.
Fix reads directory count, requires exactly one published entry, then copies
only that64-byte prefix. No blanket initialization of unused capacity, no
sanitizer filter/suppression, no production-kernel change.
Actual initcheck RED is the regression evidence; hardware replay pending.
