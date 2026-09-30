# Boundary diagnostics v38 and v39

Both runs pin runtime `c7488ca6d901afc1545edba4e25ea28f915f8206`.
Neither run supplies the missing origin-rank abort stack or proves a runtime fix.

- v38: actual `UNSUPPORTED_HOST`, two T4 with bidirectional P2P disabled.
  CUDA-GDB parent-launch preflight succeeded, but no dependent build/runtime
  test executed. Kaggle COMPLETE is not a gate PASS.
- v39: actual `DIAGNOSTIC_ONLY`, inner diagnostic TIMEOUT, supported P2P host.
  All recorded application groups are empty at both scheduled samples.
  The debugger followed a torch/Python driver child into `NvDebugAgent`
  (`/tmp/1747/session1/NvDebugAgent`) instead of a BFS rank. No application
  stack or mapped NCCL identity was obtained. The 27-second supervision ended
  with debugger kill; this is not graceful runtime termination evidence.

Logs retained under `test_results/kaggle_nonblocking_boundary_v38` and `v39`.
The existing parent-launch MI protocol is retained; the LSA host-stack mode
now uses ordinary GDB rather than CUDA-GDB to avoid driver debugger injection.
No attach, ptrace/security change, healthy host barrier or runtime backend
fallback is introduced. Tool availability/preflight failure remains explicit.

Local parser test RED then GREEN; four debugger helper tests PASS. Full Python
suite: 186 tests, eight skipped, PASS with `PYTHONUTF8=1`. Without UTF-8 mode,
`test_export_hf_dataset.test_exports_every_unique_state_and_replay_metadata`
failed due to Windows subprocess cp1251 decoding; this environmental failure
was observed, then the complete suite was rerun with UTF-8 mode enabled.
