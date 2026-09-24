# Nonblocking NCCL LSA window registration: two-T4 isolation v61

Private Kaggle `trydotatwo/mgbfs-lsa-full-bfs-gate-t4` version 61, source
`db360452b972344af916d217a5e47d5975711aad`, completed on two physical
Tesla T4s with bidirectional P2P enabled. The first attempt, version 60,
stopped at preflight because its selected host had P2P disabled; it did not
exercise the fixture.

The standalone `experiments/nccl_window_isolation.cu` fixture compared a
blocking communicator with `ncclCommInitRankConfig(blocking=0)`. Each mode
created the communicator, allocated NCCL symmetric memory, registered the
window on both ranks, deregistered it and exited with code 0. The downloaded
summary reports two registered ranks and return code 0 for each mode. No
Compute Sanitizer was run (`zero_sanitizer_errors=false` means unavailable,
not a reported sanitizer error).

Raw evidence: `test_results/kaggle_nccl_window_nonblocking_v61/lsa-bfs-gate/`
(`summary.json`, `window-blocking.log`, `window-nonblocking.log`). This is a
leaf NCCL/LSA prerequisite, not a BFS correctness, failure-propagation or
performance result.
