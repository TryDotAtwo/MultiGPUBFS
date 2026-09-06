# Native NCCL abort integration: T4 regression v31

Kaggle `trydotatwo/mgbfs-distributed-sanitizer` v31 completed on two distinct
Tesla T4 GPUs. Source: `4ddf27138ca7e2c59624807b16fd7463b3a2ac3e`.

Raw log reconciliation passed: 35 runtime/tool logs, 36 measured rank JSONs,
36 warmup rank JSONs and 36 archive verifiers. Each sanitizer log has its
required zero-error summary; racecheck also reports zero hazards/warnings.

- The 12-fixture runtime suite passed plain, memcheck, racecheck, initcheck and
  synccheck. It includes compact generation5 archives, HASH_FIRST capacity
  failure with terminal rank state and incomplete archives, and group admission.
- Nonidentity macro source and macro archive fixtures each passed all five
  modes. Five leaf fixtures passed all four sanitizers.
- All 24 S4 profile selections (1/2 ranks, DENSE/HASH_FIRST, CUB/BMMA,
  applicable generation modes, pre-dedup OFF/ON) reproduce
  `[1,3,5,6,5,3,1]` with archived outputs.
- Linux CPU bootstrap/control/epoch/pump/failure contracts passed before build.

This covers normal GPU paths and the existing capacity-error path after
connecting communicator abort to failed native advancement. It does **not**
inject a broken NCCL link, exercise a watchdog, or execute GPU exchange through
the TCP control pump. The later `mgbfs_nccl_poll` ABI is outside this source pin.
Continuous overlap and production CLI remain incomplete.

Evidence: `test_results/distributed-sanitizer-v31/distributed-sanitizer/` and
`test_results/distributed-sanitizer-v31-summary/distributed-sanitizer/summary.json`.
The downloaded files are logs and small metadata only; no state dataset was
downloaded. Published S13 was not rerun.
