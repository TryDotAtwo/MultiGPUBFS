# Physical two-T4 protocol validation

Source: clean public 6fed353; all-API tracing replay: f1a7546.
Vast instance 53928026, machine 152289: two physical Tesla T4, 15360 MiB
each, sm75, driver 595.91.07, CUDA 12.9.86. Bidirectional CUDA peer access
was verified. NCCL 2.29.7 with the pinned arch-guard and explicit-POSIX
candidate patches; NCCL_MNNVL_ENABLE=0. This is not an upstream NCCL fix.

## Results

- DENSE and HASH_FIRST, LSA + CUCO_RANK: two independent rank processes,
  full canonical state/depth oracle and archive verification passed.
- Both profiles passed all 18 asymmetric injected failures (nine failure
  points on each rank): startup, constructor, late constructor, owner,
  archive admission, archive write, archive sync, archive finish, capacity.
  Both processes exited naturally, no false COMPLETE, no forced cleanup;
  observed failure completion was 1.005--2.459 seconds.
- Whole DENSE BFS passed unfiltered memcheck, racecheck, synccheck.
  **initcheck remains FAIL** during NCCL registration (CUDA 719), despite
  ERROR SUMMARY reporting zero. Exit failure is not waived or filtered.
- S8: 40320 canonical states and all 29 layer sets match the independent
  oracle for CUB_SORT_MERGE and BMMA_BUCKET, DENSE and HASH_FIRST, reversed
  rank map, pre-dedup OFF. HASH_FIRST used INT_MMA_SM75 in these cases.
- U4 modulo 2,3,4,5,6: DENSE and HASH_FIRST full two-process oracle passed
  (64,729,4096,15625,46656 states). These runs used seed 20260828; other
  seeds are not credited by this campaign.
- CUCO owner DAG capture launched successfully on both physical ranks
  for DENSE and HASH_FIRST; healthy full-state verification passed.

## Full BFS timeline

Pinned Nsight Systems 2025.3.2, cuda/nvtx/osrt, all CUDA APIs enabled,
memory/sync/other callchains enabled. S8, batch 256, both profiles.
Each rank had 100 `mgbfs.batch` ranges. Matching used NVTX globalTid and
interval containment; version suffixes `_vNNNN` were normalized.

Inside batch ranges: zero recorded synchronous memcpy or host CUDA
stream/event/device synchronization calls. DENSE had 196/192 asynchronous
copies (rank 0/1), all archive D2H. HASH_FIRST had the same archive D2H
plus 600 device-to-device copies per rank. No counts/control D2H observed.
Copy directions were checked using correlated CUPTI activities and the
exported ENUM_CUDA_MEMCPY_OPER (2=D2H, 8=D2D).

The driver API table is absent even with all-API tracing: this does not
prove absence of unrecorded driver calls. This small diagnostic does not
prove saturation, optimal overlap, scaling, or production performance.
Finalization/initialization and archive-worker waits remain outside the
claimed batch scope. No performance A/B claim is made.

## Retained evidence

Local ignored artifacts: build/two-t4-posix-20261003/
validation-before-all-api.tar.gz and validation-all-api.tar.gz. They contain
per-case summaries, raw logs, full-state archives and Nsight reports/SQLite.
Local Python suite after tracing change: 203 passed, 8 expected skips;
the tracing flag regression was observed failing before the implementation.

Outstanding: resolve initcheck registration failure without suppression;
extend four-tool gates to other modes; seeds; paired A/B, larger workloads,
production dispatch/macro/HF and DB/framework stage. Overall goal stays open.
