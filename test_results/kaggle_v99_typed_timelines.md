# v99 typed timeline and repeated initcheck

Source 5a4ea73f5dd58a5ed58e0178caedc60b991514e6; two physical P2P-enabled
T4 GPUs, independent rank processes, CUCO_RANK, completion window3,
route banks3 for timelines. Worker COMPLETE, report INCOMPLETE.
Evidence retained on D under build/kaggle-v99-observation: four original
SQLite exports and all selected logs/summaries (152 retained outputs).

Initcheck: 2/18 repeats pass, both DENSE/banks3 (repeats0 and2).
The other16 fail; no supervisor timeout. Failed logs contain NCCL CUDA719
unspecified launch failure during activation, window_register or
device_comm_create. This does not establish its underlying cause and
does not pass/waive the four-sanitizer gate.

Both full-state timeline cases pass U4(F2), 64 unique states, layer sizes
[1,3,5,8,11,13,13,8,2], both rank exits zero, group COMPLETE, no forced
cleanup. Actual route-bank reuse is11/12 for ranks0/1.

Thread-matched NVTX/runtime analysis of each of the four exports:

| Profile | Rank | Batch host sync/synchronous memcpy | Same-depth inter-batch sync/memcpy | Batch D2H copies/bytes | Batch D2D copies/bytes |
|---|---:|---:|---:|---:|---:|
| DENSE |0|0|0|66 /1056|0|
| DENSE |1|0|0|62 /992|0|
| HASH_FIRST |0|0|0|66 /1056|216 /9504|
| HASH_FIRST |1|0|0|62 /992|216 /9504|

The async D2H counts equal the contained archive_d2h API counts: no
counts/control readback observed in these batches. Same-depth gaps use
LEAD(batch.start) on the same globalTid and exclude every gap intersecting
FinalizeDepth. Each rank still has45 stream synchronizations and108
synchronous memcpy inside FinalizeDepth, the retained semantic boundary.
Archive-worker synchronization elsewhere must not be conflated with
batch-thread waits. Callchains are UNAVAILABLE; do not invent callsites.

Recorded multi-stream activity / GPU busy time (whole captured span):
DENSE rank0 2.030/56.527ms, rank1 1.458/55.755ms;
HASH_FIRST rank0 1.769/103.512ms, rank1 1.774/107.495ms.
This tiny batch1 trace is not a throughput, occupancy or Pareto result.
It confirms measured readback removal at this scope, not maximum overlap
or absence of launch gaps at production scale. Updated b47c270 phase
orchestration has not been hardware-validated by these old-source traces.
