# Native/library rank gate v5: full BFS initcheck remains incomplete

Source `cb4603cd975dd71ac8402ae436fe49a70140e6d1`. Two physical T4s with
P2P enabled in both directions. Driver 580.178.04, nvcc12.9 V12.9.86,
Compute Sanitizer2025.1.0.0 build35583870. Worker COMPLETE; report
NATIVE_PROCESS_OR_SANITIZER_GATES_FAILED. Root summary SHA256
`0e9a8072df0d6b9de3efa18d65f616171a41e2bd43dd9cc3571071afba596eb8`.

Root report contains 24 single-GPU full-state oracles and 24 successful
two-process replays (three owners × two profiles × pre-dedup × rank map).
Full-runtime sanitizers: 19/24 successful processes. The five failures are
initcheck for CUCO_RANK DENSE/HASH_FIRST, CUB_SORT_MERGE DENSE, and BMMA_BUCKET
DENSE/HASH_FIRST. CUB HASH_FIRST initcheck passes this time. This does not
establish deterministic initcheck acceptance for that backend.
Detailed general-log retention is still running; do not overwrite v5 yet.

## Full S8 timeline evidence retained and inspected

All six timeline summaries, twelve analysis JSONs and twelve SQLite exports
are retained in `build/kaggle-native-v5-timeline-priority/lsa-bfs-gate`.
Each profile/owner has independent rank processes, full 40320-state archive
oracle, 29 layers and route-bank reuse counts [132,130]. All layer sets
match the CPU oracle, not merely the final total. Batch size128.

For all twelve traces, thread-matched fully-contained `mgbfs.batch` intervals
have zero cudaStream/Event/DeviceSynchronize and zero synchronous cudaMemcpy.
An independent SQLite LEAD(start) query also finds zero such APIs in
same-thread gaps between successive batches when gaps intersecting
FinalizeDepth are excluded. FinalizeDepth synchronizations remain.

Correlating GPU copy events with runtime API and NVTX ranges, all batch D2H
copies are contained in archive_d2h: rank0 352 copies/1621680 bytes, rank1
346 copies/1603920 bytes, for every owner/profile. HASH_FIRST additionally
has 1068 D2D copies/46992 bytes per rank. No counts/control D2H outside
archive was observed in these batch ranges.

Recorded GPU interval union (milliseconds; includes instrumented capture,
not paired unprofiled search time):

| Owner/profile | Rank0 busy | Rank1 busy | Rank0 multistream | Rank1 multistream |
|---|---:|---:|---:|---:|
| BMMA DENSE |517.620|514.308|16.384|16.403|
| BMMA HASH_FIRST |734.676|736.186|14.367|13.552|
| CUB DENSE |519.770|556.626|17.864|16.274|
| CUB HASH_FIRST |635.080|722.851|16.060|14.016|
| CUCO DENSE |483.244|571.863|16.365|16.703|
| CUCO HASH_FIRST |737.034|715.771|15.155|14.909|

Only about 2–3% of recorded GPU-busy union has multiple active streams here.
This is not evidence of full utilization, optimal overlap, occupancy or
FLOPS. Small batches, initialization/finalization and profiler overhead
limit throughput interpretation. Callchains are UNAVAILABLE. No source
callsite attribution is invented.

These traces support the no-readback healthy-batch property for the six
reference-dispatch owner/profile combinations on this workload. They do
not prove production macro-depth, arbitrary world sizes, all four sanitizer
gates or performance superiority. Paired A/B remains outstanding.
