# v11: batch-scoped Nsight evidence

Source: aa97205007b62b7247d73cc226efef839dc196ec. Notebook:
trydotatwo/mgbfs-native-rank-owner-t4, v11. Eight exported SQLite traces
downloaded under test_results/kaggle_native_rank_v11_traces/lsa-bfs-gate/.

All four S8 executions (CUB_SORT_MERGE/BMMA_BUCKET x DENSE/HASH_FIRST)
finished with both independent rank processes returning zero and the archived
full-state CPU oracle reporting 40,320 states across 29 layers. These are
profiled correctness runs, not paired performance benchmarks.

## Attribution

Select runtime APIs issued inside NVTX mgbfs.batch on the same globalTid,
excluding nested mgbfs.archive_d2h. Correlate GPU memcpy using correlationId,
not GPU execution timestamps. Each trace contains 178 batch ranges.

All eight traces have zero selected Synchronize APIs and zero synchronous
Memcpy APIs. DENSE has no selected memcpy. HASH_FIRST has 1,068 asynchronous
copies per rank totaling 46,992 bytes, all Device-to-Device (copyKind=8, verified
against the trace ENUM_CUDA_MEMCPY_OPER). No selected count/control D2H.

This does not prove that the host never waits. Bounded epoch credit uses
cudaEventQuery, cancellation checks, NCCL polling and a 1 ms sleep while its
event is not ready. Not-ready records occur in every trace. Both versioned and
unversioned EventQuery records appear; raw counts are not unique physical waits.
The runtime remains host scheduled: absence of D2H is not ideal overlap.

## Open gates

Full BFS memcheck and initcheck still fail. racecheck and synccheck pass for
the four combinations. Failed gates are not waived. Archive and FinalizeDepth
are excluded only from this narrow attribution, not end-to-end timing.
GPU overlap, launch gaps, paired A/B and sanitizer startup isolation remain.

All four downloaded two-rank S8 archive pairs were independently rechecked
locally with verify_process_archives(n=8, world=2): 40,320 states, 29 exact
layers each, checksum/config/rank-count validation and global uniqueness.
Sanitizer case manifests confirm owner_dag_capture_requested=false: an initial
suspected capture-environment leak was disproved, not fixed or used as a cause.
