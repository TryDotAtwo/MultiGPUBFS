# Scoped CUDA callsite diagnostics, Kaggle v38/v39

Both private `trydotatwo/mgbfs-lsa-full-bfs-gate-t4` runs completed an
archive-verified S10 `CUCO_RANK`/LSA DENSE BFS on two P2P T4s. The profiler
captured the measured search range, not startup/warmup/final archive drain.
v38 used source `96cb3ae`; v39 used `b93d2a2` and built the Rust executable
with release debug symbols. v39 also requested Nsight symbol resolution and
CUDA backtraces for synchronization and memory calls.

The v39 SQLite export contains 1,462 `cudaStreamSynchronize` calls and 1,740
`cudaMemcpy_v3020` calls across both ranks, matching the earlier scoped v37
counts. The CUDA API tables contain both versioned and unversioned
`cudaStreamSynchronize` records for the same calls: adding their totals would
double-count. Backtraces were captured for the unversioned sync calls, but
their `CUDA_CALLCHAINS` symbols remained virtual addresses even with debug
symbols. Versioned memcpy records had no callchain in this export. Thus these
runs **do not** establish source callsites or per-stage critical-path time.

The result rules out treating aggregate Nsight API counts as a sufficient
guide for deleting a specific wait. The next diagnostic must use explicit
stage markers or another demonstrated address-to-symbol mapping on the
remote host. It must also distinguish archive/fatal votes from semantically
required `FinalizeDepth` work.

Compact summaries and callchain exports (not the large profiler reports):
`test_results/kaggle_sync_callsites_v38/lsa-bfs-gate/` and
`test_results/kaggle_sync_callsites_v39/lsa-bfs-gate/`.
