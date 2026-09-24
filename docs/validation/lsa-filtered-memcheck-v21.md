# LSA filtered memcheck v21: unresolved timeout

- Kaggle `trydotatwo/mgbfs-lsa-full-bfs-gate-t4`, version 21, source
  `0ef6ceb6f3c3cf18e09d5a422516d79e4ae088ec`, two peer-accessible T4s.
- Plain full-state LSA/CUCO_RANK DENSE fixture matched the CPU oracle and
  verified rank archives before sanitizer started.
- `compute-sanitizer --tool memcheck --target-processes application-only
  --report-api-errors no` with filters for `lsa_publish_count`,
  `lsa_copy_exact`, and `import_transport_fatal` exceeded its 300 s limit.
  The last flushed application markers were both ranks entering
  `advance depth=0`; no sanitizer error report or test completion appeared.
  Raw log: `test_results/kaggle_lsa_sanitizer_v21/lsa-bfs-gate/
  lsa-single-fixture-filtered-memcheck.log`.

This is neither a sanitizer pass nor proof of a CUDA defect. The next probe
enables existing route-stage markers to identify the first stalled phase,
with a 120 s diagnostic timeout. An unfiltered four-tool gate remains open.

Version 22 used the same source and sanitizer filters, added
`MGBFS_TRACE_ROUTE=1`, and timed out after 120 s. Both ranks logged depth-0
`exchange_begin`; neither logged `owner_begin`. The persisted trace is at
`test_results/kaggle_lsa_sanitizer_trace_v22/lsa-bfs-gate/
lsa-single-fixture-filtered-memcheck.log`. This narrows the phase to exchange,
retirement, or pre-owner vote, but does not distinguish them. Version 23 adds
three enqueue markers at those boundaries, with no algorithmic change.

Version 23 had no P2P and was `UNSUPPORTED_HOST`. Version 24 ran source
`74bc500d46138e09185f8fed4aa0b7935503e88a` on two P2P-capable T4s.
Plain full-state BFS passed; filtered memcheck again timed out after 120 s.
Rank 1 reached `exchange_begin` but not `lsa_exchange_queued`; rank 0 reached
`route_begin` but not `route_end`. The instrumentation did not yield a stable
single blocking call across v22/v24, so the evidence remains compatible with
sanitizer-induced progress or NCCL interaction. Do not claim a code deadlock
or a clean sanitizer gate. Raw v24 summary/log:
`test_results/kaggle_lsa_sanitizer_trace_v24/lsa-bfs-gate/`.
