# Disk worker to search-sideband failure propagation

## Integrated change

The distributed reference launcher creates one failure token before archive
admission. The existing pinned archive worker and search-sideband thread share
that token. A disk/CUDA/return-queue error now publishes failure directly from
the worker, without waiting for a producer `acquire`, `submit` or next batch.

The existing constructor failure guard was generalized to FailureReportGuard,
not duplicated. Explicit error notification precedes active-slot destruction;
the guard also covers unsuccessful worker exit/unwind before queued messages
are destroyed. Each error retains an original device/error diagnostic.
Successful durable RunCommit disarms the guard. Worker startup failures are
included in ArchiveAdmission if already observed; later ones are preserved in
the same token supplied to sideband startup and communicator cancellation.

The worker never calls NCCL. The existing rank dispatcher owns abort and reader
retirement. There is no new healthy-batch count readback, synchronization,
queue, payload padding, GPU/pinned allocation or fallback. Late archive.finish
still participates in the existing group ArchiveCommitted vote before output
or group COMPLETE publication.

## RED and GREEN

- A real two-rank TCP-sideband test publishes the shared token without calling
  report_failure/advance/acquire on that rank. The old disconnected token failed
  the peer-notification assertion; shared wiring passes. The fixture performs
  retirement/cleanup even on its RED path. This uses two host threads, not two
  independent GPU rank processes.
- Three tests run the actual pinned archive worker, CUDA-pinned slots/events
  and archive codec with an external Extent injecting write/final-sync errors.
  The old worker failed both notification assertions (token remained zero).
  Both faults and the healthy durable commit now pass. Producer remains idle
  while the write-error token is observed; there is no producer polling of the
  archive. Test slots have completed/unrecorded events, not a blocked D2H copy.
- Complete default-members cargo test --locked: exit 0, session 78648. The
  earlier CPU failure-guard dead-code warnings disappear through feature gating.
- All 14 CUDA/library-owner runtime unit tests: exit 0, session 89227.
- All 11 Linux CUDA/library-owner CLI tests: pass, session 68233. Existing
  reference_bench unused-mut warning remains; it was not silently modified.
- The three actual worker tests pass under each unfiltered Compute Sanitizer
  tool: memcheck/initcheck/synccheck zero errors, racecheck zero errors/warnings/
  hazards, exit 0. No kernel filters or CUDA API-error suppression. Raw logs:
  test_results/archive_worker_20261002/{memcheck,initcheck,racecheck,synccheck}.log.

## Full BFS non-regression

Fresh local CUCO_RANK+LSA S8 runs use DENSE/HASH_FIRST and pre-dedup OFF/ON,
mandatory matrix_u8 archive, batch 1024, buckets 8, shards 4, CUCO pool 64 MiB,
state capacity 40320, ring capacity 80640, archive rows 512/slots 128.
All four processes return 0 with group COMPLETE. Independent CPU full-state
archive comparison verifies all 40320 states at all 29 depths in every run.
Raw logs/archives are test_results/archive_worker_20261002/s8-<profile>-<pre>/.

Rebuilt CLI SHA-256:
79b81ca4e5352ce7f348ba611893af7aa85d54501d0bc28bd911fd3fb44d6470.
Hardware remains the single local RTX3070 Laptop (sm86), with the previously
pinned CUDA 12.9/RAPIDS/NCCL architecture-guard candidate. These instrumented
worker tests do not execute NCCL registration and cannot close its T4 initcheck
issue. This is not a two-T4 full-BFS sanitizer gate or performance comparison.

## Open acceptance

Fresh two-independent-rank queued-D2H worker/descriptor failures, bounded group
termination, all full-BFS sanitizer tools (including NCCL registration), full
two-T4 timeline and paired A/B remain required. Distributed macro depth and
remaining DB/framework/HF work are unchanged and still part of the full goal.
No rental/remote job was created or modified during this local change.
