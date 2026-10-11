# Resident tuner lifecycle validation (2026-10-11)

Validated on one host with two physical RTX 3060 GPUs, CUDA 13 runtime, SM86 native library. No B200/B300 or eight-rank acceptance is claimed.

Generic and HOST_SIZED_NCCL SHARD_AB jobs now use one resident rank group. Compatible NCCL communicators are parked after all-rank publication; incompatible buffer shapes are retired through the rank-local CUDA pool. NCCL_LSA still uses a separate fresh-process path because its window generation/reset reuse has not been accepted.

Job-boundary phase receipts record preparing, bfs, retiring and done/failed. Resident job identity contains PID and sequence. Specialized measured deadlines begin after native setup. Host watchdogs separately bound preparation and retirement (600 seconds); the rental's absolute guard remains authoritative. A control thread polls cancellation markers; BFS only reads its atomic flag at layer boundaries. Controlled specialized probes use a GPU collective vote at those boundaries, with no per-batch disk polling, CPU state generation or CPU deduplication.

A cancellation/deadline writes an INCOMPLETE group-job-stop receipt with rank hashes and a committed selected terminal snapshot, then parks the healthy communicator. Native fatal errors still terminate poisoned workers. No blind restart is performed.

Failure evidence retains the failed tuner workspace and original per-rank producer errors together with secondary archive-reader errors. Unknown native/correctness failures are propagated.

Evidence gates:
- shared4-check: seven generic/HASH/generic/SORT_MERGE transitions; same two PIDs, communicator hits, exact independent CPU layer/state oracle.
- phase-check: same transitions with preparing/bfs/retiring/done ordering and identity checks.
- deadline-check: HASH and SORT_MERGE stop at a 0.0001-second measured BFS limit; correct archive; next generic run completes in the same processes.
- cancel-check: cancellation requested only on rank 0; both modes stop consistently and retain exact prefix states; next generic run completes in the same processes.
- autotune-check: n8r1 real automatic selection and complete GPU calculation, exact CPU oracle; selection took 4.883496 seconds on this host. This is not a large-graph speedup claim.
- diagnostic-check: synthetic producer failure preserves primary errors and secondary EOF separately; not physical GPU fault injection.
- retention-check and unit regressions: failed temporary workspace survives exception; successful workspace cleans up.

Raw logs, receipt JSON, reproduction scripts, source bundle and runtime hashes are archived separately on HF. The previous B300 initiating error remains unknown because its detailed producer logs were lost before instance deletion. These fixes address reproduced lifecycle defects; they do not prove that all B300 launch failures have been eliminated.
