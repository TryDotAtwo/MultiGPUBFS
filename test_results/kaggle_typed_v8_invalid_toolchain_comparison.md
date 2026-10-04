# Typed v8: instrumenter-comparison result is invalid

Worker COMPLETE; source a136204ab4a377e0655216d9a9e11640c51b5526.
Root status INCOMPLETE, 12/16 replays pass; all four initcheck labels fail
without outer timeout. Root summary SHA256:
40dc96a9fa2501ca7a6ef4a363efbb5e01925c7ffa79f388ea7bf7f50248c3ca.
Root retained under `build/kaggle-typed-v8-summary`; full small-output
retention completed under `build/kaggle-typed-v8-observation`:
17 summaries and all32 rank logs, before notebook reuse.

The pinned executable version command reports 2025.2.1.0/build35969825,
but this is NOT evidence that it instrumented a rank. At this exact source,
`instrument_rank_command` uses the absolute host executable
`/usr/local/cuda/bin/compute-sanitizer`; the notebook changed PATH only.
Consequently every replay, including cuda129-labelled cases, launches the
host tool. Treat them as repeated host-tool observations, never evidence
for a cross-version conclusion. Full BFS initcheck is still open.

Fix c221b93 passes MGBFS_COMPUTE_SANITIZER explicitly into both rank
commands, records actual executable SHA256/version, and refuses to pass
a comparison row whose actual executable identity does not match selection.
Regression tests failed before the fix and passed after it. Python239
tests OK/8 skipped; scripts65 OK/3 skipped. No CUDA runtime algorithm,
NCCL resources, sanitizer filters or suppressions changed.
