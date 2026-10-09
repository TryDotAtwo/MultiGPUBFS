# Paired warm-process follow-up

v112 is a cold-process search comparison, not steady-state admission: MGBFS_BENCH_WARMUP was0, and an earlier fresh-process run does not prime per-process lazy kernel initialization. Its measured regression remains valid for that cold configuration.

The prepared launcher keeps both algorithm revisions and graph unchanged. Both ranks perform a full untimed search inside their own process, destroy the warmup runtime, then construct a fresh measured runtime. The old example receives only measurement-wrapper/archive-disabling edits; the current runtime uses its existing MGBFS_BENCH_WARMUP=1 path. Raw warmup records and measured records must both be retained and checked.

This launcher has not been built or executed. Do not claim warm-process numbers or causal attribution. Vast admission remains pending a confirmed spending bound; no rental created.
