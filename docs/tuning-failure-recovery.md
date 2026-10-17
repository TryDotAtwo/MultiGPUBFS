# Autotuning candidate failures and measured selection

Version 0.2.0.dev26 removes the session-wide unmeasured fallback introduced in dev24. A failed optional probe does not establish that any profile is fast or safe.

## Candidate rejection

When a generic optional candidate records a primary CUDA 999 in durable rank phase evidence:

1. Preserve that candidate's phase files, first native fault, cleanup faults and copied error path.
2. Close the entire failed rank group and invalidate its admission ticket and memory inventory.
3. Exclude that candidate without retrying it. Re-admit and measure remaining configurations on a fresh group containing the same selected GPUs.
4. Compare only successful, identical layer prefixes. A measured choice requires at least two successful candidates and the existing substantial-prefix gate. The failed candidate contributes no timings.

The result retains `candidate_failures`, `failed_candidate_policy=REJECT_AND_MEASURE_REMAINING` and the successful `pilots`. No `TUNING_PROBE_FAILED_CONSERVATIVE_PROFILE` or session-wide failure latch remains.

If the remaining candidates cannot establish a substantial measured comparison, tuning raises `AUTOTUNE_NO_MEASURED_PROFILE_AFTER_FAILURE` with durable evidence. Unproven rank failures, timeouts, other primary CUDA errors and real main-BFS errors still propagate; they are not silently skipped. Main BFS is never automatically restarted.

## Measurement and reuse

`selection_measured` describes the comparison of execution geometry/modes. `size_profiles_measured` describes whether the size-tier table contains independent measured choices. An individually unmeasured tier remains marked as such; it does not erase a measured geometry selection or prevent healthy persistent-session reuse. Normal small-graph live tuning remains distinct from fault handling.

Use `multigpubfs.persistent_graph_workers()` around repeated public CayleyPy calls. A successful measured profile may be reused under the normal native/environment/geometry identity checks, while memory admission remains live. Rejected candidates are not installed as winners.

## Evidence boundary

Candidate rejection fixes the selector's premature termination and dev24 fallback policy. It does not identify the cause of the original production CUDA 999 on eight GPUs. The reporting CUDA/NCCL call may expose an earlier asynchronous error. CPU tests and two-GPU acceptance do not establish correctness or performance on that server.


## Bounded probes with insufficient completed work

A slow configuration may hit its probe deadline before it has enough completed work for a meaningful comparison. Its completed layer vector is still checked against the other reports. If correct but below the existing depth/work gate, it is recorded in `excluded_prefix_candidates` with its native reason and coverage. It cannot reduce the comparison window of candidates that did complete enough work. An INCOMPLETE probe with sufficient completed work still participates at the common completed depth. At least two eligible candidates are required for a measured selection; after a captured candidate failure, insufficient remaining comparisons produce the explicit error above. This does not extrapolate large-layer performance or treat a short probe as fully tested.
