# CLI full-state gate

The existing S4 native owner-capture and hardware-conditional Tensor
HASH_FIRST CLI fixtures now compare every archived canonical state at every
depth with the independent full-byte CPU matrix successor oracle, after the
real archive checksum verifier. Duplicate rows are rejected explicitly.

A mutation fixture writes a valid checksummed archive through the production
writer, swapping states across depths while preserving layer counts and the
global unique set. The structural verifier accepts it; the full-state gate
rejects it. This covers a defect that the preceding counts-only fixture missed.

Local Linux/CUDA run: four native_lsa_capture tests pass (5.79s), including
real DENSE S4 owner capture and archive-worker failures. Local hardware is
sm86, so the Tensor fixture exercises only early unsupported-hardware
rejection. The newly strengthened positive sm75 branch remains unexecuted.

The full runtime/CLI library-owner suite was also attempted. CLI tests and
20 runtime unit tests passed, but bootstrap integration stopped at
archive_admission_failure_reaches_both_ranks_before_nccl_setup with
BOOTSTRAP_TIMEOUT on both threads. Isolated rerun passed in 0.01s. The cause
of that intermittent failure is not established; the full suite is not
credited as passing. Existing unused_mut warning at reference_bench.rs:578
remains unrelated and unchanged.

No two-rank T4 correctness, sanitizer, registration or A/B result is inferred
from these one-GPU tests. Runtime arithmetic and NCCL teardown were unchanged.
