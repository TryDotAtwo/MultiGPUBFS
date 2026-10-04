# Typed runtime gate v97: incomplete

Notebook: trydotatwo/mgbfs-lsa-full-bfs-gate-t4, worker COMPLETE.
Source: 5a4ea73f5dd58a5ed58e0178caedc60b991514e6.
Root summary SHA256: 76d8b449abcff225760b8ba715500e69125f717bf1710f80a578e38a48a3a3b5.

The report contains 78 typed replays: 72 pass and six fail. The passing
groups are 48 ordinary S4 configurations, six U4(F2) bank-reuse cases,
and six cases each for memcheck, racecheck and synccheck. Initcheck fails
all six profile/bank combinations. This is not completion of the four
sanitizer gates and is not a performance result.

All twelve failing rank logs were retained under
build/kaggle-v97-initcheck/lsa-bfs-gate. Failures occur during LSA
activation, before search: DENSE/banks2 and HASH_FIRST/banks3,4 report
window_register; DENSE/banks3,4 and HASH_FIRST/banks2 report
device_comm_create. Rank1 reports an asynchronous NCCL unhandled CUDA
error; rank0 reports cancellation or timeout without terminal NCCL
result. Both ranks enter failure teardown and return application errors.
The sanitizer ERROR SUMMARY is zero, but application failure prevents
the gate from passing. ARCHIVE_INCOMPLETE is a failure-path consequence,
not evidence of a successful search or durable commit.

The logs do not establish the underlying CUDA/NCCL/instrumentation cause.
Do not waive this failure based on the reduced activation probe passing
on a different notebook host. A diagnostic follow-up needs detailed
NCCL activation logging and the real full-runtime timeline.

MODE=typed_rank_gate does not run Nsight Systems. No timeline evidence
is claimed for v97. The full small-output download remains in progress;
the six targeted failure summaries and twelve rank logs are already
retained. Do not replace this notebook until the outstanding download
has completed. Nested fault-scenario counts require separate verification.
