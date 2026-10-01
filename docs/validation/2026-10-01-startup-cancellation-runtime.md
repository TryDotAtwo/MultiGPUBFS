# Startup cancellation runtime change

The actual NCCL wrapper now accepts a cancellation callback before
ncclCommInitRankConfig. Legacy create delegates with no callback. The
NCCL-calling thread alone checks cancellation and aborts; output remains
null on cancelled/failed initialization, including pre-cancelled startup.
The callback context belongs to an Arc kept alive through constructor
cleanup and, on success, stored in DistributedNativeBfs.

reference_bench passes its already-running TCP sideband token into library,
native and compact-LRX constructors. Thus LSA preparation and constructor
collectives no longer precede cancellation binding. This does not remove
remaining HASH_FIRST host counts or change healthy payload sizes.

RED: the actual production wrapper, compiled with existing deterministic
NCCL/CUDA doubles, failed the missing startup-cancellable-create assertion.
GREEN: an indefinitely in-progress initialization cancels without returning
a handle and aborts once; pre-cancelled startup does not call NCCL init.
All existing wrapper ordering/cleanup assertions pass.

On Vast 53673926 (two RTX A4000), the final candidate patch SHA-256 is
45f780db117558472846c082a4eb955ad091c1bf9d5a8499bb6ba52936e76186
against 705684ffce9dfd2dde3f63e1c0de798b412319d6. The dirty patch also
contains the already published bounded replay extension. The previous
unvalidated fence/volatile diagnostic hypothesis was excluded and saved
locally in test_results/nccl_transport-startup-with-old-fence-candidate.cpp.

Nine independent-process cases pass: healthy full-state S4 and startup,
owner, archive-admission, archive-finish faults on each rank. Startup faults
terminate both ranks with exit 1 in approximately 0.5 seconds, with no
forced cleanup or group-complete. Full-state U/S/map/pre-dedup oracle and
asymmetric capacity fixture also pass (the latter fixtures use rank threads).

Initcheck 2025.2.1 full-BFS diagnostic on the preceding library-path
candidate now ends both processes with exit 1 in 11.49 seconds instead of
requiring forced cleanup at 120 seconds. No group-complete exists. Both
sanitizer summaries report zero instrumented errors, but application/NCCL
fails (including unspecified launch failure): this is NOT an initcheck pass.
Native/compact constructor wiring added afterward is compile-checked and
covered by the final ordinary replay, not by this initcheck execution.

Artifacts: test_results/vast_a4000_cancel_20261001/
mgbfs-a4000-startup-cancel-candidate2,
mgbfs-a4000-startup-cancel-allpaths,
mgbfs-a4000-startup-initcheck129.
T4 acceptance, unfiltered memcheck/initcheck and remaining profiles stay open.
