# S8 independent-process LSA check, 2026-10-01

Hardware: Vast instance 53673926, two RTX A4000, driver 580.173.02.
This is not the required T4 acceptance gate or a paired performance result.

Runtime commit: 705684ffce9dfd2dde3f63e1c0de798b412319d6.
Only the replay script was modified; captured source.patch SHA-256:
622159ed3604174e3d687a3162f47fa995587a757f03e0056d5f24ea69fc9622.
No local nccl_transport.cpp fence candidate was included.

Configuration: DENSE, NCCL_LSA, CUCO_RANK, pre-dedup ON, batch 128,
matrix_u8 states/archive, four shards, eight buckets. The bounded oracle
reserves 40320 records per rank and 80640 future records conservatively;
these settings are not an optimized production memory plan.

Two independent rank processes completed S8. All 40320 canonical states,
their depths, rank archive checksums and local counts matched the independent
permutation CPU oracle. The layer sizes are:

1,3,6,12,23,44,80,142,247,411,662,1019,1481,2059,2745,3465,4126,
4633,4913,4777,4163,3079,1612,488,94,25,6,3,1.

Uninstrumented process wall time was 1.707 seconds. This includes process
startup and verification boundary overhead and is NOT search time or an
A/B speed claim. The separate Nsight execution also passed full-state checks.

Actual runtime NVTX ranges identify 178 batch ranges on each rank and 29
FinalizeDepth ranges. Joining CUPTI runtime records to a batch range on
the same globalTid, with complete start/end containment, found zero
Synchronize APIs and zero non-Async Memcpy APIs inside healthy batches.
Rank 0 has 176 cudaMemcpyAsync and 176 cudaMemcpy2DAsync calls in those
ranges; rank 1 has 173 of each. These asynchronous archive transfers must
not be described as absence of D2H traffic. Each rank has 133 event-query
credit checks (Nsight exposes duplicate API-name variants).

This trace excludes post-owner host drain on this observed execution. It
does not establish peak overlap, all workloads/modes, or sanitizer success.
HASH_FIRST still uses host pending counts, request-size exchanges and
publication readbacks; enabling device_epoch alone would be incorrect.

Artifacts: test_results/vast_a4000_cancel_20261001/mgbfs-a4000-s8-batch128
and mgbfs-a4000-s8-nsys128, including source.patch, summary.json, two rank
logs, lossless archives, .nsys-rep and SQLite exports.
