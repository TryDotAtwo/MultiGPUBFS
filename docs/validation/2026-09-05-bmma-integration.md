# BMMA integration checkpoint (not implemented)

The current gate uses CUDA 12.8.93 and real T4s, not a Hopper build.
Primary instruction reference, matched to CUDA 12.8:
[PTX 8.7 matrix fragments](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-88128).
The single-bit m8n8k128 XOR.POPC operation consumes one packed b32 A fragment,
one packed b32 B fragment and two s32 accumulators per lane. Accumulator zero
identifies equal 128-bit rows/columns. Warp participation and fragment mapping
must be tested, including partial tiles; padding must never create a match.

Concrete integration site: `cuda/bounded_owner.cu` currently runs validate,
adjacent incoming duplicate flags, three merge-based membership passes (prev,
curr, accepted), compact, and finish_compare. OwnerCommit later merges accepted
hashes and publishes survivor indices. BMMA should replace membership passes,
not reservation, commit, archival ownership or StateReady semantics.

Keep the same duplicate-category precedence: incoming duplicate, prev, curr,
accepted, survivor. Compare flags, counts and selected indices against the CUB
backend, then run full archived BFS with both profiles. Include all-equal keys,
empty references, partial tiles, matches at each boundary and sticky fatal.

The architecture also requires deterministic prefix refinement before oversized
bucket comparisons. A quadratic scan of arbitrarily large buckets would not
satisfy the backend contract, even if a small tile test passed. Refined ranges
must be flat, preallocated and bounded; identical full hashes need segmented
reduction. No fallback to CUB under the BMMA backend name. Default remains CUB
until measured end-to-end evidence supports changing it.

RED confirmed on Kaggle `trydotatwo/mgbfs-bmma-owner-gate` v1 at cdb0fd4:
both CUDA translation units compile, then linkage fails at both calls to
missing `mgbfs_bounded_owner_create_backend`. The same fixture exercises
category precedence, stable survivor indices, accepted merge, repeat-job
deduplication, invalid ranges and insufficient grants/capacity. Evidence:
`test_results/bmma-owner-v1/bmma-owner/build.log`. This establishes the missing
implementation, not GPU BMMA correctness. The separate v19 distributed gate
continues independently; no active run was restarted for this RED check.
