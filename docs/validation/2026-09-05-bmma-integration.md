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

Implementation c65d9fc replaces membership passes with SM75 XOR.POPC while
retaining the existing compact/reserve/commit contract. Each bucket CTA uses a
fixed shared array of 129 range descriptors (24 bytes each), enough for binary
prefix depth 128. Only nonempty candidate/reference intersections are pushed.
Splits use the highest differing endpoint bit; each bounded leaf uses 8x8
comparisons with masked tails. Identical endpoint runs use linear marking.
No descriptor allocation occurs in a run. The constructor requires the declared
refinement bound to cover candidate capacity and rejects non-SM75 hardware.
One additional u32 error per bucket is allocated before execution; a separate
kernel publishes errors to avoid concurrent read/write of owner control.

Kaggle BMMA gate v2 pins c65d9fc and runs plain plus all four sanitizers. This
is not yet integrated into DistributedNativeBfs, nor profiled. One CTA per
bucket and serial metadata refinement are explicit performance risks requiring
measurement; the implementation is not claimed to saturate the GPU.

V2 completed at c65d9fc. BOUNDED_OWNER_PASS appears in plain and all four
sanitizer logs; zero errors and zero race hazards/warnings. The notebook had
two real T4s, but this leaf fixture executes on one device: this is single-GPU
owner correctness, not multi-GPU BFS evidence. Downloaded artifacts:
`test_results/bmma-owner-v2/bmma-owner/`. Next required gate is full archived
DistributedNativeBfs using explicit BMMA selection with DENSE and HASH_FIRST.

At f8f4b35, `new_reference_with_owner` freezes the existing OwnerBackend enum
and BMMA tile limit before constructing the owner plan. Existing constructors
remain CUB. HASH_FIRST capacity validation is shared by all constructors.
The new full archived fixture was first compile-RED (missing constructor),
then CUDA-feature cargo check passed after integration. Distributed gate v20
pins this source and checks both profiles with BMMA, seed 20260828, owner map
[0,1], pre-dedup OFF, plus the existing CUB/configuration/failure fixtures.
GPU result is pending; no performance or complete-architecture claim.
