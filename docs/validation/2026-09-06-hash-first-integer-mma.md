# HASH_FIRST integer MMA: experimental generation backend

The canonical child state never enters a global child-state buffer. Each warp
owns a candidate, iterates 8x8 output tiles and K tiles of 16, and accumulates
unsigned-byte matrix products in two signed 32-bit registers per lane.
The modular epilogue immediately contributes to four 64-bit affine hash sums.
Warp reductions write only Hash128 and OriginRef. The host rejects non-SM75
devices rather than selecting the scalar backend.

This accelerates matrix multiplication with Tensor Cores; the hash projection
still uses integer register reductions, **not a second Tensor Core GEMM**.
It is not evidence of a speedup or of the final overlapping pipeline.
There is no new device allocation compared to the scalar hash-only leaf.
Current experimental entry checks device properties each call; moving that
check into a validated preflight plan is still required before hot-path tuning.

Fragment mapping and instruction:
[CUDA 12.8.1 PTX ISA, mma.m8n8k16](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-8816).
The accumulated matrix product is bounded by n*255^2 and the full projection
by 33025*255*(p-1)+offset, within s32 and u64 respectively under the manifest.

## Evidence

Kaggle `trydotatwo/mgbfs-hash-first-tc-gate` v1 at 79e7967 was RED:
the fixture compiled, linkage failed solely at the missing TC entry point.

V2 at b3929cc completed on T4. Plain, memcheck, racecheck, initcheck and
synccheck each report HASH_FIRST_GENERATE_PASS and zero sanitizer errors;
racecheck also reports zero warnings/hazards. SASS contains
`IMMA.8816.U8.U8`. The fixture covers frozen 2x2 vectors, absolute OriginRefs,
capacity failure without stores, zero-count batches, and full CPU matrix/hash
oracles for n=1,3,4,8,9,16,17 and modulus=2,5,256.
Evidence: `test_results/hash-first-tc-v2/hash-first-tc/`.

At 6e98fcf the explicit `new_hash_first_tc_with_owner` constructor connects
this leaf to existing HASH_FIRST routing, owner commit, regeneration and
archive lifetimes. Selection is fixed before allocation; scalar constructors
remain unchanged. CUDA-feature compile check passes. A new two-rank full-state
archive fixture uses both CUB and BMMA owners. Its hardware result is pending;
the single-GPU leaf result must not be presented as that integration gate.
