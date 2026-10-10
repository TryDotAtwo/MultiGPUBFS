# Exact permutation GEMM and CUDA comparison

Permutation graphs with degree 1..64, 1..64 generators and labels 0..255 can use `MGBFS_GENERIC_GENERATOR=gemm`. The generator matrices are exact one-hot matrices. A large parent matrix is multiplied by the concatenated generator matrices, using unsigned-byte Tensor Core GEMM with exact int32 accumulation. Permutations retain their compact byte state storage. Existing modular-matrix GEMM eligibility is unchanged.

The parent packing and result routing use preallocated device buffers. The same exact hash, packed exchange, history and dedup contracts are retained. A forced single-GPU GEMM request uses the generic distributed implementation rather than the legacy CUDA-only shortcut.

The autotuner now considers permutation GEMM before optional sorted-owner lane variants. Family caches accept eligible permutation GEMM profiles. Live candidate timing drains shared retirement/reseed work at a collective boundary before starting the first candidate timer. This boundary runs only when profiling a new frontier-size bucket; normal chunks do not gain a synchronization.

## RTX 3060 evidence, 2026-10-10

Two RTX 3060 12 GB, CUDA 13.2, SM86. Baseline source 12f10379f50d06e7d74a1943f893ab6582f27424 plus this feature. GPU route records match exactly on both devices, including byte label 255, tail batches and batches through 1,048,576 parents. Complete (9,1) layer tables and retained terminal states match an independent CPU oracle (362,880 states). Single-GPU forced GEMM, degree33/64 permutations and the existing degree8 modular-matrix path also pass exact CPU layer/state checks. Python suite: 531 tests completed, one skipped.

| Measurement | CUDA | GEMM |
| --- | ---: | ---: |
| GPU0 generation + packing + hashes + routing, 1,048,576 parents | 1.942 ms | 3.359 ms |
| GPU1 same scope | 1.999 ms | 3.419 ms |
| (14,1), large completed-layer prefix, median of 3 alternating repetitions | 3.376134 s | 3.569562 s |

The heavy comparison fixes capacity50,000,000/card, batch349525, shards1, HASH history and full transport for both generators. Completed transitions from depth23 through depth36 have parent frontiers at least1,080,005; the largest completed output layer contains98,340,264 states. All six layer tables agree. Both modes stop with RESOURCE_258 on the next transition. The failed transition and setup/output wall time are excluded from the completed-layer comparison. This is not a complete (14,1) BFS timing.

On this configuration GEMM is about5.7% slower for the measured large prefix and about1.7x slower for generation+hash+route alone. No B300 speedup is inferred. Native timing identity is recorded in the raw reports; the subsequent live-timing fix changes only profile boundaries, not forced CUDA/GEMM generation or the fixed-mode measurements.

Live tuning remains a sampling heuristic on different parent chunks, not identical-work replay and not a guarantee of the fastest configuration. Preserve the raw candidate scores and frontier-size coverage when interpreting a selected profile.

After the collective timing-boundary fix, a fresh (14,1) automatic profile run matches all36completed transitions and selects CUDA at every live frontier bucket tested (3,746,333 through74,757,850 parents). Raw before/after candidate scores are preserved.
