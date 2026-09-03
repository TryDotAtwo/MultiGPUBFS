# Large generation launch limit found by the scaling benchmark

Kaggle single-GPU comparison v2, source 7b7137a, failed native U4 m16 during
warmup at batch 1048576 with `NATIVE_CUDA_STATUS_5` (exit -6). CayleyPy completed
16,777,216 states at the same batch, 1.597 s measured (one probe, not a median).
The notebook worker itself returned COMPLETE, but the experiment summary was
`NO_COMMON_COMPLETED_GRAPH`. This is NOT an OOM result and NOT a completed A/B.

Root cause isolated without changing CUDA kernels:

- `cuda/generate.cu` uses GEMM tile 64x32 and identity threadblock swizzle N=1.
- For U4 the GEMM N dimension is 4 * parent_count, so grid.y=ceil(parent_count/8).
- CUTLASS `can_implement` and plan creation do not reject this grid overflow;
  the launch fails and the generation C ABI maps CUTLASS launch failure to 5.
- Local direct C ABI reproduction on RTX 3070 Laptop with the same compiled
  kernel: queried cudaDevAttrMaxGridDimY=65535; count 524280 returns 0,
  count 524281 returns 5, count 1048576 returns 5. Synchronization returns 0
  because the invalid-grid kernel was never launched.

The reproduction script is retained privately at
`test_results/diagnose_generation_grid.py`. No core algorithm fix is included
in this benchmark-only task. Runtime preflight/launch geometry correction is
still needed to support larger requested batches safely and diagnostically.

The next experiment uses native batches 65536/262144/524280; CayleyPy keeps
65536/262144/1048576. These are independent preselected worker configurations,
not within-run fallback. On a failed size probe, test the other configurations
before declaring no common completed graph at that size. Fixed capacities and
the 1 GiB native reserve remain unchanged.

V2 evidence: `test_results/kaggle_single_gpu_benchmark_v2/`.
