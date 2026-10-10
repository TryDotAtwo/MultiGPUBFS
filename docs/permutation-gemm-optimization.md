# CUTLASS permutation GEMM optimization — 2026-10-10

Implemented remotely on Vast 55203640 / machine 92763, 2×RTX 3060 12 GiB. Baseline source: 968eab2. CUDA 13.2.86, driver 595.71.05, CUTLASS ffa119a1255d78998536107466cc7097ecefa393, SM86. No B200/B300 were rented or benchmarked in this stage.

## Change

For one-hot generator rows, keep INT32 Tensor Core accumulation and emit exact UINT8 products. A one-hot row selects one input byte, so label 255 is preserved; non-one-hot matrices retain INT32 intermediates. The default narrow case (degree ≤16, padded output columns ≤64) uses CTA 128×64×32, warp 64×32×32, MMA 16×8×16 and two stages. Other shapes retain the baseline default. All buffers remain preallocated; shape/one-hot selection is cold context setup. No CPU generation/dedup was introduced into the hot path.

The matrix output buffer shrinks from 192 MiB to 48 MiB at 1,048,576 parents / 48 padded columns. This is allocation size, not a measured DRAM-traffic counter. Input packing and a separate hash/route kernel remain. Memory admission still reserves a conservative workspace upper bound.

Seven tile/output variants were measured. K32 with MMA K32 was compile-rejected by the even-iteration mainloop contract; K32 with MMA K16 compiled and won. A wider UINT8 epilogue vector was compile-rejected and was not shipped. The supported forcing values are MGBFS_GEMM_VARIANT=0..6; 0 selects the old baseline, 6 the optimized narrow byte variant. Non-one-hot matrices fall back to INT32 even under byte variants. Profile identities hash the CUDA library/configuration; distributed resume identity also includes the requested variant.

## Measured generation + packing + hashing + routing

These event timings exclude exchange/dedup. Full route records, including states and indices, were SHA-matched against CUDA on both cards. Seven alternating samples per mode; medians below use 1,048,576 parents / 3,145,728 children.

| GPU | CUDA | old GEMM | optimized GEMM |
|---|---:|---:|---:|
| 0 | 1.940480 ms | 3.309248 ms | 2.804736 ms |
| 1 | 1.997824 ms | 3.365888 ms | 2.859712 ms |

The optimized GEMM path reduces this time by about 15%. Separately measured packing is ~0.388 ms; the GEMM operation decreases from ~0.99 ms to ~0.531 ms. Isolated phase timings are diagnostic measurements, not additive end-to-end throughput guarantees.

## End-to-end BFS large completed prefix

Graph (14,1), two ranks, HASH history, full transport, one shard/rank, capacity 50,000,000/rank, batch 349,525. Three alternating repetitions per mode. All nine layer-size tables agree through completed depth 36. Compare transitions 23..35; parents range 1,080,005..74,757,850, final completed output 98,340,264.

| Mode | Median completed large-prefix seconds |
|---|---:|
| CUDA | 3.369032233 |
| old GEMM | 3.564713009 |
| optimized GEMM | 3.502405574 |

Optimized GEMM throughput is 1.01779× old GEMM (about 1.78% faster). Its time is 3.96% higher than CUDA. The graph did not complete: the next transition stopped RESOURCE_258. Failed-transition time is excluded; these values must not be called full-graph times.

Fresh automatic tuning reaches the same layer prefix and selects CUDA in all 6 observed large-frontier buckets (3,746,333..74,757,850 states). Initial plan.generator_backend=gemm is a cold/global plan; actual online size-profile winners override it. Live disjoint chunks are a tuning heuristic, not matched-work replay.

## Correctness and limits

- Full records agree for all 48 initial candidates, 12 K16 checks and 12 final-native paired checks, including odd tails/count 33 and label 255.
- Independent CPU oracle: complete (9,1), 362,880 states, full layer counts and retained terminal states agree with CUDA/GEMM.
- Single GPU, degree 33/64 regeneration, non-one-hot coefficient-2 matrix fallback, parent transport and a two-column one-hot matrix all complete and match CPU layers/retained states.
- Final CPU suite: 531 tests, one skip (see cpu-tests.log).
- Input packing/epilogue+route fusion are not removed; this is a measured improvement of the existing exact GEMM path, not a claim of globally optimal BFS.
- Astra consultation did not return a confirmed response. No expert approval is claimed.
- The standalone generation/route translation unit also cross-compiles for SM100a; this is compile evidence only, not a full B300 build or device execution. B300 correctness/performance of this revision remains unverified.

Raw reports, saved states, configs, binaries, source bundle and SHA manifest are archived on private HF with readback before rental deletion.
