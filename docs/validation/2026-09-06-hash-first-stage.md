# HASH_FIRST generation stage: scalar versus integer MMA

Kaggle `trydotatwo/mgbfs-hash-first-tc-gate` v3, source
`c2ae53244b24a2c4382d2925071fb60165134e18`, Tesla T4 device 0.
This is synthetic canonical binary matrix arithmetic, **not exhaustive BFS**.
Neither routing, deduplication, materialization nor archive is timed.

The independent frozen/oracle leaf tests passed normally and under memcheck,
racecheck, initcheck and synccheck (zero errors/warnings). SASS contains
`IMMA.8816.U8.U8`. Each benchmark shape additionally compares all scalar/Tensor
hashes and OriginRefs before measurement and checks emitted count and fatal.

Three warmups per backend, five paired samples with alternating backend order;
each sample averages ten launches. CUDA events span the launch batch; CPU wall
times are also retained. Clocks were not locked. All 120 individual samples,
including wall time and explicit device payload, are in
[the raw measurement file](data/2026-09-06-hash-first-stage.json).

| Matrix n | Parents | Moves | Scalar median ms | MMA median ms | Scalar/MMA |
|---:|---:|---:|---:|---:|---:|
| 4 | 16384 | 6 | 0.4992 | 1.7535 | 0.285 |
| 4 | 16384 | 24 | 1.6085 | 1.8966 | 0.848 |
| 4 | 65536 | 6 | 1.6557 | 1.9624 | 0.844 |
| 4 | 65536 | 24 | 6.6439 | 7.4171 | 0.896 |
| 12 | 16384 | 6 | 1.0037 | 1.4360 | 0.699 |
| 12 | 16384 | 24 | 4.3966 | 4.9017 | 0.897 |
| 12 | 65536 | 6 | 4.4172 | 4.9797 | 0.887 |
| 12 | 65536 | 24 | 18.5657 | 18.1734 | 1.022 |
| 16 | 16384 | 6 | 1.8270 | 1.6697 | 1.094 |
| 16 | 16384 | 24 | 7.4985 | 6.0888 | 1.232 |
| 16 | 65536 | 6 | 7.6806 | 6.1391 | 1.251 |
| 16 | 65536 | 24 | 30.6258 | 22.4670 | 1.363 |

For 65536 parents / 24 moves, GPU-time MAD is respectively scalar/MMA:
n4 0.0236/0.0758 ms; n12 0.1367/0.1543 ms; n16 0.0473/0.1255 ms.
Both backends use the same explicit device payload: 51,380,892 / 59,774,620 /
67,119,132 bytes for those three shapes. This excludes allocation alignment,
CUDA context, events and driver storage; it is **not total VRAM consumption**.

Decision: keep Tensor generation explicitly experimental, not the default.
The measured n16 advantage does not establish a complete-BFS speedup. The
current one-warp-per-child implementation pads small matrices and still hashes
with register integer reductions; these are optimization candidates, not
proven bottleneck attribution. The n4/small-batch anomaly needs profiling before
explaining it. No automatic backend switching is introduced.

## Separate distributed runtime gate

`trydotatwo/mgbfs-distributed-sanitizer` v24 completed at source
`1b44a0e0a265c580bff55f73b80ae746d536f708`. All ten distributed archive fixtures
passed normally and under all four sanitizers, including
`tensor_hash_first_preserves_archived_layers_with_both_owner_backends`.
That Tensor fixture covers UT(3,3), seed 20260828, owner map [1,0], pre-dedup ON,
and both CUB/BMMA owners with full archived-state/hash comparisons. This does
not establish the entire Tensor seed/map/pre-dedup matrix or large-graph speed.
Individual logs in `test_results/distributed-sanitizer-v24/distributed-sanitizer/`
show zero sanitizer errors and zero racecheck warnings/hazards.
