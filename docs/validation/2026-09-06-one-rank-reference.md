# One-rank path in the distributed reference engine (GPU pending)

The same `DistributedNativeBfs` now accepts world size 1 or 2. Single-rank
mapping is `[0,0]`: both high-bit hash ranges belong to rank 0. Global bucket
and shard counts are not halved. It uses the full-prefix bucket directory,
routes all sorted records to local owner commit, and issues no nonexistent
peer send/recv or remote materialization round trip. Generation, deduplication,
StateRing, archive, failure guards and finalization remain the shared code path.
The example launcher accepts `WORLD_SIZE=1` and reports world size explicitly.

CPU topology tests verify geometry and invalid maps. CUDA-feature Rust check
passes. A new real-GPU archive fixture runs the three matrix profile/generation
choices times two owners times two pre-dedup choices, plus four compact DENSE
variants: 16 one-rank configurations. Each decodes every archived state/hash
and compares complete layers with the independent small-group oracle, as the
existing two-rank fixture does. Full hardware gate completion is pending;
partial live evidence is recorded below.

Scope limitations: not an arbitrary-N-rank router; conservative bounded peer
buffers still exist in the one-rank allocation plan. No speed or VRAM improvement
is claimed from compilation.

## Shared benchmark orchestration

Commits `8b48cf4` and `b02bf3f` add explicit
`MGBFS_BENCH_WORLD_SIZE=1|2` selection to the Python profile panel. Default is
still two ranks. Both choices retain 12 native configurations, five measured
runs each, and the three-batch CayleyPy calibration. Native archive verification
and cleanup enumerate the selected rank count. The launcher exposes physical
GPU 0 for one rank, GPUs 0 and 1 for two ranks; the external sampler charges
only those device indices. Missing device samples stay unknown, not zero.

At one rank the worker uses the existing single-GPU CayleyPy measurement
function, with a full graph warmup and no distributed barrier. The pinned
CayleyPy dispatcher selects ordinary `BfsAlgorithm` when `WORLD_SIZE=1` and
`num_gpus=1`; its torchrun BFS requires `WORLD_SIZE>1`. Two-rank behavior remains
the existing distributed measurement. Native still uses a world-one NCCL
communicator in its shared engine, so this is not a claim of identical internal
communication overhead.

Twelve CPU metric/orchestration tests pass. GPU subprocesses are substituted
at the test boundary; these tests prove launch configuration, retained sample
inventory and metric aggregation, not GPU correctness or speed. Actual paired
hardware measurements with this orchestration remain pending.

Version 29 of `trydotatwo/mgbfs-distributed-sanitizer`, script version
`347583042`, was observed RUNNING on two T4s with source
`b520a788e1ad6de3202e77dadf7bf59662f941c3`. Its live logs show all 12 runtime
fixtures passing plain, memcheck (zero errors), and racecheck (zero errors or
warnings), including the new one-rank archive fixture. The remaining tools and
24 process-level smokes still require completed-output verification. The
unversioned notebook URL showed older version 28 output; it is not evidence
for version 29.
