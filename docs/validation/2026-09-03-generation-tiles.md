# Generation layout and tile experiment

## Fixed alternatives (no runtime switching)

| ID | GEMM orientation | CTA tile M,N,K | Warp tile | Child output |
|---|---|---|---|---|
| 0 | generators * parents | 64,32,64 | 32,32,64 | legacy scalar bytes |
| 1 | parents^T * generators^T | 64,32,32 | 32,32,32 | scalar bytes |
| 2 | parents^T * generators^T | 128,32,32 | 64,32,32 | scalar bytes |
| 3 | parents^T * generators^T | 64,32,64 | 32,32,64 | scalar bytes |
| 4 | same as ID 1 | 64,32,32 | 32,32,32 | U4 vector output |

ID 4 explicitly requires n=4. IDs 0..3 support the existing general square-matrix
primitive. Existing `mgbfs_generate_create` and default Rust constructors still
choose ID 0: no winner has been selected without target measurements.
New `mgbfs_generate_create_variant` and `new_pipelined_with_generation` select
one immutable alternative before allocation. Unknown variants are errors.

## Dataflow

For U4, original GEMM is [24,16] * [16,4B]. Its long dimension maps to grid.y.
Transposed GEMM is [4B,16] * [16,24], with the same packed parent/generator byte
buffers, now interpreted as the opposite operands/layouts. Its long dimension
maps to grid.x. No extra tensor transpose or device allocation is added in run.
Output indexing converts back to the exact original parent-major/move-major,
canonical row-major u8 state format. General generator-column output is padded
to a multiple of four where necessary; padding weights are initialized to zero.

ID 4 loads four aligned uint4 vectors (the four columns of each result),
transposes/reduces them in registers, and stores a complete 16-byte state.
Consecutive moves consume adjacent source vectors. It preserves the same
modulus arithmetic and output format, including modulus 256.

The CTA 128x32x32 candidate initially used four 32x32x32 warps. CUTLASS rejected
that configuration: a 32x32 u8 B tile cannot supply a 16-byte vector to each of
128 loading threads. The implemented candidate uses two 64x32x32 warps instead.
This compile-time rejected candidate is not a benchmark failure or a fallback.

All variants check grid bounds before enqueue/output writes; legacy oversized
U4 batches now return explicit status 7 rather than failing inside CUTLASS.
Transposed variants handle 2^20 parents without the previous grid.y overflow.

## Correctness evidence and scope

- RED: native test at 524281 parents failed with legacy status 5 versus expected 0.
- New transposed variants: 2^20 identity parents, every generated child verified
  against the generator matrix. All four alternatives pass locally.
- Dense nontrivial inputs, n=2,3,4,5, partial batches 1/3/16/67, non-power-of-two
  moduli and modulus 256: exact CPU successors and Hash128 agree for all applicable
  variants. The n=3 single-involution case exercises padded generator columns.
- Invalid variant and legacy grid rejection preserve the output sentinel.
- Full self-fed BFS U4 m2..6, pre-dedup ON/OFF, all four alternatives: every state
  in every layer matches the full-state CPU oracle, through graph exhaustion.
- Smaller m2..3 full-feedback fixture is used for sanitizer runs; large-generation
  test is plain-only. Sanitizer coverage is not claimed for every large workload.

## Measurement protocol

Rust `generation_bench` has fixed allocations before warmup, at least 200 ms
warmup, seven samples of 50 ordinary calls per configuration, and CUDA event timing.
Each ordinary call includes packing, GEMM, modular conversion, and child writes.
Separate instrumented calls record pack/GEMM/write times using four caller-owned
events. Those instrumented samples are not substituted for ordinary-call timing.
Arithmetic sample checks are outside timing. Modulus is 256, dense canonical
input values, batches 4096/65536/262144/524280/1048576. Legacy's unsupported largest
batch is retained as UNSUPPORTED_GRID, not silently removed or marked successful.

The dedicated private Kaggle package uses two physical T4s for correctness:
plain + four sanitizer modes, generation + feedback, 20 test invocations total.
Timing uses only GPU 0. Full BFS comparison is U4 m16 (16,777,216 states), fixed
batch 262144 and capacity 16^6 for ALL variants, five fresh-process repetitions,
alternating variant order. Each process warms a whole same-workload BFS first.
Only generation changes; hash, owner, sorting, materialization and streams stay
the same. This isolates the end-to-end effect from a larger batch opportunity.

Neither executor is production archived BFS. No archive is disabled to inflate
performance. Full-state digest checks use m5, in addition to CPU oracle fixtures;
large m16 comparisons check all layer counts and known total cardinality.

Nsight Compute full materialization-kernel reports are attempted for IDs 0 and 4
if the tool/counter access is available. Availability/failures are recorded; stage
times alone do not prove transaction efficiency or optimal occupancy.

## Current status

Local build/plain/oracle tests passed. All eight local sanitizer invocations
passed: generation and small full-feedback fixtures under memcheck/racecheck/
initcheck/synccheck. Racecheck feedback took 141.72 s, zero hazards; all other
tools reported zero errors. Logs: `test_results/generation-tiles-local/`.
Target T4 experiment is in preparation. No T4 speedup or default change is claimed.
An earlier local diagnostic suggested output conversion dominates GEMM, motivating
ID 4; local laptop timing does not select a T4 winner.
