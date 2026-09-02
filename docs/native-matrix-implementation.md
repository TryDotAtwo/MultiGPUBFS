# Native matrix BFS implementation status

This is the standalone implementation approved on 2026-09-02. CayleyPy at
`f0f2b8e` is an external, unchanged torchrun baseline. Following explicit approval
on 2026-09-02, GitHub is the source of truth:
`https://github.com/TryDotAtwo/MultiGPUBFS`, branch `codex/native-matrix-bfs`.
The repository was initially private and was made public on 2026-09-02 at the
user's explicit request. Kaggle runs must check out an immutable commit from
GitHub anonymously; no GitHub token or Kaggle Secret is required for source
downloads. GitHub Actions checks CPU contracts only, not GPU correctness or
performance.

## Required gates (not claims of completion)

- [x] Matrix/config manifests, digest roundtrip, exact matrix oracle, frozen GEMM hash vectors
- [ ] StateTraits integration and frozen complete config serialization vectors
- [ ] Allocation ledger and complete kernel-query-backed memory planner
- [x] StateRing / owner-commit / exchange-epoch CPU protocol models
- [x] Checksummed archive codec, Linux physical preallocation and fault injection
- [ ] Archive pinned ring, worker, shard/bucket directories and group RunCommit
- [ ] One-GPU streaming DENSE/CUB implementation
- [x] CUTLASS unsigned Tensor Core generation and GEMM hash, local RTX 3070 tests
- [x] CUB full-128-bit source sort and stable pre-dedup ON/OFF, local RTX 3070 tests
- [ ] Actual sm75 hardware validation (cross-compilation is not this gate)
- [ ] Real 2xT4 native NCCL exchange with overlap
- [ ] HASH_FIRST rematerialization and source leases
- [ ] BMMA bucket implementation and equivalence
- [x] Local primitive memcheck/racecheck/initcheck/synccheck (generation/hash/route)
- [ ] Full runtime / real 2xT4 Compute Sanitizer gates
- [ ] Five-repeat m12/m16 A/B and m20/m24 capacity probes

## Boundaries

Rust owns contracts, CPU verification, host scheduling and artifact management.
C++/CUDA owns GPU kernels and minimal CUDA/NCCL C ABI. Python only launches and
exports benchmark fixtures. CPU models are never a fallback for `run`.

The runtime will expose preflight/run/calibrate/verify/bench. Until a native
backend exists, missing functionality must return a failure, never a synthetic
success report. No CUDA performance or multi-GPU correctness claim is valid from
CPU tests alone.

First matrix profile: square, invertible canonical u8 matrices, modulus 2..256,
inverse-closed generators. Owner commit is irreversible. Native output archive
is mandatory; the baseline has a weaker output contract. Report search and
durable completion times separately.

## Benchmark contract

Independent full-state CPU oracle: U4(Z/mZ), m=2..6. Baseline parity: m=5..8,
known diameters 10,13,12,14 and cardinalities m^6. Seeds 0,1,20260828; rank maps
identity and swap. Calibrate m12 before selecting immutable production profiles.
Measure m12/m16 in five fresh process groups, after CUDA/NCCL warmup. Probe
m20/m24 once, with 30-minute external timeout and explicit failure rows.

Keep raw repetitions, median/MAD, per-depth counters/times, routed bytes,
per-rank imbalance, explicit allocation bytes, CUDA free-memory readings and
external device-memory samples. Report PyTorch allocated/reserved separately.
Do not compare first-run NCCL setup with warmed native kernels.

## Verified development slice, 2026-09-02

The default Rust workspace tests contain 25 contract tests. The independent
full-state oracle exhausts U4 moduli 2 through 6 and checks the three-layer edge
window; it confirms diameters 10 and 13 for moduli 5 and 6. This is **not** a
native-vs-CayleyPy comparison.

Three GPU integration tests cover:

- hash: 4 state widths x 3 seeds x 5 batch lengths (60 combinations);
- generation followed by hash without intermediate host synchronization:
  4 matrix/modulus combinations x 4 batch lengths (16 combinations);
- CUB routing: all 128 key bits, stable OriginRef retention, duplicate keys,
  pre-dedup ON/OFF, zero-sized input, partial batches and capacity failure.

These GPU tests passed locally on an RTX 3070 Laptop (8 GiB), CUDA 12.8.93,
driver 572.70, CUTLASS commit `ffa119a1255d78998536107466cc7097ecefa393`.
All three passed the four sanitizer tools with zero reported errors/hazards.
None of this establishes a speedup or correctness of the missing BFS runtime.

`OwnerModel`, `StateRing` and `Sequencer` are CPU protocol models only. The
archive codec currently constructs a bounded CPU payload and writes
synchronously; it must run on a disk worker with preallocated pinned buffers
before it can satisfy the production no-backpressure contract. No production
`run`, `calibrate` or `bench` executable has been supplied as a pretend success.

## Reproduce in the existing Docker toolchain

Mount this checkout at `/src`, use `/src` as working directory, pass `--gpus all`
and run `bash scripts/native-check.sh` in `multigpubfs-rust-toolchain:dev`.
This captures environment/build/test logs in ignored `test_results/native/`.
It compiles sm75 and sm86 by default; set `MGBFS_CUDA_ARCHITECTURES` explicitly
for another **preselected** build target. `CUTLASS_ROOT` defaults to `/opt/cutlass`.

Run `bash scripts/native-sanitize.sh <GPU-test-executable> ...` to apply all four
tools to the exact executables printed by Cargo. Logs are retained under
`test_results/native-sanitizer/`. The script propagates nonzero test and sanitizer
statuses. A local test executable path is not portable to a different build.

## Next implementation dependency

Implement and test the GPU owner accepted store and irreversible per-bucket
commit, then connect device generation/hash/source-sort/owner stages into the
bounded one-GPU DENSE runtime. NCCL exchange, HASH_FIRST materialization,
BMMA_BUCKET, complete memory sizing, CLI, mandatory archive integration and
the real 2xT4 A/B are still required by the approved plan.
