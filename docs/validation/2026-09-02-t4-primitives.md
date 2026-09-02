# Native matrix primitives: real Kaggle T4 validation

Status: **PASS for primitives, NOT a complete multi-GPU BFS gate**.

- Kernel: [mgbfs-native-matrix-primitives-t4](https://www.kaggle.com/code/trydotatwo/mgbfs-native-matrix-primitives-t4), private version 1.
- Source: public GitHub checkout at `95747c376355fa0fd9568f3d2b0d4a0720f0900b`.
- Hardware: two distinct physical Tesla T4 GPUs, 15360 MiB each, initially
  14912 MiB free each. Tests run on each physical device in separate processes.
- Compiler: CUDA 12.8.93, target sm75, Rust 1.75.0.
- CUTLASS: `ffa119a1255d78998536107466cc7097ecefa393`.
- CPU: 25 contract tests passed in the Kaggle worker.

Four GPU test executables (`generate`, `hash`, `route`, `owner`) each ran on both
GPUs, once normally and once per Compute Sanitizer tool: memcheck, racecheck,
initcheck, synccheck. All 40 executable invocations passed. Raw sanitizer logs
report zero errors; racecheck reports zero hazards and warnings.

The GPU owner test covers old-layer membership, current-epoch duplicates,
cross-epoch duplicates, stable OriginRef retention, tiled merge boundaries,
maximum Hash128, empty epochs, device-count overflow, capacity exhaustion,
unsorted input and non-increasing epoch rejection. Overflow leaves the
persistent accepted span unchanged and sets a sticky fatal status.

The GPU tests are launched separately with `CUDA_VISIBLE_DEVICES=0` and `1`.
There is no NCCL exchange or two-rank scheduler in this gate. No BFS-time or
VRAM Pareto claim follows from it. In particular, isolated primitive success
does not prove archive overlap, global FinalizeDepth or distributed termination.

## Raw evidence (kept private and outside git)

`test_results/kaggle_native_primitives_v1/native-primitive-gate/` contains
`summary.json`, GPU inventory, exact checkout SHA, compiler versions, build log,
CPU contract log and all 40 per-GPU test logs. The parent directory includes the
Kaggle notebook worker log. Artifacts were downloaded after Kaggle reported
COMPLETE; both summary rows and raw passing-test/sanitizer summaries were checked.

## Subsequent local integration test

`crates/mgbfs-cuda/tests/pipeline.rs` connects generation, hashing, source routing
and owner commit in one CUDA stream without intermediate host synchronization.
For U4 moduli 2..6, first two expansion depths, and pre-dedup ON/OFF, survivor
hashes match independent full-state CPU oracle layers; each retained OriginRef
replays to the corresponding full state. It passed on the local RTX 3070 with
all four sanitizers. That integration test was added after the version 1 source
commit and is **not** part of the T4 result above. Frontiers are supplied by the
CPU oracle; the test is not a native exhaustive BFS runtime.

## Version 2: connected pipeline also verified on both T4s

Private kernel version 2 checked out
`d63683f62837cf54ba778226db38f1cc2ac088ab` and included the `pipeline` executable.
Kaggle reported COMPLETE, and downloaded artifacts report PASS_PRIMITIVE_GATE.
All **50** executable invocations passed: 5 executables x 2 physical T4s x
(plain + memcheck + racecheck + initcheck + synccheck). All 50 raw logs contain
passing tests; all 40 sanitizer logs report zero errors/hazards/warnings.
The 25 CPU contract tests also passed again in this Kaggle worker.

This confirms the connected generation/hash/route/owner test described above
on real sm75 hardware, including device-side route counts and pre-dedup ON/OFF.
It still does **not** include a two-rank NCCL exchange, native frontier
materialization/scheduling, archive overlap, or an exhaustive native BFS run.

Raw evidence: `test_results/kaggle_native_primitives_v2/native-primitive-gate/`.
SHA256 of the downloaded `summary.json`:
`d36ce8c52240558fda328f2534cb9b2c9a5bde3af35ff7c83c3705a7523eec95`.
Version 1 summary SHA256:
`31d4f81ebe592fcd467906d74dacfcbec1f3a0b664f2c1c1c3caf7bc49e7afaa`.
