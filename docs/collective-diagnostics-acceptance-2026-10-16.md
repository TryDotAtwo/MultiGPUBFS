# Collective diagnostics acceptance, 2026-10-16

Package: `0.2.0.dev22`, Python source `d31de6d341b37c130e2598d1079f19a4793829e1`.
Wheel SHA-256: `b691739ae60e917a7a4a677453edc666b9aaa38a34b29931eb0db37deda2a267` (140396 bytes).
Native index remains `98cb1c35e660d454b48ba65dbc684e1c4baedf6320fe6c644fe9461ef4d5050b`; native binaries and CUDA/Rust
algorithms were not rebuilt or changed.

The change follows the requested allocation policy: an eight-GPU selection
stays one eight-rank test, not eight independent one-GPU BFS executions.
Primitive API checks have concurrent per-rank children and combined results;
NCCL/BFS run the real distributed implementation. Checked runtime handles are reused across constructors. Catalog constructors reuse
one process/context per rank. Recipe search no longer falls back to one GPU.

GitHub CI: 777 tests passed, including 14 regressions. The first 11 regressions initially failed (3 failures, 8 errors); three
runtime-reuse regressions subsequently failed (1 failure, 2 errors) before
checked runtime handles were reused across constructors. A report-summary
field mismatch found during integration was fixed before packaging. The tests
include real eight-child concurrent CPU process/barrier evidence and retained
rank-5 failure details; this is scheduling evidence, not eight physical GPUs.

Installed public wheel, physical 2xTesla T4: ordinary doctor passed on both
cards together, including per-rank context/memory/kernel/persistent probes,
NCCL, exact BFS and the executed READY launcher. The entire finite catalog
ran in two concurrent workers with reused contexts; each supported successor
was checked against the CPU oracle. See the
[sanitized machine-readable receipt](acceptance/collective-diagnostics-20261016.json).

The ordinary doctor took 106.57 seconds;
the joint catalog took 9.62 seconds.
These are synthetic diagnostic timings on this 2xT4 run, not BFS throughput
benchmarks or predictions for another server.

Physical eight-GPU execution of the new diagnostic scheduler has not been
repeated here. The previous dev20 server archive separately demonstrated
successful eight-GPU small BFS/NCCL. The real production calculation's rank-5
failure is not diagnosed by this change or by the old synthetic acceptance.

Reproduce with [the synthetic private-test notebook](../examples/acceptance_collective_doctor.ipynb)
or run `python -m multigpubfs doctor` inside the allocated job after updating.
`doctor --deep --total-timeout 1800` additionally runs the controlled matrix.
Optional diagnostic build controls may require their own cached components;
production prefetch/cache semantics are unchanged.

[CPU/package CI](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38953099289).
