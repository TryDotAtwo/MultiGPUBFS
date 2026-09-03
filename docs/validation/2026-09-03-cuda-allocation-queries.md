# CUDA generation/hash allocation queries

Implemented: host C ABI `mgbfs_generate_query` and `mgbfs_hash_query`, sharing
geometry with the actual CUDA create paths. Queries enumerate every internally
owned generator/packed-parent/product or weight/offset/partial allocation.
The compiled CUTLASS GEMM supplies workspace size and can-implement status.
These fixed split-K-one policies reject a nonzero workspace before allocation;
they do not pretend an unsupported policy has zero scratch. Input/output banks,
CUDA context, allocator rounding and driver overhead are explicitly excluded.
Allocation queries are not a full run preflight: launch grid/device capability,
canonical input data and subsequent stages have separate checks.

Both create paths now use the query's sizes directly for cudaMalloc, removing
duplicate size formulas. C ABI has frozen sizes 48/40 bytes; Rust FFI tests
check sizes and independently derived literal values. Invalid queries clear
the output. Null output, invalid dimensions/modulus/variant, padded layout and
32-bit count limits have fixtures. Owner/CUB/NCCL query adapters remain pending;
the rank-wide ledger is not yet supplied by a complete production provider.

Local verification on Windows:

- Geometry test: observed RED with stub, then `ALLOCATION_SHAPE_PASS` using MSVC.
- Public query test: observed RED with stub, then `ALLOCATION_SHAPE_PASS` after
  compiling actual generate.cu/hash.cu and linking the C ABI. CUDA12.5, target
  sm75, CUTLASS ffa119a1255d78998536107466cc7097ecefa393, MSVC17.14.
  nvcc needed `-allow-unsupported-compiler`; this is NOT the Linux/T4 gate.
  Public query execution launches no GPU kernels and does not validate runtime
  numeric results or performance.
- Python hardware/source guards: 2 passed; Rust formatting passed.
- Initial local Rust regression could not run: Docker Desktop failed during startup
  on inaccessible sailor-ingest.sock; native cargo dependency download failed
  with Schannel SEC_E_NO_CREDENTIALS. No Docker reset, WSL shutdown or network
  reconfiguration was performed. Retrying native `cargo test --locked` outside
  the sandbox resolved Schannel access and passed all 55 Windows CPU tests
  (the Linux-only file extent test is not compiled on Windows).

Linux CI passed on df42c51 and launcher pin288d43d: all 56 Rust CPU tests,
two Python guards and the new g++ geometry fixture. Private Kaggle kernel
`trydotatwo/mgbfs-native-matrix-primitives-t4`, version7, was launched against
df42c51df6d3b44b8ba620de6edcf0b3e0f68e5f. Its results must be checked separately;
RUNNING does not certify either GPU identity or test success.

Version7 hardware evidence: two distinct Tesla T4 UUIDs, 15360MiB each,
CUDA12.8.93. Linux CUDA build and public-query CTest passed. GPU0 generation
plain and memcheck passed (including new FFI query test); memcheck reported
zero errors. Generation racecheck printed four passing Rust tests in186.14s,
but the enclosing180s watchdog killed the sanitizer before final exit/summary.
Therefore racecheck is NOT counted PASS and the run is INCOMPLETE. Preserved
artifacts: `test_results/native-query-v7/native-primitive-gate/`.
The repeated gate increases generation timeout to900s, keeps all tests and all
sanitizer modes, and keeps the exact same CUDA source commit. No algorithm
change or skipped sanitizer is used to resolve this harness timeout.

Rust report adapter (subsequent change): `mgbfs_cuda::allocation` exposes named
`QueryResult` values directly consumable as the Generation/Hash entries in
`RankQueries`. It transfers returned byte counts; it does not repeat geometry
formulas, invent missing owner/transport groups or certify a full rank plan.
The optional-CUDA wrappers call the native queries and propagate failures.
Report conversion passed RED/GREEN with frozen 48/40-byte ABI fixtures and named
plane sizes. Windows validation: 55 existing CPU tests +1 adapter test, two
Python guards, and `cargo check -p mgbfs-cuda --features cuda --tests` passed.
This type-check does not link/run CUDA. Kaggle v8 remains pinned to df42c51:
it validates the same C++ backend, not the subsequently added Rust report wrapper.

`scripts/verify_primitive_gate.py` checks the saved final summary against the
requested full source SHA, two distinct T4 devices and all80 unique combinations.
It also requires nonempty passing Rust suites and final zero-error sanitizer
summaries, plus the allocation-query CTest result. Synthetic validation fixtures
pass; the real incomplete v7 archive is rejected with exit1. This checker audits
evidence only; it does not run BFS or convert an incomplete gate into a pass.

The shared C++ geometry fixture is added to CPU CI. CMake also builds a public
query executable and registers a CTest test; the private Kaggle primitive gate
runs it before CUDA test executables. No production multi-rank result is claimed.

Version8 is INCOMPLETE: 22/80 combinations passed on GPU0, then the900s
ping_pong/racecheck watchdog fired inside generation_variants_preserve_full_layers.
Generation/hash/materialize/owner passed all five modes, including final clean
sanitizer summaries; ping_pong passed plain and memcheck. GPU1 was not tested.
Artifacts: test_results/native-query-v8/native-primitive-gate/.
The generation timeout adjustment is verified: its racecheck finished164.66s
with zero hazards. This is distinct from the new ping_pong fixture-selection bug.

The sanitizer selector predates the new generation-variant tests: it excluded
full_u4_pipelined_sweep but unintentionally included the second m2..m6 sweep.
Its summary label also understated actual coverage. The next launcher explicitly
selects generation_variants_small_feedback (m2..m3, all four variants, pre ON/OFF),
slot-reuse and capacity-failure fixtures for ping_pong/racecheck. The larger
variant sweep remains in plain/memcheck/initcheck/synccheck; no sanitizer tool
is removed. This is bounded racecheck coverage, NOT m2..m6 racecheck certification.
A RED/GREEN Python regression locks this selection; the immutable CUDA source
remains df42c51. The required repeat subsequently completed as version9.

Version9 PASSED the independently verified80-combination matrix on source
df42c51df6d3b44b8ba620de6edcf0b3e0f68e5f:

- GPU0 Tesla T4: GPU-b117f3d5-c2d0-79dc-7bbd-c06d719ffa8c.
- GPU1 Tesla T4: GPU-cdfa6d50-b30c-1d6f-86e4-b093ce0459b6.
- Each device15360MiB total,14912MiB free at initial inventory.
- All eight primitive/feedback executables passed plain, memcheck, racecheck,
  initcheck and synccheck on both devices. All sanitizer final summaries were
  zero-error; racecheck also had zero warnings/hazards. Query CTest passed.
- The verifier checked the exact fixture names and nonempty all-passed Rust
  summaries, including interleaved diagnostic output, not just exit codes.
- ping_pong/racecheck: three bounded fixtures,290.30s on GPU0; dense_device
  racecheck:943.61s on GPU0 and938.44s on GPU1. These are instrumentation times,
  not BFS performance measurements. The m2..m6 sweeps remain in other modes;
  bounded m2..m3 racecheck is not claimed to cover the larger graphs.
- Preserved artifacts: test_results/native-query-v9/native-primitive-gate/.
  Verification command: python scripts/verify_primitive_gate.py
  test_results/native-query-v9/native-primitive-gate --source
  df42c51df6d3b44b8ba620de6edcf0b3e0f68e5f.

This certifies the generation/hash C ABI implementation at that source commit,
not the later Rust report adapter or CUB route query. Those subsequently passed
the separate version10 gate on77bc9a1; see
[route allocation query](2026-09-03-route-allocation-query.md).
Production archived multi-rank BFS remains outside this
primitive gate. The CayleyPy baseline was rechecked clean at
f0f2b8e5ee61173039ab9742f3a7756c9b6365e6.
