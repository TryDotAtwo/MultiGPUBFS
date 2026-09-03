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
