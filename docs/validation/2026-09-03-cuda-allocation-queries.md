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
- Local Rust regression could not run: Docker Desktop failed during startup
  on inaccessible sailor-ingest.sock; native cargo dependency download failed
  with Schannel SEC_E_NO_CREDENTIALS. No Docker reset, WSL shutdown or network
  reconfiguration was performed. Linux CI and target tests are required.

The shared C++ geometry fixture is added to CPU CI. CMake also builds a public
query executable and registers a CTest test; the private Kaggle primitive gate
runs it before CUDA test executables. No production multi-rank result is claimed.
