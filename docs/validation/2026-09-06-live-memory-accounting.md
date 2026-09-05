# Live memory accounting integration (partial)

The reference distributed runtime now allocates all 29 shared `Buffer` planes
using `distributed_memory::shared_buffers`, including real ABI sizes and
checked products. The constructor computes this ledger before creating streams
or allocating those buffers; each allocation consumes its named payload size.
The ledger uses 256-byte accounting alignment but does not claim CUDA driver
residency or allocator granularity. No buffer lifetime or BFS scheduling changes.

The owner C ABI now provides `mgbfs_bounded_owner_query`, and owner constructors
consume that same query for flags, indices, merged accepted output and optional
BMMA refinement errors. This differs from the generic architecture-v2
`bounded_owner_ledger`, which is not an account of this reference implementation.

Verified locally:

- MSVC C++17 `tests/owner_memory_query.cpp` with `cuda/bounded_owner_query.cpp`:
  literal shapes, invalid shapes, overflow, failed-output clearing pass.
- `cargo test --locked -p mgbfs-runtime --lib --tests`: CPU tests pass; CUDA
  fixtures compile out in this command and are not GPU evidence.
- `cargo test --locked -p mgbfs-cli`: four tests pass, still no CUDA dependency
  required for offline preflight/verify.
- `cargo check --locked -p mgbfs-runtime --features cuda --test distributed_archive`
  passes with the configured library search directory; this does not link or run CUDA.

Pending: GPU validation of the shared-buffer wiring; joining shared, library,
HASH_FIRST-only, NCCL, pinned archive and disk accounting into one preflight;
coordinated pre-allocation rejection on any rank; untouched VRAM reserve across
the complete runtime. Do not label the partial ledgers a complete rank budget.

Further integration: `owned_memory()` now combines the shared Buffer ledger
with actual generation/hash/route/owner queries and the HASH_FIRST profile
ledger, all before creating runtime streams or those allocations. HASH_FIRST
Buffer allocations now consume their named ledger entries too. CUDA/NCCL
internal residency and the separately constructed pinned archive remain outside
this explicit-device-allocation ledger. No resource-capacity admission is
claimed yet. The generation-query report adapter previously rejected supported
compact variant 5; a failing compact-query test reproduced that stale guard,
and the adapter now accepts 5 while rejecting 6. Five CUDA report adapter tests
and three shared-memory composition tests pass; GPU gate remains pending.

Admission integration: after communicator construction, a tiny temporary vote
buffer performs the actual warmup all-reduce. Free VRAM is then queried; every
rank votes on `explicit aligned device bytes + declared untouched reserve <=
free`. Any rejection returns `VRAM_PREFLIGHT_GROUP` on both ranks before large
runtime/library allocation. The benchmark and smoke defaults reserve 1 GiB.
The new asymmetric fixture requests an impossible reserve on rank 1 only and
requires both ranks to reject. Its CPU arithmetic tests pass and its CUDA Rust
code checks; real two-T4 execution is pending. Backend/query errors before
communicator establishment and unexpected CUDA/NCCL failures are not thereby
proved coordinated. Admission does not reserve resources against other GPU
processes or account for future driver/NCCL allocations; pinned/disk admission
and complete rank startup protocol remain separate unfinished requirements.

Archive startup now derives `ArchiveRingPlan` (slot bytes, total pinned payload,
descriptor count) with checked arithmetic and validates width before allocating
slots. Physical disk extent reservation/header initialization precedes pinned
host allocation, so ENOSPC is not discovered only after pinning the full ring.
Nine CPU archive tests pass, including the new ring geometry cases, and the
CUDA runtime Rust check passes. This does not query Kaggle's platform quota or
the OS pinned-memory allowance, and this archive-order change still needs its
next hardware gate.
