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
