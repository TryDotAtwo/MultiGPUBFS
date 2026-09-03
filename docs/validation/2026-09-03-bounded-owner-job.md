# Bounded owner job admission (CPU contract, not a GPU gate)

Implemented the 64-byte, 64-byte-aligned BucketJob descriptor specified by
matrix-runtime-architecture-v2, with matching Rust and C/C++ declarations.
Offsets are independently asserted in both languages: 0, 4, 8, 24, 40, 56, 60.

`mgbfs_core::owner_job::admit` checks a caller-owned descriptor slice without
allocating or modifying it. Each job contains one shard, one lane and one
generation. Incoming ranges are nonempty, packed from zero, ordered by distinct
bucket ID, and bounded by I. Descriptor count is bounded by J. Each prev, curr
and accepted bucket count is bounded by K; prev/curr ranges are checked against
their arena lengths with overflow detection. Incoming count may exceed K:
deduplication can still produce a valid committed bucket. Admission returns
live rows rather than the lane's capacity.

## Evidence

- RED: the initial stub failed three behavioral tests (one ABI test passed).
- GREEN: seven owner-job tests pass, including invalid topology, stale
  generation, wrong shard/lane, overlapping/gapped input, arithmetic overflow,
  out-of-arena ranges and capacity rejection.
- `cargo test --locked --offline`: 62 CPU tests passed on Windows. Linux-only
  file archive test is not part of this local count.
- `cargo fmt -p mgbfs-core -p mgbfs-runtime -p mgbfs-cuda -- --check`: passed.
- C++17 descriptor assertions compiled and ran with MSVC `/W4 /WX`.
- GitHub CPU workflow now also builds this fixture with GCC C++17.

## Remaining boundary

This is a host-side admission contract, not the CUDA owner implementation or
its scheduler integration. Caller must own the shard's exclusive writer lease
and refresh accepted counts after each commit. Admission does not inspect hash
sortedness, authenticate directory ranges, acquire a lease, or reserve commit
credits. The GPU compare/compact/reserve/merge-copyback path, bounded job
splitting and production multi-rank scheduling remain unimplemented. No new
GPU correctness, sanitizer or performance result is claimed by this change.
The old global-owner CUDA prototype is unchanged.
