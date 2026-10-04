# FIFO output limit regression

The reference unit-cost and macro-depth dispatchers now distinguish the
sequential FIFO wire-offset bound from a physically reserved file extent.
FIFO mode uses the checked u64 offset limit; its bounded external consumer
remains responsible for staging capacity. File mode retains the full-orbit
extent calculation and overflow rejection. Result records expose both
`archive_wire_limit_bytes` and `disk_reserved_bytes` (zero for FIFO).

Evidence on 2026-10-04:

- Regression initially failed to compile because the output-limit contract
  did not exist; after implementation the archive suite passed 13 tests.
- Invalid width, zero state bound and zero capacity remain rejected in both
  modes; a u64::MAX orbit can be streamed but cannot reserve a file extent.
- Linux CUDA/library-owner test binary rebuilt successfully. Initial link
  attempt lacked the existing RAPIDS dependency paths; retry with the full
  LD_LIBRARY_PATH succeeded. No dependency or runtime workaround was added.
- RTX 3070 Laptop: both `cuco_lsa_route_bank*` fixtures passed (10.20 seconds),
  covering banks 2/3/4, DENSE/HASH_FIRST, full-state oracle and archived payload,
  plus capacity failure/teardown. This is a world=1 check, not multi-rank proof.

Not established: end-to-end FIFO consumer/HF publication on new code, two-rank
transport/fault gates, sm75 acceptance or a performance improvement. Existing
typed production file-extent admission is unchanged.
