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

## Native CLI/FIFO gate

`cargo test -p mgbfs-cli --features library-owner --test fifo_native_gpu --
--nocapture --test-threads=1` passed on the same RTX 3070 Laptop (6.82 seconds).
It launches eight independent CLI processes: DENSE/HASH_FIRST each with a
healthy run and archive-admission, owner-host and archive-finish failures.
Healthy U4/F2 streams match every full CPU-oracle layer and every seeded hash,
and their rank results report zero disk reservation and u64::MAX wire bound.
The group marker explicitly declares `fifo_flush`, not durable HF publication.
All six injected failures exit before the 60-second per-process deadline and
leave neither a rank COMPLETE result nor a group COMPLETE marker.

The first owner-fault run exposed a test hook which only targeted a remote
owner job: it was silently unused for world=1. The debug-only hook now targets
the local first job for world=1; the existing multi-rank remote placement is
unchanged. The extended gate failed before this correction and passed after it.

Not established: end-to-end HF publication on new code, two-rank
transport/fault gates, sm75 acceptance or a performance improvement. Existing
typed production file-extent admission is unchanged.

## Worker failure extension

The same CLI gate additionally injects archive worker write and sync errors
for both profiles. All twelve subprocess cases passed in 11.49 seconds on
the local GPU. Each failure checks its exact injected error string rather
than accepting any generic failure; rank and group COMPLETE must be absent.
This exercises actual asynchronous worker failure reporting, but still does
not establish asymmetric cancellation between two independent ranks.
