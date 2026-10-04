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

## Actual owner DAG capture

Healthy CLI runs now enable `MGBFS_TEST_OWNER_DAG_CAPTURE` and require
multiple successful graph launches across batches, followed by the same
full-state/hash/archive oracle checks. The twelve-case gate passed in
11.42 seconds. `commit_rank_library_batch` captures candidate conversion,
CUCO compare, shard counts, StateRing reservation, persistent commit and
either state materialization/extent publication or HASH_FIRST request creation.
This is actual graph execution, not the older drained-completion leaf test.
Transport, HASH_FIRST responses and retirement are outside this capture scope;
capture does not replace the real multi-rank timeline requirement.
The fixture is debug-only because its injection/probe hooks are debug-only.

## Native CUB rank-owner coverage

The same gate now runs both CUCO_RANK and CUB_SORT_MERGE, each with DENSE
and HASH_FIRST, healthy captured owner DAG execution and five failure points.
All 24 independent CLI processes passed in 22.12 seconds on the local GPU.
The CUB configuration deliberately omits the library pool: an initial attempt
was correctly rejected as REFERENCE_UNUSED_LIBRARY_POOL before search.
Healthy outputs for all four combinations match full CPU-oracle states and
seeded hashes. Failure cases require the precise injected error and absence
of rank/group COMPLETE. This does not validate BMMA, multi-rank execution,
sm75 or paired performance.

## Independent-rank runner deadline

The existing `scripts/replay_lsa_cancel_candidate.py` remains the two-process
acceptance runner, not this world1 CLI fixture. Its former fixed 120-second
instrumented deadline was shorter than the observed 369.90-second local
racecheck. The runner now defaults racecheck cases to 600 seconds; ordinary
fault cases remain at 45 seconds and other instrumented cases at 120 seconds.
An explicit 1..3600-second override is validated before launching and recorded
globally and per case. Forced cleanup remains failure, never sanitizer success.
The deadline contract failed before implementation and all 23 replay unit
tests passed after it. These CPU tests do not establish rank cleanup on Linux,
GPU sanitizer success or hardware readiness. Rental stop deadlines are separate
and must cover any authorized instrumented run; this change provisions nothing.
