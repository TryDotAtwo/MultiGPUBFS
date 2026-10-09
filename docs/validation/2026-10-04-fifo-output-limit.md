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

## Replay cleanup isolation

The runner formerly signalled and waited on rank0 before touching rank1;
a reaping timeout or exit/signal race could skip the second rank entirely.
Cleanup now signals all live owned sessions first, attempts bounded reaping
for every rank, tolerates ProcessLookupError and records any remaining errors
as CANDIDATE_CLEANUP_FAILED. Forced cleanup is still not acceptance.
All 26 replay CPU tests passed before the Linux-only addition; four cleanup
tests then passed in Linux Docker, including two real independent sessions
whose PIDs were confirmed absent after reaping. No CUDA/NCCL cancellation
claim follows from this OS-process test. Runtime cancellation is unchanged.

## Local pre-dedup OFF/ON

The CLI gate now covers both pre-dedup settings for both owners and both
profiles: 48 independent subprocesses passed in 45.80 seconds on sm86.
U4/F2 deliberately has duplicate inverse generators; OFF therefore exercises
owner-side duplicate removal instead of relying on source-side suppression.
All eight healthy capture/FIFO runs match the full-state/hash oracle; forty
injected failure runs retain the same bounded-exit/no-COMPLETE checks. This
does not extend the source pin of the already running Kaggle v88, nor replace
its two-rank acceptance. It is not a speed comparison.

## Explicit BMMA hardware gate and current regression results

The existing CLI/FIFO fixture has a separate, explicitly ignored sm75-only
`bmma_cli_fifo_preserves_full_layers_and_wire_budget` entrypoint. It uses the
same real CLI, full-state/hash archive verification, owner DAG capture and
five failure cases as the CUCO/CUB entrypoint, with both profiles and both
pre-dedup settings (24 independent processes). Run it explicitly on physical
sm75 with `cargo test -p mgbfs-cli --features cuda,library-owner --test
fifo_native_gpu bmma_cli_fifo -- --ignored --nocapture --test-threads=1`.
It has **not** passed on T4 in this change.

An initial unconditional BMMA attempt on the local RTX3070Laptop failed
before search with `NATIVE_PLAN_CREATE_FAILED status=3`. The backend's
existing admission policy deliberately accepts only sm75; no architecture
check was weakened and no CUB fallback was added. The BMMA entrypoint is
therefore separated rather than falsely counted as a local success.

After extraction, the CUCO/CUB gate passed all 48 independent CLI processes
in 45.62s; the test report has one passed entrypoint and one ignored BMMA
entrypoint. `cargo test --workspace --exclude multigpubfs-gpu` also returned
zero (CPU-only workspace; GPU fixtures do not execute in that command).
Linux Docker ran the replay deadline, cleanup and protocol-supervisor suites:
17 tests passed in 2.830s, including the real independent-process checks.

The separate full `library_bfs_gpu` invocation returned failure: four tests
passed, but `library_bfs_layers_match_full_state_oracle_in_both_profiles`
stopped at `HASH_FIRST_TC_DEVICE_UNSUPPORTED` on sm86. Its mixed fixture
does not complete all indexed/cuDF/Tensor combinations on this host. This is
not a green full CUDA suite and does not close target-sm75 correctness.

Kaggle v88 remains the sole active notebook, pinned to 582c065. Its live
viewer confirms the exact source, two Tesla T4s and successful native/library
builds after a roughly 14-minute NCCL build. Runtime test acceptance is still
pending; the current test-only changes are not in v88's pinned source.
