# Main branch tail archive port

Scope: standalone archive implementation and its five existing behavioral
tests selectively imported from `codex/bfs-tail-archive` at `210c1dd`.
The distributed runtime was not replaced with the older branch version.

Files: `scripts/bfs_tail_archive.py`, `tests/test_bfs_tail_archive.py`.

Before import, the test collection rejected the missing module. After import,
all five tests passed. Independent rerun of the project Python suite with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests scripts -x -q
-p no:cacheprovider` completed: 254 passed, 14 skipped, 69.57 seconds.
An earlier non-fail-fast suite run was interrupted after showing a failure;
the rerun did not reproduce it. This does not diagnose that earlier failure.

This module is not yet connected to the main runtime/export launcher. These
tests do not prove archive/HF production integration or any GPU acceptance.
Existing unrelated changes remain untouched.

Follow-up port adds `scripts/tail_wire.py` and `tests/test_tail_wire.py`:
the bounded MGBFSAR1 reader validates checksums/order before forwarding a
completed layer. Native stream emission already exists in the main runtime.
An integration fixture forwards the checksummed wire into TailArchive,
verifies literal packed output and consumes the temporary rank spool.
Ten focused tests passed; the full scoped Python suite then passed 259 tests,
14 skipped, in 14.54 seconds. This is CPU wire/tail integration, not a native
GPU producer/consumer run or proof of HF publication.

Current runtime inspection confirms the LSA rank-owner device fatal votes,
bounded epoch credits and owner-consumed event are present. This is source
evidence, not proof of two-rank failure handling or measured overlap.

The local GPU retirement regression could not start: Docker's Linux engine
named pipe was absent. Docker Desktop was started; its processes appeared,
but the engine was still unavailable at subsequent checks. No cloud rental
or competing Kaggle run was started. Windows CPU runtime tests were launched
separately: `cargo test -p mgbfs-runtime --lib` passed all 12 tests, including
archive submission failure ordering, constructor failure publication and
post-search archive-failure escalation. This CPU-only feature selection does
not compile or execute the CUDA distributed path.
