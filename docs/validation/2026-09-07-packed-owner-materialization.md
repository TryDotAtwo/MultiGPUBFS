# Packed owner materialization: verified leaf, runtime gate pending

Source: `b9cd0acf4b3b6d14430e5bd057ec78e09003cd90`.
Kaggle: `trydotatwo/mgbfs-state-commit-t4`, version 7; package `f08e373`.

The DENSE reference owner now materializes its already-packed span directly.
`selected` is span-local and input is advanced by `source_begin * stride`.
The packed specialization does not load `identity_refs`; source-order and
HASH_FIRST paths retain their existing index mapping. The identity allocation
still exists for other runtime uses, so this change does **not** reduce the
reported allocation plan. No speedup has been measured.

Local validation: shared CPU/GPU index helper observed failing before
implementation, then `STATE_INDEX_PASS` under MSVC `/W4 /WX`; complete
`cargo test --locked` exited 0; 25 Python gate tests passed; CUDA-feature
library/native-scatter Rust type check passed. The benchmark example cannot
be checked on Windows because its archive constructor is Linux-only. Linux
CI for b9cd0ac and package f08e373 completed successfully.

## Actual hardware evidence

Two independent Tesla T4 GPUs, 15360 MiB each:

- `GPU-bcb1b727-c94c-09d6-3e40-d14d7c77856f`
- `GPU-285a69fb-d6db-140b-ee96-f86e2ba6f6ab`

Downloaded logs and summary were reconciled by exact source, GPU UUID,
binary/tool identity, completion marker and all sanitizer summaries:
**20/20** = two GPUs × state-commit/archive-pack × plain, memcheck,
racecheck, initcheck, synccheck. No reported sanitizer errors or race warnings.
Raw files remain under `test_results/state-commit-v7/state-commit-gate/`.

The state fixture tests valid packed selection from a nonzero source-span
offset, invalid selection with no partial state writes, the original indexed
path, ring reservation/reclamation, and existing tiny full-layer owner tests.
This is two independent single-device leaf suites, **not** multi-rank BFS.

## Still pending

Commit `47415a3` extends the real framed NCCL fixture: retain its consumer,
materialize directly from the validated payload view on another stream, wait
for its generation-bound event, then release the consumer. It supplies a
synthetic already-committed owner result; it does not implement owner compare
or the full dispatcher. This extension still needs its hardware gate.

The full reference runtime now calls the new packed entry point, but its
full-runtime GPU regression must run at this source or newer. The previous
distributed sanitizer v43 pins a7378e3 and cannot verify this change.
The production asynchronous owner dispatcher remains incomplete. S13 was
neither recalculated nor uploaded again.
