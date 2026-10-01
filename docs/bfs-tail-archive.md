# BFS tail archive integration contract

Branch: `codex/bfs-tail-archive`. The active BFS checkout is not modified.

`scripts/bfs_tail_archive.py` implements SSD retention and publication ordering.
`scripts/run_tail_bfs.py` wires it to the existing CUDA/NCCL archive FIFO and
the existing global layer completion boundary. `tail_wire.py` validates the
native archive chain, removes routing hashes and packs words on CPU. A layer
is admitted only after every rank has both a checked archive layer commit
and the existing completed-advance marker. Candidate states are not admitted.

Packing uses little-endian words: symbol i occupies bits 4*i; padding is zero.
Width is 8 bytes for n<=16 and 16 for 17<=n<=32, provided symbols fit [0,15].
Larger alphabets are rejected; do not silently truncate symbols.

Retention uses decimal GB. The local tail retains at least three entire layers
and at least 10 GB, or all layers back to depth zero. Each layer updates counts,
whole-layer wall duration, per-GPU observed VRAM peaks, and an INCOMPLETE snapshot
of at most 1 GB. Its oldest selected layer may be partial; offsets are explicit.
COMPLETE snapshots contain only whole layers. All layer statistics survive tail
eviction. `COMPLETE` must be requested only after graph exhaustion; archive
success alone cannot establish search completion.

The producer must supply the exact start, L/R/X definitions, program commit and
full launch configuration. Time is measured at its existing completion boundary.
VRAM peaks must come from a separate monitor whose interval is recorded; they
are sampled peaks rather than guaranteed physical maxima. No batch-level CUDA
synchronization is introduced by this module.

`tail_upload.Publisher` is one background worker with a 25 GB pending-byte
limit. It pins files with hard links immediately when enqueued, verifies
checksums, uploads payloads, and publishes the manifest last. Failed uploads
retain pinned inputs and propagate an error. Run paths are
`tail-runs/<run_id>/...`; use a fresh run_id. Remote obsolete files may remain
but are not referenced by the latest manifest. Live HF publication was validated
on 2026-10-02 for COMPLETE (9,4) and an INCOMPLETE capacity failure; see the
validation report for remote manifests and checksum readback evidence.

The separate nvidia-smi monitor requests samples every 50 ms. The manifest
uses host receipt timestamps and the earliest rank BEGIN/latest rank END
window. A short layer with no observation records a null peak explicitly;
it does not invent an exact physical maximum. Raw CUDA kernel/duplicate/path
metrics are not part of the tail manifest.

Remaining integration gates:

- Cancellation still needs a dedicated run; native capacity failure retained
  completed layers 0..3 in the isolated two-GPU validation.
- Physical SSD stress wrote 13 decimal GB, retained four full layers totaling
  10.4 GB, evicted the oldest layer and verified retained SHA-256 by readback.
  The INCOMPLETE suffix was exactly 1 GB and its checksum also matched.
  This synthetic payload check is not a >=10 GB real BFS/GPU result.
- Compare archive on/off on (15,4), with identical configuration and hardware.

Validation: `python -m unittest discover -s tests -p 'test_*tail*.py'`.
GPU full-word oracle: `lrx_multiset_two_rank_cuco_full_state_oracle`.
Archived full-word oracle: `python scripts/verify_tail_oracle.py <saved-root>`.
Matched benchmark: `scripts/tail_remote_panel.py` (isolated rental paths).

Finite grid driver: `scripts/sweep_tail_bfs.py`, where `m` maps to native `r`.
Pass explicit `--n-min`, `--n-max`, optional `--m-min`/`--m-max`, a positive
`--deadline-seconds`, and the same config/source/runtime-env arguments as the
single-run driver. `--plan-only` prints the grid without launching GPUs.
Each pair gets its own run directory and HF run ID. `sweep.json` records
COMPLETE/INCOMPLETE, last completed layer and reason per attempted pair,
unsupported pairs, and the remaining unstarted pairs. Restarting with the
identical configuration resumes pending pairs without overwriting earlier
runs. The alphabet must fit four bits and the native orbit count must fit u64;
unsupported pairs are explicitly recorded, never claimed as completed.
The grid driver has CPU tests; a complete GPU grid has not yet been executed.

Storage stress: `python scripts/stress_tail_archive.py <fresh-directory>`;
requires at least 18 decimal GB free. Recorded physical validation is in
`docs/validation/2026-10-02-physical-tail-stress.json`. Payloads are explicitly
synthetic, not claimed as reachable graph layers or search-completion proof.

For an existing Windows DPAPI-encrypted token, the saved-run publisher accepts
`--token-dpapi <path>`; the credential requires the original Windows account.
Keep this file outside Git. Linux GPU runs use the standard HF_TOKEN secret
environment or HF login cache. On this Windows host, set HF_HUB_DISABLE_XET=1
before invoking the publisher to bypass a local Xet cache access failure.
