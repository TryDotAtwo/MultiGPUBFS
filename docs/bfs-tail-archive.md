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
but are not referenced by the latest manifest. Live HF publication remains
unverified because no local write credential was available.

The separate nvidia-smi monitor requests samples every 50 ms. The manifest
uses host receipt timestamps and the earliest rank BEGIN/latest rank END
window. A short layer with no observation records a null peak explicitly;
it does not invent an exact physical maximum. Raw CUDA kernel/duplicate/path
metrics are not part of the tail manifest.

Remaining integration gates:

- Complete live HF upload/receipt validation with a scoped write credential.
- Exercise native capacity/cancellation failure and retain completed layers.
- Large (>=10 GB) physical archive stress remains untested; scaled retention
  thresholds and complete small real GPU archives are validated separately.
- Compare archive on/off on (15,4), with identical configuration and hardware.

Validation: `python -m unittest discover -s tests -p 'test_*tail*.py'`.
GPU full-word oracle: `lrx_multiset_two_rank_cuco_full_state_oracle`.
Archived full-word oracle: `python scripts/verify_tail_oracle.py <saved-root>`.
Matched benchmark: `scripts/tail_remote_panel.py` (isolated rental paths).
