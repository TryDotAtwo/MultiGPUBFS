# BFS tail archive integration contract

Branch: `codex/bfs-tail-archive`. The active BFS checkout is not modified.

`scripts/bfs_tail_archive.py` implements SSD retention and publication ordering.
It is not yet wired into the CUDA/NCCL producer. It accepts already packed
bytes from all owners, after the global completion vote. Do not feed candidate
states, a partially completed rank, or an unverified future frontier.

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

`publish_snapshot` accepts a synchronous acknowledged upload callback; invoke it
in a background worker using the existing HF credentials. It checks checksums,
uploads payload files, then publishes the manifest. The integration must pin or
copy a snapshot generation before enqueueing and retain it until upload receipt;
otherwise the next snapshot can delete the worker's input. Publication paths
must include a unique run prefix. Remote obsolete files may remain but must not
be referenced by the current manifest. No HF upload has been performed here.

Remaining integration gates:

- Wire globally completed packed state chunks without retaining extra VRAM.
- Add independent per-GPU VRAM sampling and carry the samples to layer records.
- Add bounded background uploads with generation pinning and failure reporting.
- Exercise cancellation and disk/upload failures without upgrading INCOMPLETE.
- Compare archive on/off on (15,4), with identical configuration and hardware.

Validation: `python -m unittest discover -s tests -p test_bfs_tail_archive.py`.
