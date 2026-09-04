# Native S8 streaming publication gate (2026-09-04)

Kaggle notebook `trydotatwo/mgbfs-hugging-face-publisher`, version 21,
completed on a physical Tesla T4 and published a native exhaustive S8 archive
to Hugging Face without retaining the Parquet payload on the workstation.

- Status: `PASS`
- Source commit used by the run: `3d5d287ee0b7292bef170ed3405ca3566f0d6132`
- Run id: `s8-native-stream-20260904-183411`
- Dataset: <https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results>
- Atomic dataset commit: `c5bb47a45d32278fc79e298aff56e5d8d6619205`
- Unique states: 40,320
- Layers: 29 (`0..28`)
- State Parquet shards: 5
- Native search completion: 0.061579825 s
- Durable native run commit: 1.234658341 s
- Requested device allocation: 121,863,020 bytes
- Observed CUDA peak: 246,284,288 bytes
- Preallocated pinned archive ring: 20,971,520 bytes (64 x 4,096 rows)
- Preallocated Parquet staging ring: 8 x 16 MiB; peak five live slots
- Native disk reservation: 70,334,464 bytes

The notebook verified the final default-branch file set through the authenticated
Hugging Face API before deleting the staging branch.  It required the run,
layer, and verification metadata plus all five final state shards.  The archive
chain digest is
`0f2dca01d0666795584efe384e5b338c00b494640bc38d7876f7eeb794895f98`.

This proves the bounded S8 streaming and atomic-promotion path.  It does not
prove that the same fixed rings can absorb an S13 producer burst, nor that the
network sink sustains production BFS bandwidth.  Larger runs still require a
preflighted host/disk staging-capacity plan; ring exhaustion remains fatal and
never silently introduces backpressure or drops states.

Downloaded evidence is under `artifacts/kaggle/hf-stream-v21/hf-stream/` and
contains logs and metadata only, not the Parquet state payload.
