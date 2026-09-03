# Macro settlement: physical 2xT4 gate

- Kaggle kernel: `trydotatwo/mgbfs-macro-settle-t4`, version 3
- source: `bf035c0c950894632a34d5f7f9418e49753ef354`
- CUTLASS: `ffa119a1255d78998536107466cc7097ecefa393`
- GPU 0: Tesla T4, `GPU-887c3943-2f2d-280a-2ccd-60ff8bc93777`
- GPU 1: Tesla T4, `GPU-0ab8b2dc-d9bc-a341-6ac3-12a6725f7db5`

Both physical devices independently passed the Rust/CUDA macro-settlement
contract under plain execution and Compute Sanitizer `memcheck`, `racecheck`,
`initcheck`, and `synccheck`: 10/10 checks passed.

The test covers strict monotonic epochs, sticky fatal state, capacity failure,
sorted-run validation, filtering against all `2K` history runs, same-future
deduplication, dense survivor compaction, and preservation of `StateRef`.
