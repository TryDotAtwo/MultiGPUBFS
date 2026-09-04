# Macro allocation-query T4 gate

Source commit under test: `076d6fe4f99181f1e17150d9196b70abe6bd5b32`.
Kaggle kernel: `trydotatwo/mgbfs-native-matrix-primitives-t4`, version 12.

Result: `PASS_PRIMITIVE_GATE` on two distinct Tesla T4 devices. Each device
reported 15,360 MiB total and 14,912 MiB free before the gate.

The gate built the CUDA/CUTLASS library from the immutable source commit and
passed the complete primitive inventory. In particular, the new
`mgbfs_materialize_query` and `mgbfs_future_merge_query` contracts matched the
actual create paths, and the `generate`, `future_merge`, `macro_settle`,
`materialize`, `hash`, `route`, `owner`, `pipeline`, `ping_pong` and
`dense_device` tests passed on both GPUs under:

- plain execution;
- Compute Sanitizer `memcheck`;
- Compute Sanitizer `racecheck`;
- Compute Sanitizer `initcheck`;
- Compute Sanitizer `synccheck`.

This proves the primitive and single-bucket feedback scope stated by the gate.
It does not prove production archived multi-rank BFS or NCCL correctness.
