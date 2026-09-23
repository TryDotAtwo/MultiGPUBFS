# Weighted DENSE scatter, 2×T4

- Kaggle notebook: `trydotatwo/mgbfs-weighted-scatter-gate-2xt4`, version 1, private.
- Pinned source: `2facb2c79af6641433ea565f9160e0d302e3bc11`.
- Fixture: `admitted_adapter_native_scatter_and_depth_rollover`.
- Hardware inventory: two Tesla T4, 15360 MiB each.
- Plain torchrun: PASS. Compute Sanitizer memcheck, racecheck, initcheck,
  synccheck: PASS, zero errors in each log.
- Evidence: Kaggle output `macro-scatter-gate/summary.json` and corresponding
  `scatter-*.log` files, downloaded locally under
  `target/macro-scatter-gate-v1/` (ignored build output).
- Scope: weighted frame packing, rank scatter, received read lease, and
  metadata/state inspection. Does not establish full distributed macro BFS,
  owner settlement, throughput, or memory advantage.
