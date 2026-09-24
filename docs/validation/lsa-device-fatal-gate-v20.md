# LSA device-side pre-owner fatal gate: two-T4 correctness

- Source: `2b2e930e8ff974caceabe6bd57b68fa8fe63cc77`.
- Kaggle: `trydotatwo/mgbfs-lsa-full-bfs-gate-t4`, version 20, `rounds_gate`.
- Hardware: two Tesla T4, peer access enabled in both directions.
- Result: `COMPLETE`; CUCO_RANK+DENSE+LSA full layer sets and rank archives
  matched the CPU oracle. The HostSized two-rank and CUCO_RANK control fixtures
  also passed. Machine-readable report:
  `test_results/kaggle_lsa_device_gate_rounds_v20/lsa-bfs-gate/summary.json`.
- Earlier device-fatal leaf gate passed on the same source family in Kaggle v19;
  v20 validates the integrated no-fault path, not injected runtime failure.

This does **not** establish a CPU-free owner/transport/retirement pipeline,
sanitizer cleanliness, performance improvement, or correctness under an
injected one-rank runtime failure. The remaining host-sized and HASH_FIRST
paths are not made device-driven by this gate.
