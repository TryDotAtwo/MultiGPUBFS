# Macro history event gate, physical 2xT4, Kaggle v10

- Private Kaggle kernel: `trydotatwo/mgbfs-weighted-scatter-gate-2xt4`, version 10, status `COMPLETE`.
- Source: `0efee986ac59fd195ef45e5e656d782cec3239b9`; CUTLASS: `ffa119a1255d78998536107466cc7097ecefa393`.
- Hardware: two Tesla T4 GPUs, 15,360 MiB each.
- `admitted_adapter_native_scatter_and_depth_rollover`: plain, memcheck, racecheck, initcheck and synccheck all PASS. Settlement completion is polled through a real owner-stream CUDA event before `FinalizeDepth` ACK.
- Compact future-bucket CUDA leaf: the same five modes PASS.
- Downloaded output: ignored local `test_results/macro-scatter-gate-v10/macro-scatter-gate/summary.json` and Kaggle logs.

This proves the test fixture's event-to-control transition and sanitizer cleanliness, **not** a full weighted multi-GPU BFS. The production weighted scheduler is not yet connected to these callbacks; archive and reader events still need production ownership and full-graph correctness gates.
