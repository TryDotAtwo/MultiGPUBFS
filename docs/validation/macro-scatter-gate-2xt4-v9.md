# Macro scatter control gate, physical 2xT4, Kaggle v9

- Private Kaggle kernel: `trydotatwo/mgbfs-weighted-scatter-gate-2xt4`, version 9, status `COMPLETE`.
- Git source: `edc738be1498fe2b1b1bd5dca881a88f80da5ceb`; CUTLASS: `ffa119a1255d78998536107466cc7097ecefa393`.
- Hardware inventory: two Tesla T4 GPUs, 15,360 MiB each.
- Rust fixture `admitted_adapter_native_scatter_and_depth_rollover`: plain PASS; Compute Sanitizer memcheck, racecheck, initcheck, synccheck PASS. Racecheck reported zero hazards/errors/warnings.
- CUDA compact future bucket fixture: plain PASS; the same four sanitizer modes PASS, with zero reported errors/hazards.
- Exact local downloaded output: `test_results/macro-scatter-gate-v9/macro-scatter-gate/summary.json` and associated Kaggle logs (ignored build/test artifacts).

Scope: the two-rank fixture checks NCCL scatter, rollover and the new macro-history control gate. It supplies an already committed owner row and therefore **does not** validate an end-to-end weighted BFS, an exhaustive matrix Cayley graph, durable archival, or a performance claim. The future bucket test is a separate leaf. Next gate is wiring real CUDA completion events to settlement, reader and archive-copy notifications in the production scheduler, followed by a full weighted graph oracle.
