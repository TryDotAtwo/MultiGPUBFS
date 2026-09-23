# BFS completion ledger

This file tracks implemented contracts separately from hardware evidence and full-run acceptance. The original requirements remain in `ARCHITECTURE_NEED.md`; library and database alternatives are tracked separately in `db-framework-stage.md` and `db-interface-audit.md`.

| Area | Current evidence | Remaining gate |
|---|---|---|
| Single-depth CPU control and capacity contracts | Runtime/core/CLI CPU tests pass; fixed-budget admission, fail-fast, and archive contracts have tests. | Re-run the full CUDA workspace suite on target hardware after each integration. |
| Three-layer macro history | `MacroHistoryWindow` has depth-generation, reader and archive-lease rules. `AdmittedBuffers` now withholds `FinalizeDepth` ACK until the next history layer is settled and its replacement slot is free, then publishes on the admitted command. CPU integration test passes. | Bind settlement and archive-copy notifications to actual CUDA completion events in the production scheduler. Current physical two-rank fixture exercises the control boundary but not a complete weighted BFS. |
| Bounded owner/macro GPU primitives | Separate 2xT4 CUB/BMMA leaf checks and four Compute Sanitizer modes passed at `2300f5c`; see `docs/validation/bounded-owner-compact-layout-2xt4.md`. | Integrate into a full weighted owner pipeline; repeat correctness and sanitizers after integration. |
| Profiles and owner backends | DENSE/HASH_FIRST, local pre-dedup and CUB/cuCollections/BMMA components exist with partial tests and benchmarks. | Full parity matrix on real 2xT4 and larger graphs, no unsupported performance generalization. |
| Macro depth | Weighted generation/settlement and compact future-bucket leaf exist. | End-to-end weighted scheduler, exact layer reconstruction and exhaustive graph oracle. |
| Streaming archive/HF catalog | Existing pinned archive and Parquet/HF tooling are separate deliverables. | Demonstrate full durable output and resumable publication at target graph scale; search-only counts are not full graph archives. |
| Ready-made GPU DB/framework stage | Component/API analysis and bounded cuCollections comparisons are recorded in `db-framework-stage.md` and `db-interface-audit.md`. | Device-driven cuCollections owner, closed-loop Sirius/HeavyDB probes if warranted, fair end-to-end A/B with bounded VRAM. |

The native two-rank macro control gate is running on Kaggle version 9 from source `edc738be1498fe2b1b1bd5dca881a88f80da5ceb`; result is **pending**, not proof of passage. The CUDA suite cannot run on this Windows checkout without a built `MULTIGPUBFS_CUDA_LIB_DIR`/CUDA runtime. Do not treat leaf sanitizer checks as end-to-end validation.
