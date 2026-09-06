# Native single-source scatter: two-T4 gate

Kaggle `trydotatwo/mgbfs-distributed-sanitizer`, version 32, completed on source
`bf486f1ee2e7a04663b42f7943bc6b1f6f20c9f9`.

Two distinct Tesla T4 devices, 15,360 MiB each, were recorded in the run
manifest. Downloaded metadata/logs were reconciled against the recorded source
SHA and all per-rank results. No state payloads were downloaded locally.

## Verified scope

- Native scatter sends from rank 0 and rank 1, checks exact received device
  bytes, keeps the source's local segment as a view, and drains zero-byte
  epochs. Health polling and repeated terminal abort pass.
- This fixture passes plain, memcheck, racecheck, initcheck and synccheck.
  All sanitizer error summaries are zero; racecheck reports zero errors,
  warnings and hazards.
- The existing 12 full-runtime fixtures pass in all five modes, including
  DENSE/HASH_FIRST, compact generation5, archive validation and deliberate
  group-terminal capacity failure.
- Single-device macro and macro-archive fixtures pass in all five modes.
- All 24 reference profile selections pass at one/two ranks. Reconciled
  per-rank measured results, warmups and archive verifiers each total 36.
  Their global S4 layer counts are `[1, 3, 5, 6, 5, 3, 1]`.

Audit result: `RAW_GATE_RECONCILED`, 40 tool logs, 36 measured ranks,
36 warmup ranks, 36 verifiers. Local evidence:
`test_results/distributed-sanitizer-v32/distributed-sanitizer/`.

## Limits

This is correctness evidence, not a throughput or overlap measurement. The
scatter fixture supplies matching byte counts directly; it does not validate
distributed ticket admission or the complete CUDA-event-driven control pump.
The later explicit receive-capacity ABI guard and its device fixture
(`4f73acd` through `1c05b85`) are not part of this pinned run. Their CPU/CI
checks do not substitute for a subsequent hardware run. Published S13 was
neither recomputed nor uploaded again.
