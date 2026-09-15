# CUDA CLI reference gate: raw acceptance

Kaggle `trydotatwo/mgbfs-distributed-sanitizer`, version 46.
Runtime source: `676b842c6ab540a23fd63b7b4c2b181252c81ee6`.
Raw artifact reconciliation completed on 2026-09-15, after missing metadata
downloads were recovered. No graph datasets were downloaded locally.

The raw auditor returned `RAW_GATE_RECONCILED`: 40 tool logs, 36 measured
rank records, 36 warmup rank records and 36 archive verifier outputs.
All 24 profile selections matched their requested profile, owner, generation
and pre-dedup settings; their global S4 layers were `[1,3,5,6,5,3,1]`.
The pinned source SHA matched the downloaded source-sha log.

Plain execution and memcheck/racecheck/initcheck/synccheck each passed the
12-test distributed archive fixture and the three-test native scatter fixture.
The raw macro and native primitive logs also passed their strict test-count,
PASS-marker and sanitizer-summary checks. Linux CUDA CLI contracts passed
three tests, including rejection of archive disabling and implicit reference
selection.

Local evidence roots:

- `test_results/distributed-sanitizer-v46/distributed-sanitizer`
- `test_results/distributed-sanitizer-v46-summary/distributed-sanitizer`

This is correctness evidence for the reference CLI runtime only. It is not
production-dispatcher completion, performance evidence or validation of the
later DENSE lookahead commit `013ed5c`. That commit has its own independent
Kaggle gate and adds a thirteenth distributed archive test.
