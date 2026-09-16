# N-peer runtime: physical T4 regression

Kaggle `trydotatwo/mgbfs-library-capacity-t4`, version 12, completed PASS.
Evidence retrieved 2026-09-16. Source commit:
`fce7de137d2ffb121854759efcea330b7c25a803`.

Two distinct physical Tesla T4 devices were recorded in summary.json.
All fifteen BFS test logs (gpu0, gpu1, two-gpu; plain and four sanitizers)
report one test passed, zero failed. All twelve sanitizer logs report zero
errors; all three racecheck logs additionally report zero warnings/hazards.
The two-GPU suite reports one ignored test: the explicit eight-device test.
This gate does not validate an eight-GPU NCCL run.

Individual CLI artifacts were also inspected: twelve rank records COMPLETE and
twelve archive checksum/count verifications VERIFIED. The six two-process cases
cover cuCollections/cuDF each on S4 DENSE, S4 HASH_FIRST, and U4m2 DENSE.
This does not cover U4m2 HASH_FIRST or the native CUB CLI backend. Checksums/counts
alone are not a proof of full-state-set equality.
No performance screen was requested in this run.

Raw evidence, ignored by git:

- `test_results/library-capacity-v12/library-owner/summary.json`
- `test_results/library-capacity-v12/library-owner/bfs-*.log`
- `test_results/library-capacity-v12/mgbfs-library-capacity-t4.log`

Next hardware gate: eight physical devices, explicit ignored test, followed by
separate-process torchrun smoke and bounded performance measurement. Only after
those gates may a full LRX13 run be reported as ready.
