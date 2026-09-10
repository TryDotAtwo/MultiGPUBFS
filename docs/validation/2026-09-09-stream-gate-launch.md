# Stream-separated runtime: physical gate launched

Kaggle accepted `trydotatwo/mgbfs-distributed-sanitizer` version 45 on
2026-09-09; a subsequent authoritative status query returned RUNNING.
The package pins `b3c191a5c32654a779e325d4b3bbe8e2f8226362`.
Python AST and kernel metadata parsing passed before submission.
This is pending hardware evidence, not a sanitizer pass or speedup claim.

Both paired S11 benchmark kernels were independently observed COMPLETE:
`mgbfs-distributed-bench` (prepared v13, one rank) and
`mgbfs-distributed-bench-s11` (prepared v5, two ranks).
The one-rank summary and environment were downloaded and identify source
`66d82d03cb055daa08dae328978208efda7c8ede`, baseline
`f0f2b8e5ee61173039ab9742f3a7756c9b6365e6`, and summary COMPLETE.
The two-rank metadata and notebook-log download also completed. Neither benchmark kernel
has been overwritten. Full repeat-level reconciliation remains required.
Only metadata and notebook logs are requested, not graph archives.

Next: finish both result downloads and reconcile configurations, repeats,
layer counts, archive verifier outcomes and memory observations; collect v45
terminal sanitizer artifacts before accepting the stream change.
Do not repeat the published S13 run.

## Terminal observation

The authoritative Kaggle status subsequently changed to COMPLETE. Downloaded
v45 summary pins the expected b3c191a source and reports COMPLETE, with PASS
for plain, memcheck, racecheck, initcheck and synccheck. Raw fixture/rank/
archive logs are downloading to `test_results/distributed-sanitizer-v45`.
Until the strict auditor reconciles those artifacts, this is a terminal
summary observation only, not final acceptance of the stream change.

## Raw reconciliation completed 2026-09-10

After the interrupted download processes were confirmed absent, missing
sort-origins logs were retrieved without rerunning the GPU workload.
The strict auditor completed successfully against source b3c191a:
40 tool logs, 36 measured ranks, 36 warmup ranks and 36 archive verifiers.
All 24 profile selections reproduce global layers `[1,3,5,6,5,3,1]`.
This covers the compact full-runtime correctness/sanitizer gate for the
separate exchange stream, including 12 distributed archive tests and three
native scatter tests per tool mode. It does not establish performance gain,
actual timeline overlap, large-graph acceptance or production dispatcher
completion. The S11 tables still measure the earlier 66d82d0 source.
