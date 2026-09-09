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
