# Batched DENSE owner: full-runtime T4 regression verified

Source `66d82d03cb055daa08dae328978208efda7c8ede`.
Kaggle `trydotatwo/mgbfs-distributed-sanitizer`, v44; package `032fcf5`.

Physical Tesla T4 devices, 15360 MiB each:

- `GPU-d24b66e1-cf39-d6dc-1b09-d2d91e2ce64a`
- `GPU-19303d9f-499a-8556-b3d6-ed6ccb7ee541`

All logs/JSON were downloaded without state archives and reconciled against
the exact source and suite expectations. Result: **40 tool logs, 36 measured
rank records, 36 warmup rank records, 36 archive verifiers**. All 24 reference
profile selections completed with `[1,3,5,6,5,3,1]` global layers. Plain plus
memcheck/racecheck/initcheck/synccheck completed with clean summaries.

The full-runtime fixture has 12 passing tests per mode, including compact
generation5 S4 archives, DENSE/HASH_FIRST full-state oracle comparisons,
owner-map/seed/pre-dedup variations and group-terminal capacity failure.
The capacity test now also exercises DENSE's queued jobs: sticky ring fatal
prevents result publication, both ranks become terminal, and archives do not
acquire a successful RunCommit. The three native transport fixtures pass per
mode, including framed payload materialization on a separate stream with the
consumer held through its completion event.

Raw files: `test_results/distributed-sanitizer-v44/distributed-sanitizer/`.
Reconciliation command:

```text
python test_results/audit_sanitizer_v30.py test_results/distributed-sanitizer-v44/distributed-sanitizer test_results/distributed-sanitizer-v44/distributed-sanitizer/summary.json 66d82d03cb055daa08dae328978208efda7c8ede 3
```

This verifies the current reference runtime changes, not a completed admitted
asynchronous BFS dispatcher. The transport fixture supplies an already
committed owner result; its success does not prove full owner integration.
No performance improvement is claimed from sanitizer timings.

## Fresh S11 profile panels launched

Package `230ae90`; both pin the verified source above and CayleyPy baseline
`f0f2b8e5ee61173039ab9742f3a7756c9b6365e6`:

- `trydotatwo/mgbfs-distributed-bench`, v13: one active T4.
- `trydotatwo/mgbfs-distributed-bench-s11`, v5: two T4, equal-global capacity.

Both were confirmed RUNNING on 2026-09-08. Existing five-repeat profile
panels and baseline batch sweeps are unchanged. Native archive creation and
verification remain mandatory; baseline does not create archives. Only
metadata/logs will be downloaded locally. Results and performance comparisons
are pending. S13 was not rerun or reuploaded.
