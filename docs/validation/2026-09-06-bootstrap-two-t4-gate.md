# Bootstrap integration: real T4 gate v30

Kaggle `trydotatwo/mgbfs-distributed-sanitizer` v30 completed on two distinct
Tesla T4 GPUs. Source aebc057dd14fa2ace9c021ab6263c8f74839274d.

Linux bootstrap, handshake, connection and codec tests passed before CUDA
build. New reference rendezvous was exercised by 24 S4 process selections:
world sizes 1/2, DENSE and HASH_FIRST scalar/integer-MMA, CUB/BMMA,
pre-dedup OFF/ON. All reproduce literal layers [1,3,5,6,5,3,1].

Raw reconciliation covers 36 measured rank JSONs, 36 COMPLETE warmup JSONs,
36 VERIFIED archive logs and the source SHA. The full runtime suite reports
12 passed under plain and all four sanitizers. Macro nonidentity-source and
macro-archive fixtures each report 1 passed in all five modes. Five leaf
fixtures pass all four sanitizers. All 35 runtime/tool logs contain required
success markers and zero error/hazard summaries where applicable.

This validates setup integration and existing GPU paths, **not TCP-driven
exchange epochs**. The reference holds control sockets but its GPU dispatcher
does not use them yet. Continuous overlap and production CLI remain unfinished.
The later bucket-capacity fix 68e2a866e302125828f90d50df4bcebd5820dd79 is not
covered by this pin; its one-T4 v12 measurement is separate.

Evidence: `test_results/distributed-sanitizer-v30/distributed-sanitizer/` and
`test_results/distributed-sanitizer-v30-summary/distributed-sanitizer/summary.json`.
No state dataset downloaded; S13 was not recomputed.
