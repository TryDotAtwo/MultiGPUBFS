# CUCO_RANK full-BFS sanitizer panel

Inspected HEAD c9f6428; runtime unchanged since 12dee48. Direct invocation of
the real mgbfs CLI on one RTX 3070 Laptop sm86, CUDA 12.9 native build,
Compute Sanitizer 2025.1.0.0, diagnostic minimum-arch NCCL library.

S4/batch7, CUCO_RANK+NCCL_LSA, DENSE and HASH_FIRST scalar generation,
pre-dedup ON, rank/world/map 0/1/0, pool64MiB, record capacity64/ring128,
buckets8/shards4/job2/bucket capacity32, matrix_u8 state/archive,
seed20260828, archive rows3/slots128, warmupOFF/archiveON/macro1,
owner-DAG captureON. No filters or suppressed CUDA API reports.

Each profile passed memcheck, racecheck, initcheck and synccheck. All eight
direct BFS processes exited0 within the120s external deadline. Memcheck,
initcheck and synccheck summaries:0errors; racecheck:0hazards/errors/warnings.
All logs reached the real owner-DAG capture launch and all eight archives
passed production verification. Independent full-byte CPU oracle then
confirmed all24unique states and all7layers [1,3,5,6,5,3,1] in each archive.

Artifacts: test_results/cuco-full-bfs-sanitizer-c9f6428/cuco-s4-*/.
CLI SHA256 f2cf9beded7f1096d80d0a5d332ed4d0b6ae91250780a12a4cfedba47944ca90
Native SHA256 bc61042eed619230675cd3760c07f78fc88edfb215298a42cc76daccd42866ce

Evidence scope: full one-rank CUCO BFS integration, including archives.
Not two-rank failure propagation, T4 NCCL registration initcheck, Tensor/BMMA
execution, large-graph capacity or paired A/B performance. Full goal open.
