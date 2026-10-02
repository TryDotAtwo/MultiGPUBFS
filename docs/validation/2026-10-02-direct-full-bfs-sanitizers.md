# Direct full-BFS sanitizer gate

Source c807c0f. One physical RTX 3070 Laptop sm86, diagnostic NCCL minimum-arch
build, CUDA 12.9 runtime/native build, Compute Sanitizer 2025.1.0.0.
No kernel/API filters or suppressions. Each tool directly launched mgbfs,
not a parent test harness. error-exitcode=97 and external timeout=120s;
all four commands returned zero without timeout.

S4, batch 7, DENSE, CUB_SORT_MERGE, NCCL_LSA, pre-dedup ON, world=1/map=0,
matrix_u8 states/archive, seed 20260828, capacity 64/ring 128, buckets 8,
shards 4/job buckets 2/bucket capacity 32, archive rows 3/slots 128,
warmup OFF, archive ON, macro depth 1, owner-DAG capture ON.

memcheck/initcheck/synccheck: zero errors. Racecheck: zero hazards, errors,
warnings. All logs reached MGBFS_OWNER_DAG_CAPTURE launched. Each archive
passed production checksum/count verification and independent Python
full-state oracle: 24 states, all layers [1,3,5,6,5,3,1].

Raw logs, archives and rank/group results:
test_results/full-bfs-sanitizer-c807c0f/direct-s4-*/.

Actual SHA-256:
- CLI f2cf9beded7f1096d80d0a5d332ed4d0b6ae91250780a12a4cfedba47944ca90
- native bc61042eed619230675cd3760c07f78fc88edfb215298a42cc76daccd42866ce

This does not establish two-rank correctness, peer-window lifetime, T4 NCCL
registration initcheck, Tensor/BMMA execution or performance. A prior harness
target-processes run returned zero but is not the credited direct-BFS evidence.
Kaggle push was rejected by weekly 30h GPU quota; no new worker admitted.
The full project goal remains active and incomplete.
