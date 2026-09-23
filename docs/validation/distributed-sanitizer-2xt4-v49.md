# Full small-graph two-T4 regression, Kaggle v49

Private `trydotatwo/mgbfs-distributed-sanitizer` version 49 completed on two
physical Tesla T4s, pinned to source
`a272a17d47320136d8288f4dbfada6732f03c851`. Downloaded
`test_results/distributed-sanitizer-v49/distributed-sanitizer/summary.json`
reports `COMPLETE`; the notebook log is alongside it.

Plain execution and all four Compute Sanitizer modes (`memcheck`,
`racecheck`, `initcheck`, `synccheck`) passed the small distributed archive,
macro and CUDA leaf fixtures. Then 24 independent CLI smoke configurations
passed: one/two ranks; DENSE scalar and HASH_FIRST scalar/SM75 INT MMA;
CUB/BMMA owner; pre-dedup OFF/ON. Each verified its mandatory rank archives
and identical global S4 layer counts `[1,3,5,6,5,3,1]` (24 states), with
1 GiB untouched VRAM reserve. The CLI launch used torchrun processes, unlike
some earlier two-device in-process leaf fixtures.

This establishes regression correctness of that pinned small workload. It is
not a full-state S10/S13 acceptance, an 8-rank gate, a performance comparison,
or proof that the CPU-driven owner/transport/retirement hot path is gone.
