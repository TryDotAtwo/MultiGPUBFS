# cuCO rank-batch C ABI: two-T4 gate

The private `trydotatwo/mgbfs-cuco-dynamic-shard-refs-t4` v6 was RED at
`47befd4bdac1b12d08414f4bad279331651a077b`: `cuco_owner_probe` reached
linking and failed on the missing `mgbfs_library_rank_*_v1` symbols.

Version 7 built `fadcdcd88dce2258b9fd4111e377bd8c0a77ce54` and reported
PASS on two physical T4s. The existing rank-batch fixture now calls the C ABI
for creation, captured compare/commit, completion, shard export and destruction.
It still checks deterministic first sources, second-batch persistent keys and
capacity-overflow atomicity. Plain, memcheck, racecheck, initcheck and synccheck
passed on both cards; the owner-probe logs show zero errors/hazards. Raw output:
`test_results/kaggle_cuco_rank_v6_red/` and
`test_results/kaggle_cuco_rank_v7/`.

The ABI does not itself wire the Rust scheduler or prove an end-to-end BFS.
Result pointers remain borrowed until stream/event-ordered readers complete;
calling `complete` before that is an invalid caller protocol. There is no
evidence yet for a host-readback-free owner→transport→retirement timeline.
