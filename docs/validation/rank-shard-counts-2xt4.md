# GPU shard counts from selected hashes: T4 gate

Private Kaggle `trydotatwo/mgbfs-state-commit-t4` v9 was the RED gate at
`2f3ca42e34d003d0107220541dd63e9deccf711f`: the state-commit test failed
to link specifically because `mgbfs_owner_shard_counts` did not exist.

Version 10 built `ec988b7b1c797815ac63af125516572d7053ccf5` and finished
COMPLETE on two physical T4s. It reports 20/20 checks, including state-commit
and archive-pack plain plus memcheck, racecheck, initcheck and synccheck on each
GPU. The state-commit test covers sorted survivor indices across four local
shards, an empty selection, duplicate/out-of-range/out-of-order indices,
wrong-owner hashes and count overflow. Invalid inputs leave count/offset
outputs untouched and set sticky fatal without moving the StateRing.

Raw outputs: `test_results/kaggle_state_commit_v9_red/` and
`test_results/kaggle_state_commit_v10/`. This validates a device-side shard
directory leaf, not a full cuCO rank-batch owner or end-to-end CPU-free BFS.
