# Device-count DENSE materialization: two-T4 gate

The private Kaggle `trydotatwo/mgbfs-state-commit-t4` v11 was the RED gate at
`9c0e8462b6b8ddd05eff0bdfa351ac5afc48d373`: the state-commit test reached
the linker and failed only on the missing `mgbfs_state_materialize_rank_batch`.

Version 12 built `4dcc48f65c1c7009e0759d99a73dd5aa725e3ae1` and finished
COMPLETE on two physical T4s. It passed all 20 plain/sanitizer checks for the
state-commit and archive-pack binaries on both cards. The new test checks
source order `[3,0]`, an out-of-range source index, a selected-count/extent
mismatch, and a device source count above admitted capacity. Failure sets
owner/ring fatal before any state write and never publishes StateReady.

Raw outputs are in `test_results/kaggle_state_commit_v11_red/` and
`test_results/kaggle_state_commit_v12/`. This is a GPU leaf and Rust FFI
declaration, not owner/runtime integration. It does not exercise NCCL,
retirement, HASH_FIRST requests, or full BFS layers.
