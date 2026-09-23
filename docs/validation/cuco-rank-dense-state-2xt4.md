# cuCO owner to DENSE StateRing: two-T4 integration gate

Private `trydotatwo/mgbfs-cuco-dynamic-shard-refs-t4` v8 built the C++/CUDA
fixture at `9a6b85e317ea35e947d1c28dc482fe2f57171ca3`, but stopped during
the Rust build. Its pinned Rust toolchain does not implement `Default` for raw
pointers; no owner materialization test ran in v8. The unnecessary derive was
removed at `abceacffb3891d74f6befe0de21d0e080a6c84a8`.

Version 9 at that exact source reported PASS on two physical T4s. The captured
CUDA Graph now contains C-ABI cuCO compare, GPU shard counts, all-shard ring
reserve, persistent commit and device-count DENSE materialization. The fixture
checks actual 16-byte state rows from the first and second batches, then
checks that accepted-capacity overflow leaves ring tail and the next output
row unchanged. The owner probe passed plain plus memcheck, racecheck,
initcheck and synccheck on both GPUs with zero errors/hazards. The pinned Rust
ABI tests also passed. Raw output:
`test_results/kaggle_cuco_rank_v8/` and
`test_results/kaggle_cuco_rank_v9/`.

This gate exercises one owner rank on each T4 independently. It does not
exercise NCCL, archive, retirement, multi-rank BFS, or the production Rust
scheduler. The complete owner→transport→retirement hot-path claim remains
unproven.
