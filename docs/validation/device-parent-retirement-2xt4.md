# Value-form parent retirement

The distributed runtime used to upload a 64-byte host extent into a device
buffer before every parent retirement. `Buffer::put()` synchronizes its CUDA
stream to protect the temporary host slice, so this imposed a host wait even
when the parent extent was already known. The new value-form CUDA ABI passes
the immutable extent as a launch argument and reuses the existing FIFO,
descriptor and sticky-fatal checks. The Rust DENSE and HASH_FIRST call sites
now use that ABI; archive/generation reader events still precede retirement.

TDD evidence on two physical Kaggle T4s:

- Private `trydotatwo/mgbfs-state-commit-t4` v15, source
  `d5a55dae103bd48b7b8ac907da85a2b63768f741`, failed at the expected
  undefined `mgbfs_state_retire_dense_prefix_value` linker symbol. Raw
  `test_results/kaggle_retire_value_red_v15/`.
- V16, source `002157b187889b717afded5441de0a4c974bf6e9`, completed all 20
  state-commit/archive-pack checks on both T4s: plain plus memcheck,
  racecheck, initcheck and synccheck. All reported zero errors; racecheck
  reported zero hazards and warnings. The fixture checks normal FIFO release,
  wrap padding and stale-descriptor rejection without ring mutation. Raw
  `test_results/kaggle_retire_value_green_v16/`.

The Rust integration is commit `0e3d1d655e5c56e6b15c7116600aadb5824e24bc`.
Local CUDA-feature typecheck and the complete CPU no-default-features test
suite passed. A separate full BFS 2xT4 runtime gate is in flight in private
`trydotatwo/mgbfs-rank-retirement-gate-t4` v1. The leaf result must not be
confused with that still-unverified integration. The runtime still drains the
stream and reads ring fatal after retirement; transport counts and NCCL sizes
also remain CPU-driven. No end-to-end latency or VRAM improvement is claimed.
