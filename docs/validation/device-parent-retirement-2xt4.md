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
suite passed. A separate private
`trydotatwo/mgbfs-rank-retirement-gate-t4` v1 at the integration source
completed with `summary.status=PASS`: both physical T4s passed the one-GPU
full-state/capacity tests (3/3 each), and the two-GPU NCCL rank-owner layer
and archive fixture passed (2/2; one unsupported case ignored). The
two-process CLI S4/U4m2 scenarios also completed, but on that pinned source
they exercise cuDF/cuCO-indexed, not CucoRank. Raw logs:
`test_results/kaggle_retire_value_full_bfs_v1/`. V2 at the same exact source
completed the full plain/four-sanitizer 1/2-T4 BFS fixture matrix: 3/3
one-GPU tests on each T4 and 2/2 two-GPU NCCL rank-owner layer/archive tests
in every mode. memcheck/initcheck/synccheck reported zero errors; racecheck
reported zero hazards/errors/warnings. The eight-GPU case was intentionally
ignored. Raw `test_results/kaggle_retire_value_full_bfs_sanitizers_v2/`.

An audit found that DENSE could return on a local retirement FIFO fatal before
its peer reached the next failure collective. Commit
`b71c16b5ba95cadaea227671acfb44a1bbddf4ce` adds an all-rank pre-owner
failure vote after the P2P completion event, including an empty local batch.
Local test-first coverage, CUDA-feature Rust typecheck and the CPU suite
passed. Private `trydotatwo/mgbfs-rank-retire-fatal-gate-t4` v2 at that exact
source completed on two physical T4s: the per-device full-state/capacity
fixtures passed 3/3 on each GPU, and the two-GPU NCCL rank-owner layer/archive
fixture passed 2/2 (one eight-GPU case ignored). This is a normal-path gate:
it did **not** inject a retirement FIFO fatal, and it did not run sanitizers.
The additional collective is a correctness guard, not a throughput improvement;
it must be replaced by a device-driven fatal protocol before claiming the
requested CPU-free pipeline. Raw logs:
`test_results/kaggle_retire_fatal_vote_v2/`.

The next cut derives the 0/1 NCCL fatal input word on GPU from the sticky
StateRing control, eliminating the separate ring-fatal D2H in the DENSE and
HASH_FIRST retirement votes. Source `1c29487` produced the expected RED
undefined-symbol link error on private state-commit notebook v17. CUDA/Rust
integration `8aacb24df124503a7fa0b9ba079c9858729412c7` passed local
CUDA-feature typecheck and the full CPU suite. Private state-commit v18
completed 20/20 plain and four-sanitizer checks on two physical T4s, with
zero errors and racecheck hazards/warnings. Raw:
`test_results/kaggle_ring_fatal_vote_red_v17/` and
`test_results/kaggle_ring_fatal_vote_green_v18/`. Full 2×T4 BFS gate v3 at
the integration source passed its plain single-device and two-device
full-state/layer/archive fixtures; raw
`test_results/kaggle_ring_fatal_vote_full_bfs_v3/`. A separate v4 full-BFS
four-sanitizer gate completed at source `8aacb24` with 3/3 one-GPU tests
on each physical T4 and 2/2 two-GPU tests in plain and every sanitizer
mode (one eight-GPU case ignored). memcheck/initcheck/synccheck reported
zero errors; racecheck reported zero hazards/errors/warnings. Raw logs:
`test_results/kaggle_ring_fatal_vote_full_bfs_sanitizers_v4/`.
Neither v3 nor v4 nor the leaf test injects a
retirement FIFO error. The group vote still synchronizes the host
to branch before owner commit, so this is not CPU-free retirement.

A focused two-device fault-injection fixture at `4bd474b` passed in private
`trydotatwo/mgbfs-rank-retirement-gate-t4` v3. It sends an invalid FIFO
descriptor to rank 0 only: rank 0 observes sticky fatal 17, rank 1 has no
local fatal, and both receive group fatal 1 from NCCL without hanging. The
two-GPU test binary passed 3/3 plain tests (one eight-GPU case ignored). This
proves the CUDA retirement → device word → NCCL vote chain under an injected
local fault; it does not inject the fault through the full BFS scheduler or
measure its latency. Raw `test_results/kaggle_retire_fifo_fault_v3/`.

The follow-on source `ced41ab` removes a redundant DENSE stream wait after
the first-round fatal vote, whose own completion already covers the stream.
Private full-BFS notebook `trydotatwo/mgbfs-rank-retire-fatal-gate-t4` v5
was launched for a plain 1/2-T4 gate at this exact source; its result is
pending.

The runtime still has host-side completion and branch dependencies after
retirement;
transport counts and NCCL sizes also remain CPU-driven. No end-to-end latency
or VRAM improvement is claimed.
