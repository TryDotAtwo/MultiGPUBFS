# Native reference benchmark CLI

`mgbfs bench --reference` calls the same library entry point as the legacy
`distributed_bench` example. There is no subprocess wrapper or second BFS
implementation. Build on Linux with CUDA/NCCL and the native library:

```sh
MGBFS_CUDA_LIB_DIR=/absolute/native/build cargo build --locked --release -p mgbfs-cli --features cuda
```

With that build directory and CUDA libraries on `LD_LIBRARY_PATH`, an S4
two-rank smoke launch is:

```sh
MGBFS_BENCH_WARMUP=1 MGBFS_BENCH_CAPACITY=64 MGBFS_FUTURE_CAPACITY=128 \
torchrun --standalone --nproc-per-node=2 --no-python \
  target/release/mgbfs bench --reference s4 7 /tmp/unique-bootstrap \
  /tmp/unique-archive /tmp/unique-results
```

Choose unused bootstrap/archive/output paths for each independent run.
Arguments after `--reference` are group (`sN`), parent batch, bootstrap file,
archive prefix and result directory. Both ranks write their own archive and
result JSON. The native data plane does not import PyTorch. Rank topology is
read from torchrun. Use `mgbfs verify <rank-archive>` after completion.

All existing reference environment selectors remain available, including
`MGBFS_PROFILE`, `MGBFS_OWNER_BACKEND`, `MGBFS_PRE_DEDUP`,
`MGBFS_HASH_FIRST_GENERATION`, capacities, rank map and archive settings.
The CLI rejects `MGBFS_BENCH_SKIP_ARCHIVE` except when absent or `0`;
the test-only no-archive switch remains confined to the legacy example.
Warmup is explicit (`MGBFS_BENCH_WARMUP=1`); with it, the runner completes
an independent untimed pass before recording the measured pass. Warmup
requires file archives, not the external streaming pipe.

This reference is limited to symmetric-group fixtures and one or two ranks.
It retains the current reference scheduling, seed and backend restrictions.
It does not implement production `run`, general manifests, macro-depth
dispatch, calibration/sweep orchestration or the admitted multi-slot
dispatcher. Its CLI wrapper has not yet passed a target-GPU run; previous
hardware evidence belongs to the source revisions recorded in validation
reports. A non-CUDA or non-Linux build rejects this command without a CPU
fallback. `preflight --offline` and `verify` remain available without CUDA.
