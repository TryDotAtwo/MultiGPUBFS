# Weighted DENSE frame pack: local GPU leaf gate

Scope: `mgbfs_macro_exchange_pack_frame`, not the distributed BFS runtime.
Source branch: `codex/library-first-bfs`, 2026-09-23. The test builds from
`cuda/exchange_pack.cu` and `cuda/tests/macro_exchange_pack_test.cu`.

- `nvcc.exe -std=c++17 -arch=sm_75 ...` compiled and linked the leaf test.
- Physical execution on NVIDIA GeForce RTX 3070 Laptop GPU, compute capability
  8.6, driver 572.70: `MACRO_EXCHANGE_PACK_PASS`, exit 0.
- The test compares every output word against the scalar frame specification,
  checks hash/ref/state planes and zero padding, and observes sticky fatal=1
  for an out-of-range child reference. Zero weight is rejected before enqueue.
- `cargo test -p mgbfs-core --test macro_exchange_memory` passes three shape,
  overflow, and owner-partition-bound tests.
- Linux CUDA cross-target `cargo check -p mgbfs-runtime --features cuda`
  passes. This is typechecking, not a Linux GPU execution.

Compute Sanitizer 2024.2 on this Windows host returned `Error launching target
app` (exit 13) even for the passing executable, including its ASCII 8.3 path.
No sanitizer pass is claimed. The actual T4/sm75 execution, four sanitizer
modes there, framed NCCL exchange, multi-rank owner settlement, memory
admission, and end-to-end performance remain open.
