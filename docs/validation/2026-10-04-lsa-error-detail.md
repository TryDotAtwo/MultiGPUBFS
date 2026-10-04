# LSA activation failure diagnostics

The Rust startup vote previously dropped the native error buffer; commit
bc6a8ce preserves its bounded detail without changing the wrapper status code.
The C++ activation path also described the original submission result even
when the nonblocking waiter observed a different terminal async error.

The waiter now optionally returns the terminal submit/query/async NCCL code.
Activation diagnostics record operation, wrapper progress code, submitted
code and actual terminal code. Cancellation/timeout without a terminal NCCL
result is explicitly distinguished. The return codes, progress loop, abort
ownership and exchange ordering are unchanged.

## Verification

- Deterministic C++ test of the actual production waiter failed RED because
  async code 9 was discarded; GREEN preserves async 9, query 8 and submit 5,
  and clears stale detail on success. Existing cancellation/group/abort tests
  in the same executable also pass. This uses a NCCL test double, not hardware.
- Linux nvcc compilation of the full nccl_transport.cpp with MGBFS_NCCL_LSA=1,
  sm75 and pinned NCCL 2.29.7 wheel headers passes. No link/run claim follows.
- CPU cargo workspace, excluding legacy multigpubfs-gpu, passes.
- Earlier Rust CUDA-feature cargo check passes (existing unused_mut warning).

Kaggle v90 remains pinned to 749c691; it does not include these diagnostic
changes. Registration root cause and full initcheck acceptance remain OPEN.
