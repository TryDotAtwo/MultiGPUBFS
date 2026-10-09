# Runtime recheck and Kaggle footer result

Main checkout: `79fcca7`; native runtime code remains the physical-bank and
single-rank retirement implementation already present at `75494be`.

Docker engine is available outside the restricted shell. Earlier missing-pipe
checks inside that shell were not evidence of an unavailable host engine.
WSL reports docker-desktop Running and Docker reports version 29.7.2.

Local CUDA regression `one_rank_lsa_abort_retires_local_readers_without_peer_callback`
passed; terminal NCCL abort returned zero. Existing two unused-mut warnings
remain, not suppressed.

CUCO LSA tests `cuco_lsa_route_bank*` were first invoked concurrently by the
Rust test harness. Capacity fixture passed, while full-state fixture failed
at process-global library pool creation (`CUDA_STATUS_-1`). Both tests were
then rerun in the same binary with `--test-threads=1`: 2 passed, 0 failed.
This establishes sequential fixture success, not supported concurrent runtime
instances sharing one process-global RMM resource.

Sequential coverage: U4(F2), one GPU, DENSE/HASH_FIRST scalar, 2/3/4 physical
banks, K=3, batch=1. Six full-state/archive cases and six deliberate capacity
failure cases passed after actual bank reuse. Capacity failures latched and
abort returned zero. This is not a two-rank/T4 failure gate, a sanitizer run,
timeline or performance measurement. Local archive extent is memory-backed.

Kaggle `trydotatwo/mgbfs-s11-hf-stream` currently reports COMPLETE. Its only
listed output was footer-summary.json (896 bytes); output/log downloaded to
ignored `build/kaggle-status-20261004/`, no state dataset downloaded.
The report is VERIFIED_FOOTERS for historical S13 run
`s13-native-2xt4-20260905-152407`, HF revision
`d43c3aa640ef12935ff12f986e53d3e6fef6e92f`:
6228 files, 6227020800 rows, rank rows 3113547204 and 3113473596.
This checks metadata/footer row totals, not state uniqueness, content hashes,
CPU-oracle equality or the current native runtime. No new notebook launched.
