# Current local CPU regression

Checkout: 2f92a6d (runtime CUDA code unchanged from 5a4ea73).
Existing Docker image multigpubfs-rust-toolchain:dev; Rust 1.75.
Command: cargo test --locked --workspace --exclude multigpubfs-gpu.
CARGO_HOME and CARGO_TARGET_DIR explicitly reside under D:/MultiGPUBFS/build.
Exit zero: 401 passing tests across 99 test-result blocks, zero failed,
zero ignored. Includes epoch/control/buffer-lifetime/failure contracts;
does not execute CUDA kernels or establish multi-rank hardware acceptance.
Log: build/rust-contract-regression-20261004.log.
SHA256: 499a66b6b3b8bb5c5611403795f40eceffcef99bb93cdcceda4a5acee39f77f2.

Python tests: 233 OK, eight skipped. Scripts: 57 OK, three skipped.
An initial Windows HF test failure was reproduced independently: the
child interpreter emitted a non-UTF8 traceback, causing parent decoding
to fail and stderr to become None. Explicit child -X utf8 in 2f92a6d
restores the real DUPLICATE_STATE assertion; isolated three-test and
full 233-test runs pass. No exporter/verifier behavior was weakened.

Main Kaggle v99 and secondary v3 remain RUNNING at the last live check.
Initcheck, updated full-runtime timeline, paired performance/VRAM,
production weighted macro dispatch and remaining publication gates are
not completed by this local regression.
