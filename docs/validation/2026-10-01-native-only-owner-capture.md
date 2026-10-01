# Native-only owner capture regression

Local hardware: one real RTX 3070 Laptop (sm86), CUDA 12.8.93. Existing native CUDA sources were built with MGBFS_NCCL_LSA=ON against pinned official nvidia-nccl-cu12 2.29.7; no paid rental and no second Kaggle notebook.

An actual DENSE/CUB S4 CLI run with NCCL_LSA completed, committed its archive and matched the independent full-state oracle: 24 unique states, layers [1,3,5,6,5,3,1]. However the requested debug owner-DAG capture was silently ignored in a cuda-only Rust build. Its declarations and native call sites were incorrectly conditional on library-owner.

The new native_only_cli_executes_requested_owner_capture integration test reproduced that defect RED against the real CLI. Removing only the irrelevant library-owner condition produced GREEN: the graph is captured, instantiated and launched; S4 layer counts and group RunCommit remain correct. Debug opt-in is retained. No runtime allocation, communication protocol, host readback or production-default scheduling change is introduced by this fix.

Full Linux CLI CUDA suite first exposed two additional test-contract failures: public_bench_cannot_disable_the_archive_output_contract and macro_depth_must_not_be_silently_ignored_by_reference_bench. Those fixtures had applied unavailable-build CLI expectations to native launch. They now exercise native admission with an isolated valid single-rank launch identity, retain the exact archive-rejection assertion, and check invalid macro values through the native validator; positive single-rank macro depth remains supported. No production validation was moved ahead of group bootstrap.

Final commands: cargo test --locked --offline -p mgbfs-cli --features cuda -- --test-threads=1 (11 tests pass, including real LSA capture); fresh ordinary Windows cargo test --locked (exit 0). Existing unused-mut/dead-code warnings remain. CPU suite cfg-gates the new GPU test out; it is executed separately by the Linux command above.

This is a single-GPU debug-capture correctness result, not two independent ranks, CUCO_RANK, T4 sanitizer closure, a full BFS Nsight timeline or paired performance acceptance. Existing two-T4 gates remain required.

## Unfiltered memcheck follow-up

The same production DENSE/CUB LSA CLI was run under compute-sanitizer --tool memcheck --error-exitcode 97, with a 120-second process timeout and no filters/suppressions. The observation completed (not timed out): launcher exit 1, ERROR SUMMARY: 12 errors. All reported errors are cudaErrorNoKernelImageForDevice (209) at cudaFuncGetAttributes/cudaGetLastError inside ncclInitKernelsForDevice during communicator initialization.

Seven owner-DAG capture launches and archive finalization were reached. Independently verifying that run's archive again yields 24 canonical S4 states and exact layers [1,3,5,6,5,3,1]. That does not make the memcheck gate pass. Raw log: test_results/local_lsa_20261001/cub-dense-memcheck/rank-0.log.

The pinned upstream NCCL source b91894bd5b190c874d98a017f93f5daa515b65d0 was fetched for investigation, without changing it. Its kernel generators already derive minimum CUDA architecture, while initialization probes function attributes and tolerates unsupported-image failures. Whether an explicit architecture admission guard can avoid these probes without changing supported kernel selection still requires implementation and real validation. No vendor patch or custom-library gate is claimed here.
