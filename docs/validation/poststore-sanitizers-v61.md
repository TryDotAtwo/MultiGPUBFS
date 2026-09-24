# Post-store CUDA sanitizer gate, Kaggle v61

Kernel `trydotatwo/mgbfs-library-owner-t4`, version 61, completed with
`status=PASS` on two physical Tesla T4 cards. The notebook pinned source
`6dd41bb1f669a7008e08ab2c2b0769ad64fe71b7` and ran Compute Sanitizer
`memcheck`, `racecheck`, `initcheck`, and `synccheck`.

The runtime BFS suites on GPU 0 and GPU 1 each ran three tests under every
tool. The two-GPU default suite ran three tests under every tool:
`cuco_rank_two_gpu_dense_layers_and_archives_match_oracle`,
`library_two_rank_layers_and_archives_match_oracle`, and
`retirement_fifo_fault_votes_group_fatal_on_two_devices`. Each test process
reported `0 failed`. `memcheck`, `initcheck`, and `synccheck` each reported
`ERROR SUMMARY: 0 errors`; `racecheck` reported `0 hazards displayed
(0 errors, 0 warnings)`. The separate native CUDA, cuCO and control-transfer
test suites were also included in the notebook's four-tool pass. The notebook
ran small two-process torchrun CLI S4/U4m2 cases and verified both rank
archives for the configured library backends and profiles.

Scope limit: the default two-GPU Rust suite explicitly ignored seven tests,
including LSA-specific full BFS and injected host/capacity failures. Thus v61
is **not** a four-tool LSA full-pipeline sanitizer gate. The earlier two-T4
LSA correctness runs are separate evidence, not a substitute for that gate.
This is also not a scale or speed measurement. The diagnostic CUDA-profiler
instrumentation added after `6dd41bb` is not covered by v61.

Raw summary and logs are in
`test_results/kaggle_poststore_sanitizers_v61/library-owner/`.
