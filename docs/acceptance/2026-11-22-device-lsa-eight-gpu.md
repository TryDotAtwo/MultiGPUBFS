# Experimental device-driven BFS on eight GPUs

Scope: standalone research prototypes, not the library's production BFS,
the production segmented24/warp owner, Explore, archive export, or a
performance acceptance. Their finite fixed-size buffers are test fixtures.

The pinned source used for every binary in this run is commit
`b83f842e8c97945adf80165c585ea031ef460f5c` (files under
`tests/experimental_device_*_world_lsa_gpu.cu`). The tests used eight
separate rank processes, one GPU per rank, and NCCL 2.30.7. The GPU was
A100-SXM4-80GB, CUDA 12.8.93, driver 595.91.07. CUDA reported cooperative
launch and native peer atomic support in both directions for every one of
the 56 distinct device pairs.

For both normal and reversed device order, the bitmap fixture passed 26
ranked cases per GPU. Its independent CPU FIFO oracle compares complete
per-rank multisets of (depth, full packed state), including multiplicities.
Fixtures cover S5, S6, repeated symbols, directed rotation, empty ranks,
queue capacities 1 and 7, and resource-limited completed prefixes. A
previously seen child at a full owner did not become a false hard-full.

The separate serial hash prototype passed 64 ranked cases per GPU for each
device order. Tests include deliberately equal full hashes, exact state
comparison, two shards, bounded queues and full completed-layer multiset
comparison against the CPU oracle. This does not validate the production
concurrent segmented24/warp hash path.

The bitmap prototype also completed S8 on all eight devices, for normal and
reversed order: all 40,320 states and all 29 layers matched the CPU oracle.
The normal-order run recorded 282,240 generated edges. These tiny tests
retain output for verification and do not establish large-graph performance.

Nsight Systems 2025.1.1 traced the normal-order S8 experiment. Each GPU had
one `exact_layers` kernel. For each rank, the SQLite trace shows one host
`cudaStreamSynchronize` overlapping that kernel and no CUDA Runtime or
driver API call starting during the corresponding kernel interval. This is
a trace of this experiment and does not establish the same property for the
existing production BFS.

Evidence: `device-evidence-8a100-20261122.tgz`, SHA-256
`9cd3b3aa478c906da3d94f1ce7838f885ad06c7b8b03c754a2229cb04f27ff4f`.
The archive contains binaries and hashes, exact inputs, device topology,
raw test logs, Nsight report and exported SQLite, and the runtime interval
analysis. Compute Sanitizer was run on the earlier four-GPU bitmap prototype,
not on this eight-GPU run. GPU memory pressure beyond the small fixture,
failure cancellation and recovery, real 2^32-row boundaries, production
persistent reuse, and large workloads remain open.
