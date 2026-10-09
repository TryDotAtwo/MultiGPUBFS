# B300 compact automatic sweep
The image contains prebuilt native SM103 libraries, NCCL LSA support, CLI, Graph32 primitive, Python dependencies and source. No compilation per graph, credentials, graph states or compiler caches.

Default startup waits for configuration. Pass arguments to the entrypoint:

    --root /results/run-unique-id --repo-id TryDotAtwo/multigpubfs-bfs-results --deadline-unix UNIX_WORK_DEADLINE --token-file /run/secrets/hf_token

Mount persistent /results and read-only HF credential file. Use all 8 GPUs, sufficient host RAM, unlimited memlock, host IPC and shared memory. An external lease guard must enforce a separate hard deletion deadline including disk/traffic expenses. The image neither rents machines nor changes budgets.

Admission requires exactly 8 distinct B300 SM103 GPUs, HF write/readback, Graph32 primitive and complete independent 24-state CPU oracle across native ranks. --expected-gpus 1 is explicitly only single-card staging. --preflight-only runs admission without production.

Production runs two seeds in compact last_complete_small_1000 retention, end publication with disk pressure handling, native VRAM admission and bounded pinned RAM. It resumes the same root with existing immutable ledger; completed pairs are preserved. Resource-limited first seed skips the second; fixed-r heuristic pruning is recorded separately from computed COMPLETE graphs.

SIGTERM requests native shutdown, snapshot and publication. Do not equate SSH timeout with program completion. HF VERIFIED receipt and all_checksums_verified are required before discarding unique results.

CPU/build validation does not certify B300 correctness or speed. Hardware startup gates run on the actual rented B300s before the sweep. No H200-equivalent throughput claim has been measured.
