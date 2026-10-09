# External ranks without a shared filesystem

The same run_graph call detects WORLD_SIZE>1. Start one Python process per GPU with RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR and MASTER_PORT set by your cluster launcher (for example torchrun). The package itself has no torch dependency. One explicit device per process can override LOCAL_RANK. All ranks must use identical graph/native artifacts and parameters.

The rank-zero control server uses MGBFS_CONTROL_PORT (default MASTER_PORT+1); this port and NCCL transport must be reachable on the trusted cluster network. MGBFS_RUN_ID identifies the run, and MGBFS_CONTROL_TOKEN can supply the same authentication token on every rank. This small standard-library HTTP control service exchanges only inventory, native/graph identity, launch parameters, NCCL bootstrap and bounded terminal receipts. Production state generation, routing, communication and exact dedup remain native GPU operations. A shared filesystem is not required; output is local to each worker and the bounded merged receipt is written on each node.

Free VRAM is gathered per rank and independently admitted by the same weighted planner. Four bounded collective same-graph profiles select a common shard/batch configuration when autotune is enabled and the admitted workload is substantial. Every profile and final calculation creates a separate native NCCL communicator; profile time is deducted from the work budget. There is currently no persistent external profile cache.

Rank errors are propagated to peers and no BFS restart is attempted. Failed observation does not authorize a restart. Graph/native/parameter mismatches are rejected before calculation. COMPLETE/INCOMPLETE and state SHA256 are agreed before writing global receipts.

Acceptance: two actual RTX3060, separate worker directories, real TCP control and NCCL; exact permutation and matrix oracle, resource snapshot, deliberate mismatch rejection and collective tuning on40320 states. This is not physical multi-host, heterogeneous-model or8/128-GPU hardware acceptance.
