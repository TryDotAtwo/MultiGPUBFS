# Exact-key domain and native package repair, 2026-10-09

The near-VRAM n14r1 run reproduced SHARD_AB_PREPARE_FATAL_102. Four residual legacy hash-domain checks rejected legal32-bit words >=4294967291. They are removed: hash occupancy uses row indices and sort padding is identified by UINT32_MAX row reference; real all-ones keys survive min-reference reduction. Both GPUs passed boundary words in every coordinate, all-ones duplicate/state identity, pipeline, indexed requests and invalid owner/order/range protection.

Native library search also preferred system NCCL over the supplied CUDA12 package. On this host it selected a library requiring a newer driver. Packaged CUDA/NCCL now precede the real-driver system directory, which still precedes inherited compatibility stubs. Regression and ldd/GPU validation passed.

## Measured large prefix

Two RTX3060, HASH, capacity72694960 perrank, batch65536, shard target1048576, same geometry and binary; only key producer changes between private legacy fingerprint timing baseline and new lossless keys. Four alternating cases;37completed transitions, ending layer127996579states. Old times [19.67088, 19.533752]; new times [17.87136, 17.944131]; ratio of means 1.094628. All completed layer counts match. Each run is INCOMPLETE RESOURCE_STOP, not full graph time or exhaustive large-state proof. Terminal compact archive is committed. New first run was reused, not rerun.

Full n8r1 HASH/SORT_MERGE and n25r22 SORT_MERGE match independent CPU layer counts and terminal/previous-small sets. A fresh installed environment repeats HASH n8r1 and wide SORT_MERGE n25r22 and resolves packaged NCCL.

## Package

https://huggingface.co/datasets/TryDotAtwo/multigpubfs-native-releases/resolve/017880f66fd1f41bb18b6a7330750ed9ee09c920/linux-x86_64-sm86-cuda12.8/cfae5a1ae6ebb3451db9fa9e454b0b746f3afd3d/multigpubfs-0.2.0.dev0-py3-none-linux_x86_64.whl

SHA256 7527a3ab4fdc4e32959b5d01e84b00b7ad96576f768eae857c3486cdad9ac893. Anonymous immutable readback verified. Binary unchanged from fa4883c builtCUDA12.6; repaired library builtCUDA12.8; same CUDA runtime major12, accepted driver570.211.01. SM86 target, LSA compiledOFF. Earlier fa4883c package is superseded because of legal-key rejection. Blackwell/physical4/8/128GPU acceptance and global optimal tuning are not claimed. Broader thread objective remains active: general persistent sorted-history integration and scope-wide scaling/acceptance audit remain unfinished.

## General installed package follow-up

General one/twoGPU inverse/directed permutation/matrix, overflow inverse fallback, deadline, actual native SIGTERM, resource compact samples, two external TCP/NCCL ranks and collective/local tuning passed. An omitted peer transport and explicit default auto initially produced different direct tuner cache identities. The planner normalizes this default without mutating the supplied environment, preserves forced policy, and passes2CPU regressions plus freshly reinstalled cold/cached GPU acceptance. No unchanged large graph benchmarks were repeated. Latest package source 407a5d7c796e548f6cf88de58c10c345a864475c; native library unchanged from cfae5a1, timing above unchanged. Goal remains active.
