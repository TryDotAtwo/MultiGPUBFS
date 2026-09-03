# Weight-grouped macro generation: two-T4 gate

Kaggle kernel `trydotatwo/mgbfs-generation-tiles-t4`, private version 2,
completed against source `6f2ebaaf97b097eaf057bb99ee359bbf489e73b5`
and CUTLASS `ffa119a1255d78998536107466cc7097ecefa393`.

Physical devices:

- Tesla T4 `GPU-68c59f7b-d2db-0196-6f33-78b211f76edf`, 15360 MiB;
- Tesla T4 `GPU-97468642-758d-b973-d9e8-ecc7e44db44a`, 15360 MiB.

The CUDA build compiled the new macro generation ABI. The `generate` test
binary, including the independent S8 K3 matrix-product comparison and
weight-grouped move-major layout check, passed on both devices under plain,
memcheck, racecheck, initcheck and synccheck. The unchanged ping-pong generation
suite also passed the same matrix. Total recorded hardware gates: 20/20.

This proves the current one-GEMM layout primitive and backward compatibility of
the tested generation path. It does not prove all K values, macro owner
settlement, NCCL execution or end-to-end macro BFS performance; those gates stay
open.
