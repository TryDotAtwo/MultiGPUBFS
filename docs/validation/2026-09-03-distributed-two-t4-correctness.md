# Native NCCL BFS: physical two-T4 correctness gate

Kaggle kernel `trydotatwo/mgbfs-distributed-native-smoke`, version 3, completed
on two physical Tesla T4 devices from source commit
`8e62b116c15b20fd544f562282b48216ed0b1fa8`.

The test exhausted U4(Z/2Z) twice, with logical-owner maps `[0,1]` and `[1,0]`.
For every depth, the union of the two rank archives was compared as full states
against an independent CPU oracle.  Both maps passed.  Global layer sizes were
`[1,3,5,8,11,13,13,8,2]`.

The physical UUIDs recorded by the run were:

- `GPU-396977ea-e2bd-a33a-bc74-edfcd65a43e1`
- `GPU-28b02c60-ceea-9085-5adb-10c7ea6bbe96`

NCCL 2.25.1 used the CUDA 12.8 runtime.  The logs record direct send/receive
transport and rank-local result files; torchrun acted only as the process
launcher.  This gate establishes exact distributed semantics and owner-map
invariance, not the still-separate performance result.

The raw evidence is retained in ignored local artifacts under
`artifacts/kaggle/distributed-smoke-v3/distributed-native-smoke/`.
