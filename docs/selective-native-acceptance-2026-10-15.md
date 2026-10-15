# Selective native delivery: measured acceptance, 2026-10-15

Status: PASS for installed physical 2×T4. [Machine-readable receipt](acceptance/selective-native-20261015.json), SHA256 `91177812ce9f75f4757bb4a8eff5aa55de39ecf63b2e4885ef120fdb8eca98cf`. This is one finite delivery/correctness validation, not B200/B300 execution or a global performance optimum.

The Python package, native executable and CUDA component have distinct source identities. Python/wheel source: `914b2b3a250e94e54dcc5a30c9867d7855481c1f`. Native CLI: `0257cd3bd2f9aa2fbb1911f5c3115a4d20a112bb`. Per-SM CUDA build: `5746c36524cf3125efa7ccdb2cf9a9ad166f1b4b`. All CUDA kernel/helper inputs retain fingerprint `196427e16607984984b3f534bd0eb7eebec91a9c0561df9874f0b683538e644c`; no BFS algorithm change was introduced for distribution.

| Check | Measured result |
|---|---|
| Ordinary Python wheel | 135,304 bytes; no native executables/shared libraries inside |
| Cold explicit activation, 2×T4 | 51.013 s; 317,591,823 bytes, common + cuda-sm75 only |
| Warm ordinary doctor | Actual GPU/BFS and executed READY command; {"PASS": 29, "SKIPPED": 2}; zero newly downloaded bytes |
| Fresh-process offline cache | Exact 2-GPU S3 layers [1,2,2,1], no download |
| shard_ab_hash / shard_ab_sort_merge | Both real 2-GPU runs pass independent CPU layers and terminal-state oracles |
| Deferred alternatives | shared-universal and static-sm120 each execute exact oracle BFS and printed public command |
| Production identity during contrasts | Manifest SHA unchanged; unused legacy payloads absent |
| Persistent S11 | 39,916,800 states, diameter55; BFS 2.880441 s; total call 5.750977 s |
| Persistent S12 | 479,001,600 states, diameter66; BFS 35.928731 s; total call 36.029513 s |
| Between S11/S12 | Same two worker PIDs, zero hardware-probe subprocesses, zero new transfers |
| Diagnostic archive | Every manifest size/SHA checked; graph-state/job payloads excluded; owned processes reaped |

Every S11/S12 layer was checked against exact Mahonian counts, totals against n!, and the terminal state against the reverse permutation. Timing columns distinguish BFS from total API call; cold preparation is listed separately. Deferred-recipe candidate restriction is an explicit test fixture; the chosen library and public command still ran on real GPUs.

[Package CI](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38761978972) passed **744 CPU tests**, built the pure Python wheel and installed both wheel and GitHub source in clean no-GPU environments. [Native build](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38752777505) compiled all eight individual SM targets and checked 24 CUDA modules per target. [Assembly](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38757641731) checked ELF linkage, cubin/PTX inventory, complete source identities and all component hashes. [Release relay](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38762333092) transferred assets directly on CI; native payloads were not downloaded to the workstation.

Wheel: `multigpubfs-0.2.0.dev19-py3-none-any.whl`; SHA256 `5deedf22ac24ac48c0cd8f6aeb909c18197cf3444dd73b668db5dbd4891e8986`. Component-index SHA256 `98cb1c35e660d454b48ba65dbc684e1c4baedf6320fe6c644fe9461ef4d5050b`. [Component release](https://github.com/TryDotAtwo/MultiGPUBFS/releases/tag/native-components-20261015-v1) contains immutable checked assets for SM75/80/86/89/90/100/103/120, common runtime and deferred controls.

The new selection/download flow was executed on T4/SM75 only. Other target libraries, including B200/SM100, B300/SM103 and SM120, are compiled/inventoried; this report does not prove their execution, fix a particular server driver, or establish the root cause of an earlier native-count hang. `doctor` on the actual allocated server remains the execution gate. Ordinary matching cubins avoid packaged-kernel PTX JIT, while vendor-library initialization can still occur.
