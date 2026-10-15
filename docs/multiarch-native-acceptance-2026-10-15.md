# Multi-architecture native release acceptance, 2026-10-15

Release [native-pip-20261015-v18](https://github.com/TryDotAtwo/MultiGPUBFS/releases/tag/native-pip-20261015-v18), package `0.2.0.dev18`.

* Wheel SHA256: `1eda27dcd590528dc972dedffcf978c26854832e70703cb8e3d3233910769fa9`; bytes: 614758379.
* Python/package source: `fd248e3979f69655417fab92541feb864272c60f`.
* Native CLI source: `0257cd3bd2f9aa2fbb1911f5c3115a4d20a112bb`; CUDA library source: `8a2ad1003d96527783681d1b553dc8bb8347bee7`.
* Production CUDA library SHA256: `663a2a8d7a69a090d51e4d6c43cb6772b1537a912af22f9c0dfaceb44d3cd802`.
* [Sanitized real-GPU receipt](acceptance/multiarch-native-20261015.json).

## Compiled coverage

Both shared production and static control contain native SM75/80/86/89/90/100/103/120 and ordinary compute80 PTX. `cuobjdump` verifies every target against every CUDA module, not just a declaration in metadata. Native module counts: `{"100": 24, "103": 24, "120": 24, "75": 24, "80": 24, "86": 24, "89": 24, "90": 24}`.

Native [build CI](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38736138908) and [packaging/clean-install CI](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38747274003) succeeded; the full CPU regression suite passed 713 tests. The package gates native/runtime linkage, private library resolution, artifact hashes and identical CUDA kernel/helper inputs. Complete CUDA trees are recorded per variant: the old control's CMake defaults differ; this is not hidden as identical source-tree provenance. The [diagnostic regressions](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38747250792) include negative tests rejecting changed kernel/helper input or a mixed source matrix.

The historical IDs `shared-sm120` and `static-sm120` are retained for profile compatibility. They now identify multi-architecture libraries; read their actual target arrays.

## Real execution on 2×T4

The private notebook installed the exact public GitHub wheel with SHA verification and no Kaggle Inputs. It verified:

* Ordinary doctor: all non-skipped components PASS, all GPUs selected, exact permutation/repeated/matrix BFS and executed public command accepted.
* Injected native-count TIMEOUT: independent real driver/runtime discovery recovered both logical GPUs; real native BFS and the printed-command gate passed. The original report remained FAIL with a separate READY recipe. This is fault injection for branch coverage, not reproduction of the target server's unexplained hang.
* Real two-GPU `shard_ab_hash` and `shard_ab_sort_merge` with independent CPU layer and terminal-state oracles.
* Static-runtime/fresh-worker/socket recipe with actual command execution; candidate restriction was a test fixture.
* Persistent CayleyPy API on S11/S12, full Mahonian layer vectors, `n!` state counts, diameter and terminal reverse state; same two resident worker PIDs across both calls.
* Archive SHA256 manifests, architecture/proof inclusion, exclusion of graph-state/job payloads and complete owned-group cleanup.

| Graph | States | Diameter | BFS seconds | API wall seconds |
|---|---:|---:|---:|---:|
| S11 | 39916800 | 55 | 2.905319 | 7.623074 |
| S12 | 479001600 | 66 | 36.351246 | 36.452345 |

These are acceptance measurements on this T4 run, not a controlled speedup claim. Other architectures, including B200/SM100, B300/SM103 and RTX Blackwell/SM120, have code-inventory evidence for this wheel; they were not executed in this run.

## User workflow

Use the [one-command update/check and server API examples](server-quickstart-ru.md). CUDA selects a compatible cubin inside the one wheel, or ordinary PTX where applicable; no architecture-specific manual download/build is required. NVIDIA documents [cubin/PTX compatibility](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html).

Both doctor modes preserve native-count failures and attempt independently discovered GPUs. READY requires actual exact BFS and successful execution of the reported command. It establishes the selected finite working path, not unlimited graph capacity, optimal throughput or a unique explanation of an earlier timeout. A driver or administrator GPU restriction cannot be repaired by packaging alone.
