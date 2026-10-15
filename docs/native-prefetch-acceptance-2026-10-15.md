# GPU-free native prefetch acceptance, 2026-10-15

Status: PASS. Package0.2.0.dev20; Python source `144c4a44029e423dff8ec39d2b3cd547c75a48ba`; wheel SHA256 `2a3213f46e02256dc3eeddf094a80488ceeddaba2f2bc70cebb85ee6c69ba7cb`; size 137,885 bytes. Native-index SHA256 `98cb1c35e660d454b48ba65dbc684e1c4baedf6320fe6c644fe9461ef4d5050b` is unchanged from dev19; no CUDA/Rust/BFS algorithm changes or new native compilation were introduced. [Exact public receipt](acceptance/native-prefetch-20261015.json), SHA256 `8cb86a9b511fe9f0c1b717c5875278944d5e1d03712800152cd9850c7c57c7d4`.

| Gate | Evidence |
|---|---|
| Regression | 19 new prefetch tests first failed for the absent feature, then passed; complete suite763 tests passed |
| Clean package | Both wheel and GitHub source installed/imported without GPU; anonymous public extras install passed |
| Real CPU-only host | No NVIDIA device nodes; CUDA library loading, hardware probes and subprocess creation forbidden during prefetch |
| Selected actual archives | SM100/SM120 + common:327,579,748 bytes downloaded once on CPU host |
| All production archives | Then186,883,601 additional bytes, total514,463,349; common was not downloaded again |
| Integrity | Every cached file independently checked against pinned size and SHA256 |
| Repeat / fresh CLI | Offline prefetch in parent and a fresh installed-CLI process:0 newly downloaded bytes |
| Concurrent selection | Three CPU-fixture processes requesting different SM sets downloaded common once; no corrupt/partial publication |
| GPU prefetch boundary | On2×T4, prefetch with no visible GPU and forbidden driver/library/process access populated SM75/SM100 |
| Parallel real GPU jobs | Separate jobs on physical GPU0 and GPU1 completed exact S3 [1,2,2,1],0 new bytes |
| Real2-GPU persistent | Two LRX-S4 jobs matched independent CPU layers [1,3,5,6,5,3,1] and terminal states; same2 worker PIDs;0 new bytes |
| Multiple SM files | SM100 remained hash-valid after T4 execution; no package reinstall or SM eviction |

[Full package CI](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38782373175); [CPU real-file acceptance](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38783081409); [anonymous extras install](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38783083416). CPU receipt is the same result object recorded in the successful CI log marker; formatting is normalized for publication. [CPU reproduction script](../scripts/accept_native_prefetch.py); [physical2×T4 reproduction notebook](../examples/acceptance_prefetch_t4.ipynb).

The first private GPU harness stopped before BFS because it assumed the CLI script lived beside Kaggle's Python executable. The harness was corrected to invoke `python -m multigpubfs`; a separately detected unavailable oracle helper was replaced with the explicit independent CPU traversal shown in the notebook. Installed package/native bytes were unchanged. The final version2 run completed all six GPU checks. This is startup/cache/correctness acceptance, not a throughput benchmark.

All production SM payloads and universal code were downloaded and hash-verified on the CPU host. Actual GPU execution here is limited to2×T4/SM75. B200/SM100, B300/SM103 and SM120 have the existing compilation/cubin inventory, not new execution acceptance. CPU-fixture selection under synthetic hardware descriptions is software evidence only. Deep diagnostic Static/legacy alternatives are separate from the production `--all` set.

The shared cache must be persistent and visible to all participating jobs/nodes. Separate or wiped filesystems cannot provide cross-job reuse. `MGBFS_NATIVE_OFFLINE=1` makes missing native components fail explicitly instead of downloading. Prefetch does not install a driver, alter Torch/NCCL dependencies, discover a GPU, set global environment variables or claim a ready hardware configuration.
