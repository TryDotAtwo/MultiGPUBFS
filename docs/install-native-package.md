# Install MultiGPUBFS with one pip command

The same ordinary pip command installs the Python API from GitHub on Kaggle or a Linux x86_64 GPU server accessed through SSH. No compiler, standalone bootstrap, manual CUDA paths, HF account or Kaggle Inputs are required.

## Linux / SSH

```bash
python -m pip install "multigpubfs[cayleypy] @ https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-20261019-v31/multigpubfs-0.2.0.dev31-py3-none-any.whl#sha256=9620fda20aff293e143f973d68916edbf91e487032aac7ccee8225fdad795670"
```

## Kaggle

Enable **Internet** and **GPU (2×T4)**, then run:

```python
%pip install "multigpubfs[cayleypy] @ https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-20261019-v31/multigpubfs-0.2.0.dev31-py3-none-any.whl#sha256=9620fda20aff293e143f973d68916edbf91e487032aac7ccee8225fdad795670"
```

After an upgrade, start a new Python process/session before importing the package.

```python
import multigpubfs, cayleypy
from multigpubfs.generic_session import persistent_graph_workers
multigpubfs.enable_native()
with persistent_graph_workers():
    for n in (10, 11, 12):
        graph = cayleypy.CayleyGraph(cayleypy.PermutationGroups.coxeter(n), device="cpu")
        result = graph.bfs()
        print(n, result.bfs_completed, result.layer_sizes)
```

`device="cpu"` constructs the graph definition; native BFS uses the GPUs visible to the allocated job. Use `multigpubfs.NativeOptions(devices=(0,1))` in `enable_native(...)` to select logical devices explicitly. Import alone does not initialize CUDA, download native files or replace CayleyPy methods.

## Shared/login host and offline GPU jobs

To prepare before GPUs are allocated, use `python -m multigpubfs prefetch --sm 100,120` or `python -m multigpubfs prefetch --all`. No GPU/driver discovery is performed. `--all --dry-run` reports the remaining archive bytes without downloads/writes. Put the cache on persistent storage visible to every job and set `MGBFS_NATIVE_CACHE` consistently. Set `MGBFS_NATIVE_OFFLINE=1` in GPU jobs to prohibit missing-component downloads. [Commands, sizes, Python API and concurrency](native-prefetch.md).

## What downloads and when

The wheel is **142,874 bytes** and contains Python code plus a pinned component index. On a GPU node, explicit `enable_native()` detects the visible GPUs through a bounded NVIDIA Driver API child and prepares the native runtime before the first BFS clock. Installation and graph setup can also happen on a CPU/login/container-build host: preparation is deferred to the first real native operation on a GPU node. `doctor` has a separate preparation stage.

Only the common executable/private CUDA12.9.79/NCCL2.30.7 component and one compatible CUDA component are fetched from pinned **GitHub Releases**. Native targets are SM75/80/86/89/90/100/103/120. A same-major compatible image can cover newer minor capabilities; mixed or other forward-compatible devices can use the universal component with ordinary compute80 PTX. The package records the actual selection. No target-server compilation is used for matching native cubins.

The current common component is 304,497,488 bytes. A clean SM75/T4 selection needs 317,597,550 component bytes; SM100/B200 needs 316,431,478 according to the manifest. These byte counts are delivery sizes, not GPU performance or B200 execution evidence. Ordinary BFS does not download the optional diagnostic alternatives. Deep investigation or a needed recipe contrast fetches those separately.

Downloads are streamed, size/SHA256 checked, extracted against a strict file list and atomically published under `~/.cache/multigpubfs/native` (or `$XDG_CACHE_HOME/multigpubfs/native`). Processes share a locked content-addressed cache. `MGBFS_NATIVE_CACHE` can point to an existing writable persistent location. A complete cache supports `MGBFS_NATIVE_OFFLINE=1`; a missing component fails explicitly in offline mode. The first preparation requires outbound GitHub HTTPS. A download failure leaves the original CayleyPy method intact.

Private CUDA/NCCL libraries belong to native child processes. Pip does not replace Torch's pinned NVIDIA packages. No NVIDIA driver is installed. Normal graph transitions reuse the selected runtime without another hardware-discovery process or network transfer.

## Checks and limits

```bash
multigpubfs doctor
# Extended controlled alternatives, with a 30-minute execution budget:
multigpubfs doctor --deep --total-timeout 1800
```

The diagnostic checks actual GPU kernels, NCCL, persistent workers and exact tiny BFS against independent CPU oracles. `working_configuration: READY` requires successful BFS and execution of its printed command. A cached file or a detected SM alone is not GPU acceptance. A native-count timeout retains its original evidence; agreeing independent CUDA probes allow isolated recipe checks to continue. See [server instructions](server-quickstart-ru.md) and [controlled build comparisons](diagnostic-build-contrasts.md).

Python3.10+ is required. Native execution requires Linux x86_64, GLIBC2.34+, GLIBCXX3.4.22+, CXXABI1.3.9+ and a compatible NVIDIA driver. Windows-native GPU execution is unsupported. BFS has no work deadline by default: it runs until graph completion, resource exhaustion or explicit cancellation. Positive max_seconds remains optional; startup/query/retirement timeouts and scheduler job limits are separate. Unknown graph families keep ordinary memory admission and measured tuning; profiles are not claimed to be globally optimal.

The current prefetch flow passed actual selected/all component preparation on a CPU-only host and physical **2×T4** parallel/persistent offline BFS. Earlier native delivery versions have the S11/S12, shard-engine and diagnostic-contrast acceptance recorded separately. Other SM components, including SM100/B200, SM103/B300 and SM120, have compilation/cubin inventory evidence; execution of this delivery flow on them remains unmeasured. [Current prefetch acceptance, hashes and scope](native-prefetch-acceptance-2026-10-15.md).

Import [the basic notebook](../examples/cayleypy_basic.ipynb) or [the pair sweep](../examples/cayleypy_pair_sweep.ipynb) from GitHub into Kaggle. Their input lists remain empty. For a loop over graphs, retain one `persistent_graph_workers()` context around the loop.

Release: [native-pip-20261019-v31](https://github.com/TryDotAtwo/MultiGPUBFS/releases/tag/native-pip-20261019-v31). Wheel SHA256 `9620fda20aff293e143f973d68916edbf91e487032aac7ccee8225fdad795670`; native-index SHA256 `423a491a802ec807b0ecc38fbdb694af5abf8738fbdba186c6d7943856972f8d`. Older pinned releases and their acceptance reports remain historical and are not overwritten.

Native CUDA/NCCL error context, retained failed-rank phase files and opt-in call tracing: [server failure evidence](native-failure-evidence.md).


В dev30 исправлено зависание восьми рангов при большом числе шардов: группы обмена NCCL ограничены без отключения автоподбора. Общий компонент dev29 переиспользуется из кэша; один раз скачивается новая CUDA-библиотека для нужного SM. Старые диагностические контроли с отличающимися исходниками исключены; рабочий SM и универсальная сборка остаются доступны.

## Work limits and persistent tuning (dev31)

Use NativeOptions(max_seconds=None), or omit max_seconds. The CLI accepts --seconds 0 for unlimited work. Diagnostics retain bounded stages so a hung probe cannot consume an entire allocation. A correct truncated prefix is inconclusive, not a wrong BFS result; unmeasured tuning is not a verified production pass.

Persistent workers cache tuning by graph, runtime, devices and configuration. Identical repeated calls reuse the profile with fresh admission checks; a changed graph or geometry may need new bounded probes. Online layer-size adaptation is distinct from repeating the full tuner.

[dev31 regression and 2-GPU acceptance](acceptance/unlimited-dev31.json). Eight-GPU server acceptance for this release is not inferred from these tests.
