# Install MultiGPUBFS with one pip command

The same GitHub wheel installs on Kaggle and a Linux x86_64 GPU server accessed through SSH. No bootstrap, compiler, separate CUDA toolkit installation, manual library paths, chmod, HF account or Kaggle Inputs are needed.

## Linux / SSH

```bash
python -m pip install "multigpubfs[cayleypy] @ https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-20261013-v4/multigpubfs-0.2.0.dev4-py3-none-linux_x86_64.whl#sha256=7b1a2de550bdd268c1628c314c01fc2f8fc09d315c0503e48de1fdff5a0d944c"
```

## Kaggle

Enable **Internet** and **GPU (2Г—T4)**, then run:

```python
%pip install "multigpubfs[cayleypy] @ https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-20261013-v4/multigpubfs-0.2.0.dev4-py3-none-linux_x86_64.whl#sha256=7b1a2de550bdd268c1628c314c01fc2f8fc09d315c0503e48de1fdff5a0d944c"
```

Use the package after installation:

```python
import multigpubfs, cayleypy
multigpubfs.enable_native()
graph = cayleypy.CayleyGraph(cayleypy.PermutationGroups.lrx(5), device="cpu")
result = graph.bfs()
```

`device="cpu"` creates the graph definition. Native BFS uses the visible GPUs. To select them explicitly, pass `native_options=multigpubfs.NativeOptions(devices=(0,1))` to `graph.bfs()`. Activation is explicit and reversible; importing the package alone does not replace CayleyPy methods.

The `[cayleypy]` extra installs CayleyPy and its ordinary Python dependencies using pip. CUDA runtime12.9.79 and NCCL2.30.7 are bundled under `multigpubfs/_native/lib`, with licenses and checked hashes. They are used only by native child processes; the wheel does not declare NVIDIA runtime package dependencies or replace Torch's pinned NCCL. Executable permissions are carried by the wheel.

## Scope and identity

Release: [native-pip-20261013-v4](https://github.com/TryDotAtwo/MultiGPUBFS/releases/tag/native-pip-20261013-v4). Package `0.2.0.dev4`; SHA256 `7b1a2de550bdd268c1628c314c01fc2f8fc09d315c0503e48de1fdff5a0d944c`. Pip verifies this hash through the URL fragment.

Requires Python3.10+, Linux x86_64 with the Ubuntu22.04-or-newer system ABI and a compatible NVIDIA driver. No NVIDIA driver is shipped in the wheel. This wheel compiles SM75/SM86 (T4/RTX3060); it does not claim B200/B300 acceptance or Windows-native support. Graph size remains limited by memory/time.

The native binaries were rebuilt on Ubuntu22.04 to avoid the old artifact's GLIBC2.38/GLIBCXX3.4.32 dependency. Exact runtime requirements and source/build hashes are recorded in the release manifest. [Acceptance evidence](pip-install-acceptance-2026-10-13.md) separates clean Linux installation, GPU results and hardware limits.

## Examples

Import [the basic notebook](../examples/cayleypy_basic.ipynb) or [the pair sweep](../examples/cayleypy_pair_sweep.ipynb) from this GitHub repository into Kaggle. Their dataset/kernel/competition/model input lists are empty. Results are written locally; no external artifact-storage account is needed.

The earlier pinned documentation and release are historical. Use this version's command; do not combine it with old global CUDA/NCCL installation commands. If an earlier command already changed the notebook environment, start a fresh Kaggle session/environment before testing this version.

For a sweep over different graphs, keep one `persistent_graph_workers()` context around the whole loop. See [geometry and catalogue acceptance](geometry-catalogue-acceptance-2026-10-13.md). The Coxeter bound is certified; unknown graph families retain ordinary memory admission and measured tuning. No profile is claimed to be a global optimum.
