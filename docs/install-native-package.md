# Install MultiGPUBFS from GitHub

Our source, notebook setup and native wheel are published in this GitHub repository and its Releases. No Kaggle utility, datasets or external artifact-storage account is needed. Standard Python dependencies (CayleyPy and NVIDIA runtime packages) are installed using pip.

## Kaggle: no inputs, no compilation

Use [the basic notebook source](../examples/cayleypy_basic.ipynb) or [the pair-sweep notebook source](../examples/cayleypy_pair_sweep.ipynb). Import the GitHub notebook into Kaggle (File → Import Notebook), enable **Internet** and **GPU (2×T4)**, and Run All. The notebook has a setup cell that fetches the bootstrap at a pinned GitHub commit, checks its SHA256 and installs the checksummed wheel from GitHub Releases. All dataset/kernel/competition/model source lists are empty.

For an existing Kaggle notebook, run this setup cell in a fresh environment before importing the package:

```python
import hashlib, urllib.request
url = "https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/591efb9136c25e269c4ac1d36bc391b87f285997/examples/kaggle_native_setup.py"
with urllib.request.urlopen(url, timeout=90) as response:
    source = response.read()
assert hashlib.sha256(source).hexdigest() == "d528afd0422f8fde9c614b79b81df14404cd7a7737e6ccc14700522a3305870b", "GitHub bootstrap checksum mismatch"
namespace = {}
exec(compile(source, "github-native-setup", "exec"), namespace)
namespace["setup"]()
```

Then use the CayleyPy API:

```python
import multigpubfs, cayleypy
multigpubfs.enable_native()
graph = cayleypy.CayleyGraph(cayleypy.PermutationGroups.lrx(5), device="cpu")
result = graph.bfs()
```

The setup isolates native CUDA/NCCL under `/tmp/multigpubfs-runtime`; it preserves Kaggle's Torch-pinned NCCL installation. `device="cpu"` creates graph definitions; the native BFS uses all visible GPUs. Select GPUs explicitly with `native_options=multigpubfs.NativeOptions(devices=(0,1))`. Activation is explicit; import alone does not patch CayleyPy.

If the older global pip command already changed Kaggle's dependencies, use a fresh notebook session/environment. A Python-only kernel restart does not restore installed dependencies. Do not replace Kaggle's GPU Torch with CPU Torch.

## Release identity and supported hardware

[Native release](https://github.com/TryDotAtwo/MultiGPUBFS/releases/tag/native-419fa34f-cuda12.9-sm75-sm86): source `419fa34f6ab4f35543bbd37f918faca85aed6c8f`, Linux x86_64, CUDA12.9, SM75/SM86 (T4/RTX3060). Wheel SHA256 `f85b5d3f90bb6c870bc60b3fbc76ef920284a83a443185090aa81643f3bee9aa`. Anonymous GitHub asset readback matched this checksum. The runtime also verifies native executable/library hashes. This artifact is not a B200/B300 binary. A compatible NVIDIA driver and host libraries are required.

The unchanged native wheel passed clean one/twoRTX3060 acceptance and physical2×T4 independent CPU comparisons for LRX/M2. The GitHub-only distribution path is separately checked on Kaggle; its current receipt is recorded in [the pair-sweep acceptance report](pair-sweep-acceptance-2026-10-12.md). Publication is not itself a new target-hardware acceptance.

## Standalone Linux: fresh virtual environment

This route is for a separate Linux environment, not Kaggle's preinstalled environment. CPU Torch here is only for graph definitions; native BFS runs on GPUs.

```bash
python -m venv bfs-venv
source bfs-venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "cayleypy==0.2.0" "nvidia-cuda-runtime-cu12==12.9.79" "nvidia-nccl-cu12==2.30.7"
python -m pip install --no-deps "https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-419fa34f-cuda12.9-sm75-sm86/multigpubfs-0.2.0.dev0-py3-none-linux_x86_64.whl"
python -c "import pathlib,sysconfig; (pathlib.Path(sysconfig.get_paths()['purelib'])/'multigpubfs/_native/bin/mgbfs').chmod(0o755)"
```

CayleyPy0.2.0 is the tested API dependency; arbitrary GitHub HEAD is not the same validation target. See [adapter details](cayleypy-native-integration.md) for option and result contracts.

## Older document links

The document URL pinned to commit165e6c499bea5c324ec22c5e3c9d75a8a25d4b93 is immutable and references an older wheel without enable_native(). Use this current document and the GitHub Release above. Historical release provenance is retained in a separate archive document; it is not an installation requirement.
