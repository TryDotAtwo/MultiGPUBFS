# Install MultiGPUBFS with one pip command

The same GitHub wheel installs on Kaggle and a Linux x86_64 GPU server accessed through SSH. No bootstrap, compiler, separate CUDA toolkit installation, manual library paths, chmod, HF account or Kaggle Inputs are needed.

## Linux / SSH

```bash
python -m pip install "multigpubfs[cayleypy] @ https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-20261015-v18/multigpubfs-0.2.0.dev18-py3-none-linux_x86_64.whl#sha256=1eda27dcd590528dc972dedffcf978c26854832e70703cb8e3d3233910769fa9"
```

## Kaggle

Enable **Internet** and **GPU (2×T4)**, then run:

```python
%pip install "multigpubfs[cayleypy] @ https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-20261015-v18/multigpubfs-0.2.0.dev18-py3-none-linux_x86_64.whl#sha256=1eda27dcd590528dc972dedffcf978c26854832e70703cb8e3d3233910769fa9"
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

Release: [native-pip-20261015-v18](https://github.com/TryDotAtwo/MultiGPUBFS/releases/tag/native-pip-20261015-v18). Package `0.2.0.dev18`; SHA256 `1eda27dcd590528dc972dedffcf978c26854832e70703cb8e3d3233910769fa9`. Pip verifies this hash through the URL fragment.

Requires Python3.10+, Linux x86_64 with GLIBC 2.34+, GLIBCXX 3.4.22+ and CXXABI 1.3.9+ and a compatible NVIDIA driver. No NVIDIA driver is shipped in the wheel. The ordinary native library uses Shared CUDA runtime linkage and native SM75/80/86/89/90/100/103/120 cubins plus ordinary compute80 PTX. The CUDA loader selects compatible code inside the same wheel; users do not select or rebuild an architecture package. Legacy alternatives are isolated diagnostic contrasts. This release has real 2×T4 acceptance; B200/B300/SM120 have compiled-code coverage, not execution acceptance for this wheel. Windows-native execution is unsupported. Graph size remains limited by memory/time.

The native binaries were rebuilt on Ubuntu22.04 to avoid the old artifact's GLIBC2.38/GLIBCXX3.4.32 dependency. Exact runtime requirements and source/build hashes are recorded in the release manifest. [Acceptance evidence](pip-install-acceptance-2026-10-13.md) separates clean Linux installation, GPU results and hardware limits.

## Examples

Import [the basic notebook](../examples/cayleypy_basic.ipynb) or [the pair sweep](../examples/cayleypy_pair_sweep.ipynb) from this GitHub repository into Kaggle. Their dataset/kernel/competition/model input lists are empty. Results are written locally; no external artifact-storage account is needed.

The earlier pinned documentation and release are historical. Use this version's command; do not combine it with old global CUDA/NCCL installation commands. If an earlier command already changed the notebook environment, start a fresh Kaggle session/environment before testing this version.

For a sweep over different graphs, keep one `persistent_graph_workers()` context around the whole loop. See [geometry and catalogue acceptance](geometry-catalogue-acceptance-2026-10-13.md). The Coxeter bound is certified; unknown graph families retain ordinary memory admission and measured tuning. No profile is claimed to be a global optimum.

## Server component diagnostic

For a single broad server investigation run `python -m multigpubfs.diagnostics --deep` from a fresh process inside the assigned GPU job. It has a 30-minute execution budget, four same-kernel CUDA runtime/native-code build comparisons and one metadata/log archive. [Coverage and interpretation](diagnostic-build-contrasts.md); [measured acceptance](diagnostic-build-acceptance-2026-10-14.md). The shared multi-architecture library is the normal default (its historical identifier remains `shared-sm120`). Production API/resident/BFS checks and broad modes run before legacy contrasts. Owned-group cleanup is bounded in both normal and deep doctor, with retained evidence and no false PASS for an unreaped child.

Run `multigpubfs doctor` after installation. It checks each visible local GPU, CUDA/native libraries, resident-worker file requests, NCCL and a tiny exact BFS, then writes a readable report and diagnostic ZIP. For Python-only submissions use `from multigpubfs.diagnostics import run_diagnostics; run_diagnostics()`. [Instructions](diagnostics.md) and [measured acceptance](diagnostics-acceptance-2026-10-13.md).

The current diagnostic also probes NVIDIA Driver API directly without `nvidia-smi`. [Driver diagnostic acceptance](driver-diagnostics-acceptance-2026-10-14.md). After upgrading inside an already-running Python session, invoke `multigpubfs doctor` in a fresh process; previously imported Python code may otherwise remain cached. Python-only job submissions can use `subprocess.run([sys.executable, "-m", "multigpubfs.diagnostics"])`.

The current `doctor` includes broad user-space evidence, isolated loader contrasts and `analysis.json`/`analysis.txt`: [forensic diagnostic guide](forensic-diagnostics.md), [acceptance](forensic-diagnostics-acceptance-2026-10-14.md). The primary failure, observed findings, hypotheses, exclusions and inaccessible evidence are separate.

For a ready update-and-check command, Python-only job submission and actual calculation see [server quickstart](server-quickstart-ru.md). [Current production acceptance](multiarch-native-acceptance-2026-10-15.md).

A failed or timed-out native GPU count no longer ends both diagnostic modes early when independent driver/runtime probes agree. The original failure remains visible; only actual exact BFS and execution of the printed command can produce `working_configuration: READY`. Architecture coverage and separate native/CUDA/Python source identities are stored in `architecture.json` and the report.
