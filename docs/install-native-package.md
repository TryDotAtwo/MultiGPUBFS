# Install the verified native development package

Latest implementation: `73cdef732ba116c4540491b405d659a103ee91e4`. Linux x86_64, CUDA13 runtime and a compatible driver are required. The wheel contains SM86 and PTX targets; actual acceptance is one/two RTX3060, not Blackwell or physical8/128GPU scaling.

```bash
python -m pip install --force-reinstall "https://huggingface.co/datasets/TryDotAtwo/multigpubfs-native-releases/resolve/bd87478b1df62b573f655a25bb25cd2efd6ac7f6/linux-x86_64-sm86-cuda13.2/73cdef732ba116c4540491b405d659a103ee91e4/multigpubfs-0.2.0.dev0-py3-none-linux_x86_64.whl" "nvidia-nccl-cu13==2.30.7"
```

Wheel SHA256: `dc6f7215d18d757c58895f094c7fc02c6cc0b5151ae51a5798b61e994cb4be66`. Anonymous public readback matched the hash. Native artifact identity is checked before launch. Development wheels share a version, so force reinstall is required when changing immutable artifacts.

```python
from multigpubfs import run_graph
report = run_graph(cayleypy_graph, "results")
```

Pass either a CayleyPy0.2.0 CayleyGraph object, its CayleyGraphDef, or multigpubfs.GraphDefinition. CayleyPy itself is optional for explicit GraphDefinition use; install `cayleypy==0.2.0` separately for its catalog. All visible local GPUs are selected by default. Use the same API under the external rank environment for cluster launch. See [unified launch](unified-launch.md), [external ranks](external-rank-launch.md), and [current acceptance audit](universal-acceptance-2026-10-09.md).

The fresh installation passed one/two-GPU exact graph, deadline/cancel/resource, network and collective tuning gates. Actual CayleyPy wrappers passed CPU neighbor comparisons and installed native complete BFS for inverse-closed permutations, directed permutations and modular rectangular matrices. General graph state identity is the exact integer vector, not its hash. Compact output contains a maximum1000-state last-layer sample and the previous layer only if globally smaller than1000. INCOMPLETE means enumeration stopped before frontier exhaustion.

## Historical artifact records

The records below preserve previous versions and their measured acceptance. Use the latest artifact above for the current implementation.

# Install the verified development wheel

Linux x86_64 with compatible NVIDIA driver, CUDA runtime12 and NCCL2 shared libraries:

```sh
python -m pip install "https://huggingface.co/datasets/TryDotAtwo/multigpubfs-native-releases/resolve/8b2680daf1cd1f932cde41204bec76b2a8b3c3fd/linux-x86_64-sm86/3aaf1862dab3d6cf7dc8f6e9d3642d27ebf25c2d/multigpubfs-0.2.0.dev0-py3-none-linux_x86_64.whl"
```

```python
from multigpubfs import run_graph
report = run_graph(cayleypy_graph, "results")
```

The package contains the native executable, CUDA library and manifest. It verifies their SHA256 before launch and requires no Rust/CUDA compilation on the target machine. Default placement inventories local visible GPUs, independently admits free VRAM, and measures bounded shard/batch profiles. External-rank environments use the same API with network bootstrap; see external-rank-launch.md.

Artifact SHA256: `8623ffbf9908d79971dae96ea56a357de2fb5f1c0476878a43e587b59eae1c6a`.
Source compiled/packaged functional checkpoint: `3aaf1862dab3d6cf7dc8f6e9d3642d27ebf25c2d`.
The public URL was downloaded without credentials and checked against the on-rental wheel checksum. A clean installation outside the source checkout passed native GPU smoke, regular tuning and external network tuning gates. CUTLASS redistribution notice is bundled.

Actual GPU acceptance is one/two RTX3060. The library contains SM86 code and PTX; this is not B200/B300 or other-architecture hardware acceptance. Separate physical nodes and8/128 GPU configurations are unverified. Native runtime dependencies are external; this wheel does not install drivers. General directed retained-history, automatic specialized compact-history dispatch and survivor-only transport boundaries are documented in unified-launch.md.

## Packed-candidate CUDA13 release

For Linux x86_64 with CUDA runtime13, NCCL2 and a compatible driver, the updated packed-candidate wheel is available without a token:

```bash
python -m pip install "https://huggingface.co/datasets/TryDotAtwo/multigpubfs-native-releases/resolve/34b7b907773f83dadf89976ef1743e7f3a626f83/linux-x86_64-sm86-cuda13.2/ceb125a47bd4d5c0224404c9835a57c9e8de8bfa/multigpubfs-0.2.0.dev0-py3-none-linux_x86_64.whl"
```

SHA256: `85f8570c1e2753d7a21833a439e4d76c4baebc5a94f3dcd4ada94b8f6ef3c998`. The CUDA12 wheel above remains the earlier implementation. The updated wheel passed clean two-RTX3060 installation, exact layer/resource/collision tests, network rank control and four-profile autotuning. CUDA13 compatibility and these GPU tests do not establish acceptance on Blackwell or physical multihost clusters.

## Automatic three-bank build (latest verified implementation)

Source implementation: `9a30c0212063a6f595b251ef1733ef0a80c20b49`. This CUDA13.2/SM86+PTX artifact passed clean installed one/two-GPU exact graph, resource/deadline/cancel, external-rank network and collective tuning gates. Other GPU models and physical multi-host configurations remain unverified.

```bash
python -m pip install --force-reinstall "https://huggingface.co/datasets/TryDotAtwo/multigpubfs-native-releases/resolve/a04c7ba9b108981678c3140de7258c865fe66fad/linux-x86_64-sm86-cuda13.2/9a30c0212063a6f595b251ef1733ef0a80c20b49/multigpubfs-0.2.0.dev0-py3-none-linux_x86_64.whl" "nvidia-nccl-cu13==2.30.7"
```

The host must already provide compatible CUDA runtime13 and an NVIDIA driver. Wheel SHA256: `9f62296da744202d46e76743b7af88ec8769b03a96fffbd41a92f9913e090a11`. The embedded manifest checks the executable and native library before launch. Force reinstall is intentional because these development artifacts share a package version; immutable URLs and source manifests distinguish builds.
