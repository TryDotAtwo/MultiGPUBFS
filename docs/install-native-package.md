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
