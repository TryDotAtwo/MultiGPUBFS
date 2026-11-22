# Experimental segmented and bitmap BFS

Version 0.2.0.dev44 provides opt-in segmented hash (`segmented24`) and permutation bitmap (`permutation_bitmap`) backends. The existing `legacy` backend remains the default. This is a development release: see the [GPU acceptance and limitations](acceptance/2026-11-22-dev44-advisory.md) before relying on it for a large calculation.

## Install

On Linux x86-64, in the Python environment used by the job:

```bash
python -m pip install --upgrade "multigpubfs[cayleypy] @ https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/b2ed1712e6c79f7402b45f961355899cb2e80c3f/wheels/multigpubfs-0.2.0.dev44-py3-none-any.whl#sha256=29b1ffdf6d31b194184e0d377e83e22b0f8b8c7b1355b36ff7b09eda5bdfd10b"
```

Alternatively, install the verified source snapshot directly from GitHub (requires Git and downloads the repository):

```bash
python -m pip install --upgrade "multigpubfs[cayleypy] @ git+https://github.com/TryDotAtwo/MultiGPUBFS.git@5ed4b90e6ab3ff8ab16c78e36722e9c41fb5cd70"
```

The wheel is smaller to download. Its SHA-256 is `29b1ffdf6d31b194184e0d377e83e22b0f8b8c7b1355b36ff7b09eda5bdfd10b`; pip verifies it via the URL fragment. The public URL was installed in a clean environment and that installation was exercised on eight RTX 3060 GPUs; see the acceptance report for the exact scope. GPU-independent prefetch dry-run was also checked. The Git source command is pinned to the matching dev44 source snapshot.

Start a new Python process after updating. CUDA native components are selected from the visible GPUs and downloaded into the cache on first use. The first download includes a shared runtime of about 304 MB, the executable and the selected architecture component. Later jobs reuse checksum-verified cached files.

## Prefetch on a login node without GPUs

Choose the targets needed on the compute nodes, or use `--all` to include every available target and the universal component:

```bash
export MGBFS_NATIVE_CACHE="$HOME/.cache/multigpubfs"
python -m multigpubfs prefetch --sm 100,120 --cache "$MGBFS_NATIVE_CACHE"
# Alternative: python -m multigpubfs prefetch --all --cache "$MGBFS_NATIVE_CACHE"
```

In the compute job set the **same shared path** and, once the necessary components are cached, require offline use:

```bash
export MGBFS_NATIVE_CACHE="$HOME/.cache/multigpubfs"
export MGBFS_NATIVE_OFFLINE=1
```

Use `python -m multigpubfs prefetch --help` for supported targets, `--dry-run` and download sizes. If the chosen cache lacks a needed component, offline mode reports an error rather than downloading it.

## CayleyPy API and persistent workers

```python
from cayleypy import CayleyGraph, PermutationGroups
from multigpubfs import NativeOptions, enable_native, persistent_graph_workers

enable_native(NativeOptions(
    memory_backend="segmented24",   # alternative for supported permutation actions: "permutation_bitmap"
    memory_preset="balanced",       # alternative: "max_capacity"
    segment_strategy="warp",        # alternatives: "atomic", "batch"
    devices=None,                   # all visible GPUs; or tuple(range(8)) if eight are visible
))

with persistent_graph_workers():
    for n in (4, 5):
        graph = CayleyGraph(PermutationGroups.coxeter(n), device="cpu")
        result = graph.bfs()
        print(n, result.report["status"], result.layer_sizes, result.output)
        if result.bfs_completed:
            print("diameter:", result.diameter())
```

`device="cpu"` here constructs the CayleyPy graph definition on the CPU; the selected native BFS uses the requested GPUs. The persistent context reuses workers across calls, but it does not promise that every different graph skips all admission and tuning work. `max_capacity` compares measured candidates using an estimated admitted storage capacity, then speed; it is a heuristic, not a proof of the maximum graph that fits. The live memory check still runs, and changes in effective free memory can trigger retuning.

Each segment/bank permits up to `segment_rows=2**32` rows by default; `NativeOptions(segment_rows=...)` accepts a smaller value. Multiple segments can be allocated subject to the actual memory budget. This local segment limit is not a promise that a particular graph fits. `bfs_completed=False` means only the published completed prefix is valid; inspect `result.report` and `result.output` for details. The native result exposes exact layer counts and retained samples, not CayleyPy's complete edge lists or all states.

The provided binaries include SM100 and SM120 components, but the tests in the acceptance report were performed on RTX 3060 (SM86), not B200. A physically filled `2**32`-row segment and a complete S14 on B200 have not been tested.
