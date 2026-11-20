# Install MultiGPUBFS with pip

This is an experimental release for Linux x86_64 NVIDIA GPU jobs. The same
command works on a login host and a GPU worker; no target-host compilation is
required. After upgrading, start a new Python process.

```bash
python -m pip install --upgrade "multigpubfs[cayleypy] @ https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-20261120-v38/multigpubfs-0.2.0.dev38-py3-none-any.whl#sha256=6a7c41fae04209f00429ac35a035fe4980c687c82fe4d37c09af4562db142097"
```

For Kaggle use the same quoted argument after `%pip install --upgrade`,
with Internet and GPU enabled. Import alone does not download native files.

## Prefetch on a login host without a GPU

Select all SM versions you expect, or use `--all` for every target including
the universal fallback. The shared location must be visible and writable to
the later GPU jobs.

```bash
export MGBFS_NATIVE_CACHE=/path/to/shared/multigpubfs-cache
python -m multigpubfs prefetch --sm 86,100,120
# Alternatively: python -m multigpubfs prefetch --all
```

Use `--all --dry-run` to inspect download sizes. The shared CUDA/NCCL archive
is content-addressed and reused if already cached from the prior package.
On every job set the same cache path and `MGBFS_NATIVE_OFFLINE=1` to forbid
downloads; missing preloaded components fail explicitly.

## CayleyPy persistent BFS

```python
import cayleypy
import multigpubfs
from multigpubfs.generic_session import persistent_graph_workers

multigpubfs.enable_native(multigpubfs.NativeOptions(
    devices=(0, 1),  # set to the logical devices allocated to this job
    memory_backend="segmented",
    segment_rows=2**32,  # maximum rows per bank, configurable
))
try:
    with persistent_graph_workers():
        for n in (4, 5, 6):
            graph = cayleypy.CayleyGraph(cayleypy.PermutationGroups.coxeter(n), device="cpu")
            result = graph.bfs()
            print(n, result.bfs_completed, result.layer_sizes)
            if result.bfs_completed:
                print("diameter:", result.diameter())
            else:
                print("last completed depth:", result.completed_depth)
finally:
    multigpubfs.disable_native()
```

`device="cpu"` constructs a CayleyPy graph definition; the native BFS uses
the GPUs specified in `NativeOptions`. The alternate exact
`permutation_bitmap` backend supports permutation actions only. The existing
`legacy` backend remains the default for compatibility. The CLI accepts `--memory-backend segmented --segment-rows 4294967296`
or `--memory-backend permutation_bitmap`. Changing the configured maximum
does not force allocation of a full segment on a smaller GPU.

`INCOMPLETE` is not a completed graph or diameter. Segmented storage does
not guarantee that a graph fits. The bitmap covers the entire multiset
permutation domain. A cache hit, successful compilation, or a reported SM
does not establish correctness or speed on a target GPU.

## Diagnosis and limits

```bash
python -m multigpubfs doctor
python -m multigpubfs doctor --deep --total-timeout 1800
```

The new backends have exact small-oracle and S12/S13 evidence on RTX GPUs;
SM100/B200 and SM103/B300 were compiled but have not been hardware-accepted
for this release. More than 2^32 simultaneously resident states on one GPU
and complete S14 remain unverified. Python 3.10+, Linux x86_64, a compatible
NVIDIA driver and the GLIBC/GLIBCXX versions in the native manifest are
required. [Architecture, timings and limitations](experimental-memory-backends.md).

[Release and checksums](https://github.com/TryDotAtwo/MultiGPUBFS/releases/tag/native-pip-20261120-v38)
