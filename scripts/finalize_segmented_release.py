"""Update install documentation after an immutable wheel has been assembled."""
import argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TAG='native-pip-20261119-v36'
def main():
 p=argparse.ArgumentParser();p.add_argument('--wheel',type=Path,required=True);a=p.parse_args()
 wheel=a.wheel
 if wheel.name!='multigpubfs-0.2.0.dev36-py3-none-any.whl':raise RuntimeError('UNEXPECTED_WHEEL_NAME')
 sha=hashlib.sha256(wheel.read_bytes()).hexdigest()
 url=f'https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/{TAG}/{wheel.name}#sha256={sha}'
 text=f'''# Install MultiGPUBFS with pip

This is an experimental release for Linux x86_64 NVIDIA GPU jobs. The same
command works on a login host and a GPU worker; no target-host compilation is
required. After upgrading, start a new Python process.

```bash
python -m pip install --upgrade "multigpubfs[cayleypy] @ {url}"
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
    memory_backend="segmented24",
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
`legacy` backend remains the default for compatibility. The CLI also accepts
`--memory-backend segmented24` or `--memory-backend permutation_bitmap`.

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

[Release and checksums](https://github.com/TryDotAtwo/MultiGPUBFS/releases/tag/{TAG})
'''
 (ROOT/'docs/install-native-package.md').write_text(text)
 path=ROOT/'docs/experimental-memory-backends.md'
 data=path.read_text()
 old='''These modes require a native executable **and** CUDA library built from this
source. The existing dev35 GitHub native assets predate these modes. Installing
the source while continuing to use those old assets is not enough. The
launcher queries key-info and raises NATIVE_MEMORY_BACKEND_UNAVAILABLE
before starting BFS if the matching executable lacks the requested mode. It
also checks the actual backend in the receipt. A new cross-architecture
published native package is not part of this source change.'''
 new='''These modes require a matching native executable **and** CUDA library.
The dev36 GitHub release provides independently checksummed components for
SM75/80/86/89/90/100/103/120 and a universal build. Existing dev35 native
assets do not contain the new kernels. The launcher checks the requested
native capability before BFS and checks the actual backend in the receipt.
See [installation and offline prefetch](install-native-package.md).'''
 if old not in data:raise RuntimeError('DOC_BUILD_REQUIREMENT_CHANGED')
 data=data.replace(old,new)
 old='''A separately built SM86 wheel was installed offline in a clean environment;
CayleyPy persistent S3 and S4 passed with both modes on 2x RTX 3060. The
wheel is an unpublished test artifact. The complete CUDA library was also
compiled and its SM100 inventory checked in GitHub Actions; this is not a
B200 runtime acceptance or a published installable package.'''
 new='''An installed SM86 candidate and the selective offline-cache layout passed
CayleyPy persistent S3-S5 on 2x RTX 3060 with both modes. All advertised CUDA
components, including SM100 and SM103, were compiled and inventory-checked;
this is not a B200 or B300 runtime acceptance.'''
 if old not in data:raise RuntimeError('DOC_ACCEPTANCE_TEXT_CHANGED')
 path.write_text(data.replace(old,new))
 print('DOCS_READY wheel_sha256='+sha,flush=True)
if __name__=='__main__':main()
