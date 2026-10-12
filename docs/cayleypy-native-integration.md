# CayleyPy integration

The adapter is implemented in [cayleypy_backend.py](../multigpubfs/cayleypy_backend.py), exported by [__init__.py](../multigpubfs/__init__.py), and covered by [dispatch tests](../tests/test_cayleypy_backend.py).

With an installed Linux CUDA native package ([installation](install-native-package.md)):

```python
import multigpubfs, cayleypy
multigpubfs.enable_native()
graph = cayleypy.CayleyGraph(cayleypy.PermutationGroups.lrx(5), device="cpu")
result = graph.bfs()
```

`enable_native()` explicitly installs a process-wide wrapper on `cayleypy.CayleyGraph.bfs`. Importing the package alone does not modify CayleyPy. The wrapper keeps the original method and sends default calls through `from_cayleypy()` and `run_graph()` to the native GPU runtime. `graph.bfs(backend="torch")` calls the original implementation; `disable_native()` restores it. Another integration replacing the method is detected rather than silently overwritten.

`device="cpu"` keeps CayleyPy's generator/start description on CPU. It does not enumerate the graph and does not select the native BFS computation device. Native GPU selection is controlled by `NativeOptions(devices=(0, 1))`; the default delegates device selection to the native launcher.

`NativeBfsResult` exposes exact `layer_sizes`, completion status, a report and output path. Compact state samples are separate from full retained layers. It is not a drop-in implementation of every CayleyPy `BfsResult` operation: full edges, all hashes and path reconstruction are unsupported. `backend="auto"` can use the original method for unsupported options, but a failed native run is not silently restarted on CPU.

For repeated-label graphs, the maximum BFS depth is the eccentricity of the supplied start; the `diameter()` compatibility method does not establish the global graph diameter. It rejects INCOMPLETE runs.

One-shot `multigpubfs.bfs(graph)` is available without patching the class. Adapter tests use mocked native execution and do not constitute a new GPU benchmark.
