"""Explicit, reversible CayleyPy BFS integration, analogous to beam search."""
from dataclasses import dataclass, field
from functools import wraps
import json
from pathlib import Path
import tempfile
import threading

from .graph_definition import from_cayleypy, GraphDefinition, _flatten
from .launch import run_graph


@dataclass(frozen=True)
class NativeOptions:
    output_root: str | Path | None = None
    devices: tuple[int, ...] | None = None
    max_seconds: int | None = None
    autotune: bool = True


@dataclass(frozen=True)
class NativeBfsResult:
    """Exact layer counts with explicitly separated compact state samples.

    This is not CayleyPy's full-state BfsResult: the native runtime does not
    retain all small layers, edge lists, hashes or path reconstruction data.
    """
    bfs_completed: bool
    layer_sizes: list[int]
    graph: object
    report: dict
    output: Path
    layers: dict = field(default_factory=dict)
    layer_samples: dict = field(default_factory=dict)
    backend: str = field(default="native", init=False)

    def num_vertices(self):
        return sum(self.layer_sizes)

    def diameter(self):
        if not self.bfs_completed:
            raise RuntimeError("BFS_INCOMPLETE: only completed_depth is known")
        return self.completed_depth

    @property
    def completed_depth(self):
        return len(self.layer_sizes) - 1

    def get_layer(self, layer_id):
        if layer_id not in self.layers:
            raise KeyError("Full layer not retained; use layer_samples for compact samples")
        return self.layers[layer_id].cpu().numpy()

    def last_layer(self):
        return self.get_layer(self.completed_depth)


_LOCK = threading.RLock()
_ORIGINAL = _INSTALLED = None


def _unsupported(kwargs):
    defaults = dict(max_layer_size_to_store=1000,
                    max_layer_size_to_explore=10**12, max_diameter=1000000,
                    return_all_edges=False, return_all_hashes=False,
                    stop_condition=None, disable_batching=False)
    for key, value in kwargs.items():
        if key == "start_states":
            continue
        if key not in defaults or value != defaults[key]:
            return key
    return None


def _search(original, graph, backend, options, output, kwargs):
    if backend not in ("native", "torch", "auto"):
        raise ValueError("backend must be native, torch or auto")
    if not isinstance(options, NativeOptions):
        raise TypeError("native_options must be NativeOptions")
    if backend == "torch":
        if output is not None:
            raise ValueError("output belongs to the native backend")
        return original(graph, **kwargs)
    unsupported = _unsupported(kwargs)
    if unsupported:
        if backend == "auto" and output is None:
            return original(graph, **kwargs)
        raise NotImplementedError(f"Native BFS does not support {unsupported}; use backend='torch'")
    definition = from_cayleypy(graph)
    if kwargs.get("start_states") is not None:
        start = _flatten(kwargs["start_states"])
        if len(start) != len(definition.start):
            if backend == "auto" and output is None:
                return original(graph, **kwargs)
            raise NotImplementedError("Native BFS supports a single decoded start state")
        value = json.loads(definition.to_json())
        value["start"] = start
        definition = GraphDefinition.from_dict(value)
    if output is None:
        root = Path(options.output_root or Path.cwd() / "bfs-results")
        root.mkdir(parents=True, exist_ok=True)
        # The parent persists for the receipt; run_graph creates its child.
        output = Path(tempfile.mkdtemp(prefix="run-", dir=root)) / "result"
    output = Path(output).absolute()
    from . import generic_session
    import contextlib
    # Share rank contexts between admission, pilots and the final BFS.
    # An existing sweep owns its pool; a single API call owns a scoped pool.
    scope = (generic_session.persistent_graph_workers()
             if options.autotune and generic_session._active is None
             else contextlib.nullcontext())
    with scope:
        report = run_graph(definition, output, devices=options.devices,
                           max_seconds=options.max_seconds, autotune=options.autotune)
    import torch
    snapshot = json.loads((output / "states.json").read_text())
    sizes = report["layer_sizes"]
    samples, layers = {}, {}
    for depth, name in ((len(sizes)-1, "current"), (len(sizes)-2, "previous_small")):
        rows = snapshot.get(name, [])
        if depth < 0 or not rows:
            continue
        tensor = torch.tensor(rows, dtype=torch.int64)
        if definition.action["kind"] == "matrix":
            tensor = tensor.reshape(-1, definition.action["rows"], definition.action["cols"])
        samples[depth] = tensor
        if len(rows) == sizes[depth]:
            layers[depth] = tensor
    return NativeBfsResult(report["status"] == "COMPLETE", sizes,
                           getattr(graph, "definition", graph), report, output, layers, samples)


def bfs(graph, *, backend="native", native_options=None, output=None, **kwargs):
    """Dispatch once without modifying CayleyPy."""
    from cayleypy import CayleyGraph
    with _LOCK:
        original = _ORIGINAL or CayleyGraph.bfs
    return _search(original, graph, backend, native_options or NativeOptions(), output, kwargs)


def enable_native(options=None, *, default_backend="native"):
    """Enable graph.bfs() explicitly; importing multigpubfs has no side effects."""
    from cayleypy import CayleyGraph
    global _ORIGINAL, _INSTALLED
    if default_backend not in ("native", "auto", "torch"):
        raise ValueError("Invalid default_backend")
    selected = options or NativeOptions()
    if not isinstance(selected, NativeOptions):
        raise TypeError("options must be NativeOptions")
    with _LOCK:
        if _ORIGINAL is not None and CayleyGraph.bfs is not _INSTALLED:
            raise RuntimeError("CayleyGraph.bfs was replaced by another integration")
        if default_backend == "native":
            from .native_distribution import prepare_for_activation
            prepare_for_activation()
        original = _ORIGINAL or CayleyGraph.bfs
        @wraps(original)
        def installed(graph, **kwargs):
            return _search(original, graph, kwargs.pop("backend", default_backend),
                           kwargs.pop("native_options", selected), kwargs.pop("output", None), kwargs)
        _ORIGINAL, _INSTALLED = original, installed
        CayleyGraph.bfs = installed


def disable_native():
    """Restore the original method without overwriting another integration."""
    from cayleypy import CayleyGraph
    global _ORIGINAL, _INSTALLED
    with _LOCK:
        if _ORIGINAL is not None:
            if CayleyGraph.bfs is not _INSTALLED:
                raise RuntimeError("CayleyGraph.bfs was replaced by another integration")
            CayleyGraph.bfs = _ORIGINAL
            _ORIGINAL = _INSTALLED = None
