"""Small public interface for repeated exact BFS and path searches."""

from contextlib import nullcontext
from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading

from . import generic_session
from .artifact_paths import artifact_path
from .graph_definition import GraphDefinition, _flatten, _normalize_graph
from .launch import (
    _PACKET_ADDR, _PATH_COMPACT, _PATH_SEGMENTED_NEAREST, _SEGMENTED_NEAREST_CAPABILITY,
    _compact_path_capability, _require_memory_backend, run_graph,
)
from .memory_admission import resolve_traversal_memory
from .native_distribution import native_runtime
from .process_control import query as native_query
from .process_control import startup_timeout, startup_timeout_error


_session_lock = threading.RLock()
_active_session = None


def _preset(value):
    if value == "max_capacity":
        return "maxmemory"
    if value not in ("balanced", "maxmemory"):
        raise ValueError("INVALID_MEMORY_PRESET")
    return value


def _validate_compact_path_request(world, distributed_enabled):
    if world is not None and world < 2:
        raise ValueError("NATIVE_PATH_COMPACT_REQUIRES_MULTIPLE_GPUS")
    if not distributed_enabled:
        raise ValueError("NATIVE_PATH_COMPACT_DISTRIBUTED_DISABLED")


def _require_compact_path_capabilities(capabilities, *, direction, window):
    bidirectional = direction == "bidirectional"
    nearest = "graph_path_nearest_bidirectional_v1" if bidirectional else "graph_path_nearest_distributed_v1"
    required = (_compact_path_capability(bidirectional), nearest)
    if window:
        required += (f"graph_path_window_{direction}_v1",)
    if any(capabilities.get(key) is not True for key in required):
        raise RuntimeError("NATIVE_PATH_COMPACT_UNAVAILABLE: rebuild the native executable and CUDA library")


def _record_graph(output, graph):
    path = output / "graph.json"
    if not path.exists():
        path.write_text(graph.to_json(), encoding="utf-8")


@dataclass(frozen=True)
class TraversalResult:
    output: Path
    graph: GraphDefinition
    report: dict
    layers: dict[int, list[list[int]]]
    layer_samples: dict[int, list[list[int]]]

    @property
    def layer_sizes(self):
        return self.report["layer_sizes"]

    @property
    def bfs_completed(self):
        return self.report["status"] == "COMPLETE"

    @property
    def completed_depth(self):
        return len(self.layer_sizes) - 1

    def num_vertices(self):
        return sum(self.layer_sizes)

    def get_layer(self, depth):
        return self.layers[depth]

    def last_layer(self):
        return self.get_layer(self.completed_depth)


def _traversal_result(graph, output, report, policy):
    snapshot = json.loads((output / "states.json").read_text(encoding="utf-8"))
    sizes = report["layer_sizes"]
    layers, samples = {}, {}
    for depth, key in ((len(sizes) - 1, "current"), (len(sizes) - 2, "previous_small")):
        rows = snapshot.get(key, [])
        if depth >= 0 and rows:
            samples[depth] = rows
            if len(rows) == sizes[depth]:
                layers[depth] = rows
    _record_graph(output, graph)
    report["session_policy"] = policy
    temporary = output / "report.json.session-tmp"
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output / "report.json")
    return TraversalResult(output, graph, report, layers, samples)


def _explicit_transport_env(env, capabilities, world, transport):
    if world < 2:
        raise ValueError("TRAVERSAL_TRANSPORT_REQUIRES_MULTIPLE_RANKS")
    feature = {
        "exact": "segmented_exact_exchange_v1",
        "compact": "segmented_packet_transport_v1",
    }[transport]
    if capabilities.get(feature) is not True:
        raise RuntimeError(f"TRAVERSAL_TRANSPORT_UNAVAILABLE: {transport}")
    env = dict(env, MGBFS_SEGMENT_EXACT_EXCHANGE="1")
    env.pop("_MGBFS_PRESET_IMPLICIT_PACKET", None)
    if transport == "exact":
        env.pop("MGBFS_SEGMENT_PACKET_CONTROL_ADDR", None)
        env.pop("MGBFS_SEGMENT_PACKET_BIND_ADDR", None)
    return env


def _traversal_exchange_env(env, capabilities, world, transport, preset, autotune, packet_address, *,
                            external=False):
    env = dict(env)
    if transport is not None:
        env = _explicit_transport_env(env, capabilities, world, transport)
    packet_key = "MGBFS_SEGMENT_PACKET_CONTROL_ADDR"
    implicit = (
        transport is None and preset == "maxmemory" and world > 1 and packet_address is not None
        and capabilities.get("segmented_packet_transport_v1") is True
        and all(key not in env for key in (
            packet_key, "MGBFS_SEGMENT_PACKET_BIND_ADDR", "MGBFS_SEGMENT_EXACT_EXCHANGE",
        ))
    )
    if (implicit or transport == "compact") and packet_key not in env:
        if packet_address is None:
            raise RuntimeError("CLUSTER_COMPACT_TRANSPORT_REQUIRES_PACKET_PORT")
        env[packet_key] = packet_address()
        env["MGBFS_SEGMENT_EXACT_EXCHANGE"] = "1"
        if implicit:
            env["_MGBFS_PRESET_IMPLICIT_PACKET"] = "1"
    if (external or not autotune) and world > 1 and capabilities.get("segmented_exact_exchange_v1") is True:
        env.setdefault("MGBFS_SEGMENT_EXACT_EXCHANGE", "1")
    return env


class BfsSession:
    """Reuse native ranks and their buffers across sequential calls in one context."""

    def __init__(self, *, gpus=None, devices=None, output_root=None, tune="auto",
                 executable=None):
        if gpus is not None and (type(gpus) is not int or gpus < 1):
            raise ValueError("INVALID_GPU_COUNT")
        if devices is not None:
            devices = tuple(devices)
            if (gpus is not None or not devices or len(set(devices)) != len(devices)
                    or any(type(device) is not int or device < 0 for device in devices)):
                raise ValueError("INVALID_DEVICES")
        if tune not in ("auto", "off", "full"):
            raise ValueError("INVALID_TUNING_POLICY")
        self._gpus = gpus
        self._devices = devices
        self._output_root = Path(output_root) if output_root is not None else artifact_path("bfs-results")
        self._tune = tune
        self._executable = executable
        self._local_devices = None
        self._packet_address = None
        self._lock = threading.RLock()
        self._scope = None
        self._pool = None
        self._path_ran = False

    def __enter__(self):
        global _active_session
        from .cluster import current_cluster

        with _session_lock:
            if _active_session is not None:
                raise RuntimeError("BFS_SESSION_ALREADY_OPEN")
            pool = generic_session._active
            scope = nullcontext(pool)
            if pool is None and current_cluster() is None:
                scope = generic_session.persistent_graph_workers()
            self._pool = scope.__enter__()
            self._scope = scope
            _active_session = self
        return self

    def __exit__(self, *exc):
        global _active_session
        with self._lock, _session_lock:
            scope, self._scope = self._scope, None
            self._pool = None
            self._local_devices = None
            self._packet_address = None
            try:
                return scope.__exit__(*exc)
            finally:
                _active_session = None

    def _release_path_memory(self):
        if self._path_ran:
            import torch

            torch.cuda.empty_cache()
            self._path_ran = False

    def _output(self, requested):
        if self._scope is None:
            raise RuntimeError("BFS_SESSION_REQUIRES_CONTEXT")
        if requested is not None:
            path = Path(requested).absolute()
            if path.exists():
                raise FileExistsError(path)
            return path
        self._output_root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="run-", dir=self._output_root)).absolute() / "result"

    def _native_devices(self, native, env):
        if self._devices is not None:
            return self._devices
        if self._local_devices is None:
            count = native_query([native, "graph-count"], env=env, capture_output=True, text=True)
            if count.returncode:
                raise RuntimeError("NATIVE_GPU_DISCOVERY_FAILED: " + count.stderr[-4000:])
            available = int(count.stdout.strip())
            requested = available if self._gpus is None else self._gpus
            if not 1 <= requested <= available:
                raise ValueError(f"CUDA_GPU_COUNT_UNAVAILABLE: requested {requested}, visible {available}")
            self._local_devices = tuple(range(requested))
        return self._local_devices

    def _local_packet_address(self):
        if self._packet_address is None:
            with socket.socket() as reserve:
                reserve.bind(("127.0.0.1", 0))
                self._packet_address = f"127.0.0.1:{reserve.getsockname()[1]}"
        return self._packet_address

    def _memory_policy(self, graph, devices, native, env, preset, options):
        capabilities = _require_memory_backend(native, env, options.get("memory_backend", "segmented24"))

        def load_inventory():
            probe_env = dict(env, MGBFS_EXPERIMENTAL_SEGMENTED24="1", MGBFS_GENERIC_HISTORY="hash")
            try:
                return self._pool.inventory(native, devices, probe_env, startup_timeout(probe_env))
            except subprocess.TimeoutExpired as error:
                raise startup_timeout_error(error) from error

        return resolve_traversal_memory(graph, preset, capabilities, env, options, load_inventory), capabilities

    def _native_nearest(self, devices, direction="forward", *, window=False, maxmemory=False):
        distributed_enabled = os.environ.get("MGBFS_NATIVE_DISTRIBUTED_NEAREST") != "0"
        compact = os.environ.get(_PATH_COMPACT) == "1"
        eligible = direction == "forward" and not window and (devices is None or len(devices) > 1)
        segmented_nearest = eligible and os.environ.get(_PATH_SEGMENTED_NEAREST) == "1"
        if compact and not segmented_nearest:
            _validate_compact_path_request(None if devices is None else len(devices), distributed_enabled)
        elif not segmented_nearest and not distributed_enabled and (
            direction == "bidirectional" or window or maxmemory or devices is not None and len(devices) != 1
        ):
            return None
        try:
            native, env = native_runtime(self._executable)
            env = dict(env)
            segmented_nearest = env.pop(_PATH_SEGMENTED_NEAREST, None) == "1" and eligible
            compact = env.get(_PATH_COMPACT) == "1"
            capabilities = _require_memory_backend(native, env, "segmented24")
            if not (compact or segmented_nearest) and capabilities.get("graph_path_nearest_v1") is not True:
                return None
            devices = self._native_devices(native, env)
        except (OSError, RuntimeError):
            if compact or segmented_nearest:
                raise
            return None
        if segmented_nearest and len(devices) > 1:
            if not distributed_enabled:
                raise ValueError("NATIVE_PATH_SEGMENTED_NEAREST_DISTRIBUTED_DISABLED")
            if capabilities.get(_SEGMENTED_NEAREST_CAPABILITY) is not True:
                raise RuntimeError("NATIVE_PATH_SEGMENTED_NEAREST_UNAVAILABLE: rebuild the native executable and CUDA library")
            env.pop(_PATH_COMPACT, None)
            env[_PATH_SEGMENTED_NEAREST] = "1"
            if _PACKET_ADDR not in env:
                env[_PACKET_ADDR] = self._local_packet_address()
            return native, env, tuple(devices), True
        if not compact and capabilities.get("graph_path_nearest_v1") is not True:
            return None
        if compact:
            _validate_compact_path_request(len(devices), distributed_enabled)
            _require_compact_path_capabilities(capabilities, direction=direction, window=window)
            if _PACKET_ADDR not in env:
                env = dict(env, **{_PACKET_ADDR: self._local_packet_address()})
            return native, env, tuple(devices), True
        if window:
            return (native, env, tuple(devices), True) if capabilities.get(f"graph_path_window_{direction}_v1") is True else None
        if direction == "bidirectional":
            return (native, env, tuple(devices), True) if capabilities.get("graph_path_nearest_bidirectional_v1") is True else None
        if len(devices) == 1 and not maxmemory:
            return native, env, devices[0], capabilities.get("graph_path_nearest_resident_v1") is True
        if distributed_enabled and capabilities.get("graph_path_nearest_distributed_v1") is True:
            return native, env, tuple(devices), True
        return None

    def traverse(self, graph, *, start=None, preset="balanced", output=None, max_seconds=None,
                 native_options=None):
        with self._lock:
            graph = _normalize_graph(graph, start)
            preset = _preset(preset)
            allowed = {"memory_backend", "segment_strategy", "bitmap_pack4", "shards",
                       "capacity", "batch", "segment_rows", "bitmap_owner_page",
                       "queue_capacity", "owner_lanes", "resident_alphabet8", "transport"}
            if native_options is not None and (not isinstance(native_options, dict)
                                              or native_options.keys() - allowed):
                raise ValueError("INVALID_TRAVERSAL_NATIVE_OPTIONS")
            manual = dict(native_options or {})
            if "transport" in manual and manual["transport"] not in ("exact", "compact"):
                raise ValueError("INVALID_TRAVERSAL_TRANSPORT")
            if manual.get("resident_alphabet8") is True:
                manual.setdefault("memory_backend", "segmented24")
            if "memory_backend" in manual and manual["memory_backend"] not in ("segmented24", "permutation_bitmap"):
                raise ValueError("INVALID_TRAVERSAL_MEMORY_BACKEND")
            if "batch" in manual:
                manual["_batch"] = manual.pop("batch")
            output = self._output(output)
            from .cluster import current_cluster

            cluster = current_cluster()
            if cluster is not None:
                report = (cluster.traverse(self, graph, output, preset, max_seconds, native_options=manual)
                          if manual else cluster.traverse(self, graph, output, preset, max_seconds))
                policy = {"preset": preset, "memory_backend": report["cluster_memory_backend"], "placement": "external_ranks"}
                if "cluster_memory_selection_reason" in report:
                    policy["selection_reason"] = report["cluster_memory_selection_reason"]
                if preset == "maxmemory":
                    policy["capacity_slowdown_status"] = "UNVERIFIED"
                return _traversal_result(graph, output, report, policy)
            if os.environ.get("WORLD_SIZE", "1") != "1":
                raise RuntimeError("BFS_SESSION_EXTERNAL_RANKS_REQUIRE_CLUSTER_RUNNER")
            self._release_path_memory()
            self._pool.release_nearest_memory()
            native, env = native_runtime(self._executable)
            if _PATH_SEGMENTED_NEAREST in env:
                env = dict(env)
                env.pop(_PATH_SEGMENTED_NEAREST)
            devices = self._native_devices(native, env)
            memory, capabilities = self._memory_policy(graph, devices, native, env, preset, manual)
            backend = memory.backend
            manual = memory.launch_options
            transport = manual.pop("transport", None)
            autotune = self._tune != "off" and not native_options
            env = _traversal_exchange_env(
                env, capabilities, len(devices), transport, preset, autotune, self._local_packet_address,
            )
            tuning = {"capacity_slowdown_limit": 4.0}
            if self._tune == "auto":
                tuning.update(pilot_limit=3, pilot_wall_seconds=5)
            report = run_graph(
                graph, output, devices=devices, executable=native, backend="generic",
                autotune=autotune, max_seconds=max_seconds,
                memory_backend=backend, segment_strategy=memory.strategy, memory_preset=preset,
                _native_env=env, _profile_options=tuning, **manual,
            )
            policy = {"preset": preset, "memory_backend": backend, "selection_reason": memory.reason}
            if preset == "maxmemory":
                slowdown = report.get("autotune", {}).get("capacity_slowdown", {})
                if slowdown:
                    policy["same_backend_pilot_slowdown_status"] = slowdown.get("status", "UNVERIFIED")
                policy["capacity_slowdown_status"] = "UNVERIFIED"
                if backend == "permutation_bitmap":
                    policy["cross_backend_slowdown_status"] = "UNVERIFIED"
            return _traversal_result(graph, output, report, policy)

    bfs = traverse

    def solve(self, graph, *, start=None, targets, direction="forward", goal_mode="nearest",
              extra_steps=0, preset="balanced", output=None, max_seconds=None,
              max_paths=None, max_output_bytes=None, lineage_dir=None,
              lineage_cache_bytes=None, max_lineage_bytes=None):
        with self._lock:
            graph = _normalize_graph(graph, start)
            preset = _preset(preset)
            output = self._output(output)
            from .cluster import current_cluster
            cluster = current_cluster()
            options = {"direction": direction, "goal_mode": goal_mode, "extra_steps": extra_steps,
                       "preset": preset, "max_seconds": max_seconds, "max_paths": max_paths,
                       "max_output_bytes": max_output_bytes}
            if lineage_dir is not None:
                options["lineage_dir"] = os.fspath(lineage_dir)
            if lineage_cache_bytes is not None:
                options["lineage_cache_bytes"] = lineage_cache_bytes
            if max_lineage_bytes is not None:
                options["max_lineage_bytes"] = max_lineage_bytes
            if cluster is not None:
                options["targets"] = [_flatten(target) for target in targets]
                return cluster.solve(self, graph, output, options)
            if os.environ.get("WORLD_SIZE", "1") != "1":
                raise RuntimeError("BFS_SESSION_EXTERNAL_RANKS_REQUIRE_CLUSTER_RUNNER")
            from .path_search import solve

            devices = self._devices or self._local_devices
            if devices is None and self._gpus is not None:
                devices = tuple(range(self._gpus))
            native = None
            native_mode = goal_mode in ("nearest", "window")
            compact = os.environ.get(_PATH_COMPACT) == "1"
            segmented_nearest = (os.environ.get(_PATH_SEGMENTED_NEAREST) == "1"
                                 and direction == "forward" and goal_mode == "nearest"
                                 and devices is not None and len(devices) > 1)
            if (direction in ("forward", "bidirectional") and native_mode
                    and graph.generator_count < 0x7fff_ffff):
                native = self._native_nearest(devices, direction, window=goal_mode == "window",
                                              maxmemory=preset == "maxmemory")
                if native is not None:
                    options["native"] = native
            if native is None:
                if segmented_nearest:
                    raise RuntimeError("NATIVE_PATH_SEGMENTED_NEAREST_UNAVAILABLE_FOR_QUERY")
                if compact and direction in ("forward", "bidirectional") and native_mode:
                    raise RuntimeError("NATIVE_PATH_COMPACT_UNAVAILABLE_FOR_QUERY")
                if not self._path_ran:
                    self._pool.release_memory()
                self._path_ran = True
            else:
                self._release_path_memory()
                if isinstance(native[2], tuple):
                    self._pool.release_nearest_memory()
                if not native[3]:
                    self._pool.release_memory()
            result = solve(graph, targets=targets, devices=devices, output=output, **options)
            _record_graph(output, graph)
            return result


def open_session(*, gpus=None, devices=None, output_root=None, tune="auto",
                 executable=None):
    """Open a session with all visible GPUs, a count, or explicit device IDs."""
    return BfsSession(gpus=gpus, devices=devices, output_root=output_root, tune=tune,
                      executable=executable)
