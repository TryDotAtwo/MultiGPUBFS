"""Run one coordinator script on a preallocated SLURM or torchrun job."""

import argparse
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import tempfile
import time

from .network_control import ControlStore


_active = ContextVar("multigpubfs_cluster", default=None)


def current_cluster():
    return _active.get()


def _path_native_runtime(job):
    from .launch import _PACKET_ADDR, _PATH_COMPACT, _PATH_SEGMENTED_NEAREST, _PATH_SEGMENTED_WINDOW
    from .native_distribution import native_runtime

    native, env = native_runtime(job["executable"])
    address = job.get("compact_path_packet_address")
    modes = (_PATH_COMPACT, _PATH_SEGMENTED_NEAREST, _PATH_SEGMENTED_WINDOW)
    if address is not None or any(mode in env for mode in modes):
        env = dict(env)
        for mode in modes:
            env.pop(mode, None)
        if address is not None:
            mode = _PATH_SEGMENTED_NEAREST if job.get("segmented_nearest") or job.get("segmented_window") else _PATH_COMPACT
            env.update({mode: "1", _PACKET_ADDR: address})
            if job.get("segmented_window"):
                env[_PATH_SEGMENTED_WINDOW] = "1"
    return native, env


@dataclass(frozen=True)
class Placement:
    rank: int
    local_rank: int
    world: int
    host: str
    coordinator_port: int
    native_port: int
    path_port: int
    session_id: str
    packet_port: int | None = None

    @classmethod
    def from_environment(cls, environment):
        if "RANK" in environment and "WORLD_SIZE" in environment:
            rank, local, world = (int(environment[name]) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
            host = environment["MASTER_ADDR"]
            base_port = int(environment["MASTER_PORT"])
            identity = environment.get("TORCHELASTIC_RUN_ID", "torchrun")
        elif "SLURM_PROCID" in environment:
            rank, local, world = (int(environment[name]) for name in ("SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS"))
            host = environment.get("MASTER_ADDR") or environment.get("SLURM_LAUNCH_NODE_IPADDR")
            # Prefer the first task node over the srun launch host, which can be a login node.
            if "MASTER_ADDR" not in environment and "SLURM_JOB_NODELIST" in environment:
                output = subprocess.check_output(["scontrol", "show", "hostnames", environment["SLURM_JOB_NODELIST"]], text=True)
                host = output.splitlines()[0] if output.splitlines() else None
            identity = environment.get("SLURM_JOB_ID", "slurm") + "." + environment.get("SLURM_STEP_ID", "0")
            job_id = int(environment.get("SLURM_JOB_ID", "0"))
            base_port = int(environment.get("MASTER_PORT", 20000 + 5 * (job_id % 9000)))
        else:
            raise ValueError("CLUSTER_RUNNER_REQUIRES_TORCHRUN_OR_SLURM_TASKS")
        coordinator = int(environment.get("MGBFS_SESSION_PORT", base_port + 1))
        native = int(environment.get("MGBFS_CONTROL_PORT", base_port + 2))
        paths = int(environment.get("MGBFS_PATH_PORT", base_port + 3))
        packet = int(environment.get("MGBFS_SEGMENT_PACKET_PORT", base_port + 4))
        ports = base_port, coordinator, native, paths, packet
        if (not host or not 1 <= world <= 128 or not 0 <= rank < world or local < 0
                or len(set(ports)) != len(ports) or not all(1 <= port <= 65535 for port in ports)):
            raise ValueError("INVALID_CLUSTER_PLACEMENT")
        return cls(rank, local, world, host, coordinator, native, paths,
                   environment.get("MGBFS_RUN_ID", f"mgbfs-{identity}-{base_port}-{world}"), packet)


class Cluster:
    def __init__(self, placement, store, scratch):
        self.placement = placement
        self.store = store
        self.scratch = Path(scratch)
        self.sequence = 0
        self.worker = None
        self.worker_settings = None
        self.path_ran = False
        self.failure = None

    def _worker(self, native, env, device):
        from .autotune import _stat_identity
        from .generic_session import Worker

        family = env.get("MGBFS_EXPERIMENTAL_SEGMENTED24") == "1"
        startup = tuple(env.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_MODULE_LOADING", "CUDA_DISABLE_PTX_JIT",
            "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "MGBFS_CUDART_LIBRARY", "MGBFS_CUDA_LIB_DIR",
        ))
        transport = tuple(sorted((key, value) for key, value in env.items() if key.startswith("NCCL_")))
        settings = native, _stat_identity(native), device, startup, transport, family
        if self.worker is not None and settings[:-1] == self.worker_settings[:-1] and family != self.worker_settings[-1]:
            self._close_worker()
        if self.worker is None:
            worker_root = Path(tempfile.mkdtemp(prefix="worker-", dir=self.scratch)) / "resident"
            self.worker = Worker(native, env, self.placement.rank, device, worker_root, "generic")
            self.worker_settings = settings
        elif settings != self.worker_settings:
            raise RuntimeError("CLUSTER_RESIDENT_RUNTIME_OR_DEVICE_CHANGED")
        return self.worker

    def _close_worker(self):
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.close()

    def launch_rank(self, command, env, log):
        worker = self._worker(command[0], env, int(env["LOCAL_RANK"]))
        offset = (worker.root / "worker.log").stat().st_size
        job = worker.submit(command[1], command[2:], env)
        job.receipt_log = log
        job.log_offset = offset
        return job

    def inventory(self, native, env, device):
        worker = self._worker(native, env, device)
        job = worker.submit("inventory", [], env)
        if job.wait(60):
            raise RuntimeError(f"CLUSTER_RESIDENT_INVENTORY_FAILED: {job.response!r}")
        return job.response["result"]

    def _local_device(self):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        return 0 if visible is not None and len(visible.split(",")) == 1 else self.placement.local_rank

    def _packet_address(self):
        return f"{socket.gethostbyname(self.placement.host)}:{self.placement.packet_port}"

    def _run(self, job, output):
        from .graph_definition import GraphDefinition
        graph = GraphDefinition.from_dict(job["graph"])
        if job["mode"] == "solve":
            from .path_search import solve

            native_window = job.get("native_window")
            native_nearest = job.get("native_nearest") or job.get("native_forward_nearest")
            if native_window:
                return self._run_external_window(job, graph, output)
            if native_nearest:
                return self._run_external_nearest(job, graph, output)
            if self.worker is not None and not self.path_ran:
                release = self.worker.submit("release-memory", [], {})
                if release.wait(30):
                    raise RuntimeError("CLUSTER_RESIDENT_MEMORY_HANDOFF_FAILED")
            placement = self.placement
            distributed = {"rank": placement.rank, "world_size": placement.world, "timeout_seconds": 120,
                           "init_method": f"tcp://{placement.host}:{placement.path_port}", "persistent": True}
            self.path_ran = True
            return solve(graph, devices=(self._local_device(),), output=output, distributed=distributed, **job["options"])

        self.close_path()
        from .api import _traversal_exchange_env
        from .distributed_launch import _native_artifact_hashes
        from .launch import _PATH_SEGMENTED_NEAREST, _require_memory_backend, run_graph
        from .memory_admission import resolve_traversal_memory
        from .native_distribution import native_runtime

        native, env = native_runtime(job["executable"])
        if _PATH_SEGMENTED_NEAREST in env:
            env = dict(env)
            env.pop(_PATH_SEGMENTED_NEAREST)
        try:
            capabilities = _require_memory_backend(native, env, "segmented24")
        except RuntimeError as error:
            if not str(error).startswith("NATIVE_MEMORY_BACKEND_UNAVAILABLE:"):
                raise
            capabilities = {}
        rank, world = self.placement.rank, self.placement.world
        prefix = f"job/{self.sequence}/memory-policy/"
        manual = dict(job.get("native_options") or {})
        transport = manual.pop("transport", None)
        policy_keys = (
            "MGBFS_SEGMENT_PACK4", "MGBFS_BITMAP_BANKS", "MGBFS_SEGMENT_RESIDENT_ALPHABET8",
            "MGBFS_SEGMENT_EXACT_EXCHANGE", "MGBFS_SEGMENT_PACKET_CONTROL_ADDR",
        )
        policy_env = {key: env[key] for key in policy_keys if key in env}
        policy_env["bounded_hash"] = env.get("MGBFS_SEGMENT_HASH_SERIES") != "0"
        if manual.get("bitmap_pack4") is not None:
            policy_env.pop("MGBFS_SEGMENT_PACK4", None)
        if manual.get("resident_alphabet8") is not None:
            policy_env.pop("MGBFS_SEGMENT_RESIDENT_ALPHABET8", None)
        if transport is not None:
            policy_env.pop("MGBFS_SEGMENT_EXACT_EXCHANGE", None)
        if transport == "exact":
            policy_env.pop("MGBFS_SEGMENT_PACKET_CONTROL_ADDR", None)
        if transport is None and "MGBFS_SEGMENT_PACKET_CONTROL_ADDR" not in env:
            policy_env["packet_bind_configured"] = "MGBFS_SEGMENT_PACKET_BIND_ADDR" in env
        strategies = capabilities.get("segmented_strategies_v1")
        artifacts = _native_artifact_hashes(native, env) if world > 1 else None
        self.store.put(prefix + f"preflight/{rank}", {
            "features": [key for key, value in capabilities.items() if value is True],
            "strategies": [key for key, value in strategies.items() if value is True]
            if isinstance(strategies, dict) else [],
            "env": policy_env,
            "artifacts": artifacts,
        })
        if rank == 0:
            peers = [self.store.get(prefix + f"preflight/{peer}") for peer in range(world)]
            if world > 1 and any(peer["artifacts"]["cuda"] is None for peer in peers):
                raise RuntimeError("EXTERNAL_NATIVE_LIBRARY_IDENTITY_UNAVAILABLE")
            if any(peer["artifacts"] != artifacts for peer in peers):
                raise RuntimeError("EXTERNAL_GRAPH_NATIVE_OR_PARAMETERS_MISMATCH")
            features = sorted(set.intersection(*(set(peer["features"]) for peer in peers)))
            self.store.put(prefix + "preflight", {
                "strategies": sorted(set.intersection(*(set(peer["strategies"]) for peer in peers))),
                "features": features, "env_agrees": all(peer["env"] == policy_env for peer in peers),
            })
        preflight = self.store.get(prefix + "preflight")
        if not preflight["env_agrees"]:
            raise RuntimeError("CLUSTER_MEMORY_POLICY_ENV_MISMATCH")
        capabilities = dict.fromkeys(preflight["features"], True)
        capabilities["segmented_strategies_v1"] = dict.fromkeys(preflight["strategies"], True)
        network = capabilities.get("segmented_network_agreement_v1") is True
        preset = job["preset"]
        autotune = job["tune"] == "full" or (network and job["tune"] == "auto")
        if transport is not None and not network:
            raise RuntimeError("CLUSTER_TRANSPORT_REQUIRES_SEGMENTED_NETWORK_RUNTIME")
        if preset == "maxmemory" and not network:
            raise RuntimeError("CLUSTER_MAXMEMORY_REQUIRES_SEGMENTED_NETWORK_RUNTIME: rebuild or update the native runtime")

        inventory = None

        def load_inventory():
            nonlocal inventory
            if inventory is not None:
                return inventory
            inventory_env = dict(env, MGBFS_EXPERIMENTAL_SEGMENTED24="1", MGBFS_GENERIC_HISTORY="hash")
            local = self.inventory(native, inventory_env, self._local_device())
            self.store.put(prefix + f"inventory/{rank}", local)
            if rank == 0:
                self.store.put(prefix + "inventory", [
                    self.store.get(prefix + f"inventory/{peer}") for peer in range(world)
                ])
            inventory = self.store.get(prefix + "inventory")
            return inventory

        recipe = profile_status = expected = None
        registry = job.get("profile_registry")
        if registry is not None and network:
            from .process_control import ProfileBudgetExpired, profile_budget, profile_timeout
            from .profile_selection import (cluster_hardware, cluster_rank_hardware, registry_run_receipt,
                                            select_registry_recipe)

            # Inventory is required by either policy. Bound the optional phase separately.
            inventory = load_inventory()
            deadline = time.monotonic() + 5
            try:
                with profile_budget(deadline):
                    metadata = cluster_rank_hardware(env, socket.gethostname())
                    self.store.put(prefix + f"profile-hardware/{rank}", metadata,
                                   deadline=time.perf_counter() + profile_timeout(5))
            except (ProfileBudgetExpired, TimeoutError):
                pass
            if rank == 0:
                def hardware():
                    control_deadline = time.perf_counter() + profile_timeout(5)
                    try:
                        metadata = [self.store.get(prefix + f"profile-hardware/{peer}", deadline=control_deadline)
                                    for peer in range(world)]
                    except TimeoutError as error:
                        raise ProfileBudgetExpired("REGISTRY_RANK_HARDWARE_DEADLINE") from error
                    return cluster_hardware(metadata, inventory)

                recipe, profile_status = select_registry_recipe(
                    registry, graph, preset, native, env, capabilities, deadline,
                    hardware=hardware, inventory=load_inventory,
                    packet_address=self._packet_address if self.placement.packet_port is not None else None,
                )
                resolved = None
                if recipe is not None:
                    memory, settings, expected, launch_preset = recipe
                    resolved = {"memory": vars(memory), "settings": settings, "geometry": expected,
                                "preset": launch_preset}
                self.store.put(prefix + "profile-selection", {"recipe": resolved, "receipt": profile_status})
            else:
                # Only the coordinator may choose a recipe or the fallback for this job.
                profile = self.store.get(prefix + "profile-selection")
                profile_status = profile["receipt"]
                if profile["recipe"] is not None:
                    from .memory_admission import TraversalMemory

                    payload = profile["recipe"]
                    recipe = (TraversalMemory(**payload["memory"]), payload["settings"], payload["geometry"],
                              payload["preset"])
        launch_preset = preset
        if recipe is None:
            memory = resolve_traversal_memory(
                graph, preset, capabilities if network else {}, env, manual, load_inventory,
                default_backend="segmented24" if network else "legacy",
            )
        else:
            memory, settings, expected, launch_preset = recipe
            env = dict(env, **settings)
            autotune = False
        if network:
            env = _traversal_exchange_env(
                env, capabilities, world, transport, launch_preset, autotune,
                self._packet_address if self.placement.packet_port is not None else None, external=True,
            )
        report = run_graph(graph, output, device=self._local_device(),
                           executable=native, backend="generic", memory_backend=memory.backend,
                           memory_preset=launch_preset, segment_strategy=memory.strategy, _native_env=env,
                           autotune=autotune, max_seconds=job["max_seconds"], **memory.launch_options)
        report["cluster_memory_backend"] = memory.backend
        report["cluster_memory_selection_reason"] = memory.reason
        if profile_status is not None:
            report["cluster_profile_registry"] = registry_run_receipt(profile_status, expected, report)
        return report

    def _run_external_nearest(self, job, graph, output):
        from math import ceil
        from time import perf_counter
        from .launch import _require_memory_backend, _segmented_nearest_capability, run_graph
        from .path_search import (
            _active_path_targets, _deadline, _empty_native_path_receipt, _graph_and_targets, solve,
        )

        started = perf_counter()
        self.close_path()
        native, env = _path_native_runtime(job)
        capabilities = _require_memory_backend(native, env, "segmented24")
        options = job["options"]
        direction = options.get("direction", "forward")
        bidirectional = direction == "bidirectional"
        if job.get("segmented_nearest"):
            capability = _segmented_nearest_capability(bidirectional)
        elif bidirectional:
            capability = "graph_path_nearest_bidirectional_v1"
        else:
            capability = "graph_path_nearest_distributed_v1"
        if capabilities.get(capability) is not True:
            raise RuntimeError("CLUSTER_NATIVE_DISTRIBUTED_NEAREST_UNAVAILABLE")
        graph, targets = _graph_and_targets(graph, None, options["targets"])
        roots = [state for _, state in _active_path_targets(graph, targets, direction)]
        deadline = _deadline(options.get("max_seconds"), started)
        key = f"job/{self.sequence}/nearest-budget"
        if self.placement.rank == 0:
            self.store.put(key, None if deadline is None else max(0.0, deadline - perf_counter()))
        seconds = self.store.get(key)
        if seconds == 0:
            receipt = _empty_native_path_receipt("TIME_LIMIT")
        elif not roots:
            receipt = _empty_native_path_receipt()
        else:
            with tempfile.TemporaryDirectory(prefix="nearest-ranks-", dir=self.scratch) as directory:
                receipt = run_graph(
                    graph, Path(directory) / "result", device=self._local_device(), executable=native,
                    backend="generic", autotune=False, shards=1, _native_env=env,
                    max_seconds=None if seconds is None else ceil(seconds), _path_targets=roots,
                    _path_bidirectional=bidirectional, memory_preset=options.get("preset", "balanced"),
                )
        if self.placement.rank != 0:
            return None
        devices = tuple(f"rank:{rank}" for rank in range(self.placement.world))
        return solve(graph, output=output, devices=devices, _native_receipt=receipt, _started=started, **options)

    def _run_external_window(self, job, graph, output):
        from math import ceil
        from time import perf_counter

        from .launch import _require_memory_backend, _require_segmented_path_capabilities, _window_shards, run_graph
        from .path_search import (
            _active_path_targets, _deadline, _empty_native_path_receipt, _graph_and_targets, solve,
        )
        from .path_window_files import WindowOwnerFiles, serve_window_files

        started = perf_counter()
        self.close_path()
        native, env = _path_native_runtime(job)
        options = job["options"]
        direction = options.get("direction", "forward")
        capabilities = _require_memory_backend(native, env, "segmented24")
        if job.get("segmented_window"):
            _require_segmented_path_capabilities(capabilities, direction == "bidirectional", window=True)
        elif capabilities.get(f"graph_path_window_{direction}_v1") is not True:
            raise RuntimeError("CLUSTER_NATIVE_WINDOW_UNAVAILABLE")
        graph, targets = _graph_and_targets(graph, None, options["targets"])
        roots = [state for _, state in _active_path_targets(graph, targets, direction)]
        deadline = _deadline(options.get("max_seconds"), started)
        rank, world = self.placement.rank, self.placement.world
        prefix = f"job/{self.sequence}/window/"

        lineage_parent = self.scratch
        if rank == 0 and options.get("lineage_dir") is not None:
            lineage_parent = Path(options["lineage_dir"])
        with tempfile.TemporaryDirectory(prefix="window-ranks-", dir=lineage_parent) as directory:
            root = Path(directory)

            def launch(folder, config, seconds):
                return run_graph(
                    graph, root / folder, device=self._local_device(), executable=native,
                    backend="generic", autotune=False, shards=_window_shards(env), _native_env=env,
                    max_seconds=seconds, _path_targets=roots, _path_window=config,
                    _path_bidirectional=direction == "bidirectional",
                    memory_preset=options.get("preset", "balanced"),
                )

            if rank == 0:
                first_budget = None if deadline is None else max(0.0, deadline - perf_counter())
                self.store.put(prefix + "nearest-budget", first_budget)
            first_budget = self.store.get(prefix + "nearest-budget")
            if first_budget == 0:
                nearest = _empty_native_path_receipt("TIME_LIMIT")
            elif not roots:
                nearest = _empty_native_path_receipt()
            else:
                nearest = launch("nearest", None, None if first_budget is None else ceil(first_budget))
            receipt = None
            stop_reason = None
            if nearest["status"] == "COMPLETE":
                if rank == 0:
                    remaining = None if deadline is None else max(0.0, deadline - perf_counter())
                    self.store.put(prefix + "remaining", remaining)
                remaining = self.store.get(prefix + "remaining")
                if remaining == 0:
                    stop_reason = "TIME_LIMIT"
                else:
                    limit = nearest["d_min"] + options["extra_steps"]
                    if not 0 <= limit <= 0xffff_ffff:
                        raise ValueError("NATIVE_WINDOW_DEPTH_RANGE")
                    budget = options.get("max_lineage_bytes")
                    rank_budget = ((1 << 64) - 1 if budget is None else budget) // world
                    config = {"max_length": limit, "disk_budget": rank_budget}
                    if direction == "bidirectional":
                        config["min_length"] = nearest["d_min"]
                    receipt = launch("window", config,
                                     None if remaining is None else ceil(remaining))
            window_root = root / "window"
            files_prefix = prefix + "files/"
            if rank != 0:
                if receipt is not None:
                    serve_window_files(self.store, files_prefix, window_root, rank, receipt)
                return None
            devices = tuple(f"rank:{index}" for index in range(world))
            files = (WindowOwnerFiles(self.store, files_prefix, window_root, world, deadline)
                     if receipt is not None else nullcontext())
            with files as owner_files:
                return solve(graph, output=output, devices=devices, _started=started,
                             _native_window_data=(nearest, receipt, window_root, stop_reason),
                             _native_window_files=owner_files,
                             _output_reserved=job.get("window_output_reserved", False), **options)

    def close_path(self):
        if self.path_ran:
            import torch
            from .path_network import close_path_group

            close_path_group()
            torch.cuda.empty_cache()
            self.path_ran = False

    def _validate_session(self, session):
        self._check_generation()
        if session._devices is not None:
            raise ValueError("CLUSTER_DEVICE_PLACEMENT_IS_SET_BY_LAUNCHER")
        if session._gpus not in (None, self.placement.world):
            raise ValueError(f"CLUSTER_GPU_COUNT_MISMATCH: allocated {self.placement.world}")

    def traverse(self, session, graph, output, preset, max_seconds, *, native_options=None):
        self._validate_session(session)
        job = {"mode": "traverse", "graph": graph.to_dict(), "executable": None if session._executable is None else str(session._executable),
               "tune": "off" if native_options else session._tune, "max_seconds": max_seconds,
               "preset": preset, "native_options": native_options}
        if not native_options:
            from .profile_selection import registry_source

            registry = registry_source(session._profile_registry, session._profile_registry_sha256, os.environ)
            if registry is not None:
                job["profile_registry"] = registry
        return self._dispatch(job, output)

    def solve(self, session, graph, output, options):
        from .api import _require_compact_path_capabilities, _validate_compact_path_request
        from .launch import (_PACKET_ADDR, _PATH_COMPACT, _PATH_SEGMENTED_NEAREST, _PATH_SEGMENTED_WINDOW,
                             _require_memory_backend, _require_segmented_path_capabilities,
                             _supports_segmented_path)
        from .native_distribution import native_runtime
        from .path_search import (
            _deadline, _empty_output_dir, _graph_and_targets, _path_options, _reserve_output, inverse_action,
        )

        self._validate_session(session)
        graph, targets = _graph_and_targets(graph, None, options.get("targets"))
        options = dict(options, targets=targets)
        executable = None if session._executable is None else str(session._executable)
        mode = options.get("goal_mode", "nearest")
        direction = options.get("direction", "forward")
        _deadline(options.get("max_seconds"), 0.0)
        _path_options(**{
            name: options[name] for name in (
                "direction", "goal_mode", "extra_steps", "preset", "max_paths", "max_output_bytes",
                "max_lineage_bytes", "lineage_cache_bytes",
            ) if name in options
        })
        setting = os.environ.get("MGBFS_NATIVE_DISTRIBUTED_NEAREST")
        path_setting = os.environ.get(_PATH_SEGMENTED_WINDOW if mode == "window" else _PATH_SEGMENTED_NEAREST)
        segmented_nearest = mode == "nearest" and path_setting == "1" and self.placement.world > 1
        segmented_window = mode == "window" and path_setting == "1"
        segmented = segmented_nearest or segmented_window
        compact = not segmented and os.environ.get(_PATH_COMPACT) == "1"
        if segmented_nearest and setting == "0":
            raise ValueError("NATIVE_PATH_SEGMENTED_NEAREST_DISTRIBUTED_DISABLED")
        if compact or segmented_window:
            _validate_compact_path_request(self.placement.world, setting != "0")
        if direction == "bidirectional" and graph.action["kind"] == "matrix":
            inverse_action(graph)
        capabilities = None
        if (not compact and not segmented and path_setting is None and setting != "0"
                and graph.generator_count < 0x7fff_ffff and self.placement.world > 1
                and self.placement.packet_port is not None):
            try:
                native, env = native_runtime(executable)
                capabilities = _require_memory_backend(native, env, "segmented24")
            except (OSError, RuntimeError):
                pass
            if (capabilities is not None and capabilities.get("segmented_network_agreement_v1") is True
                    and _supports_segmented_path(capabilities, direction == "bidirectional", window=mode == "window")):
                segmented = True
                segmented_window = mode == "window"
                segmented_nearest = mode == "nearest"
        use_native = False
        compact_address = None
        if compact or segmented:
            if graph.generator_count >= 0x7fff_ffff:
                mode_name = "SEGMENTED_WINDOW" if segmented_window else "SEGMENTED_NEAREST" if segmented_nearest else "COMPACT"
                raise RuntimeError(f"NATIVE_PATH_{mode_name}_UNAVAILABLE_FOR_QUERY")
            if capabilities is None:
                native, env = native_runtime(executable)
                capabilities = _require_memory_backend(native, env, "segmented24")
            compact_address = env.get(_PACKET_ADDR)
            if compact_address is None and self.placement.packet_port is not None:
                compact_address = self._packet_address()
            if not compact_address:
                raise ValueError("NATIVE_PATH_COMPACT_REQUIRES_EXTERNAL_PACKET_ADDR")
            if segmented:
                _require_segmented_path_capabilities(capabilities, direction == "bidirectional", window=segmented_window)
                if capabilities.get("segmented_network_agreement_v1") is not True:
                    raise RuntimeError("NATIVE_SEGMENTED_NETWORK_AGREEMENT_UNAVAILABLE")
            else:
                _require_compact_path_capabilities(capabilities, direction=direction, window=mode == "window")
            use_native = True
        elif (setting != "0" and mode in ("nearest", "window")
                and direction in ("forward", "bidirectional") and graph.generator_count < 0x7fff_ffff):
            if setting == "1":
                use_native = True
            else:
                if mode == "window":
                    capability = f"graph_path_window_{direction}_v1"
                elif direction == "bidirectional":
                    capability = "graph_path_nearest_bidirectional_v1"
                else:
                    capability = "graph_path_nearest_distributed_v1"
                try:
                    if capabilities is None:
                        native, env = native_runtime(executable)
                        capabilities = _require_memory_backend(native, env, "segmented24")
                    use_native = capabilities.get(capability) is True
                except (OSError, RuntimeError):
                    pass
        use_window = use_native and mode == "window"
        use_native = use_native and mode == "nearest"
        output = Path(output).resolve()
        lineage = Path(options["lineage_dir"]).resolve() if options.get("lineage_dir") is not None else None
        if lineage is not None:
            options["lineage_dir"] = os.fspath(lineage)
        if use_window:
            nested_lineage = lineage is not None and lineage.is_relative_to(output)
            if lineage is not None and not nested_lineage:
                lineage.mkdir(parents=True, exist_ok=True)
            _reserve_output(output, graph)
            if nested_lineage:
                lineage.mkdir(parents=True, exist_ok=True)
        else:
            _empty_output_dir(output)
        job = {"mode": "solve", "graph": graph.to_dict(), "options": options,
               "native_nearest": use_native, "native_window": use_window,
               "window_output_reserved": use_window, "executable": executable}
        if compact_address is not None:
            job["compact_path_packet_address"] = compact_address
        if segmented_nearest:
            job["segmented_nearest"] = True
        if segmented_window:
            job["segmented_window"] = True
        return self._dispatch(job, output)

    def _check_generation(self):
        if self.failure is not None:
            raise RuntimeError(f"CLUSTER_GENERATION_CLOSED: {self.failure}")

    def _fail(self, error):
        if self.failure is None:
            self.failure = str(error)
            try:
                self.store.report_error(error, self.placement.world)
            except Exception:
                pass
            try:
                self._close_worker()
            except Exception:
                pass

    def _dispatch(self, job, output):
        self._check_generation()
        key = f"job/{self.sequence}/"
        try:
            self.store.put(key + "command", job)
            result = self._run(job, output)
            if job.get("native_window"):
                from time import perf_counter
                from .path_window_files import WINDOW_DRAIN_SECONDS

                self._acknowledge(key, deadline=perf_counter() + WINDOW_DRAIN_SECONDS)
            else:
                self._acknowledge(key)
            return result
        except BaseException as error:
            self._fail(error)
            raise

    def _acknowledge(self, key, *, deadline=None):
        wait = {} if deadline is None else {"deadline": deadline}
        self.store.put(key + f"done/{self.placement.rank}", True, **wait)
        if self.placement.rank == 0:
            for rank in range(self.placement.world):
                self.store.get(key + f"done/{rank}", **wait)
            self.store.release_prefix(key)
        self.sequence += 1

    def serve(self):
        while True:
            key = f"job/{self.sequence}/"
            job = self.store.get(key + "command")
            if job is None:
                self._acknowledge(key)
                return
            with tempfile.TemporaryDirectory(prefix="job-", dir=self.scratch) as temporary:
                self._run(job, Path(temporary) / "result")
            self._acknowledge(key)

    def stop(self):
        self._check_generation()
        key = f"job/{self.sequence}/"
        self.store.put(key + "command", None)
        self._acknowledge(key)


@contextmanager
def _cluster_scope(placement, idle_timeout):
    previous = {}
    settings = {"RANK": str(placement.rank), "LOCAL_RANK": str(placement.local_rank),
                "WORLD_SIZE": str(placement.world), "MASTER_ADDR": placement.host,
                "MASTER_PORT": str(placement.native_port - 1), "MGBFS_CONTROL_PORT": str(placement.native_port),
                "MGBFS_RUN_ID": placement.session_id}
    for key, value in settings.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        with tempfile.TemporaryDirectory(prefix="mgbfs-cluster-", dir=os.environ.get("MGBFS_GENERIC_SESSION_LOG_ROOT")) as scratch:
            store = ControlStore(placement.host, placement.coordinator_port, placement.rank,
                                 placement.session_id, idle_timeout, os.environ.get("MGBFS_CONTROL_TOKEN", ""))
            cluster = Cluster(placement, store, scratch)
            token = _active.set(cluster)
            try:
                yield cluster
            except BaseException as error:
                cluster._fail(error)
                raise
            finally:
                _active.reset(token)
                try:
                    try:
                        cluster.close_path()
                    finally:
                        cluster._close_worker()
                finally:
                    store.close()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_script(script, args):
    previous = sys.argv
    sys.argv = [str(script), *args]
    try:
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as error:
            if error.code not in (0, None):
                raise
    finally:
        sys.argv = previous


def main(argv=None):
    parser = argparse.ArgumentParser(prog="multigpubfs cluster", description=__doc__)
    parser.add_argument("--idle-timeout", type=int, default=3600)
    parser.add_argument("script", type=Path, help="Python script executed on rank zero")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.idle_timeout < 1:
        parser.error("--idle-timeout must be positive")
    placement = Placement.from_environment(os.environ)
    if placement.world == 1:
        return _run_script(args.script, args.script_args)
    with _cluster_scope(placement, args.idle_timeout) as cluster:
        if placement.rank != 0:
            cluster.serve()
        else:
            _run_script(args.script, args.script_args)
            cluster.stop()
