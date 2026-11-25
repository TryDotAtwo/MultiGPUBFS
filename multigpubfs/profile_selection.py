"""Use compatible Registry recipes only after native admission of the live ranks."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .artifact_paths import artifact_path
from .process_control import ProfileBudgetExpired, profile_budget, profile_timeout, query


def registry_source(source, sha256, env):
    if source is None:
        source = env.get("MGBFS_PROFILE_REGISTRY")
        if sha256 is None:
            sha256 = env.get("MGBFS_PROFILE_REGISTRY_SHA256")
    return {"source": os.fspath(source), "sha256": sha256} if source is not None else None


def local_hardware():
    from .autotune import _system_info

    topology = _system_info(["nvidia-smi", "topo", "-m"])
    hardware = _system_info(["nvidia-smi", "--query-gpu=uuid,driver_version,name", "--format=csv,noheader"])
    if topology.returncode or hardware.returncode:
        return None
    versions = {line.split(",")[1].strip() for line in hardware.stdout.splitlines() if line.count(",") >= 2}
    return (topology.stdout, next(iter(versions))) if len(versions) == 1 else None


def cluster_hardware(rank_metadata, inventory):
    if any(item is None for item in rank_metadata) or len(rank_metadata) != len(inventory):
        return None
    devices = [item["device"] for item in inventory]
    return inventory, devices, {"rank_metadata": rank_metadata}, None


def cluster_rank_hardware(env, host):
    from .autotune import _system_info

    result = local_hardware()
    if result is None:
        return None
    topology, driver = result
    network_keys = ("NCCL_NET", "NCCL_IB_DISABLE", "NCCL_NET_GDR_LEVEL", "NCCL_P2P_DISABLE",
                    "NCCL_SHM_DISABLE", "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA")
    metadata = {"host": host, "topology": topology, "driver": driver,
                "network": {key: env[key] for key in network_keys if key in env}}
    uuid = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if uuid.startswith("GPU-") and "," not in uuid:
        index = _system_info(["nvidia-smi", "--id", uuid, "--query-gpu=index", "--format=csv,noheader"])
        if not index.returncode and index.stdout.strip().isascii() and index.stdout.strip().isdigit():
            metadata["topology_device"] = int(index.stdout)
    return metadata


def _compatible_recipe(profile, env, capabilities, packet_address):
    from .memory_admission import TraversalMemory

    backend = profile["memory_backend"]
    strategy = profile["strategy"]
    codec = profile["resident_codec"]
    wire = profile["packet_wire"]
    required = {
        "segmented24": "segmented24_v1",
        "permutation_bitmap": "permutation_bitmap_v1",
    }[backend]
    if (capabilities.get(required) is not True
            or strategy != "atomic" and capabilities.get("segmented_strategies_v1", {}).get(strategy) is not True
            or codec == "pack4-v1" and capabilities.get("bitmap_pack4_experimental_v1") is not True
            or codec == "alphabet8-v1" and capabilities.get("segmented_resident_alphabet8_v1") is not True
            or profile["compact_transport"] and capabilities.get("segmented_packet_transport_v1") is not True
            or profile["exact_exchange"] and capabilities.get("segmented_exact_exchange_v1") is not True
            or backend == "segmented24" and strategy == "batch" and (
                profile["hash_lookup_series_limit"] != 64
                or capabilities.get("segmented_hash_series_v1") is not True)):
        return None

    settings = {
        "MGBFS_SEGMENT_EXACT_EXCHANGE": profile["exact_exchange"],
        "MGBFS_SEGMENT_OWNER_LANES": profile["owner_lanes"],
        "MGBFS_SEGMENT_QUEUE_CAP": profile["queue_capacity"],
        "MGBFS_SEGMENT_MAX_ROWS": profile["segment_rows"],
        "MGBFS_SEGMENT_COMPACT_POS": profile["compact_positions"],
        "MGBFS_SEGMENT_STRATEGY": strategy,
        "MGBFS_SEGMENT_PACK4": codec == "pack4-v1",
        "MGBFS_SEGMENT_RESIDENT_ALPHABET8": codec == "alphabet8-v1",
    }
    if backend == "permutation_bitmap":
        settings["MGBFS_BITMAP_BANKS"] = profile["banks"]
        settings["MGBFS_SEGMENT_BITMAP_OWNER_PAGE"] = profile["bitmap_owner_page"] or 0
    if backend == "segmented24" and strategy == "batch":
        settings["MGBFS_SEGMENT_HASH_SERIES"] = profile["hash_lookup_series_limit"] == 64
        settings["MGBFS_SEGMENT_HASH_PREFIX_PROBE"] = profile["batch_hash_prefix_probe"]
        if profile["batch_owner_tile"] is not None:
            settings["MGBFS_SEGMENT_BATCH_TILE"] = profile["batch_owner_tile"]
    if wire is not None:
        settings.update(
            MGBFS_SEGMENT_PACKET_RING_SLOTS=profile["compact_packet_slots"],
            MGBFS_SEGMENT_PACKET_RX_SLOTS=profile["compact_receive_slots"],
            MGBFS_SEGMENT_PACKET_ROUTE_LOOKAHEAD=profile["compact_route_lookahead"],
            MGBFS_SEGMENT_PACKET_PACK4_DIRECT=wire["pack4_direct_owner"],
            MGBFS_SEGMENT_PACKET_BYTE_DIRECT=wire["byte_direct_owner"],
            MGBFS_SEGMENT_PACKET_ROUTE_PROGRESS=wire["route_receive_progress"],
        )
        if codec == "pack4-v1":
            settings["MGBFS_SEGMENT_PACKET_PACK4"] = wire["wire_state_bytes_per_row"] == 0
        elif wire["wire_record_bytes_per_row"] == 16:
            width = wire["wire_state_bytes_per_row"]
            settings["MGBFS_SEGMENT_PACKET_BINARY"] = width == 8 and profile["state_bytes"] == 1
            settings["MGBFS_SEGMENT_PACKET_RANGE16"] = width != 0 and profile["state_bytes"] == 8
    settings = {key: str(int(value)) if isinstance(value, bool) else str(value)
                for key, value in settings.items()}
    packet_key = "MGBFS_SEGMENT_PACKET_CONTROL_ADDR"
    if (any(key in env and env[key] != value for key, value in settings.items())
            or any(key.startswith("MGBFS_SEGMENT_TEST_") for key in env)
            or "MGBFS_SEGMENT_MAX_RANK_STATES" in env or "MGBFS_SEGMENT_ROWS" in env
            or wire is None and packet_key in env):
        return None
    if wire is not None and packet_key not in env:
        if packet_address is None:
            return None
        settings[packet_key] = packet_address()
    options = {
        "shards": profile["shards"], "_batch": profile["batch"],
        "owner_lanes": profile["owner_lanes"], "queue_capacity": profile["queue_capacity"],
        "segment_rows": profile["segment_rows"],
        "bitmap_pack4": codec == "pack4-v1",
        "resident_alphabet8": codec == "alphabet8-v1",
    }
    return TraversalMemory(backend, strategy, "LIVE_ADMITTED_REGISTRY_PROFILE", options), settings


def _live_admit(graph, preset, native, env, capabilities, inventory, memory, settings, profile):
    from .distributed_launch import _match_resident_reserve
    from .launch import _apply_memory_preset

    if any(rank.get("capacity_limit_per_rank") is not None for rank in profile["rank_geometry"]):
        return "EXPLICIT_CAPACITY_RECIPE"
    minimum = [rank.get("free_bytes_at_admission") for rank in profile["rank_geometry"]]
    if any(floor is not None and now["free_bytes"] < floor for now, floor in zip(inventory, minimum)):
        return "RECIPE_VRAM_PRESSURE"
    planning = _apply_memory_preset(dict(env, **settings), memory.backend,
                                    "max_capacity" if preset == "maxmemory" else "balanced")
    planning.update(MGBFS_EXPERIMENTAL_SEGMENTED24="1", MGBFS_GENERIC_HISTORY="hash",
                    MGBFS_GENERIC_TRANSPORT="full", MGBFS_GENERIC_SORT="none",
                    MGBFS_GENERIC_GENERATOR="cuda")
    try:
        planning = _match_resident_reserve(planning, inventory)
    except RuntimeError:
        return "RECIPE_RESERVE_CONSENSUS"
    if memory.backend == "permutation_bitmap":
        planning["MGBFS_EXPERIMENTAL_BITMAP"] = "1"
        orbital = capabilities.get("permutation_bitmap_position_orbits_v1") is True
        planning["_MGBFS_BITMAP_POSITION_ORBITS_CAPABLE"] = "1" if orbital else "0"
    else:
        planning.pop("MGBFS_EXPERIMENTAL_BITMAP", None)
    with tempfile.TemporaryDirectory(prefix="mgbfs-registry-admission-") as folder:
        graph_path, inventory_path = Path(folder) / "graph.json", Path(folder) / "inventory.json"
        graph_path.write_text(graph.to_json(), encoding="utf-8")
        inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
        result = query([native, "graph-plan", str(graph_path), str(inventory_path),
                        str(profile["shards"]), "auto", str(profile["batch"])],
                       env=planning, resident=False, capture_output=True, text=True)
    if result.returncode:
        return "RECIPE_REJECTED_BY_LIVE_ADMISSION"
    admission = json.loads(result.stdout)
    plan = admission["plan"]
    defaults = {"compact_transport": False, "compact_route_lookahead": False,
                "compact_packet_slots": 1, "compact_receive_slots": 1}
    if (any(plan.get(key, defaults.get(key)) != profile[key] for key in
            ("shards", "batch", "history_layers", "state_bytes", *defaults))
            or plan["queue_capacity"] < profile["queue_capacity"]
            or profile["owner_cuts"] is not None and admission.get("owner_cuts") != profile["owner_cuts"]):
        return "RECIPE_LIVE_GEOMETRY_CHANGED"
    return None


def registry_run_receipt(status, expected, report):
    from .profile_registry import SCHEMA, _COMMON, _PLAN, _case_profile

    if expected is None:
        return status
    try:
        actual = _case_profile(report, report["devices"], SCHEMA)
    except (ValueError, TypeError, KeyError, IndexError):
        return dict(status, runtime_geometry="UNVERIFIED")
    fields = (*_COMMON, *_PLAN, "memory_backend", "owner_cuts")
    rank_fields = ("admitted_batch", "rows_per_bank", "capacity_limit_per_rank", "reserve_bytes")
    matches = (len(actual["rank_geometry"]) == len(expected["rank_geometry"])
               and all(actual[field] == expected[field] for field in fields)
               and all(left[field] == right[field] for left, right in
                       zip(actual["rank_geometry"], expected["rank_geometry"]) for field in rank_fields))
    return dict(status, runtime_geometry="MATCHED" if matches else "CHANGED")


def _load_catalog(spec, env):
    from .profile_registry import read_profile_registry

    source, digest = spec["source"], spec["sha256"]
    cache = artifact_path("profile-registry", env=env, override="MGBFS_PROFILE_REGISTRY_CACHE")
    offline = env.get("MGBFS_NATIVE_OFFLINE") == "1" or env.get("MGBFS_PROFILE_REGISTRY_OFFLINE") == "1"
    try:
        return read_profile_registry(source, sha256=digest, offline=True, cache_dir=cache)
    except (OSError, ValueError):
        if offline or not source.startswith("https://") or digest is None:
            raise
    # A process deadline bounds DNS and reads from a stalled server, including redirects.
    with tempfile.TemporaryDirectory(prefix="mgbfs-registry-import-") as folder:
        catalog = Path(folder) / "catalog.json"
        command = [sys.executable, "-m", "multigpubfs.profile_registry", "import", source,
                   "--sha256", digest, "--cache-dir", str(cache), "--output", str(catalog)]
        try:
            result = subprocess.run(command, timeout=profile_timeout(5), capture_output=True)
        except subprocess.TimeoutExpired as error:
            raise ProfileBudgetExpired("REGISTRY_DOWNLOAD_DEADLINE") from error
        if result.returncode:
            raise ValueError("PROFILE_REGISTRY_IMPORT_FAILED")
        return read_profile_registry(catalog)


def select_registry_recipe(spec, graph, preset, native, env, capabilities, deadline, *,
                           hardware, inventory, packet_address):
    from .native_distribution import runtime_identity
    from .profile_registry import hardware_key, matching_profiles

    status = {"status": "NO_COMPATIBLE_PROFILE"}
    if spec is None:
        return None, None
    try:
        with profile_budget(deadline):
            catalog = _load_catalog(spec, env)
            if not catalog["entries"] or catalog["schema"] < 2:
                return None, status
            supported = hardware()
            if supported is None:
                return None, status
            rank_inventory, devices, topology, drivers = supported
            identity = runtime_identity(native, env)
            matches = matching_profiles(catalog, graph.digest(),
                                        hardware_key(rank_inventory, devices, topology, driver_versions=drivers),
                                        identity)
            if not matches and isinstance(topology, dict) and "rank_metadata" in topology:
                # Old single-host evidence has no explicit rank topology or NCCL settings.
                legacy_key = hardware_key(rank_inventory, devices, topology, driver_versions=drivers,
                                          legacy_cluster=True)
                matches = matching_profiles(catalog, graph.digest(), legacy_key, identity)
            namespace = env.get("MGBFS_PROFILE_PRESSURE_NAMESPACE", "ambient")
            acceptable = {None, "ambient"} if preset == "balanced" and namespace == "ambient" else {namespace}
            matches = [entry for entry in matches if
                       all(arm["pressure_namespace"] == entry["profiles"]["balanced"]["pressure_namespace"]
                           for arm in entry["profiles"].values())
                       and entry["profiles"]["balanced"]["pressure_namespace"] in acceptable]
            if not matches:
                return None, status
            candidates = sorted(
                ((entry, arm, profile) for entry in matches for arm, profile in entry["profiles"].items()),
                key=lambda item: item[2]["bfs_seconds"],
            )
            baseline = candidates[0][2]
            fastest = baseline["bfs_seconds"]
            baseline_peaks = baseline["peak_vram_bytes_by_rank"]
            if preset == "maxmemory":
                candidates = [(entry, arm, profile) for entry, arm, profile in candidates
                              if arm == "max_capacity" and baseline_peaks
                              and entry["capacity_verification"] == "LOWER_PHYSICAL_PEAK"
                              and (profile is baseline or max(profile["peak_vram_bytes_by_rank"]) < max(baseline_peaks))
                              and profile["bfs_seconds"] <= 4 * fastest]
                candidates.sort(key=lambda item: (
                    max(item[2]["peak_vram_bytes_by_rank"]), item[2]["bfs_seconds"],
                ))
            for entry, arm, profile in candidates[:3]:
                profile_timeout(5)
                recipe_preset = "maxmemory" if arm == "max_capacity" else "balanced"
                candidate = _compatible_recipe(profile, env, capabilities, packet_address)
                if candidate is None:
                    status = {"status": "RECIPE_UNAVAILABLE"}
                    continue
                memory, settings = candidate
                refusal = _live_admit(graph, recipe_preset, native, env, capabilities, inventory(),
                                      memory, settings, profile)
                if refusal is not None:
                    status = {"status": refusal}
                    continue
                status = {
                    "status": "LIVE_ADMITTED_REGISTRY_PROFILE",
                    "evidence_sha256": entry["evidence_sha256"],
                    "requested_preset": preset,
                    "recipe_preset": recipe_preset,
                    "historical_verification": entry["capacity_verification"],
                    "historical_ratio": entry["profiles"]["max_capacity"]["bfs_seconds"] / fastest,
                    "scope": "FULL_BFS_HISTORICAL_PROFILE",
                }
                profile_timeout(5)
                return (memory, settings, profile, recipe_preset), status
            return None, status
    except ProfileBudgetExpired:
        return None, {"status": "REGISTRY_STARTUP_DEADLINE"}
    except (OSError, ValueError, TypeError) as error:
        return None, {"status": "REGISTRY_UNAVAILABLE", "reason": type(error).__name__}
