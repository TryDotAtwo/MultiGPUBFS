"""Use compatible Registry recipes only after native admission of the live ranks."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .artifact_paths import artifact_path
from .memory_admission import TraversalMemory, bounded_hash_available
from .process_control import ProfileBudgetExpired, profile_budget, profile_timeout, query


@dataclass(frozen=True)
class RegistryRecipe:
    memory: TraversalMemory
    settings: dict
    evidence: dict
    preset: str
    admission: dict


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


def nccl_profile_network(env):
    return {key: hashlib.sha256(value.encode()).hexdigest()
            for key, value in env.items() if key.startswith("NCCL_")}


def local_profile_hardware(devices, env):
    from .autotune import _system_info
    from .native_assets import AssetError, run_hardware_probe
    from .profile_registry import local_profile_topology

    result = local_hardware()
    if result is None or sys.platform != "linux":
        return result
    topology, driver = result
    try:
        hardware = run_hardware_probe(timeout=profile_timeout(1), env=env)
        by_device = {row["logical_id"]: row["pci_bus_id"] for row in hardware["devices"]}
        physical = _system_info(["nvidia-smi", "--query-gpu=index,pci.bus_id,mig.mode.current", "--format=csv,noheader"])
        if physical.returncode:
            return result

        def bus_key(pci):
            domain, address = pci.strip().lower().split(":", 1)
            return int(domain, 16), address

        by_bus = {}
        for line in physical.stdout.splitlines():
            index, bus, mig = (field.strip() for field in line.split(","))
            if mig != "Enabled":
                by_bus[bus_key(bus)] = int(index)
        slots = [by_bus[bus_key(by_device[device])] for device in devices]
    except (AssetError, AttributeError, OSError, ValueError, KeyError, TypeError):
        return result
    layout = local_profile_topology(topology, slots, nccl_profile_network(env))
    return (layout, driver) if layout is not None else result


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
    metadata = {"host": host, "topology": topology, "driver": driver,
                "network": nccl_profile_network(env)}
    uuid = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if uuid.startswith("GPU-") and "," not in uuid:
        index = _system_info(["nvidia-smi", "--id", uuid, "--query-gpu=index", "--format=csv,noheader"])
        if not index.returncode and index.stdout.strip().isascii() and index.stdout.strip().isdigit():
            metadata["topology_device"] = int(index.stdout)
    return metadata


def _compatible_recipe(profile, env, capabilities, packet_address):
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
            or profile["exact_exchange"] and capabilities.get("segmented_exact_exchange_v1") is not True):
        return None
    if backend == "segmented24":
        bounded = bounded_hash_available(capabilities, env)
        if strategy == "batch":
            if not bounded or profile["hash_lookup_series_limit"] != 64:
                return None
        elif bounded:
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
        return None, "EXPLICIT_CAPACITY_RECIPE"
    minimum = [rank.get("free_bytes_at_admission") for rank in profile["rank_geometry"]]
    if any(floor is not None and now["free_bytes"] < floor for now, floor in zip(inventory, minimum)):
        return None, "RECIPE_VRAM_PRESSURE"
    planning = _apply_memory_preset(dict(env, **settings), memory.backend,
                                    "max_capacity" if preset == "maxmemory" else "balanced")
    planning.update(MGBFS_EXPERIMENTAL_SEGMENTED24="1", MGBFS_GENERIC_HISTORY="hash",
                    MGBFS_GENERIC_TRANSPORT="full", MGBFS_GENERIC_SORT="none",
                    MGBFS_GENERIC_GENERATOR="cuda")
    try:
        planning = _match_resident_reserve(planning, inventory)
    except RuntimeError:
        return None, "RECIPE_RESERVE_CONSENSUS"
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
        return None, "RECIPE_REJECTED_BY_LIVE_ADMISSION"
    admission = json.loads(result.stdout)
    plan = admission["plan"]
    defaults = {"compact_transport": False, "compact_route_lookahead": False,
                "compact_packet_slots": 1, "compact_receive_slots": 1}
    if (any(plan.get(key, defaults.get(key)) != profile[key] for key in
            ("shards", "batch", "history_layers", "state_bytes", *defaults))
            or plan["queue_capacity"] < profile["queue_capacity"]
            or profile["owner_cuts"] is not None and admission.get("owner_cuts") != profile["owner_cuts"]):
        return None, "RECIPE_LIVE_GEOMETRY_CHANGED"
    return admission, None


def registry_run_receipt(status, expected, report):
    from .profile_registry import SCHEMA, _COMMON, _PLAN, _case_profile

    if expected is None:
        return status
    receipt = dict(status)
    historical_network = receipt.pop("historical_network_by_rank", None)
    try:
        actual = _case_profile(report, report["devices"], SCHEMA)
    except (ValueError, TypeError, KeyError, IndexError):
        return dict(receipt, runtime_geometry="UNVERIFIED")
    fields = (*_COMMON, *_PLAN, "memory_backend", "owner_cuts")
    rank_fields = ("admitted_batch", "rows_per_bank", "capacity_limit_per_rank", "reserve_bytes")
    matches = (len(actual["rank_geometry"]) == len(expected["rank_geometry"])
               and all(actual[field] == expected[field] for field in fields)
               and all(left[field] == right[field] for left, right in
                       zip(actual["rank_geometry"], expected["rank_geometry"]) for field in rank_fields))
    receipt["runtime_geometry"] = "MATCHED" if matches else "CHANGED"
    policy = report.get("communication_preflight", {}).get("policy") or {}
    network_matches = (historical_network is not None and all(
        network.get(key) == value for key, value in nccl_profile_network(policy).items()
        for network in historical_network
    ))
    if matches and network_matches and "capacity_comparison" in status:
        receipt["capacity_slowdown_status"] = "MEASURED_FULL_BFS_HISTORICAL_BOUND"
    return receipt


def _load_catalog(spec, env):
    source, digest = spec["source"], spec["sha256"]
    cache = artifact_path("profile-registry", env=env, override="MGBFS_PROFILE_REGISTRY_CACHE")
    command = [sys.executable, "-m", "multigpubfs.profile_registry", "import",
               "--cache-dir", str(cache), "--stdout"]
    if digest is not None:
        command.extend(("--sha256", digest))
    if env.get("MGBFS_NATIVE_OFFLINE") == "1" or env.get("MGBFS_PROFILE_REGISTRY_OFFLINE") == "1":
        command.append("--offline")
    command.extend(("--", source))
    # Local files and the pinned cache may also be on a stalled filesystem.
    try:
        result = subprocess.run(command, timeout=profile_timeout(5), capture_output=True)
    except subprocess.TimeoutExpired as error:
        raise ProfileBudgetExpired("REGISTRY_CATALOG_DEADLINE") from error
    if result.returncode:
        raise ValueError("PROFILE_REGISTRY_IMPORT_FAILED")
    return json.loads(result.stdout)


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
            profile_timeout(5)
            if "error" in catalog:
                return None, {"status": "REGISTRY_UNAVAILABLE", "reason": catalog["error"]}
            if not catalog["entries"] or catalog["schema"] < 2:
                return None, status
            supported = hardware()
            if supported is None:
                return None, status
            rank_inventory, devices, topology, drivers = supported
            identity = runtime_identity(native, env)
            rank_metadata = topology.get("rank_metadata") if isinstance(topology, dict) else None
            local_layout = isinstance(topology, dict) and "local_topology" in topology
            matches = matching_profiles(catalog, graph.digest(),
                                        hardware_key(rank_inventory, devices, topology, driver_versions=drivers),
                                        identity)
            historical_network = None
            if matches and local_layout:
                historical_network = [topology["network"]] * len(devices)
            elif matches and rank_metadata and all("topology_device" in rank for rank in rank_metadata):
                historical_network = [rank["network"] for rank in rank_metadata]
            elif matches and rank_metadata is None and nccl_profile_network(env):
                return None, status
            if not matches and local_layout and not topology["network"]:
                matches = matching_profiles(catalog, graph.digest(), hardware_key(
                    rank_inventory, devices, topology["local_topology"], driver_versions=drivers), identity)
            if not matches and rank_metadata is not None:
                # Old single-host evidence has no explicit rank topology or NCCL settings.
                legacy_key = hardware_key(rank_inventory, devices, topology, driver_versions=drivers,
                                          legacy_cluster=True)
                matches = matching_profiles(catalog, graph.digest(), legacy_key, identity)
            namespace = env.get("MGBFS_PROFILE_PRESSURE_NAMESPACE", "ambient")
            acceptable = {None, "ambient"} if namespace == "ambient" else {namespace}
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
            if preset == "maxmemory" and not any(
                entry["capacity_verification"] == "LOWER_PHYSICAL_PEAK" for entry in matches
            ):
                return None, status

            remaining_probes = 3

            def admit(item):
                nonlocal remaining_probes, status
                remaining_probes -= 1
                _, arm, profile = item
                profile_timeout(5)
                recipe_preset = "maxmemory" if arm == "max_capacity" else "balanced"
                candidate = _compatible_recipe(profile, env, capabilities, packet_address)
                if candidate is None:
                    status = {"status": "RECIPE_UNAVAILABLE"}
                    return None
                memory, settings = candidate
                admission, refusal = _live_admit(graph, recipe_preset, native, env, capabilities, inventory(),
                                                 memory, settings, profile)
                if refusal is not None:
                    status = {"status": refusal}
                    return None
                return RegistryRecipe(memory, settings, profile, recipe_preset, admission)

            # This is also the recipe a balanced request would launch for the
            # same matching context. Rejected arms cannot anchor the 4x bound.
            for index, item in enumerate(candidates[:remaining_probes]):
                recipe = admit(item)
                if recipe is not None:
                    break
            else:
                return None, status

            fastest = recipe.evidence["bfs_seconds"]
            if preset == "maxmemory":
                baseline = item
                baseline_peaks = recipe.evidence["peak_vram_bytes_by_rank"]
                baseline_pressure = recipe.evidence["pressure_namespace"]
                if not baseline_peaks or baseline_pressure is None:
                    return None, {"status": "NO_COMPATIBLE_PROFILE"}
                peak = max(baseline_peaks)
                memory_candidates = sorted(
                    (candidate for candidate in candidates[index:]
                     if candidate[1] == "max_capacity"
                     and candidate[0]["capacity_verification"] == "LOWER_PHYSICAL_PEAK"
                     and candidate[2]["pressure_namespace"] == baseline_pressure
                     and (candidate is baseline or max(candidate[2]["peak_vram_bytes_by_rank"]) < peak)
                     and candidate[2]["bfs_seconds"] <= 4 * fastest),
                    key=lambda candidate: (max(candidate[2]["peak_vram_bytes_by_rank"]), candidate[2]["bfs_seconds"]),
                )
                status = {"status": "NO_COMPATIBLE_PROFILE"}
                baseline_recipe = recipe
                for item in memory_candidates:
                    if item is baseline:
                        recipe = baseline_recipe
                        break
                    if remaining_probes:
                        recipe = admit(item)
                        if recipe is not None:
                            break
                else:
                    return None, status

            entry = item[0]
            status = {
                "status": "LIVE_ADMITTED_REGISTRY_PROFILE",
                "evidence_sha256": entry["evidence_sha256"],
                "requested_preset": preset,
                "recipe_preset": recipe.preset,
                "historical_verification": entry["capacity_verification"],
                "historical_ratio": entry["profiles"]["max_capacity"]["bfs_seconds"] / fastest,
                "scope": "FULL_BFS_HISTORICAL_PROFILE",
            }
            if preset == "maxmemory":
                status["capacity_comparison"] = {
                    "baseline_evidence_sha256": baseline[0]["evidence_sha256"],
                    "baseline_arm": baseline[1],
                    "baseline_memory_backend": baseline_recipe.evidence["memory_backend"],
                    "baseline_bfs_seconds": fastest,
                    "selected_memory_backend": recipe.evidence["memory_backend"],
                    "selected_bfs_seconds": recipe.evidence["bfs_seconds"],
                    "pressure_namespace": baseline_pressure,
                }
                status["historical_network_by_rank"] = historical_network
            profile_timeout(5)
            return recipe, status
    except ProfileBudgetExpired:
        return None, {"status": "REGISTRY_STARTUP_DEADLINE"}
    except (OSError, ValueError, TypeError) as error:
        return None, {"status": "REGISTRY_UNAVAILABLE", "reason": type(error).__name__}
