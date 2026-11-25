"""SHA-pinned profile catalogs and independently verified export evidence."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urlsplit
from urllib.request import urlopen

from .artifact_paths import artifact_path
from .graph_definition import GraphDefinition
from .workload_geometry import coxeter_frontier_bound

SCHEMA = 2
MAX_BYTES = 4 << 20
_BACKENDS = {
    "EXPERIMENTAL_SEGMENTED24": "segmented24",
    "EXPERIMENTAL_PERMUTATION_BITMAP": "permutation_bitmap",
}
_PRESETS = ("balanced", "max_capacity")
_RUNTIME = ("native_sha256", "cuda_library_sha256")
_INVENTORY = ("device", "compute_major", "compute_minor", "sm_count", "total_bytes")
_VERIFICATION = "INDEPENDENT_LAYER_COUNTS_AND_EXACT_TERMINAL_SETS"
_COMMON = ("strategy", "owner_lanes", "queue_capacity", "banks", "exact_exchange",
           "resident_codec", "segment_rows", "bitmap_owner_page", "compact_positions",
           "hash_lookup_series_limit", "batch_owner_tile", "batch_hash_merge_rows",
           "batch_hash_prefix_probe", "packet_wire")
_PLAN = ("shards", "batch", "history_layers", "state_bytes", "compact_transport",
         "compact_route_lookahead", "compact_packet_slots", "compact_receive_slots")
_PLAN_DEFAULTS = {"history_layers": 1, "compact_transport": False,
                  "compact_route_lookahead": False, "compact_packet_slots": 1,
                  "compact_receive_slots": 1}
_WIRE = ("protocol", "wire_record_bytes_per_row", "wire_state_bytes_per_row",
         "pack4_direct_owner", "byte_direct_owner", "route_receive_progress",
         "pinned_route_metadata")


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _int(value, low=1, high=(1 << 64) - 1):
    return type(value) is int and low <= value <= high


def _seconds(value):
    return type(value) in (int, float) and 0 < value < float("inf")


def _mapping(value, fields):
    return isinstance(value, dict) and set(value) == set(fields.split())


def _canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _path(value):
    path = Path(value)
    if os.name != "nt":
        return path
    # The digest filename can push an ordinary evidence/cache path over MAX_PATH.
    value = os.path.abspath(path)
    if value.startswith("\\\\?\\"):
        return Path(value)
    return Path("\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value)


def _read(path):
    with _path(path).open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("PROFILE_REGISTRY_OBJECT_TOO_LARGE")
    return data


def _rank_drivers(versions, nranks):
    if isinstance(versions, str):
        versions = [versions] * nranks
    if (not isinstance(versions, (list, tuple)) or len(versions) != nranks
            or any(not isinstance(v, str) or re.fullmatch(r"\d+(?:\.\d+){1,3}", v) is None
                   for v in versions)):
        return None
    return list(versions)


def _topology_text(value):
    if not isinstance(value, str):
        return ""
    plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
    return "\n".join(" ".join(line.split()) for line in plain.splitlines() if line.strip())


def _cluster_topology(topology, devices):
    metadata = topology["rank_metadata"]
    if set(topology) != {"rank_metadata"} or not isinstance(metadata, list) or len(metadata) != len(devices):
        return None
    hosts, nodes, placements, network, drivers = {}, [], [], [], []
    resolved_devices = {}
    for rank, device in zip(metadata, devices):
        if not isinstance(rank, dict) or set(rank) - {"host", "topology", "network", "driver", "topology_device"}:
            return None
        host, text, settings = rank.get("host"), _topology_text(rank.get("topology")), rank.get("network")
        slot = rank.get("topology_device", device)
        if (not isinstance(host, str) or not host.strip() or not text
                or not _int(slot, 0, 4095) or f"GPU{slot}" not in text.splitlines()[0].split()
                or not isinstance(settings, dict)
                or any(not isinstance(key, str) or not isinstance(value, str) for key, value in settings.items())):
            return None
        if host not in hosts:
            hosts[host] = len(nodes)
            nodes.append(text)
        node = hosts[host]
        if nodes[node] != text or [node, slot] in placements:
            return None
        explicit = "topology_device" in rank
        previous = resolved_devices.get((node, device))
        if previous is not None and not (previous and explicit):
            return None
        resolved_devices[node, device] = explicit
        placements.append([node, slot])
        network.append(settings)
        drivers.append(rank.get("driver"))
    if _rank_drivers(drivers, len(devices)) is None:
        return None
    return nodes, placements, network, drivers


def hardware_key(inventory, devices, topology, *, driver_versions=None, legacy_cluster=False):
    """Fingerprint ordered ranks, the whole selected topology and driver versions."""
    if (not isinstance(devices, (list, tuple)) or not 1 <= len(devices) <= 128
            or any(not _int(device, 0, 4095) for device in devices)
            or not isinstance(inventory, (list, tuple))):
        return None
    if isinstance(topology, dict) and "rank_metadata" in topology:
        projected = _cluster_topology(topology, devices)
        if (projected is None or len(inventory) != len(devices)
                or any(not isinstance(row, dict) or row.get("device") != device
                       for row, device in zip(inventory, devices))):
            return None
        nodes, placements, network, drivers = projected
        if legacy_cluster:
            if len(nodes) != 1 or any(network) or [slot for _, slot in placements] != list(devices):
                return None
            topology_bytes = nodes[0].encode("utf-8")
        else:
            topology_bytes = _canonical({"nodes": nodes, "ranks": placements, "network": network})
        if driver_versions is not None and _rank_drivers(driver_versions, len(devices)) != drivers:
            return None
        driver_versions = drivers
        selected = inventory
    else:
        if len(set(devices)) != len(devices):
            return None
        if isinstance(topology, str):
            topology_bytes = _topology_text(topology).encode("utf-8")
        elif isinstance(topology, dict) and topology and driver_versions is not None:
            topology_bytes = _canonical(topology)
        else:
            return None
        if not topology_bytes:
            return None
        by_device = {row.get("device"): row for row in inventory if isinstance(row, dict)}
        selected = [by_device.get(device, {}) for device in devices]
    ranks = []
    for item in selected:
        major, minor = item.get("compute_major"), item.get("compute_minor")
        sm_count, total = item.get("sm_count"), item.get("total_bytes")
        if not (_int(major, 1, 99) and _int(minor, 0, 9) and _int(sm_count, 1, 4096) and _int(total)):
            return None
        ranks.append({"sm": 10 * major + minor, "sm_count": sm_count, "vram_bytes": total})
    result = {"ranks": ranks, "topology_sha256": _sha256(topology_bytes)}
    if driver_versions is not None:
        drivers = _rank_drivers(driver_versions, len(ranks))
        if drivers is None:
            return None
        result["driver_versions"] = drivers
    return result


def _rank_values(values, nranks, *, low=0):
    return isinstance(values, list) and len(values) == nranks and all(_int(x, low) for x in values)


def _validate_profile(value, nranks, schema):
    fields = "memory_backend strategy shards owner_lanes queue_capacity exact_exchange banks bfs_seconds peak_resident_rows_by_rank"
    if schema == 2:
        fields += (" " + " ".join(sorted(set(_COMMON + _PLAN) - set(fields.split()))) +
                   " rank_geometry owner_cuts peak_vram_bytes_by_rank pressure_namespace")
    if not _mapping(value, fields):
        raise ValueError("PROFILE_REGISTRY_GEOMETRY")
    backend = value["memory_backend"]
    if (backend not in _BACKENDS.values() or value["strategy"] not in ("atomic", "warp", "batch")
            or not _int(value["shards"], 1, 4096)
            or not _int(value["owner_lanes"], 1, min(64, value["shards"]))
            or not _int(value["queue_capacity"], 1, (1 << 32) - 1)
            or type(value["exact_exchange"]) is not bool or type(value["banks"]) is not int
            or value["banks"] not in ((2, 3) if backend == "permutation_bitmap" else (1, 3))
            or not _seconds(value["bfs_seconds"])
            or not _rank_values(value["peak_resident_rows_by_rank"], nranks)):
        raise ValueError("PROFILE_REGISTRY_GEOMETRY")
    if schema == 1:
        return
    codec, wire = value["resident_codec"], value["packet_wire"]
    tx, rx = value["compact_packet_slots"], value["compact_receive_slots"]
    bitmap_page = value["bitmap_owner_page"]
    if (not _int(value["batch"], 1, (1 << 32) - 1)
            or not _int(value["state_bytes"], 1, (1 << 32) - 1)
            or not _int(value["history_layers"], 1, 3)
            or not _int(value["segment_rows"], 1, 1 << 32)
            or codec not in ("unpacked-soa", "pack4-v1", "alphabet8-v1")
            or (codec == "pack4-v1" and (backend != "permutation_bitmap" or value["strategy"] != "atomic"))
            or (codec == "alphabet8-v1" and backend != "segmented24")
            or any(type(value[key]) is not bool for key in ("compact_transport", "compact_route_lookahead",
                                                           "compact_positions", "batch_hash_prefix_probe"))
            or not _int(tx, 1, 3) or tx not in (1, 3) or not _int(rx, 1, tx) or rx not in (1, 3)
            or (not value["compact_transport"] and (tx != 1 or rx != 1 or value["compact_route_lookahead"]))
            or (bitmap_page is not None and (not _int(bitmap_page, 0, 1 << 20)
                or bitmap_page & (bitmap_page - 1)))
            or any(value[key] is not None and not _int(value[key])
                   for key in ("hash_lookup_series_limit", "batch_owner_tile", "batch_hash_merge_rows"))
            or (value["compact_transport"] != (wire is not None))
            or (wire is not None and (nranks == 1 or not _mapping(wire, " ".join(_WIRE))
                or wire["protocol"] != ("nccl_compact_credit_v2" if rx == 3 else "nccl_compact_credit_v1")
                or not _int(wire["wire_record_bytes_per_row"], high=32)
                or not _int(wire["wire_state_bytes_per_row"], 0, (1 << 32) - 1)
                or any(type(wire[key]) is not bool for key in _WIRE[3:])))):
        raise ValueError("PROFILE_REGISTRY_GEOMETRY")
    ranks, cuts, peaks = value["rank_geometry"], value["owner_cuts"], value["peak_vram_bytes_by_rank"]
    pressure = value["pressure_namespace"]
    if (pressure is not None and (not isinstance(pressure, str) or len(pressure) > 128 or not pressure.strip())
            or not isinstance(ranks, list) or len(ranks) != nranks
            or any(not _mapping(rank, "capacity admitted_batch rows_per_bank capacity_limit_per_rank reserve_bytes free_bytes_at_admission additional_reserve_bytes")
                   or not all(_int(rank[k]) for k in ("capacity", "admitted_batch", "rows_per_bank"))
                   or rank["admitted_batch"] > value["batch"]
                   or (rank["capacity_limit_per_rank"] is not None and not _int(rank["capacity_limit_per_rank"]))
                   or any(rank[key] is not None and not _int(rank[key], 0)
                          for key in ("reserve_bytes", "free_bytes_at_admission", "additional_reserve_bytes"))
                   for rank in ranks)
            or (cuts is not None and (not isinstance(cuts, list) or len(cuts) != nranks + 1
                or any(not _int(x, 0, 1 << 32) for x in cuts)
                or cuts[0] != 0 or cuts[-1] != 1 << 32
                or any(a >= b for a, b in zip(cuts, cuts[1:]))))
            or (peaks is not None and not _rank_values(peaks, nranks, low=1))):
        raise ValueError("PROFILE_REGISTRY_RANK_GEOMETRY")


def _capacity_verification(profiles):
    pressure = profiles["balanced"]["pressure_namespace"]
    if pressure is None or pressure != profiles["max_capacity"]["pressure_namespace"]:
        return "UNVERIFIED"
    balanced, capacity = (profiles[preset]["peak_vram_bytes_by_rank"] for preset in _PRESETS)
    return "LOWER_PHYSICAL_PEAK" if balanced and capacity and max(capacity) < max(balanced) else "UNVERIFIED"


def _runtime_projection(runtime):
    return {key: runtime.get(key) for key in _RUNTIME} if isinstance(runtime, dict) else None


def validate_registry(value):
    if (not _mapping(value, "schema entries") or type(value["schema"]) is not int
            or value["schema"] not in (1, SCHEMA)):
        raise ValueError("PROFILE_REGISTRY_SCHEMA")
    schema = value["schema"]
    if not isinstance(value["entries"], list) or len(value["entries"]) > 2048:
        raise ValueError("PROFILE_REGISTRY_ENTRIES")
    fields = "graph_digest hardware runtime profiles evidence_sha256 verification"
    if schema == 2:
        fields += " capacity_verification"
    seen = set()
    for entry in value["entries"]:
        if not _mapping(entry, fields):
            raise ValueError("PROFILE_REGISTRY_ENTRY")
        if (not _digest(entry["graph_digest"]) or not _digest(entry["evidence_sha256"])
                or entry["evidence_sha256"] in seen or entry["verification"] != _VERIFICATION):
            raise ValueError("PROFILE_REGISTRY_IDENTITY")
        seen.add(entry["evidence_sha256"])
        hardware = entry["hardware"]
        hardware_fields = "ranks topology_sha256" + (" driver_versions" if schema == 2 else "")
        if (not _mapping(hardware, hardware_fields) or not _digest(hardware["topology_sha256"])
                or not isinstance(hardware["ranks"], list) or not 1 <= len(hardware["ranks"]) <= 128
                or (schema == 2 and (not isinstance(hardware["driver_versions"], list)
                    or _rank_drivers(hardware["driver_versions"], len(hardware["ranks"])) is None))):
            raise ValueError("PROFILE_REGISTRY_HARDWARE")
        for rank in hardware["ranks"]:
            if (not _mapping(rank, "sm sm_count vram_bytes") or not _int(rank["sm"], 10, 999)
                    or not _int(rank["sm_count"], 1, 4096) or not _int(rank["vram_bytes"])):
                raise ValueError("PROFILE_REGISTRY_HARDWARE")
        runtime = entry["runtime"]
        if not _mapping(runtime, " ".join(_RUNTIME)) or not all(_digest(x) for x in runtime.values()):
            raise ValueError("PROFILE_REGISTRY_RUNTIME")
        profiles = entry["profiles"]
        if not _mapping(profiles, " ".join(_PRESETS)):
            raise ValueError("PROFILE_REGISTRY_PRESETS")
        for preset in _PRESETS:
            _validate_profile(profiles[preset], len(hardware["ranks"]), schema)
        if profiles["max_capacity"]["bfs_seconds"] > 4 * profiles["balanced"]["bfs_seconds"]:
            raise ValueError("PROFILE_REGISTRY_CAPACITY_TOO_SLOW")
        if schema == 2 and entry["capacity_verification"] != _capacity_verification(profiles):
            raise ValueError("PROFILE_REGISTRY_CAPACITY_VERIFICATION")
    return value


def _write_bytes(path, data):
    path = _path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pr-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def _catalog_lock(path):
    # The lock file stays in place so all publishers lock the same inode.
    lock = path.with_name(path.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _decode(data, digest=None):
    if len(data) > MAX_BYTES:
        raise ValueError("PROFILE_REGISTRY_OBJECT_TOO_LARGE")
    if digest is not None and _sha256(data) != digest:
        raise ValueError("PROFILE_REGISTRY_DIGEST")
    return validate_registry(json.loads(data))


def _github_source(url, pin):
    parts = urlsplit(url)
    fields = parts.path.strip("/").split("/")
    commit_file = (parts.netloc == "raw.githubusercontent.com" and len(fields) >= 4
                   and re.fullmatch(r"[0-9a-f]{40}", fields[2]))
    release_file = (parts.netloc == "github.com" and len(fields) == 6
                    and fields[2:4] == ["releases", "download"])
    if (not (commit_file or release_file) or any(not field for field in fields)
            or parts.query or parts.fragment or not _digest(pin)):
        raise ValueError("PROFILE_REGISTRY_GITHUB_PIN_REQUIRED")
    return fields if release_file else None


def _download_timeout(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("PROFILE_REGISTRY_DEADLINE")
    return remaining


def _public_catalog_bytes(url, deadline):
    with urlopen(url, timeout=_download_timeout(deadline)) as stream:
        read = getattr(stream, "read1", stream.read)
        data = bytearray()
        while len(data) <= MAX_BYTES:
            _download_timeout(deadline)
            chunk = read(min(65536, MAX_BYTES + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        _download_timeout(deadline)
    return data


def _authenticated_release_bytes(release, sha256, deadline):
    owner, repo, _, _, tag, filename = release
    tag = unquote(tag)
    if (any(part in (".", "..") or re.fullmatch(r"[A-Za-z0-9_.-]+", part) is None
            for part in (owner, repo)) or tag in (".", "..")):
        raise ValueError("PROFILE_REGISTRY_SOURCE")
    endpoint = f"repos/{owner}/{repo}/releases"
    env = dict(os.environ, GH_PROMPT_DISABLED="1", GH_NO_UPDATE_NOTIFIER="1")

    def github(path, *options):
        try:
            return subprocess.run(
                ["gh", "api", "--hostname", "github.com", "--method", "GET", path, *options],
                stdin=subprocess.DEVNULL, capture_output=True, env=env,
                timeout=_download_timeout(deadline), check=True,
            ).stdout
        except subprocess.TimeoutExpired as error:
            raise TimeoutError("PROFILE_REGISTRY_DEADLINE") from error
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise FileNotFoundError("PROFILE_REGISTRY_PRIVATE_RELEASE_UNAVAILABLE") from error

    exact_name = json.dumps(unquote(filename))
    selector = f".assets[] | select(.name == {exact_name}) | {{id,size,digest}}"
    metadata = github(f"{endpoint}/tags/{quote(tag, safe='')}", "--jq", selector)
    if not metadata:
        raise FileNotFoundError("PROFILE_REGISTRY_GITHUB_ASSET_NOT_FOUND")
    asset = json.loads(metadata)
    if not _int(asset.get("id")) or not _int(asset.get("size"), high=MAX_BYTES):
        raise ValueError("PROFILE_REGISTRY_GITHUB_ASSET_SIZE")
    if asset.get("digest") not in (None, f"sha256:{sha256}"):
        raise ValueError("PROFILE_REGISTRY_DIGEST")
    data = github(f"{endpoint}/assets/{asset['id']}", "-H", "Accept: application/octet-stream")
    if len(data) != asset["size"]:
        raise ValueError("PROFILE_REGISTRY_GITHUB_ASSET_SIZE")
    return data


def read_profile_registry(source, *, sha256=None, timeout=5, cache_dir=None, offline=False):
    """Read a local catalog or a GitHub object pinned to its SHA-256."""
    name = str(source)
    if "://" not in name:
        return _decode(_read(source), sha256)
    if urlsplit(name).scheme != "https":
        raise ValueError("PROFILE_REGISTRY_SOURCE")
    release = _github_source(name, sha256)
    if cache_dir is None:
        cache_dir = artifact_path("profile-registry", override="MGBFS_PROFILE_REGISTRY_CACHE")
    cache = Path(cache_dir) / (sha256 + ".json")
    try:
        return _decode(_read(cache), sha256)
    except FileNotFoundError as error:
        if offline:
            raise FileNotFoundError("PROFILE_REGISTRY_OFFLINE_CACHE_MISS") from error
    except (OSError, ValueError):
        if offline:
            raise
    if not _seconds(timeout):
        raise TimeoutError("PROFILE_REGISTRY_DEADLINE")
    deadline = time.monotonic() + timeout
    try:
        data = _public_catalog_bytes(name, deadline)
    except HTTPError as error:
        if release is None or error.code not in (401, 403, 404):
            raise
        error.close()
        data = _authenticated_release_bytes(release, sha256, deadline)
    result = _decode(data, sha256)
    try:
        _write_bytes(cache, data)
    except OSError:
        pass  # Online reads can still succeed when the optional cache is unwritable.
    return result


def matching_profiles(registry, graph_digest, hardware_key, runtime):
    """Return paired v2 recipes for the same runtime and ordered hardware."""
    if registry["schema"] != SCHEMA or hardware_key is None:
        return []
    identity = _runtime_projection(runtime)
    return sorted((entry for entry in registry["entries"]
                   if entry["graph_digest"] == graph_digest and entry["hardware"] == hardware_key
                   and entry["runtime"] == identity),
                  key=lambda entry: (entry["profiles"]["balanced"]["bfs_seconds"],
                                     entry["profiles"]["max_capacity"]["bfs_seconds"],
                                     entry["evidence_sha256"]))


def matching_profile(registry, graph_digest, hardware_key, runtime):
    return next(iter(matching_profiles(registry, graph_digest, hardware_key, runtime)), None)


def _oracle(graph):
    if coxeter_frontier_bound(graph) is None:
        layers = graph.exact_layers(32768)
        return [len(layer) for layer in layers], layers[-1], layers[-2] if len(layers) > 1 else []
    sizes = [1]
    for n in range(2, graph.state_elements + 1):
        result = [0] * (len(sizes) + n - 1)
        for k, count in enumerate(sizes):
            for shift in range(n):
                result[k + shift] += count
        sizes = result
    final = list(reversed(graph.start))
    previous = []
    for i in range(len(final) - 1):
        state = final.copy()
        state[i], state[i + 1] = state[i + 1], state[i]
        previous.append(state)
    return sizes, [final], previous


def _states_match(actual, expected, width):
    return (isinstance(actual, list) and len(actual) == len(expected)
            and all(isinstance(row, list) and len(row) == width
                    and all(_int(v, -(1 << 63), (1 << 63) - 1) for v in row) for row in actual)
            and sorted(map(tuple, actual)) == sorted(map(tuple, expected)))


def _verify_terminal(case, report, graph, sizes, final, previous):
    if not isinstance(case["states"], str):
        raise ValueError("PROFILE_EVIDENCE_CASE")
    states = case["states"].encode("utf-8")
    if _sha256(states) != report.get("states_sha256"):
        raise ValueError("PROFILE_EVIDENCE_STATE_DIGEST")
    terminal = case.get("terminal_states")
    if terminal is not None:
        if not isinstance(terminal, str):
            raise ValueError("PROFILE_EVIDENCE_CASE")
        witness = terminal.encode("utf-8")
        if _sha256(witness) != report.get("terminal_states_sha256"):
            raise ValueError("PROFILE_EVIDENCE_TERMINAL_DIGEST")
        data = json.loads(witness)
        if (not _mapping(data, "schema graph_digest depth current previous_small")
                or not _int(data["schema"], 1, 1) or data["graph_digest"] != graph.digest()
                or not _int(data["depth"], len(sizes) - 1, len(sizes) - 1)):
            raise ValueError("PROFILE_EVIDENCE_TERMINAL_SCOPE")
    else:
        data = json.loads(states)
    if (not isinstance(data, dict) or not _states_match(data.get("current"), final, graph.state_elements)
            or not _states_match(data.get("previous_small"), previous, graph.state_elements)):
        raise ValueError("PROFILE_EVIDENCE_TERMINAL_SETS")


def _packet_wire(packet):
    if packet is None:
        return None
    if not isinstance(packet, dict):
        raise ValueError("PROFILE_EVIDENCE_RANK_GEOMETRY")
    return {key: packet.get(key) for key in _WIRE}


def _case_profile(report, devices, schema):
    ranks = report.get("segment_memory_by_rank", [])
    if (not isinstance(ranks, list) or len(ranks) != len(devices)
            or any(not isinstance(r, dict) or r.get("rank") != i or r.get("device") != devices[i]
                   for i, r in enumerate(ranks))):
        raise ValueError("PROFILE_EVIDENCE_RANKS")
    profile = {"memory_backend": _BACKENDS[report["backend"]], "bfs_seconds": report["bfs_seconds"],
               "peak_resident_rows_by_rank": [rank.get("peak_live_rows") for rank in ranks]}
    names = {"strategy": "segment_strategy", "queue_capacity": "actual_queue_capacity",
             "segment_rows": "configured_segment_rows", "packet_wire": "packet_transport"}
    common = _COMMON if schema == 2 else ("strategy", "owner_lanes", "queue_capacity", "exact_exchange", "banks")
    for key in common:
        values = [rank.get(names.get(key, key)) for rank in ranks]
        if key == "packet_wire":
            values = [_packet_wire(packet) for packet in values]
        if any(value != values[0] for value in values[1:]):
            raise ValueError("PROFILE_EVIDENCE_RANK_GEOMETRY")
        profile[key] = values[0]
    plan = report.get("plan", {})
    if schema == 1:
        profile["shards"] = plan.get("shards")
    else:
        plans = report.get("rank_plans", [])
        if (not _int(report.get("size_profile_count"), 0, 0) or not isinstance(plans, list)
                or len(plans) != len(devices) or not all(isinstance(rank, dict) for rank in plans)):
            raise ValueError("PROFILE_EVIDENCE_STATIC_GEOMETRY")
        for key in _PLAN:
            profile[key] = plan.get(key, _PLAN_DEFAULTS.get(key))
            if any(rank.get(key, _PLAN_DEFAULTS.get(key)) != profile[key] for rank in plans):
                raise ValueError("PROFILE_EVIDENCE_RANK_PLAN")
        admitted = report.get("profile_admission_by_rank", [{} for _ in ranks])
        if (not isinstance(admitted, list) or len(admitted) != len(ranks)
                or not all(isinstance(item, dict) for item in admitted)):
            raise ValueError("PROFILE_EVIDENCE_RANKS")
        profile["rank_geometry"] = [
            {"capacity": p.get("capacity"),
             **{key: rank.get(key) for key in ("admitted_batch", "rows_per_bank",
                                              "capacity_limit_per_rank", "reserve_bytes")},
             **{key: admission.get(key) for key in ("free_bytes_at_admission", "additional_reserve_bytes")}}
            for p, rank, admission in zip(plans, ranks, admitted)
        ]
        profile["owner_cuts"] = report.get("owner_cuts") if profile["memory_backend"] == "segmented24" else None
        profile["peak_vram_bytes_by_rank"] = report.get("profile_peak_vram_bytes_by_rank")
        profile["pressure_namespace"] = report.get("profile_pressure_namespace")
        for rank, p in zip(ranks, plans):
            packet = rank.get("packet_transport")
            # The owner may lower an admitted producer window before allocating it.
            if (not _int(p.get("queue_capacity"), profile["queue_capacity"])
                    or (packet is not None and (packet.get("tx_exchange_slots") != profile["compact_packet_slots"]
                        or packet.get("rx_exchange_slots") != profile["compact_receive_slots"]))):
                raise ValueError("PROFILE_EVIDENCE_RANK_PLAN")
    _validate_profile(profile, len(devices), schema)
    return profile


def verify_evidence(data):
    if len(data) > MAX_BYTES:
        raise ValueError("PROFILE_REGISTRY_OBJECT_TOO_LARGE")
    evidence = json.loads(data)
    schema = evidence.get("schema") if isinstance(evidence, dict) else None
    fields = "schema graph inventory devices topology cases"
    if schema == 2:
        fields += " driver_versions"
    if type(schema) is not int or schema not in (1, SCHEMA) or not _mapping(evidence, fields):
        raise ValueError("PROFILE_EVIDENCE_SCHEMA")
    graph = GraphDefinition.from_dict(evidence["graph"])
    devices = evidence["devices"]
    topology = evidence["topology"]
    cluster = isinstance(topology, dict) and "rank_metadata" in topology
    if (not isinstance(devices, list) or not 1 <= len(devices) <= 128
            or any(not _int(x, 0, 4095) for x in devices)
            or (len(set(devices)) != len(devices) and not cluster)):
        raise ValueError("PROFILE_EVIDENCE_DEVICE_SET")
    key = hardware_key(evidence["inventory"], devices, topology,
                       driver_versions=evidence.get("driver_versions"))
    if key is None or not _mapping(evidence["cases"], " ".join(_PRESETS)):
        raise ValueError("PROFILE_EVIDENCE_HARDWARE_OR_CASES")
    hosts = [rank["host"] for rank in topology["rank_metadata"]] if cluster else None
    sizes, final, previous = _oracle(graph)
    profiles = {}
    runtime = None
    for preset in _PRESETS:
        case = evidence["cases"][preset]
        expected = "report states terminal_states" if isinstance(case, dict) and "terminal_states" in case else "report states"
        if cluster:
            expected += " inventory"
        if not _mapping(case, expected) or not isinstance(case["report"], dict):
            raise ValueError("PROFILE_EVIDENCE_CASE")
        report = case["report"]
        if cluster and (report.get("rank_hostnames") != hosts or case["inventory"] != evidence["inventory"]):
            raise ValueError("PROFILE_EVIDENCE_RANK_PLACEMENT")
        measured = report.get("layer_sizes")
        if (report.get("status") != "COMPLETE" or report.get("reason") != "FRONTIER_EXHAUSTED"
                or not _rank_values(measured, len(sizes), low=1) or measured != sizes
                or report.get("devices") != devices
                or report.get("graph_digest") != graph.digest() or report.get("backend") not in tuple(_BACKENDS)):
            raise ValueError("PROFILE_EVIDENCE_GRAPH_OR_COMPLETION")
        _verify_terminal(case, report, graph, sizes, final, previous)
        identity = _runtime_projection(report.get("runtime_identity"))
        if identity is None or not all(_digest(x) for x in identity.values()):
            raise ValueError("PROFILE_EVIDENCE_RUNTIME")
        if runtime is not None and identity != runtime:
            raise ValueError("PROFILE_EVIDENCE_RUNTIME_MISMATCH")
        runtime = identity
        profiles[preset] = _case_profile(report, devices, schema)
    entry = {"graph_digest": graph.digest(), "hardware": key, "runtime": runtime,
             "profiles": profiles, "evidence_sha256": _sha256(data), "verification": _VERIFICATION}
    if schema == 2:
        entry["capacity_verification"] = _capacity_verification(profiles)
    validate_registry({"schema": schema, "entries": [entry]})
    return entry


def _project_inventory(inventory):
    return [{key: rank[key] for key in _INVENTORY} for rank in inventory]


def _case(directory, *, cluster=False):
    directory = Path(directory)
    report = json.loads(_read(directory / "report.json"))
    fields = ("status", "reason", "graph_digest", "layer_sizes", "backend", "devices", "bfs_seconds",
              "states_sha256", "terminal_states_sha256", "owner_cuts", "size_profile_count",
              "profile_peak_vram_bytes_by_rank", "profile_pressure_namespace")
    projected = {key: report[key] for key in fields if key in report}
    projected["runtime_identity"] = _runtime_projection(report.get("runtime_identity"))
    plan_fields = _PLAN + ("capacity", "queue_capacity")
    projected["plan"] = {key: report["plan"][key] for key in plan_fields if key in report["plan"]}
    projected["rank_plans"] = [
        {key: plan[key] for key in plan_fields if key in plan} for plan in report["rank_plans"]
    ]
    if "profile_admission_by_rank" in report:
        projected["profile_admission_by_rank"] = [
            {key: rank[key] for key in ("free_bytes_at_admission", "additional_reserve_bytes") if key in rank}
            for rank in report["profile_admission_by_rank"]
        ]
    rank_fields = ("rank", "device", "segment_strategy", "owner_lanes", "actual_queue_capacity",
                   "banks", "exact_exchange", "resident_codec", "configured_segment_rows",
                   "bitmap_owner_page", "compact_positions", "hash_lookup_series_limit", "batch_owner_tile",
                   "batch_hash_merge_rows", "batch_hash_prefix_probe", "admitted_batch", "rows_per_bank",
                   "capacity_limit_per_rank", "reserve_bytes", "peak_live_rows")
    projected["segment_memory_by_rank"] = [
        {**{key: rank[key] for key in rank_fields if key in rank},
         "packet_transport": None if rank.get("packet_transport") is None else
         {key: rank["packet_transport"].get(key) for key in _WIRE + ("tx_exchange_slots", "rx_exchange_slots")}}
        for rank in report["segment_memory_by_rank"]
    ]
    value = {"report": projected, "states": _read(directory / "states.json").decode("utf-8")}
    if cluster:
        launch = json.loads(_read(directory / "launch.json"))
        if launch.get("graph_digest") != report.get("graph_digest") or launch.get("devices") != report.get("devices"):
            raise ValueError("PROFILE_EVIDENCE_RANK_PLACEMENT")
        projected["rank_hostnames"] = report.get("rank_hostnames")
        value["inventory"] = _project_inventory(launch["inventory"])
    if "terminal_states_sha256" in report:
        value["terminal_states"] = _read(directory / "terminal_states.json").decode("utf-8")
    return value


def _write_registry(path, payload):
    data = _canonical(validate_registry(payload))
    if len(data) > MAX_BYTES:
        raise ValueError("PROFILE_REGISTRY_OBJECT_TOO_LARGE")
    _write_bytes(path, data)


def export_profiles(graph, inventory, topology, baseline, capacity, registry, evidence_dir):
    """Add one independently checked pair of full traversals to a local v2 catalog."""
    cluster = isinstance(topology, dict) and "rank_metadata" in topology
    data = _canonical({"schema": SCHEMA, "graph": graph, "devices": inventory["devices"],
                       "inventory": _project_inventory(inventory["inventory"]),
                       "driver_versions": inventory.get("driver_versions"), "topology": topology,
                       "cases": {"balanced": _case(baseline, cluster=cluster),
                                 "max_capacity": _case(capacity, cluster=cluster)}})
    entry = verify_evidence(data)
    registry = _path(Path(registry).resolve())
    evidence = _path(evidence_dir) / (entry["evidence_sha256"] + ".json")
    with _catalog_lock(registry):
        payload = read_profile_registry(registry) if registry.exists() else {"schema": SCHEMA, "entries": []}
        if payload["schema"] != SCHEMA:
            raise ValueError("PROFILE_REGISTRY_EXPORT_SCHEMA")
        if evidence.exists():
            if _read(evidence) != data:
                raise ValueError("PROFILE_EVIDENCE_DIGEST")
        else:
            _write_bytes(evidence, data)
        if entry not in payload["entries"]:
            payload["entries"].append(entry)
            payload["entries"].sort(key=lambda item: item["evidence_sha256"])
            _write_registry(registry, payload)
    return entry


def verify_publication(registry, evidence_dir):
    raw = _read(registry)
    payload = _decode(raw)
    for entry in payload["entries"]:
        data = _read(Path(evidence_dir) / (entry["evidence_sha256"] + ".json"))
        if verify_evidence(data) != entry:
            raise ValueError("PROFILE_PUBLICATION_ENTRY_MISMATCH")
    return raw


def publish_github_registry(registry, evidence_dir, *, repo, release):
    """Attach one verified, content-addressed catalog to an existing GitHub release."""
    if (not isinstance(repo, str) or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) is None
            or not isinstance(release, str) or not release or release.startswith("-")):
        raise ValueError("PROFILE_GITHUB_RELEASE")
    raw = verify_publication(registry, evidence_dir)
    digest = _sha256(raw)
    asset_name = f"profiles-{digest}.json"

    def existing():
        result = subprocess.run(
            ["gh", "api", f"repos/{repo}/releases/tags/{quote(release, safe='')}"],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return next((asset for asset in json.loads(result.stdout)["assets"] if asset["name"] == asset_name), None)

    asset = existing()
    if asset is None:
        with tempfile.TemporaryDirectory(prefix="mgbfs-profile-publish-") as folder:
            base = Path(folder) / asset_name
            base.write_bytes(raw)
            upload = subprocess.run(["gh", "release", "upload", release, os.fspath(base), "--repo", repo],
                                    capture_output=True, text=True, timeout=60)
        asset = existing()
        if asset is None:
            upload.check_returncode()
            raise RuntimeError("PROFILE_GITHUB_ASSET_NOT_FOUND")
    url = asset.get("browser_download_url")
    if asset.get("digest") != f"sha256:{digest}" or not isinstance(url, str):
        raise RuntimeError("PROFILE_GITHUB_ASSET_DIGEST_MISMATCH")
    _github_source(url, digest)
    return {"url": url, "sha256": digest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    for field in ("graph", "inventory", "baseline", "capacity", "registry", "evidence-dir"):
        export.add_argument("--" + field, required=True)
    export.add_argument("--topology", required=True, help="nvidia-smi topo -m output or rank metadata JSON")
    load = commands.add_parser("import")
    load.add_argument("source")
    load.add_argument("--sha256")
    load.add_argument("--cache-dir")
    load.add_argument("--offline", action="store_true")
    load.add_argument("--output", required=True)
    verify = commands.add_parser("verify-publication")
    verify.add_argument("--registry", required=True)
    verify.add_argument("--evidence-dir", required=True)
    publish = commands.add_parser("publish-github")
    for field in ("registry", "evidence-dir", "repo", "release"):
        publish.add_argument("--" + field, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        topology = _read(args.topology).decode("utf-8")
        if topology.lstrip().startswith("{"):
            topology = json.loads(topology)
        export_profiles(json.loads(_read(args.graph)), json.loads(_read(args.inventory)),
                        topology, args.baseline, args.capacity,
                        args.registry, args.evidence_dir)
    elif args.command == "import":
        catalog = read_profile_registry(args.source, sha256=args.sha256, cache_dir=args.cache_dir,
                                        offline=args.offline)
        _write_registry(args.output, catalog)
    elif args.command == "publish-github":
        print(json.dumps(publish_github_registry(args.registry, args.evidence_dir,
                                                repo=args.repo, release=args.release), sort_keys=True))
    else:
        verify_publication(args.registry, args.evidence_dir)


if __name__ == "__main__":
    main()
