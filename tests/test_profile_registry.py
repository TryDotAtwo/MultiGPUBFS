from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from urllib.error import HTTPError

import pytest

from multigpubfs.graph_definition import GraphDefinition
from multigpubfs.profile_registry import (
    export_profiles,
    hardware_key,
    main,
    matching_profile,
    publish_github_registry,
    read_profile_registry,
    verify_publication,
)
from multigpubfs.profile_selection import cluster_hardware
from multigpubfs.traversal_receipt import merge_terminal_evidence, read_terminal_evidence, write_terminal_evidence


RUNTIME = {"native_sha256": "a" * 64, "cuda_library_sha256": "b" * 64}
TOPOLOGY = "GPU4 GPU2\nGPU4 X PHB\nGPU2 PHB X\n"
INVENTORY = {"devices": [4, 2], "driver_versions": ["535.14", "535.14"], "inventory": [
    {"device": 2, "compute_major": 8, "compute_minor": 6, "sm_count": 28, "total_bytes": 12 << 30},
    {"device": 4, "compute_major": 9, "compute_minor": 0, "sm_count": 132, "total_bytes": 80 << 30},
]}


def _write(path, value):
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _case(path, graph, layers, *, seconds, pack4=False, full=False, compact=None, devices=INVENTORY["devices"]):
    path.mkdir()
    compact = layers if compact is None else compact
    digest = graph.digest()
    terminal = {"current": layers[-1], "previous_small": layers[-2]}
    plan = {"capacity": 8192, "batch": 128, "shards": 2, "state_bytes": 8 * graph.state_elements,
            "queue_capacity": 4096, "compact_transport": pack4}
    packet = None
    if pack4:
        plan.update(compact_route_lookahead=True, compact_packet_slots=3, compact_receive_slots=3)
        packet = {"protocol": "nccl_compact_credit_v2", "wire_state_bytes_per_row": 0,
                  "wire_record_bytes_per_row": 16, "pack4_direct_owner": True,
                  "byte_direct_owner": False, "split_destination_windows": True,
                  "route_receive_progress": True, "pinned_route_metadata": True,
                  "tx_exchange_slots": 3, "rx_exchange_slots": 3}
    rows = [{"rank": i, "device": device, "segment_strategy": "atomic", "owner_lanes": 2,
             "exact_exchange": not pack4, "actual_queue_capacity": 3072 if pack4 else 4096,
             "banks": 2 if pack4 else 1, "bitmap_owner_page": 0 if pack4 else None,
             "resident_codec": "pack4-v1" if pack4 else "unpacked-soa", "configured_segment_rows": 2048,
             "compact_positions": False, "batch_hash_prefix_probe": False,
             "admitted_batch": 128 if pack4 else 64, "rows_per_bank": 512 + i,
             "peak_live_rows": 2, "packet_transport": packet} for i, device in enumerate(devices)]
    report = {
        "status": "COMPLETE", "reason": "FRONTIER_EXHAUSTED", "layer_sizes": [len(layer) for layer in layers],
        "graph_digest": digest, "devices": devices, "bfs_seconds": seconds,
        "backend": "EXPERIMENTAL_PERMUTATION_BITMAP" if pack4 else "EXPERIMENTAL_SEGMENTED24",
        "states_sha256": _write(path / "states.json", {"current": compact[-1], "previous_small": compact[-2]}),
        "plan": plan, "rank_plans": [plan, dict(plan, capacity=8191)], "segment_memory_by_rank": rows,
        "size_profile_count": 0, "runtime_identity": RUNTIME, "profile_pressure_namespace": "fixed-headroom",
        "profile_peak_vram_bytes_by_rank": [11, 12] if pack4 else [16, 13],
    }
    if full:
        ranks = [{key: rows[rank::2] for key, rows in terminal.items()} for rank in range(2)]
        write_terminal_evidence(path, merge_terminal_evidence(report, ranks))
        assert read_terminal_evidence(path, report)["depth"] == len(layers) - 1
    _write(path / "report.json", report)
    return path


def test_pair_export_pin_cache_and_exact_matching(tmp_path, monkeypatch):
    graph = GraphDefinition.permutation([[1, 0, 2], [0, 2, 1]], [0, 1, 2])
    layers = [[[0, 1, 2]], [[0, 2, 1], [1, 0, 2]], [[1, 2, 0], [2, 0, 1]], [[2, 1, 0]]]
    baseline = _case(tmp_path / "balanced", graph, layers, seconds=1.0)
    capacity = _case(tmp_path / "maxmemory", graph, layers, seconds=4.0, pack4=True)
    rerun = _case(tmp_path / "maxmemory-rerun", graph, layers, seconds=3.0, pack4=True)
    catalog, evidence = tmp_path / "catalog.json", tmp_path / "evidence"

    def export(candidate):
        return export_profiles(graph.to_dict(), INVENTORY, TOPOLOGY, baseline, candidate, catalog, evidence)

    with ThreadPoolExecutor(max_workers=3) as pool:
        first, preferred, again = list(pool.map(export, (capacity, rerun, capacity)))
    assert first == again
    verify_publication(catalog, evidence)
    assert len(read_profile_registry(catalog)["entries"]) == 2
    key = hardware_key(INVENTORY["inventory"], INVENTORY["devices"], TOPOLOGY.replace("\n", "\r\n"),
                       driver_versions="535.14")
    url = "https://raw.githubusercontent.com/owner/repo/" + "a" * 40 + "/profiles.json"
    data = catalog.read_bytes()
    calls = []

    def fetch(name, *, timeout):
        calls.append((name, timeout))
        return io.BytesIO(data)

    monkeypatch.setattr("multigpubfs.profile_registry.urlopen", fetch)
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="PIN_REQUIRED"):
        read_profile_registry(url, cache_dir=cache)
    assert calls == []
    with pytest.raises(ValueError, match="DIGEST"):
        read_profile_registry(url, sha256="f" * 64, cache_dir=cache)
    digest = hashlib.sha256(data).hexdigest()
    with pytest.raises(FileNotFoundError, match="OFFLINE_CACHE_MISS"):
        read_profile_registry(url, sha256=digest, cache_dir=cache, offline=True)
    registry = read_profile_registry(url, sha256=digest, cache_dir=cache)
    assert read_profile_registry(url, sha256=digest, cache_dir=cache, offline=True, timeout=0) == registry
    assert len(calls) == 2
    assert matching_profile(registry, graph.digest(), key, RUNTIME) == preferred
    assert matching_profile(registry, graph.digest(), key, dict(RUNTIME, native_sha256="e" * 64)) is None
    assert matching_profile(registry, graph.digest(), dict(key, driver_versions=["545.12"] * 2), RUNTIME) is None
    assert preferred["capacity_verification"] == "LOWER_PHYSICAL_PEAK"
    assert preferred["profiles"]["max_capacity"]["packet_wire"]["wire_state_bytes_per_row"] == 0
    assert preferred["profiles"]["max_capacity"]["compact_receive_slots"] == 3
    assert preferred["profiles"]["max_capacity"]["bitmap_owner_page"] == 0
    assert preferred["profiles"]["max_capacity"]["queue_capacity"] == 3072
    assert preferred["profiles"]["balanced"]["rank_geometry"][1]["admitted_batch"] == 64
    asset = {"name": f"profiles-{digest}.json", "digest": f"sha256:{digest}",
             "browser_download_url": f"https://github.com/owner/repo/releases/download/v1/profiles-{digest}.json"}
    uploads = []

    def github(command, **options):
        if command[1] == "api":
            return subprocess.CompletedProcess(command, 0, json.dumps({"assets": [asset] if uploads else []}), "")
        assert command[1:4] == ["release", "upload", "v1"] and command[-2:] == ["--repo", "owner/repo"]
        uploads.append(Path(command[4]).read_bytes())
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("multigpubfs.profile_registry.subprocess.run", github)
    publication = {"url": asset["browser_download_url"], "sha256": digest}
    assert publish_github_registry(catalog, evidence, repo="owner/repo", release="v1") == publication
    assert publish_github_registry(catalog, evidence, repo="owner/repo", release="v1") == publication
    assert uploads == [data]
    evidence_path = evidence / (preferred["evidence_sha256"] + ".json")
    if os.name == "nt":
        evidence_path = Path(f"\\\\?\\{evidence_path}")
    evidence_path.write_bytes(evidence_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="ENTRY_MISMATCH"):
        verify_publication(catalog, evidence)


def test_private_release_import_verifies_bytes_before_caching(tmp_path, monkeypatch):
    registry = {"schema": 2, "entries": []}
    data = json.dumps(registry).encode()
    digest = hashlib.sha256(data).hexdigest()
    url = "https://github.com/owner/repo/releases/download/research%2Fv1/private%20asset.json"
    fetched = []
    asset_data = data.replace(b"2", b"1")

    def anonymous(name, *, timeout):
        assert name == url and 0 < timeout <= 5
        raise HTTPError(name, 404, "Not Found", None, None)

    def authenticated(command, *, stdin, capture_output, env, timeout, check):
        assert command[:7] == ["gh", "api", "--hostname", "github.com", "--method", "GET",
                              "repos/owner/repo/releases/tags/research%2Fv1" if not fetched
                              else "repos/owner/repo/releases/assets/42"]
        assert stdin == subprocess.DEVNULL and capture_output and check
        assert env["GH_PROMPT_DISABLED"] == "1" and 0 < timeout <= 5
        if not fetched:
            assert command[7] == "--jq" and '"private asset.json"' in command[8]
            payload = json.dumps({"id": 42, "size": len(data), "digest": f"sha256:{digest}"}).encode()
        else:
            assert command[7:] == ["-H", "Accept: application/octet-stream"]
            payload = asset_data
        fetched.append(command[6])
        return subprocess.CompletedProcess(command, 0, payload, b"")

    monkeypatch.setattr("multigpubfs.profile_registry.urlopen", anonymous)
    monkeypatch.setattr("multigpubfs.profile_registry.subprocess.run", authenticated)
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="PROFILE_REGISTRY_DIGEST"):
        read_profile_registry(url, sha256=digest, cache_dir=cache)
    assert not cache.exists()
    asset_data = data
    fetched.clear()
    assert read_profile_registry(url, sha256=digest, cache_dir=cache) == registry
    assert len(fetched) == 2
    assert read_profile_registry(url, sha256=digest, cache_dir=cache, offline=True, timeout=0) == registry
    assert len(fetched) == 2


def test_registry_cli_prefetch_uses_explicit_download_timeout(tmp_path, monkeypatch):
    registry = {"schema": 2, "entries": []}
    data = json.dumps(registry).encode()
    digest = hashlib.sha256(data).hexdigest()
    url = "https://github.com/owner/repo/releases/download/v1/profiles.json"
    cache = tmp_path / "cache"
    output = tmp_path / "catalog.json"
    calls = []

    def download(source, deadline):
        import time

        assert source == url and 5 < deadline - time.monotonic() <= 20
        calls.append(source)
        return data

    monkeypatch.setattr("multigpubfs.profile_registry._public_catalog_bytes", download)
    arguments = ["import", url, "--sha256", digest, "--cache-dir", str(cache), "--output", str(output)]
    main(arguments + ["--timeout", "20"])
    assert json.loads(output.read_bytes()) == registry
    main(arguments + ["--offline"])
    assert calls == [url]


def test_cli_merges_verified_bundles_for_publication(tmp_path):
    from multigpubfs.cli import main as cli

    graph = GraphDefinition.permutation([[1, 0]], [0, 1])
    layers = [[[0, 1]], [[1, 0]]]
    baseline = _case(tmp_path / "b", graph, layers, seconds=1.0)
    sources = []
    entries = []
    for index, seconds in enumerate((4.0, 2.0)):
        root = tmp_path / str(index)
        root.mkdir()
        capacity = _case(root / "m", graph, layers, seconds=seconds, pack4=True)
        catalog, evidence = root / "registry.json", root / "evidence"
        sources.append(["--from", str(catalog), str(evidence)])
        entries.append(export_profiles(graph.to_dict(), INVENTORY, TOPOLOGY, baseline, capacity,
                                       catalog, evidence))

    catalog, evidence = tmp_path / "registry.json", tmp_path / "merged"
    command = ["profiles", "merge", "--registry", str(catalog), "--evidence-dir", str(evidence)]
    cli([*command, *sources[0]])
    cli([*command, *sources[1], *sources[0]])
    raw = verify_publication(catalog, evidence)
    assert json.loads(raw)["entries"] == sorted(entries, key=lambda entry: entry["evidence_sha256"])
    topology = hardware_key(INVENTORY["inventory"], INVENTORY["devices"], TOPOLOGY,
                            driver_versions=INVENTORY["driver_versions"])
    assert matching_profile(json.loads(raw), graph.digest(), topology, RUNTIME) == entries[1]

    entry = entries[1]["evidence_sha256"]
    witness = Path(sources[1][2]) / (entry + ".json")
    if os.name == "nt":
        witness = Path(f"\\\\?\\{witness}")
    witness.write_bytes(witness.read_bytes() + b" ")
    with pytest.raises(ValueError, match="ENTRY_MISMATCH"):
        cli([*command, *sources[1]])
    assert catalog.read_bytes() == raw


def test_cluster_export_matches_live_rank_topology_with_repeated_local_devices(tmp_path):
    graph = GraphDefinition.permutation([[1, 0]], [0, 1])
    layers = [[[0, 1]], [[1, 0]]]
    devices = [0, 0]
    inventory = [dict(row, device=0) for row in reversed(INVENTORY["inventory"])]
    metadata = [
        {"host": host, "topology": "GPU0\nGPU0 X", "driver": driver,
         "network": {"NCCL_NET": "Socket"}}
        for host, driver in (("node-a", "535.14"), ("node-b", "545.12"))
    ]
    topology = {"rank_metadata": metadata}
    launch = {"graph_digest": graph.digest(), "devices": devices, "inventory": inventory}
    baseline = _case(tmp_path / "b", graph, layers, seconds=1.0, devices=devices)
    capacity = _case(tmp_path / "m", graph, layers, seconds=2.0, pack4=True, devices=devices)
    for arm in (baseline, capacity):
        report = json.loads((arm / "report.json").read_bytes())
        _write(arm / "report.json", dict(report, rank_hostnames=["node-a", "node-b"]))
        _write(arm / "launch.json", launch)

    graph_path, topology_path = tmp_path / "graph.json", tmp_path / "topology.json"
    _write(graph_path, graph.to_dict())
    _write(topology_path, topology)
    catalog, evidence = tmp_path / "catalog.json", tmp_path / "e"
    main(["export", "--graph", str(graph_path), "--inventory", str(baseline / "launch.json"),
          "--topology", str(topology_path), "--baseline", str(baseline), "--capacity", str(capacity),
          "--registry", str(catalog), "--evidence-dir", str(evidence)])
    verified = json.loads(verify_publication(catalog, evidence))
    entry = verified["entries"][0]
    assert [rank["sm"] for rank in entry["hardware"]["ranks"]] == [90, 86]
    assert entry["hardware"]["driver_versions"] == ["535.14", "545.12"]
    assert entry["capacity_verification"] == "LOWER_PHYSICAL_PEAK"
    # CUDA ordinals and host names are receipt data. Physical rank order determines lookup.
    renamed = [dict(rank, host=f"rack-new-{i}", topology_device=0) for i, rank in enumerate(metadata)]
    renumbered = [dict(row, device=i + 3) for i, row in enumerate(inventory)]
    rows, actual, layout, drivers = cluster_hardware(renamed, renumbered)
    assert matching_profile(verified, graph.digest(),
                            hardware_key(rows, actual, layout, driver_versions=drivers), RUNTIME) == entry
    key = hardware_key(inventory[::-1], devices, {"rank_metadata": metadata[::-1]})
    assert matching_profile(verified, graph.digest(), key, RUNTIME) is None

    def export():
        return export_profiles(graph.to_dict(), launch, topology, baseline, capacity, catalog, evidence)

    report_path = capacity / "report.json"
    report = json.loads(report_path.read_bytes())
    _write(report_path, dict(report, profile_pressure_namespace="different-headroom"))
    assert export()["capacity_verification"] == "UNVERIFIED"
    _write(report_path, dict(report, rank_hostnames=["node-b", "node-a"]))
    with pytest.raises(ValueError, match="RANK_PLACEMENT"):
        export()
    _write(report_path, report)
    _write(capacity / "launch.json", dict(launch, inventory=inventory[::-1]))
    with pytest.raises(ValueError, match="RANK_PLACEMENT"):
        export()


def test_same_host_repeated_cuda_ordinals_need_distinct_topology_slots():
    topology = "GPU0 GPU1\nGPU0 X PHB\nGPU1 PHB X\n"
    ranks = [dict(host="one-node", topology=topology, driver="535.14", network={}) for _ in range(2)]
    rows = [dict(rank, device=0) for rank in INVENTORY["inventory"]]
    assert hardware_key(rows, [0, 0], {"rank_metadata": ranks}) is None
    partial = {"rank_metadata": [ranks[0], dict(ranks[1], topology_device=1)]}
    assert hardware_key(rows, [0, 0], partial) is None
    ranked = {"rank_metadata": [dict(rank, topology_device=i) for i, rank in enumerate(ranks)]}
    key = hardware_key(rows, [0, 0], ranked)
    assert key and hardware_key(rows, [0, 0], ranked, legacy_cluster=True) is None
    rows = [dict(row, device=i) for i, row in enumerate(rows)]
    assert hardware_key(rows, [0, 1], {"rank_metadata": ranks}) == key
    assert hardware_key(rows, [0, 1], {"rank_metadata": ranks}, legacy_cluster=True) == hardware_key(
        rows, [0, 1], topology, driver_versions="535.14",
    )


@pytest.mark.parametrize("modulus", [2048, 2049])
def test_full_terminal_witness_must_be_bound_when_either_layer_exceeds_sample(tmp_path, modulus):
    graph = GraphDefinition.matrix(2, 1, [([2, 0, 0, 1], modulus), ([2, 1, 0, 1], modulus)], [0, 1])
    layers = [[[0, 1]]]
    a = 1
    while a < modulus:
        layers.append([[x, 1] for x in range(a, min(2 * a, modulus))])
        a *= 2
    samples = layers[:-2] + [layers[-2] if len(layers[-2]) < 1000 else [], layers[-1][:1000]]
    sampled = _case(tmp_path / "sample", graph, layers, seconds=1.0, compact=samples)
    full = _case(tmp_path / "full", graph, layers, seconds=2.0, full=True, compact=samples)
    slower = _case(tmp_path / "too-slow", graph, layers, seconds=8.01, full=True, compact=samples)
    catalog, evidence = tmp_path / "catalog.json", tmp_path / "evidence"
    with pytest.raises(ValueError, match="TERMINAL_SETS"):
        export_profiles(graph.to_dict(), INVENTORY, TOPOLOGY, full, sampled, catalog, evidence)
    with pytest.raises(ValueError, match="CAPACITY_TOO_SLOW"):
        export_profiles(graph.to_dict(), INVENTORY, TOPOLOGY, full, slower, catalog, evidence)
    assert export_profiles(graph.to_dict(), INVENTORY, TOPOLOGY, full, full, catalog, evidence)[
        "capacity_verification"] == "UNVERIFIED"
    witness = full / "terminal_states.json"
    witness.write_bytes(witness.read_bytes() + b" ")
    with pytest.raises(ValueError, match="TERMINAL_DIGEST"):
        export_profiles(graph.to_dict(), INVENTORY, TOPOLOGY, full, full, catalog, evidence)
