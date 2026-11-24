from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

import pytest

from multigpubfs.graph_definition import GraphDefinition
from multigpubfs.profile_registry import (
    export_profiles,
    hardware_key,
    matching_profile,
    publish_github_registry,
    read_profile_registry,
    verify_publication,
)
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


def _case(path, graph, layers, *, seconds, pack4=False, full=False, compact=None):
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
             "banks": 2 if pack4 else 1,
             "resident_codec": "pack4-v1" if pack4 else "unpacked-soa", "configured_segment_rows": 2048,
             "compact_positions": False, "batch_hash_prefix_probe": False,
             "admitted_batch": 128 if pack4 else 64, "rows_per_bank": 512 + i,
             "peak_live_rows": 2, "packet_transport": packet} for i, device in enumerate(INVENTORY["devices"])]
    report = {
        "status": "COMPLETE", "reason": "FRONTIER_EXHAUSTED", "layer_sizes": [len(layer) for layer in layers],
        "graph_digest": digest, "devices": INVENTORY["devices"], "bfs_seconds": seconds,
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
