from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
import time

import pytest

from multigpubfs import GraphDefinition, open_session
from multigpubfs import api, autotune, distributed_launch, generic_session, launch, native_distribution, profile_selection
from multigpubfs.cluster import Cluster, Placement
from multigpubfs.profile_registry import hardware_key


@pytest.fixture
def registry_case(tmp_path):
    graph = GraphDefinition.permutation([[1, 0]], [0, 1])
    devices = (4, 2)
    topology = "GPU2 GPU4\nGPU2 X PHB\nGPU4 PHB X\n"
    driver = "535.14"
    reserve = 256 << 20
    inventory = [
        {"device": device, "compute_major": major, "compute_minor": minor, "sm_count": sms,
         "total_bytes": total << 30, "free_bytes": free << 30, "additional_reserve_bytes": reserve}
        for device, major, minor, sms, total, free in ((4, 9, 0, 132, 80, 79), (2, 8, 6, 28, 12, 11))
    ]
    runtime = {"native_sha256": "a" * 64, "cuda_library_sha256": "b" * 64}
    benchmark_plan = {
        "shards": 3, "batch": 2, "history_layers": 3, "state_bytes": 1, "queue_capacity": 2,
        "compact_transport": False, "compact_route_lookahead": False,
        "compact_packet_slots": 1, "compact_receive_slots": 1,
    }
    cuts = [0, 1 << 31, 1 << 32]
    profile = {
        **benchmark_plan, "memory_backend": "segmented24", "strategy": "batch",
        "owner_lanes": 2, "exact_exchange": True, "banks": 3, "bfs_seconds": 2.0,
        "peak_resident_rows_by_rank": [1, 1], "resident_codec": "unpacked-soa",
        "segment_rows": 2048, "bitmap_owner_page": None, "compact_positions": False,
        "hash_lookup_series_limit": 64, "batch_owner_tile": None, "batch_hash_merge_rows": None,
        "batch_hash_prefix_probe": True, "packet_wire": None, "owner_cuts": cuts,
        "peak_vram_bytes_by_rank": None, "pressure_namespace": None,
        "rank_geometry": [
            {"capacity": 2, "admitted_batch": 2, "rows_per_bank": 2048,
             "capacity_limit_per_rank": None, "reserve_bytes": 0, "free_bytes_at_admission": 10 << 30,
             "additional_reserve_bytes": reserve}
            for _ in devices
        ],
    }
    evidence_digest = "c" * 64
    catalog = tmp_path / "catalog.json"
    data = json.dumps({"schema": 2, "entries": [{
        "graph_digest": graph.digest(),
        "hardware": hardware_key(inventory, devices, topology, driver_versions=driver),
        "runtime": runtime,
        "profiles": {"balanced": profile, "max_capacity": dict(profile, bfs_seconds=3.0)},
        "evidence_sha256": evidence_digest,
        "verification": "INDEPENDENT_LAYER_COUNTS_AND_EXACT_TERMINAL_SETS",
        "capacity_verification": "UNVERIFIED",
    }]}).encode("utf-8")
    catalog.write_bytes(data)
    return {
        "graph": graph, "devices": devices, "topology": topology, "driver": driver, "inventory": inventory,
        "runtime": runtime, "plan": benchmark_plan, "cuts": cuts, "evidence": evidence_digest, "reserve": reserve,
        "catalog": catalog, "pin": hashlib.sha256(data).hexdigest(), "profile": profile,
        "capabilities": {
            "segmented24_v1": True, "segmented_strategies_v1": {"batch": True},
            "segmented_hash_series_v1": True, "segmented_exact_exchange_v1": True,
        },
    }


def _native_receipt(case):
    from multigpubfs.profile_registry import _COMMON

    profile = case["profile"]
    names = {"strategy": "segment_strategy", "queue_capacity": "actual_queue_capacity",
             "segment_rows": "configured_segment_rows", "packet_wire": "packet_transport"}
    common = {names.get(field, field): profile[field] for field in _COMMON}
    rank_plans, ranks = [], []
    for rank, (device, geometry) in enumerate(zip(case["devices"], profile["rank_geometry"])):
        rank_plans.append(dict(case["plan"], capacity=geometry["capacity"]))
        ranks.append(dict(common, **geometry, rank=rank, device=device, peak_live_rows=1))
    return {"status": "COMPLETE", "backend": "EXPERIMENTAL_SEGMENTED24", "bfs_seconds": 2.0,
            "devices": case["devices"], "segment_memory_by_rank": ranks,
            "size_profile_count": 0, "plan": rank_plans[0], "rank_plans": rank_plans,
            "owner_cuts": case["cuts"], "layer_sizes": [1, 1], "graph_digest": case["graph"].digest()}


def test_public_session_replays_pinned_recipe_and_keeps_manual_and_fallback_paths(tmp_path, monkeypatch, registry_case):
    case = registry_case
    graph, devices, inventory = case["graph"], case["devices"], case["inventory"]
    topology_calls, inventory_calls, plans, native_runs = [], [], [], []

    def smi(command):
        topology_calls.append(command)
        if command == ["nvidia-smi", "topo", "-m"]:
            return CompletedProcess(command, 0, stdout=case["topology"])
        assert command == ["nvidia-smi", "--query-gpu=uuid,driver_version,name", "--format=csv,noheader"]
        driver = case["driver"]
        return CompletedProcess(command, 0, stdout=f"GPU-test-2, {driver}, Test GPU\nGPU-test-4, {driver}, Test GPU\n")

    def live_inventory(pool, native, selected, env, timeout):
        assert native == "test-native" and selected == devices
        assert env["MGBFS_GENERIC_HISTORY"] == "hash" and timeout > 0
        inventory_calls.append(selected)
        return inventory

    def graph_plan(command, *, env, capture_output, text, resident):
        executable, mode, definition, ranks, *geometry = command
        assert (executable, mode) == ("test-native", "graph-plan")
        assert geometry == ["3", "auto", "2"] and capture_output and text and not resident
        assert env["MGBFS_GENERIC_VRAM_RESERVE_BYTES"] == str(case["reserve"])
        assert json.loads(Path(definition).read_text(encoding="utf-8")) == graph.to_dict()
        assert json.loads(Path(ranks).read_text(encoding="utf-8")) == inventory
        plans.append(env)
        return CompletedProcess(command, 0, stdout=json.dumps({"plan": case["plan"], "owner_cuts": case["cuts"]}))

    def native_launch(definition, output, **options):
        assert definition.digest() == graph.digest()
        native_runs.append(options)
        output.mkdir()
        (output / "states.json").write_text(
            json.dumps({"current": [[1, 0]], "previous_small": [[0, 1]]}), encoding="utf-8",
        )
        return _native_receipt(case)

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(api, "native_runtime", lambda executable: ("test-native", {}))
    monkeypatch.setattr(api, "_require_memory_backend", lambda native, env, backend: case["capabilities"])
    monkeypatch.setattr(autotune, "_system_info", smi)
    monkeypatch.setattr(generic_session.Pool, "inventory", live_inventory)
    monkeypatch.setattr(profile_selection, "query", graph_plan)
    live_runtime = case["runtime"].copy()
    monkeypatch.setattr(native_distribution, "runtime_identity", lambda native, env: live_runtime)
    monkeypatch.setattr(api, "run_graph", native_launch)

    with open_session(devices=devices, output_root=tmp_path, profile_registry=case["catalog"],
                      profile_registry_sha256=case["pin"]) as session:
        selected = session.bfs(graph)
        manual = session.bfs(graph, native_options={
            "memory_backend": "segmented24", "segment_strategy": "atomic", "shards": 1, "batch": 1,
        })
        assert (len(topology_calls), len(inventory_calls), len(plans)) == (2, 1, 1)
        live_runtime["native_sha256"] = "d" * 64
        fallback = session.bfs(graph)

    admitted, explicit, default = native_runs
    assert selected.bfs_completed and selected.layer_sizes == [1, 1]
    receipt = selected.report["session_policy"]["profile_registry"]
    assert (receipt["status"], receipt["evidence_sha256"]) == ("LIVE_ADMITTED_REGISTRY_PROFILE", case["evidence"])
    assert receipt["runtime_geometry"] == "MATCHED"
    assert (admitted["devices"], admitted["memory_backend"], admitted["segment_strategy"],
            admitted["shards"], admitted["_batch"], admitted["owner_lanes"],
            admitted["queue_capacity"], admitted["segment_rows"], admitted["autotune"]) == (
        devices, "segmented24", "batch", 3, 2, 2, 2, 2048, False,
    )
    for key, value in {
        "MGBFS_SEGMENT_EXACT_EXCHANGE": "1", "MGBFS_SEGMENT_OWNER_LANES": "2",
        "MGBFS_SEGMENT_QUEUE_CAP": "2", "MGBFS_SEGMENT_HASH_SERIES": "1",
    }.items():
        assert plans[0][key] == admitted["_native_env"][key] == value
    assert "profile_registry" not in manual.report["session_policy"]
    assert (explicit["segment_strategy"], explicit["shards"], explicit["_batch"], explicit["autotune"]) == (
        "atomic", 1, 1, False,
    )
    assert fallback.report["session_policy"]["profile_registry"]["status"] == "NO_COMPATIBLE_PROFILE"
    assert fallback.report["session_policy"]["selection_reason"] == "SEGMENTED_BOUNDED_HASH"
    assert default["autotune"] and default["segment_strategy"] == "batch"
    assert "shards" not in default and "_batch" not in default
    assert default["_profile_options"]["pilot_limit"] == 3 and "startup_deadline" in default["_profile_options"]
    assert (len(topology_calls), len(inventory_calls), len(plans)) == (4, 2, 1)


@pytest.mark.parametrize("metadata_timeout", [False, True])
def test_cluster_coordinator_admits_one_recipe_for_all_ranks(tmp_path, monkeypatch, registry_case, metadata_timeout):
    case = registry_case
    messages = defaultdict(Future)

    class Store:
        def put(self, key, value, **_):
            messages[key].set_result(json.loads(json.dumps(value)))

        def get(self, key, *, deadline=None):
            if "profile-hardware/" in key:
                assert deadline is not None
                if metadata_timeout:
                    raise TimeoutError("RANK_METADATA_NOT_READY")
            return messages[key].result(timeout=3)

    peers = [
        Cluster(Placement(rank, device, 2, "127.0.0.1", 1, 2, 3, "profile-test"), Store(), tmp_path)
        for rank, device in enumerate(case["devices"])
    ]
    capabilities = dict(case["capabilities"], segmented_network_agreement_v1=True)
    by_device = {row["device"]: row for row in case["inventory"]}
    plans, runs = [], {}

    def graph_plan(command, *, env, resident, **_):
        assert json.loads(Path(command[3]).read_text(encoding="utf-8")) == case["inventory"]
        assert not resident and env["MGBFS_GENERIC_VRAM_RESERVE_BYTES"] == str(case["reserve"])
        plans.append(env)
        return CompletedProcess(command, 0, stdout=json.dumps({"plan": case["plan"], "owner_cuts": case["cuts"]}))

    def dispatch(self, job, output):
        assert self is peers[0]
        with ThreadPoolExecutor(max_workers=2) as executor:
            tasks = [executor.submit(peer._run, job, output) for peer in peers]
            return [task.result(timeout=6) for task in tasks]

    def native_launch(graph, output, **options):
        runs[options["device"]] = options
        report = _native_receipt(case)
        for rank in report["segment_memory_by_rank"]:
            rank["owner_lanes"] = 1
        return report

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(native_distribution, "native_runtime", lambda _: ("test-native", {}))
    monkeypatch.setattr(native_distribution, "runtime_identity", lambda *_: case["runtime"])
    monkeypatch.setattr(distributed_launch, "_native_artifact_hashes", lambda *_: {"native": "bin", "cuda": "so"})
    monkeypatch.setattr(profile_selection, "local_hardware", lambda: (case["topology"], case["driver"]))
    monkeypatch.setattr(profile_selection, "query", graph_plan)
    monkeypatch.setattr(launch, "_require_memory_backend", lambda *_: capabilities)
    monkeypatch.setattr(launch, "run_graph", native_launch)
    monkeypatch.setattr(Cluster, "inventory", lambda self, native, env, device: by_device[device])
    monkeypatch.setattr(Cluster, "_dispatch", dispatch)

    session = open_session(gpus=2, profile_registry=case["catalog"], profile_registry_sha256=case["pin"])
    reports = peers[0].traverse(session, case["graph"], tmp_path / "cluster", "balanced", None)
    assert set(runs) == set(case["devices"])
    if metadata_timeout:
        assert not plans
        assert all(report["cluster_profile_registry"]["status"] == "REGISTRY_STARTUP_DEADLINE" for report in reports)
        assert all(run["autotune"] and "queue_capacity" not in run for run in runs.values())
        return
    assert len(plans) == 1
    for report, run in zip(reports, (runs[device] for device in case["devices"])):
        assert report["cluster_profile_registry"]["evidence_sha256"] == case["evidence"]
        assert report["cluster_profile_registry"]["runtime_geometry"] == "CHANGED"
        assert (run["autotune"], run["shards"], run["_batch"], run["segment_strategy"]) == (False, 3, 2, "batch")
        assert run["_native_env"]["MGBFS_SEGMENT_OWNER_LANES"] == plans[0]["MGBFS_SEGMENT_OWNER_LANES"] == "2"


def test_registry_capacity_requires_a_lower_peak_than_fastest_balanced(monkeypatch, registry_case):
    case = registry_case
    catalog = json.loads(case["catalog"].read_bytes())
    first = catalog["entries"][0]
    profile = dict(case["profile"], pressure_namespace="ambient")

    def arm(seconds, peak):
        return dict(profile, bfs_seconds=seconds, peak_vram_bytes_by_rank=[peak] * 2)

    first["profiles"] = {"balanced": arm(1, 100), "max_capacity": arm(1.5, 105)}
    catalog["entries"].append(dict(first, evidence_sha256="d" * 64,
                                   profiles={"balanced": arm(2, 200), "max_capacity": arm(3, 120)},
                                   capacity_verification="LOWER_PHYSICAL_PEAK"))
    case["catalog"].write_text(json.dumps(catalog), encoding="utf-8")
    monkeypatch.setattr(native_distribution, "runtime_identity", lambda *_: case["runtime"])
    recipe, status = profile_selection.select_registry_recipe(
        {"source": str(case["catalog"]), "sha256": None}, case["graph"], "maxmemory", "fake-native",
        {}, case["capabilities"], time.monotonic() + 5,
        hardware=lambda: (case["inventory"], case["devices"], case["topology"], case["driver"]),
        inventory=lambda: case["inventory"], packet_address=None,
    )
    assert recipe is None and status["status"] == "NO_COMPATIBLE_PROFILE"


def test_stalled_remote_catalog_returns_to_startup_fallback(tmp_path, monkeypatch, registry_case):
    def stopped_import(command, *, timeout, capture_output):
        assert command[1:4] == ["-m", "multigpubfs.profile_registry", "import"]
        assert 0 < timeout <= 0.5 and capture_output
        raise TimeoutExpired(command, timeout)

    monkeypatch.setattr(profile_selection.subprocess, "run", stopped_import)
    recipe, status = profile_selection.select_registry_recipe(
        {"source": "https://github.com/test/profiles/releases/download/v1/catalog.json", "sha256": "a" * 64},
        registry_case["graph"], "balanced", "fake-native", {"MGBFS_PROFILE_REGISTRY_CACHE": str(tmp_path)},
        registry_case["capabilities"], time.monotonic() + 0.5, hardware=lambda: pytest.fail("no catalog"),
        inventory=lambda: pytest.fail("no catalog"), packet_address=None,
    )
    assert recipe is None and status["status"] == "REGISTRY_STARTUP_DEADLINE"
