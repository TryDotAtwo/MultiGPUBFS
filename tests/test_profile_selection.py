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
from multigpubfs.profile_registry import _path, hardware_key


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


def test_public_balanced_replays_faster_capacity_recipe_and_can_fall_back(tmp_path, monkeypatch, registry_case):
    case = registry_case
    catalog = json.loads(case["catalog"].read_bytes())
    entry = catalog["entries"][0]
    balanced = dict(case["profile"], pressure_namespace="ambient", peak_vram_bytes_by_rank=[200, 200])
    wire = {"protocol": "nccl_compact_credit_v2", "wire_record_bytes_per_row": 16,
            "wire_state_bytes_per_row": 0, "pack4_direct_owner": True, "byte_direct_owner": False,
            "route_receive_progress": True, "pinned_route_metadata": True}
    plan = dict(case["plan"], compact_transport=True, compact_route_lookahead=True,
                compact_packet_slots=3, compact_receive_slots=3)
    capacity = dict(balanced, **plan)
    capacity.update(memory_backend="permutation_bitmap", strategy="atomic", resident_codec="pack4-v1",
                    banks=2, owner_cuts=None, hash_lookup_series_limit=None, batch_hash_prefix_probe=False,
                    packet_wire=wire, bfs_seconds=1.0, peak_vram_bytes_by_rank=[100, 100])
    entry.update(profiles={"balanced": balanced, "max_capacity": capacity},
                 capacity_verification="LOWER_PHYSICAL_PEAK")
    case["catalog"].write_text(json.dumps(catalog), encoding="utf-8")
    capabilities = dict(case["capabilities"], permutation_bitmap_v1=True, bitmap_pack4_experimental_v1=True,
                        segmented_packet_transport_v1=True)
    packet_key, address = "MGBFS_SEGMENT_PACKET_CONTROL_ADDR", "127.0.0.1:12000"
    reject_capacity = False
    admitted, launched = [], []

    def graph_plan(command, *, env, **_):
        preset = env["MGBFS_MEMORY_PRESET"]
        admitted.append(preset)
        if preset == "max_capacity":
            assert env[packet_key] == address and env["MGBFS_BITMAP_BANKS"] == "2"
            assert env["MGBFS_SEGMENT_PACKET_RX_SLOTS"] == env["MGBFS_SEGMENT_PACKET_RING_SLOTS"] == "3"
            actual = dict(plan, compact_receive_slots=1) if reject_capacity else plan
            receipt = {"plan": actual}
        else:
            assert preset == "balanced" and packet_key not in env
            receipt = {"plan": case["plan"], "owner_cuts": case["cuts"]}
        return CompletedProcess(command, 0, stdout=json.dumps(receipt))

    def native_launch(graph, output, **options):
        launched.append(options["memory_preset"])
        assert not options["autotune"]
        output.mkdir()
        (output / "states.json").write_text(
            json.dumps({"current": [[1, 0]], "previous_small": [[0, 1]]}), encoding="utf-8",
        )
        if options["memory_preset"] == "balanced":
            assert options["memory_backend"] == "segmented24" and packet_key not in options["_native_env"]
            return _native_receipt(case)
        assert options["memory_preset"] == "maxmemory" and options["memory_backend"] == "permutation_bitmap"
        assert options["_native_env"][packet_key] == address and options["bitmap_pack4"]
        report = _native_receipt(dict(case, profile=capacity, plan=plan, cuts=None))
        report.update(backend="EXPERIMENTAL_PERMUTATION_BITMAP", bfs_seconds=capacity["bfs_seconds"])
        for rank in report["segment_memory_by_rank"]:
            rank["packet_transport"] = dict(wire, tx_exchange_slots=3, rx_exchange_slots=3)
        return report

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(api, "native_runtime", lambda _: ("test-native", {}))
    monkeypatch.setattr(api, "_require_memory_backend", lambda *_: capabilities)
    monkeypatch.setattr(api.BfsSession, "_live_inventory", lambda *_: case["inventory"])
    monkeypatch.setattr(api.BfsSession, "_local_packet_address", lambda _: address)
    monkeypatch.setattr(api, "run_graph", native_launch)
    monkeypatch.setattr(native_distribution, "runtime_identity", lambda *_: case["runtime"])
    monkeypatch.setattr(profile_selection, "local_hardware", lambda: (case["topology"], case["driver"]))
    monkeypatch.setattr(profile_selection, "query", graph_plan)
    with open_session(devices=case["devices"], output_root=tmp_path, profile_registry=case["catalog"]) as session:
        fastest = session.bfs(case["graph"], preset="balanced")
        lowest = session.bfs(case["graph"], preset="maxmemory")
        reject_capacity = True
        fallback = session.bfs(case["graph"], preset="balanced")

    assert admitted == ["max_capacity", "max_capacity", "max_capacity", "balanced"]
    assert launched == ["maxmemory", "maxmemory", "balanced"]
    for result, requested, recipe in zip((fastest, lowest, fallback),
                                         ("balanced", "maxmemory", "balanced"),
                                         ("maxmemory", "maxmemory", "balanced")):
        policy = result.report["session_policy"]
        receipt = policy["profile_registry"]
        assert policy["preset"] == receipt["requested_preset"] == requested
        assert receipt["recipe_preset"] == recipe and receipt["runtime_geometry"] == "MATCHED"


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
    native_admission = {
        "graph_digest": case["graph"].digest(), "inventory": case["inventory"],
        "devices": list(case["devices"]), "plan": dict(case["plan"], world=2),
        "rank_plans": [dict(case["plan"], world=2, rank=rank) for rank in range(2)],
        "owner_cuts": case["cuts"], "additional_reserve_bytes": case["reserve"],
    }

    def graph_plan(command, *, env, resident, **_):
        assert json.loads(Path(command[3]).read_text(encoding="utf-8")) == case["inventory"]
        assert not resident and env["MGBFS_GENERIC_VRAM_RESERVE_BYTES"] == str(case["reserve"])
        plans.append(env)
        return CompletedProcess(command, 0, stdout=json.dumps(native_admission))

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
    assert all(run["_external_inventory"] == case["inventory"] for run in runs.values())
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
        assert run["_external_admission"] == {"plan": native_admission, "artifacts": {"native": "bin", "cuda": "so"}}


def test_external_rank_uses_prepared_admission_for_the_native_command(tmp_path, monkeypatch, registry_case):
    from multigpubfs.cluster import _active
    from multigpubfs import host_memory

    case = registry_case
    inventory = case["inventory"]
    device = case["devices"][0]
    plan = dict(case["plan"], world=2)
    admission = {
        "graph_digest": case["graph"].digest(), "inventory": inventory,
        "devices": list(case["devices"]), "plan": plan, "rank_plans": [plan, plan],
        "owner_cuts": case["cuts"], "additional_reserve_bytes": case["reserve"],
    }

    class NativeCommandObserved(Exception):
        pass

    class Store:
        def __init__(self, *args):
            self.values = {}

        def put(self, key, value):
            self.values[key] = value

        def get(self, key):
            return self.values["identity/0" if key == "identity/1" else key]

        def report_error(self, error, world):
            assert isinstance(error, NativeCommandObserved)

        def close(self):
            pass

    def launch_rank(command, env, log):
        assert command[1] == "graph-rank" and command[4] == "0"
        config = json.loads(Path(command[3]).read_text(encoding="utf-8"))
        assert config == dict(admission, max_seconds=None)
        assert env["MGBFS_GENERIC_VRAM_RESERVE_BYTES"] == str(case["reserve"])
        raise NativeCommandObserved()

    def redundant_admission(*args, **kwargs):
        pytest.fail("Production queried inventory or recalculated the admitted plan")

    worker = Cluster(Placement(0, device, 2, "127.0.0.1", 1, 2, 3, "native-plan"), None, tmp_path)
    monkeypatch.setattr(worker, "inventory", redundant_admission)
    monkeypatch.setattr(worker, "launch_rank", launch_rank)
    monkeypatch.setattr(distributed_launch, "native_query", redundant_admission)
    monkeypatch.setattr(distributed_launch, "ControlStore", Store)
    monkeypatch.setattr(distributed_launch, "_native_artifact_hashes", lambda *_: {"native": "bin", "cuda": "so"})
    monkeypatch.setattr(host_memory, "admit", lambda *args: {})
    monkeypatch.setattr(launch, "native_runtime", lambda executable: ("test-native", {}))
    monkeypatch.setattr(launch, "_require_memory_backend",
                        lambda *args: dict(case["capabilities"], segmented_network_agreement_v1=True))
    for key, value in {"RANK": "0", "LOCAL_RANK": str(device), "WORLD_SIZE": "2",
                       "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "43123"}.items():
        monkeypatch.setenv(key, value)
    context = _active.set(worker)
    try:
        with pytest.raises(NativeCommandObserved):
            launch.run_graph(case["graph"], tmp_path / "traversal", device=device, shards=3, _batch=2,
                             executable="test-native", memory_backend="segmented24", segment_strategy="batch",
                             autotune=False, _native_env={}, _external_inventory=inventory,
                             _external_admission={"plan": admission, "artifacts": {"native": "bin", "cuda": "so"}})
    finally:
        _active.reset(context)


@pytest.mark.parametrize("reject_fastest", [False, True])
def test_registry_capacity_compares_against_fastest_live_recipe(monkeypatch, registry_case, reject_fastest):
    case = registry_case
    catalog = json.loads(case["catalog"].read_bytes())
    first = catalog["entries"][0]
    profile = dict(case["profile"], pressure_namespace="ambient")

    def arm(seconds, peak):
        return dict(profile, bfs_seconds=seconds, peak_vram_bytes_by_rank=[peak] * 2)

    fastest = arm(1, 100)
    if reject_fastest:
        fastest["rank_geometry"] = [dict(rank, free_bytes_at_admission=12 << 30)
                                     for rank in fastest["rank_geometry"]]
    first["profiles"] = {"balanced": fastest, "max_capacity": arm(4, 105)}
    capacity = arm(6 if reject_fastest else 3, 120)
    catalog["entries"].append(dict(first, evidence_sha256="d" * 64,
                                   profiles={"balanced": arm(2, 200), "max_capacity": capacity},
                                   capacity_verification="LOWER_PHYSICAL_PEAK"))
    case["catalog"].write_text(json.dumps(catalog), encoding="utf-8")
    admitted = []

    def graph_plan(command, *, env, **_):
        admitted.append(env["MGBFS_MEMORY_PRESET"])
        return CompletedProcess(command, 0, stdout=json.dumps({"plan": case["plan"], "owner_cuts": case["cuts"]}))

    monkeypatch.setattr(native_distribution, "runtime_identity", lambda *_: case["runtime"])
    monkeypatch.setattr(profile_selection, "query", graph_plan)
    recipe, status = profile_selection.select_registry_recipe(
        {"source": str(case["catalog"]), "sha256": None}, case["graph"], "maxmemory", "fake-native",
        {}, case["capabilities"], time.monotonic() + 5,
        hardware=lambda: (case["inventory"], case["devices"], case["topology"], case["driver"]),
        inventory=lambda: case["inventory"], packet_address=None,
    )
    if reject_fastest:
        assert admitted == ["balanced", "max_capacity"]
        assert recipe.evidence["bfs_seconds"] == 6
        assert status["historical_ratio"] == 3
        assert status["evidence_sha256"] == "d" * 64
    else:
        assert admitted == ["balanced"]
        assert recipe is None and status["status"] == "NO_COMPATIBLE_PROFILE"


@pytest.mark.parametrize("catalog_source", ["local", "offline-cache", "remote"])
def test_stalled_catalog_returns_to_startup_fallback(tmp_path, monkeypatch, registry_case, catalog_source):
    remote = "https://github.com/test/profiles/releases/download/v1/catalog.json"
    source = str(registry_case["catalog"]) if catalog_source == "local" else remote
    digest = registry_case["pin"]
    env = {"MGBFS_PROFILE_REGISTRY_CACHE": str(tmp_path)}
    if catalog_source == "offline-cache":
        _path(tmp_path / f"{digest}.json").write_bytes(registry_case["catalog"].read_bytes())
        env["MGBFS_PROFILE_REGISTRY_OFFLINE"] = "1"

    def stopped_import(command, *, timeout, capture_output):
        assert command[1:4] == ["-m", "multigpubfs.profile_registry", "import"]
        assert "--stdout" in command and ("--offline" in command) == (catalog_source == "offline-cache")
        assert 0 < timeout <= 0.5 and capture_output
        raise TimeoutExpired(command, timeout)

    monkeypatch.setattr(profile_selection.subprocess, "run", stopped_import)
    recipe, status = profile_selection.select_registry_recipe(
        {"source": source, "sha256": digest}, registry_case["graph"], "balanced", "fake-native", env,
        registry_case["capabilities"], time.monotonic() + 0.5, hardware=lambda: pytest.fail("no catalog"),
        inventory=lambda: pytest.fail("no catalog"), packet_address=None,
    )
    assert recipe is None and status["status"] == "REGISTRY_STARTUP_DEADLINE"
