import json
from pathlib import Path

import pytest

import multigpubfs.api as api
from multigpubfs import GraphDefinition, open_session


def test_session_reuses_pool_and_records_graph_override(tmp_path, monkeypatch):
    graph = GraphDefinition.permutation([[1, 0]], [0, 1])
    definition = tmp_path / "definition.json"
    definition.write_text(graph.to_json(), encoding="utf-8")
    calls = []
    monkeypatch.setattr(api, "native_runtime", lambda override: ("fake-mgbfs", {}))
    monkeypatch.setattr(api, "_require_memory_backend", lambda *args: {"segmented_warp_hash_safe_v1": True})

    def launch(actual, output, **kwargs):
        calls.append((api.generic_session._active, actual.start, kwargs))
        output.mkdir()
        (output / "states.json").write_text(json.dumps({"current": [[0, 1]], "previous_small": [[1, 0]]}))
        return {"status": "COMPLETE", "layer_sizes": [1, 1], "graph_digest": actual.digest()}

    monkeypatch.setattr(api, "run_graph", launch)
    with open_session(devices=[2, 4], output_root=tmp_path, tune="off") as session:
        first = session.traverse(definition, start=[1, 0])
        second = session.bfs(graph, preset="maxmemory")
        with pytest.raises(RuntimeError, match="BFS_SESSION_ALREADY_OPEN"):
            with open_session(devices=[2]):
                pass

    assert calls[0][0] is calls[1][0] and calls[0][0] is not None
    assert api.generic_session._active is None
    assert calls[0][1] == [1, 0] and calls[0][2]["devices"] == (2, 4)
    assert calls[0][2]["segment_strategy"] == "warp"
    assert first.bfs_completed and first.last_layer() == [[0, 1]] and first.num_vertices() == 2
    assert first.output != second.output
    assert json.loads((first.output / "graph.json").read_text())["start"] == [1, 0]
    assert second.report["session_policy"]["capacity_slowdown_status"] == "UNVERIFIED"
    assert json.loads((first.output / "report.json").read_text()) == first.report


def test_session_passes_exact_public_path_query(tmp_path, monkeypatch):
    from multigpubfs import path_search

    graph = GraphDefinition.permutation([[1, 0]], [0, 1])
    calls = []

    def solve(definition, *, output, **options):
        output.mkdir()
        calls.append((definition.start, options))
        return output

    monkeypatch.setattr(path_search, "solve", solve)
    with open_session(gpus=2, output_root=tmp_path) as session:
        result = session.solve(graph, start=[1, 0], targets=[[0, 1]], direction="bidirectional",
                               goal_mode="window", extra_steps=3, preset="max_capacity", max_paths=2)
    assert calls[0][0] == [1, 0]
    assert calls[0][1] == {"targets": [[0, 1]], "devices": (0, 1), "direction": "bidirectional",
                           "goal_mode": "window", "extra_steps": 3, "preset": "maxmemory", "max_seconds": None,
                           "max_paths": 2, "max_output_bytes": None}
    assert (result / "graph.json").is_file()


def test_external_launcher_leaves_automatic_capacity_to_all_ranks(tmp_path, monkeypatch):
    from multigpubfs import distributed_launch, launch, workload_geometry

    graph = GraphDefinition.permutation([[1, 0]], [0, 1])
    output = tmp_path / "result"
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(workload_geometry, "certified_frontier_bound", lambda _: {"maximum_frontier": 9})
    monkeypatch.setattr(launch, "native_runtime", lambda _: ("native", {}))
    monkeypatch.setattr("multigpubfs.native_distribution.runtime_identity", lambda *args: {})

    def fake_external(definition, path, *, capacity, **kwargs):
        assert capacity is None and definition.digest() == graph.digest()
        Path(path).mkdir()
        return {"backend": "GENERIC_DISTRIBUTED_EXACT_ALL_VISITED"}

    monkeypatch.setattr(distributed_launch, "run_external", fake_external)
    report = launch.run_graph(graph, output, backend="generic")
    assert report["workload_geometry_bound"]["maximum_frontier"] == 9


def test_external_segmented_capacity_is_a_rank_cap(tmp_path, monkeypatch):
    from multigpubfs import distributed_launch, launch

    graph = GraphDefinition.permutation([[1, 0]], [0, 1])
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr("multigpubfs.cluster.current_cluster", lambda: object())
    monkeypatch.setattr(launch, "native_runtime", lambda _: ("native", {}))
    monkeypatch.setattr(launch, "_require_memory_backend", lambda *args: {
        "segmented_network_agreement_v1": True,
    })
    monkeypatch.setattr("multigpubfs.native_distribution.runtime_identity", lambda *args: {})

    def run_external(definition, path, *, capacity, native_env, shards, batch, **kwargs):
        assert definition.digest() == graph.digest() and capacity is None
        assert native_env["MGBFS_SEGMENT_MAX_RANK_STATES"] == "125000"
        assert native_env["MGBFS_SEGMENT_EXTERNAL_LINK"] == "1"
        assert native_env["MGBFS_SEGMENT_QUEUE_CAP"] == "32768"
        assert native_env["MGBFS_SEGMENT_OWNER_LANES"] == "4"
        assert (shards, batch) == (4, 4096)
        path.mkdir()
        return {"backend": "EXPERIMENTAL_SEGMENTED24"}

    monkeypatch.setattr(distributed_launch, "run_external", run_external)
    launch.run_graph(graph, tmp_path / "result", backend="generic", memory_backend="segmented24",
                     capacity=125000, shards=4, autotune=False, _batch=4096,
                     queue_capacity=32768, owner_lanes=4)


@pytest.mark.parametrize("pack4", [None, True, False])
@pytest.mark.parametrize("preset,transport", [
    ("balanced", None), ("maxmemory", None), ("maxmemory", "exact"), ("maxmemory", "compact"),
])
def test_explicit_session_geometry_reaches_the_normal_launcher(tmp_path, monkeypatch, pack4, preset, transport):
    graph = GraphDefinition.permutation([[1, 0]], [0, 1])
    monkeypatch.setattr(api, "native_runtime", lambda _: ("native", {}))
    monkeypatch.setattr(api, "_require_memory_backend", lambda *args: {
        "bitmap_pack4_experimental_v1": True, "segmented_packet_transport_v1": True,
        "segmented_exact_exchange_v1": True,
    })

    def run_graph(definition, output, **kwargs):
        assert definition.digest() == graph.digest()
        assert kwargs["devices"] == (0, 1)
        env = kwargs["_native_env"]
        assert env["MGBFS_SEGMENT_EXACT_EXCHANGE"] == "1"
        assert (env.get("_MGBFS_PRESET_IMPLICIT_PACKET") == "1") is (preset == "maxmemory" and transport is None)
        address = env.get("MGBFS_SEGMENT_PACKET_CONTROL_ADDR")
        if preset == "balanced" or transport == "exact":
            assert address is None
        else:
            assert address.startswith("127.0.0.1:") and int(address.split(":")[1]) > 0
        assert kwargs["autotune"] is False and kwargs["memory_preset"] == preset
        assert kwargs["memory_backend"] == "permutation_bitmap" and kwargs.get("bitmap_pack4", False) is (pack4 if pack4 is not None else preset == "maxmemory")
        assert (kwargs["shards"], kwargs["capacity"], kwargs["_batch"]) == (4, 125000, 4096)
        assert (kwargs["queue_capacity"], kwargs["owner_lanes"]) == (32768, 4)
        output.mkdir()
        (output / "states.json").write_text(json.dumps({"current": [[1, 0]], "previous_small": [[0, 1]]}))
        return {"status": "COMPLETE", "layer_sizes": [1, 1], "autotune": {
            "capacity_slowdown": {"status": "MEASURED_FULL_BFS_BOUND"},
        }}

    monkeypatch.setattr(api, "run_graph", run_graph)
    with open_session(devices=[0, 1], output_root=tmp_path) as session:
        result = session.traverse(graph, preset=preset, native_options={
            "memory_backend": "permutation_bitmap",
            **({"transport": transport} if transport is not None else {}),
            **({"bitmap_pack4": pack4} if pack4 is not None else {}),
            "shards": 4, "capacity": 125000, "batch": 4096,
            "queue_capacity": 32768, "owner_lanes": 4,
        })
    assert result.bfs_completed
    if preset == "balanced":
        return
    assert result.report["session_policy"]["cross_backend_slowdown_status"] == "UNVERIFIED"
    assert result.report["session_policy"]["capacity_slowdown_status"] == "UNVERIFIED"
    assert result.report["session_policy"]["same_backend_pilot_slowdown_status"] == "MEASURED_FULL_BFS_BOUND"


@pytest.mark.parametrize("degree,backend", [(12, "segmented24"), (13, "permutation_bitmap")])
def test_balanced_excludes_hash_when_certified_live_rows_cannot_fit(monkeypatch, degree, backend):
    generators = [list(range(i)) + [i + 1, i] + list(range(i + 2, degree))
                  for i in range(degree - 1)]
    graph = GraphDefinition.permutation(generators, list(range(degree)))
    capabilities = {"permutation_bitmap_v1": True, "segmented_warp_hash_safe_v1": True}
    monkeypatch.setattr(api, "_require_memory_backend", lambda *args: capabilities)
    monkeypatch.setattr(api.generic_session.Pool, "inventory",
                        lambda *args: [{"free_bytes": 12_539_920_384} for _ in range(2)])
    with open_session(devices=[0, 1], tune="off") as session:
        memory, _ = session._memory_policy(graph, (0, 1), "native", {}, "balanced", {})
    assert memory.backend == backend
    if degree == 13:
        assert (memory.strategy, memory.reason) == ("atomic", "HASH_FRONTIER_EXCEEDS_AVAILABLE_VRAM")


@pytest.mark.parametrize("dimensions, orbit_abi, pack4_abi, options, backend", [
    (24, True, False, {}, "segmented24"),
    (30, False, True, {}, "segmented24"),
    (30, True, True, {}, "permutation_bitmap"),
    (30, True, True, {"bitmap_pack4": False}, "segmented24"),
    (32, True, False, {}, "segmented24"),
    (32, True, True, {}, "permutation_bitmap"),
])
def test_maxmemory_admits_position_orbits_when_hash_cannot_fit(
    tmp_path, monkeypatch, dimensions, orbit_abi, pack4_abi, options, backend,
):
    n = 2 * dimensions
    generators = [[p ^ 1 if p // 2 == i else p for p in range(n)] for i in range(dimensions)]
    graph = GraphDefinition.permutation(generators, [-7 if p & 1 else 511 for p in range(n)])
    monkeypatch.setattr(api, "native_runtime", lambda _: ("native", {}))
    monkeypatch.setattr(api, "_require_memory_backend", lambda *args: {
        "permutation_bitmap_v1": True,
        "permutation_bitmap_position_orbits_v1": orbit_abi,
        "bitmap_pack4_experimental_v1": pack4_abi,
        "bitmap_pack4_adaptive_v2": pack4_abi,
    })
    monkeypatch.setattr(api.generic_session.Pool, "inventory",
                        lambda *args: [{"free_bytes": 12 << 30, "additional_reserve_bytes": 2 << 30}] * 2)

    def launch(actual, output, **kwargs):
        assert actual.digest() == graph.digest()
        assert kwargs["memory_backend"] == backend
        assert kwargs.get("bitmap_pack4", False) == (backend == "permutation_bitmap")
        output.mkdir()
        (output / "states.json").write_text(json.dumps({"current": [graph.start]}), encoding="utf-8")
        return {"status": "INCOMPLETE", "layer_sizes": [1]}

    monkeypatch.setattr(api, "run_graph", launch)
    with open_session(devices=[0, 1], output_root=tmp_path, tune="off") as session:
        result = session.bfs(graph, preset="maxmemory", native_options=options)
    assert result.report["session_policy"]["memory_backend"] == backend


@pytest.mark.parametrize("preset, batch_abi, series_abi, unbounded, explicit, strategy", [
    ("maxmemory", True, True, False, None, "batch"),
    ("maxmemory", False, True, False, None, "warp"),
    ("maxmemory", True, False, False, None, "warp"),
    ("maxmemory", True, True, True, None, "warp"),
    ("maxmemory", True, True, False, "atomic", "atomic"),
    ("balanced", True, True, False, None, "batch"),
    ("balanced", True, True, False, "warp", "warp"),
])
def test_session_hash_strategy_respects_preset_and_capabilities(
    tmp_path, monkeypatch, preset, batch_abi, series_abi, unbounded, explicit, strategy,
):
    start = list(range(65))
    graph = GraphDefinition.permutation([[1, 0, *start[2:]]], start)
    env = {"MGBFS_SEGMENT_HASH_SERIES": "0"} if unbounded else {}
    monkeypatch.setattr(api, "native_runtime", lambda _: ("native", env))
    monkeypatch.setattr(api, "_require_memory_backend", lambda *args: {
        "permutation_bitmap_v1": True,
        "segmented_strategies_v1": {"batch": batch_abi},
        "segmented_hash_series_v1": series_abi,
        "segmented_warp_hash_safe_v1": True,
    })

    def launch(actual, output, **kwargs):
        assert actual.digest() == graph.digest()
        assert kwargs["memory_backend"] == "segmented24"
        assert kwargs["segment_strategy"] == strategy
        output.mkdir()
        (output / "states.json").write_text(json.dumps({"current": [start]}), encoding="utf-8")
        return {"status": "INCOMPLETE", "layer_sizes": [1]}

    monkeypatch.setattr(api, "run_graph", launch)
    with open_session(devices=[0, 1], output_root=tmp_path, tune="off") as session:
        result = session.bfs(graph, preset=preset, native_options={"segment_strategy": explicit} if explicit else None)
    policy = result.report["session_policy"]
    if strategy == "batch":
        assert policy["selection_reason"] == "SEGMENTED_BOUNDED_HASH"
    if preset == "maxmemory":
        assert policy["capacity_slowdown_status"] == "UNVERIFIED"
