"""Physical acceptance of rank reuse after compact-transport resource limits."""

import json
import os
import socket
import time

import pytest

from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
from multigpubfs.native_distribution import native_runtime


@pytest.mark.skipif(
    not os.environ.get("MGBFS_TEST_NATIVE_COMPACT_BINARY"),
    reason="requires rebuilt native CUDA executable and two GPUs",
)
def test_compact_resource_limit_preserves_prefix_and_allows_next_request(tmp_path):
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{reserve.getsockname()[1]}"
    executable, env = native_runtime(os.environ["MGBFS_TEST_NATIVE_COMPACT_BINARY"])
    env["MGBFS_SEGMENT_PACKET_CONTROL_ADDR"] = address
    options = dict(
        devices=(0, 1), shards=2, autotune=False,
        executable=executable,
        memory_backend="segmented24", segment_rows=16, queue_capacity=128,
        _native_env=env,
    )
    graph = GraphDefinition.permutation(
        [[1, 0, 2, 3], [0, 2, 1, 3], [0, 1, 3, 2]], [0, 1, 2, 3],
    )
    expected = graph.exact_layers(24)
    with persistent_graph_workers() as pool:
        partial_dir = tmp_path / "partial"
        partial = run_graph(graph, partial_dir, capacity=1, **options)
        assert partial["status"] == "INCOMPLETE" and partial["reason"] == "RESOURCE_10"
        depth = len(partial["layer_sizes"]) - 1
        assert partial["layer_sizes"] == [len(layer) for layer in expected[:depth + 1]]
        committed = json.loads((partial_dir / "states.json").read_text())
        assert sorted(committed["current"]) == expected[depth]
        assert all(rank["packet_transport"] for rank in partial["segment_memory_by_rank"])
        workers = [worker.process.pid for worker in pool.workers]
        assert len(workers) == 2

        full_dir = tmp_path / "full"
        complete = run_graph(graph, full_dir, capacity=32, **options)
        assert complete["status"] == "COMPLETE"
        assert complete["layer_sizes"] == [len(layer) for layer in expected]
        for rank in complete["segment_memory_by_rank"]:
            sync = rank["global_sync"]
            assert sync["nccl_reduce_votes"] == 1
            assert sync["nccl_round_gathers"] == 0
            # Includes the last attempted layer, which has no new states.
            assert sync["nccl_layer_count_gathers"] == len(expected)
            assert sync["packet_hub_layer_commits"] == len(expected)
            assert sync["legacy_chunk_votes"] == 0
            assert sync["legacy_chunk_count_exchanges"] == 0
        committed = json.loads((full_dir / "states.json").read_text())
        assert sorted(committed["current"]) == expected[-1]
        assert sorted(committed["previous_small"]) == expected[-2]
        assert workers == [worker.process.pid for worker in pool.workers]

        limited_dir = tmp_path / "limited"
        limited = run_graph(graph, limited_dir, capacity=32, _profile_layers=2, **options)
        assert limited["status"] == "INCOMPLETE"
        assert limited["reason"] == "PROFILE_LAYER_LIMIT"
        assert limited["layer_sizes"] == [len(layer) for layer in expected[:3]]
        committed = json.loads((limited_dir / "states.json").read_text())
        assert sorted(committed["current"]) == expected[2]
        assert sorted(committed["previous_small"]) == expected[1]
        for rank in limited["segment_memory_by_rank"]:
            sync = rank["global_sync"]
            assert sync["nccl_reduce_votes"] == 1
            assert sync["nccl_layer_count_gathers"] == 2
            assert sync["packet_hub_layer_commits"] == 2
        assert workers == [worker.process.pid for worker in pool.workers]


@pytest.mark.skipif(
    not os.environ.get("MGBFS_TEST_NATIVE_COMPACT_BINARY"),
    reason="requires rebuilt native CUDA executable and two GPUs",
)
def test_compact_midlayer_deadline_preserves_prefix_and_reuses_workers(tmp_path):
    from scripts.segmented_acceptance_oracle import mahonian

    executable, env = native_runtime(os.environ["MGBFS_TEST_NATIVE_COMPACT_BINARY"])
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        env["MGBFS_SEGMENT_PACKET_CONTROL_ADDR"] = f"127.0.0.1:{reserve.getsockname()[1]}"
    options = dict(
        devices=(0, 1), shards=1, autotune=False, executable=executable,
        memory_backend="permutation_bitmap", bitmap_pack4=True, queue_capacity=256,
        _native_env=env,
    )

    def adjacent_graph(degree):
        return GraphDefinition.permutation(
            [list(range(i)) + [i + 1, i] + list(range(i + 2, degree))
             for i in range(degree - 1)],
            list(range(degree)),
        )

    with persistent_graph_workers() as pool:
        partial_dir = tmp_path / "deadline"
        partial = run_graph(adjacent_graph(11), partial_dir, max_seconds=1, **options)
        assert partial["status"] == "INCOMPLETE" and partial["reason"] == "DEADLINE"
        depth = len(partial["layer_sizes"]) - 1
        assert partial["layer_sizes"] == mahonian(11)[:depth + 1]
        frontier = json.loads((partial_dir / "states.json").read_text())["current"]
        assert len(set(map(tuple, frontier))) == len(frontier) == min(1000, partial["layer_sizes"][-1])
        assert all(sorted(row) == list(range(11)) for row in frontier)
        assert all(sum(a > b for i, a in enumerate(row) for b in row[i+1:]) == depth for row in frontier)
        workers = [worker.process.pid for worker in pool.workers]
        assert len(workers) == 2
        assert all("stage=nccl_abort_end code=0" in (worker.root / "worker.log").read_text()
                   for worker in pool.workers)

        complete_dir = tmp_path / "recovered"
        small = adjacent_graph(4)
        complete = run_graph(small, complete_dir, max_seconds=10, **options)
        assert complete["status"] == "COMPLETE"
        assert complete["layer_sizes"] == mahonian(4)
        committed = json.loads((complete_dir / "states.json").read_text())
        layers = small.exact_layers(24)
        assert sorted(committed["current"]) == layers[-1]
        assert sorted(committed["previous_small"]) == layers[-2]
        assert [worker.process.pid for worker in pool.workers] == workers

@pytest.mark.skipif(
    not os.environ.get("MGBFS_TEST_NATIVE_COMPACT_BINARY"),
    reason="requires rebuilt native CUDA executable and two GPUs",
)
@pytest.mark.parametrize("compact", [False, True], ids=["exact", "compact"])
@pytest.mark.parametrize("banks", [1, 3], ids=["append", "rolling"])
def test_hash_series_snapshot_preserves_exact_layers(tmp_path, compact, banks):
    executable, env = native_runtime(os.environ["MGBFS_TEST_NATIVE_COMPACT_BINARY"])
    env.update(
        MGBFS_SEGMENT_HASH_SERIES="1",
        MGBFS_SEGMENT_TEST_BATCH="1",
        MGBFS_SEGMENT_BATCH_TILE="2",
        MGBFS_SEGMENT_TEST_HASH_BITS="0",
        MGBFS_SEGMENT_EXACT_EXCHANGE="1",
    )
    env.pop("MGBFS_SEGMENT_PACKET_CONTROL_ADDR", None)
    if compact:
        with socket.socket() as reserve:
            reserve.bind(("127.0.0.1", 0))
            env["MGBFS_SEGMENT_PACKET_CONTROL_ADDR"] = f"127.0.0.1:{reserve.getsockname()[1]}"
    # The non-inverse-closed action forces one-bank exact history.
    env.pop("MGBFS_SEGMENT_FULL_HISTORY", None)
    generators = [[1, 2, 3, 4, 5, 0]] if banks == 1 else []
    for position in range(5):
        swap = list(range(6))
        swap[position], swap[position + 1] = swap[position + 1], swap[position]
        generators.append(swap)
    graph = GraphDefinition.permutation(generators, list(range(6)))
    expected = graph.exact_layers(720)
    options = dict(
        devices=(0, 1), shards=1, autotune=False, executable=executable,
        memory_backend="segmented24", segment_strategy="batch", segment_rows=32,
        queue_capacity=128, _batch=4, _native_env=env,
    )
    observed_snapshot = False
    observed_retirement = False
    with persistent_graph_workers() as pool:
        for depth, layer in enumerate(expected[1:], start=1):
            out = tmp_path / str(depth)
            result = run_graph(graph, out, _profile_layers=depth, **options)
            assert result["layer_sizes"] == [len(rows) for rows in expected[:depth + 1]]
            assert sorted(json.loads((out / "states.json").read_text())["current"]) == layer
            for rank in result["segment_memory_by_rank"]:
                assert rank["banks"] == banks
                assert bool(rank["packet_transport"]) is compact
                if rank["hash_snapshot_bytes"] > 0:
                    observed_snapshot = True
                for shard in rank["batch_hash_runs_by_shard"]:
                    if shard:
                        observed_retirement |= shard["snapshot_retirements"] > 0
                        assert shard["lookup_series_live"] <= shard["live"]
                        if shard["lookup_series_live"] < shard["live"]:
                            assert shard["lookup_series_max_observed"] >= 1
        out = tmp_path / "complete"
        complete = run_graph(graph, out, **options)
        assert complete["status"] == "COMPLETE"
        assert complete["layer_sizes"] == [len(rows) for rows in expected]
        assert observed_snapshot
        assert observed_retirement is (banks == 3)
        assert len(pool.workers) == 2

@pytest.mark.skipif(
    not os.environ.get("MGBFS_TEST_NATIVE_COMPACT_BINARY"),
    reason="requires rebuilt native CUDA executable and two GPUs",
)
@pytest.mark.parametrize("compact", [False, True], ids=["exact", "compact"])
def test_hash_snapshot_failure_closes_generation(tmp_path, compact):
    executable, env = native_runtime(os.environ["MGBFS_TEST_NATIVE_COMPACT_BINARY"])
    env.update(
        MGBFS_SEGMENT_HASH_SERIES="1",
        MGBFS_SEGMENT_TEST_BATCH="1",
        MGBFS_SEGMENT_BATCH_TILE="2",
        # S6 reaches 16 active runs, below the production merge threshold.
        MGBFS_SEGMENT_TEST_MAX_LOOKUP_SERIES="8",
        MGBFS_SEGMENT_EXACT_EXCHANGE="1",
    )
    env.pop("MGBFS_SEGMENT_PACKET_CONTROL_ADDR", None)
    if compact:
        with socket.socket() as reserve:
            reserve.bind(("127.0.0.1", 0))
            env["MGBFS_SEGMENT_PACKET_CONTROL_ADDR"] = f"127.0.0.1:{reserve.getsockname()[1]}"
    generators = []
    for position in range(5):
        swap = list(range(6))
        swap[position], swap[position + 1] = swap[position + 1], swap[position]
        generators.append(swap)
    graph = GraphDefinition.permutation(generators, list(range(6)))
    expected = [len(layer) for layer in graph.exact_layers(720)]
    options = dict(
        devices=(0, 1), shards=1, autotune=False, executable=executable,
        memory_backend="segmented24", segment_strategy="batch", segment_rows=32,
        queue_capacity=128, _batch=4,
    )
    failure_dir = tmp_path / "failure"
    bad_env = dict(env, MGBFS_SEGMENT_TEST_HASH_SNAPSHOT_FAIL_RANK="0")
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="NATIVE_DISTRIBUTED_RANK_FAILED"):
        with persistent_graph_workers():
            run_graph(graph, failure_dir, _native_env=bad_env, **options)
    assert time.monotonic() - started < 60
    assert "SEGMENTED_TEST_HASH_SNAPSHOT_FAILURE" in (failure_dir / "rank-0.log").read_text()
    with persistent_graph_workers():
        result = run_graph(graph, tmp_path / "fresh", _native_env=env, **options)
    assert result["status"] == "COMPLETE"
    assert result["layer_sizes"] == expected


@pytest.mark.skipif(
    not os.environ.get("MGBFS_TEST_NATIVE_COMPACT_BINARY"),
    reason="requires rebuilt native CUDA executable and two GPUs",
)
@pytest.mark.parametrize("compact", [False, True], ids=["exact", "compact"])
@pytest.mark.parametrize("banks", [1, 3], ids=["append", "rolling"])
def test_hash_series_bound_under_vram_pressure(tmp_path, record_property, banks, compact):
    executable, env = native_runtime(os.environ["MGBFS_TEST_NATIVE_COMPACT_BINARY"])
    env.update(
        MGBFS_SEGMENT_HASH_SERIES="1",
        MGBFS_SEGMENT_TEST_BATCH="1",
        MGBFS_SEGMENT_BATCH_TILE="2",
        MGBFS_SEGMENT_TEST_HASH_BITS="0",
        MGBFS_SEGMENT_TEST_MAX_LOOKUP_SERIES="8",
        MGBFS_SEGMENT_TEST_MERGE_ROWS="0",
        MGBFS_SEGMENT_HASH_INDEX_VRAM_BYTES="0",
        MGBFS_SEGMENT_HASH_HOST_BYTES=str(32 << 20),
        MGBFS_SEGMENT_TEST_BYTES=str(256 << 10),
        MGBFS_SEGMENT_EXACT_EXCHANGE="1",
    )
    env.pop("MGBFS_SEGMENT_PACKET_CONTROL_ADDR", None)
    if compact:
        with socket.socket() as reserve:
            reserve.bind(("127.0.0.1", 0))
            env["MGBFS_SEGMENT_PACKET_CONTROL_ADDR"] = f"127.0.0.1:{reserve.getsockname()[1]}"
    env.pop("MGBFS_SEGMENT_FULL_HISTORY", None)
    generators = [[1, 2, 3, 4, 5, 0]] if banks == 1 else []
    for position in range(5):
        swap = list(range(6))
        swap[position], swap[position + 1] = swap[position + 1], swap[position]
        generators.append(swap)
    graph = GraphDefinition.permutation(generators, list(range(6)))
    expected = graph.exact_layers(720)
    options = dict(
        devices=(0, 1), shards=1, autotune=False, executable=executable,
        memory_backend="segmented24", segment_strategy="batch",
        segment_rows=32, queue_capacity=128, _batch=4,
    )
    with persistent_graph_workers() as pool:
        stopped_dir = tmp_path / "no-index-budget"
        stopped = run_graph(
            graph, stopped_dir, _native_env=dict(env, MGBFS_SEGMENT_HASH_HOST_BYTES="0"), **options,
        )
        assert stopped["status"] == "INCOMPLETE" and stopped["reason"] == "RESOURCE_10"
        depth = len(stopped["layer_sizes"]) - 1
        assert 0 <= depth < len(expected) - 1
        assert stopped["layer_sizes"] == [len(layer) for layer in expected[:depth + 1]]
        assert sorted(json.loads((stopped_dir / "states.json").read_text())["current"]) == expected[depth]
        details = [rank["resource_detail"] for rank in stopped["segment_memory_by_rank"]]
        assert any(detail and detail["reason"] == "hash_lookup_series_limit" for detail in details)
        for rank in stopped["segment_memory_by_rank"]:
            for shard in filter(None, rank["batch_hash_runs_by_shard"]):
                assert shard["lookup_series_max_observed"] <= 8
        workers = [worker.process.pid for worker in pool.workers]
        assert len(workers) == 2
        full_dir = tmp_path / "host-index-budget"
        result = run_graph(graph, full_dir, _native_env=env, **options)
        assert [worker.process.pid for worker in pool.workers] == workers
    assert result["status"] == "COMPLETE"
    assert result["layer_sizes"] == [len(layer) for layer in expected]
    state = json.loads((full_dir / "states.json").read_text())
    assert sorted(state["current"]) == expected[-1]
    assert sorted(state["previous_small"]) == expected[-2]
    ranks = result["segment_memory_by_rank"]
    assert any(rank["hash_snapshot_mapped_host_installations"] > 0 for rank in ranks)
    assert all(rank["banks"] == banks and rank["hash_snapshot_bytes"] == 0 for rank in ranks)
    assert all(bool(rank["packet_transport"]) is compact for rank in ranks)
    for rank in ranks:
        for shard in filter(None, rank["batch_hash_runs_by_shard"]):
            assert shard["lookup_series_max_observed"] <= 8

    record_property("mapped_host_installations", sum(rank["hash_snapshot_mapped_host_installations"] for rank in ranks))
    record_property("max_lookup_series", max(
        shard["lookup_series_max_observed"]
        for rank in ranks for shard in filter(None, rank["batch_hash_runs_by_shard"])
    ))
