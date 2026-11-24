from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import errno
from itertools import product
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from threading import Barrier
from types import SimpleNamespace

import pytest

from multigpubfs import path_search
from multigpubfs.api import open_session
from multigpubfs import path_lineage
from multigpubfs.graph_definition import GraphDefinition
from multigpubfs.path_network import PathNetwork, close_path_group, connect


torch = pytest.importorskip("torch")
pytest.importorskip("pyarrow")


def _fixture():
    return GraphDefinition.permutation([[1, 2, 0], [1, 0, 2], [1, 2, 0]], [0, -1, -1])


def _records(result):
    return Counter(
        (row["target_index"], tuple(row["generator_ids"])) for row in result.iter_paths()
    )


def _oracle(graph, targets, max_length):
    indices = {tuple(state): i for i, state in enumerate(dict.fromkeys(map(tuple, targets)))}
    records = Counter()
    for length in range(max_length + 1):
        for word in product(range(graph.generator_count), repeat=length):
            state = graph.start
            for generator in word:
                state = graph.successor(state, generator)
            if tuple(state) in indices:
                records[indices[tuple(state)], word] += 1
    return records


@pytest.fixture
def local_torch_ranks(monkeypatch):
    # Only the test replaces the NVIDIA admission gate. Both logical owners
    # execute the same deduplication, routing and path implementation.
    monkeypatch.setattr(path_search, "_cuda_devices", lambda torch, ids: (torch.device("cpu"),) * len(ids))


@pytest.mark.parametrize("direction", ["forward", "bidirectional"])
@pytest.mark.parametrize("start_is_target", [False, True])
def test_window_keeps_alias_generators_repeated_states_and_later_targets(
    local_torch_ranks, tmp_path, direction, start_is_target,
):
    graph = _fixture()
    targets = [[-1, -1, 0], [-1, 0, -1], [7, 7, 7], [-1, -1, 0]]
    if start_is_target:
        targets.insert(1, graph.start)
    distance = 0 if start_is_target else 1
    limit = distance + 3
    result = path_search.solve(
        graph, targets=targets, output=tmp_path, devices=(0, 1),
        direction=direction, goal_mode="window", extra_steps=3,
    )
    assert result.status == "COMPLETE"
    assert result.report["d_min"] == distance
    assert result.report["completed_through_length"] == limit
    assert _records(result) == _oracle(graph, targets, limit)
    assert result.report == json.loads((tmp_path / "report.json").read_text())
    assert result.report["path_bytes"] == (tmp_path / "paths.parquet").stat().st_size


@pytest.mark.parametrize("direction", ["forward", "bidirectional"])
def test_nearest_uses_original_directed_generator_and_actual_start(local_torch_ranks, tmp_path, direction):
    graph = GraphDefinition.permutation([[1, 2, 3, 0]], [0, 1, 2, 3])
    start = [1, 2, 3, 0]
    targets = [[0, 1, 2, 3], [3, 0, 1, 2], [9, 9, 9, 9]]
    result = path_search.solve(
        graph, start=start, targets=targets, output=tmp_path,
        direction=direction, devices=(0, 1),
    )
    assert result.status == "COMPLETE"
    assert result.report["d_min"] == 2
    assert list(result.iter_paths()) == [{"target_index": 1, "length": 2, "generator_ids": [0, 0]}]
    assert result.graph.start == start


@pytest.mark.parametrize("direction", ["forward", "bidirectional"])
def test_unreachable_window_has_an_empty_artifact(local_torch_ranks, tmp_path, direction):
    result = path_search.solve(
        _fixture(), targets=[[9, 9, 9]], output=tmp_path,
        direction=direction, goal_mode="window", extra_steps=3, devices=(0, 1),
    )
    assert result.status == "UNREACHABLE"
    assert result.report["d_min"] is None
    assert list(result.iter_paths()) == []


def test_matrix_window_exact_abi_and_original_reverse_generators(local_torch_ranks, tmp_path):
    graph = GraphDefinition.matrix(2, 1, [
        ([1, 1, 0, 1], 5),
        ([1, -1, 0, 1], 5),
        ([-1, 0, 0, -1], 5),
    ], [0, 1])
    targets = [[2, 1], [-1, 1], [4, 1]]
    result = path_search.solve(
        graph, targets=targets, output=tmp_path, devices=(0, 1),
        direction="bidirectional", goal_mode="window", extra_steps=3, preset="max_capacity",
    )
    assert result.report["preset"] == "maxmemory"
    assert result.report["d_min"] == 1
    assert _records(result) == _oracle(graph, targets, 4)


def test_forward_matrix_preserves_i64_wrap_before_non_power_of_two_modulus(local_torch_ranks, tmp_path):
    graph = GraphDefinition.matrix(1, 1, [([1 << 62], 5)], [3])
    target = graph.successor(graph.start, 0)
    result = path_search.solve(graph, targets=[target], output=tmp_path, devices=(0,))
    assert result.report["d_min"] == 1
    assert _records(result) == {(0, (0,)): 1}
    with pytest.raises(ValueError, match="PATH_MATRIX_REVERSE_UNSUPPORTED"):
        path_search.solve(
            graph, targets=[target], output=tmp_path / "other", devices=(0,), direction="bidirectional",
        )


def test_path_and_output_limits_expose_only_committed_rows(local_torch_ranks, tmp_path):
    graph = _fixture()
    args = dict(devices=(0, 1), targets=[graph.start], goal_mode="window", extra_steps=4)
    limited = path_search.solve(graph, **args, output=tmp_path / "paths", max_paths=2)
    assert limited.status == "INCOMPLETE"
    assert limited.report["stop_reason"] == "PATH_LIMIT"
    assert limited.report["completed_through_length"] < 4
    assert len(list(limited.iter_paths())) == limited.report["path_count"] == 2
    tiny = path_search.solve(graph, **args, output=tmp_path / "bytes", max_output_bytes=1)
    assert tiny.status == "INCOMPLETE"
    assert tiny.report["stop_reason"] == "OUTPUT_LIMIT"
    assert tiny.report["path_count"] == 0
    assert not (tiny.output / "paths.parquet").exists()


@pytest.mark.parametrize("direction", ["forward", "bidirectional"])
def test_lineage_limit_preserves_completed_lengths_and_cleans_owned_scratch(
    local_torch_ranks, tmp_path, direction,
):
    graph = _fixture()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    marker = scratch / "keep"
    marker.write_text("unrelated")
    args = dict(devices=(0, 1), targets=[graph.start], goal_mode="window", extra_steps=4,
                direction=direction, lineage_dir=scratch, lineage_cache_bytes=128)
    complete = path_search.solve(graph, **args, output=tmp_path / "complete")
    assert complete.report["lineage_storage"] == "disk_csr"
    assert complete.report["lineage_committed_csr_bytes"] > 2 * complete.report["lineage_cache_bytes"]
    assert complete.report["lineage_peak_bytes"] >= complete.report["lineage_committed_csr_bytes"]
    assert complete.report["lineage_cache_peak_bytes"] <= complete.report["lineage_cache_bytes"]
    assert _records(complete) == _oracle(graph, [graph.start], 4)

    limited = path_search.solve(graph, **args, output=tmp_path / "limited", max_lineage_bytes=0)
    assert limited.status == "INCOMPLETE"
    assert limited.report["stop_reason"] == "LINEAGE_LIMIT"
    assert limited.report["d_min"] == 0
    assert limited.report["completed_through_length"] == 0
    assert limited.report["lineage_peak_bytes"] == 0
    assert _records(limited) == {(0, ()): 1}
    assert list(scratch.iterdir()) == [marker]


def test_lineage_disk_full_receipt_preserves_committed_output(local_torch_ranks, tmp_path, monkeypatch):
    disk_call = path_lineage._disk_call

    def out_of_space():
        raise OSError(errno.ENOSPC, "test disk full")

    def fail_write(call, *args, **kwargs):
        if call.__name__ == "write":
            return disk_call(out_of_space)
        return disk_call(call, *args, **kwargs)

    monkeypatch.setattr(path_lineage, "_disk_call", fail_write)
    graph = _fixture()
    limited = path_search.solve(
        graph, devices=(0, 1), targets=[graph.start], goal_mode="window",
        extra_steps=4, output=tmp_path / "limited", lineage_dir=tmp_path / "scratch",
    )
    assert limited.status == "INCOMPLETE"
    assert limited.report["stop_reason"] == "LINEAGE_DISK_FULL"
    assert limited.report["completed_through_length"] == 0
    assert _records(limited) == {(0, ()): 1}
    assert not list((tmp_path / "scratch").iterdir())


def test_disk_csr_multigeneration_merge_keeps_every_labeled_edge(tmp_path, monkeypatch):
    monkeypatch.setattr(path_lineage, "_FAN_IN", 2)
    monkeypatch.setattr(path_lineage, "_BLOCK", 64)
    storage = path_lineage.DiskLineage(tmp_path, None, 64)
    try:
        layer = storage.layer("forward", 1, 2)
        expected = [[] for _ in range(7)]
        for i in range(55):
            records = [[i % 7, i % 2, (1 << 34) + i, i], [(i + 3) % 7, 0, 2 * i, i + 60]]
            for child, rank, row, label in records:
                expected[child].append((rank, row, label))
            layer.add(0, torch.tensor(records, dtype=torch.int64))
        for _ in layer.seal((7, 0)):
            pass
        links, empty = layer.links
        assert empty is None
        for child, records in enumerate(expected):
            begin, end = links.bounds(child)
            assert [links.parent(index) for index in range(begin, end)] == records
        assert storage.cache.used <= storage.cache.maximum
        assert storage.peak_bytes > storage.bytes > storage.cache.maximum
        assert {path.name for path in layer.directory.iterdir()} == {"0.offsets", "0.edges", "seal"}
    finally:
        storage.close()
    assert not list(tmp_path.iterdir())


def test_disk_csr_budget_counts_unsealed_runs_and_merge_output(tmp_path):
    storage = path_lineage.DiskLineage(tmp_path, 100, 0)
    try:
        layer = storage.layer("forward", 1, 1)
        layer.add(0, torch.tensor([[0, 0, 9, 1]], dtype=torch.int64))
        layer.add(0, torch.tensor([[0, 0, 9, 2]], dtype=torch.int64))
        assert storage.bytes == 64
        with pytest.raises(path_lineage.SearchStopped, match="LINEAGE_LIMIT"):
            for _ in layer.seal((1,)):
                pass
        assert storage.peak_bytes == 64
        assert layer.links is None and not (layer.directory / "seal").exists()
    finally:
        storage.close()


def test_window_pressure_graph_has_exact_csr_size_and_reads_deep_witness(local_torch_ranks, tmp_path):
    from scripts.benchmark_path_lineage_pressure import clean_target, expected_csr, increasing_counter_graph

    graph = increasing_counter_graph(4)
    result = path_search.solve(graph, targets=[graph.start, clean_target(graph, 4, 8)],
                               devices=(0, 1), goal_mode="window", extra_steps=8,
                               lineage_cache_bytes=64 << 10, output=tmp_path)
    low, high = expected_csr(4, 8, 2)
    assert result.status == "COMPLETE" and result.report["completed_through_length"] == 8
    assert low <= result.report["lineage_committed_csr_bytes"] <= high
    assert 0 < result.report["lineage_cache_peak_bytes"] <= 64 << 10
    assert _records(result) == {(0, ()): 1, (1, (0,) * 8): 1}


def test_time_limit_does_not_claim_unreachable(local_torch_ranks, tmp_path):
    graph = _fixture()
    result = path_search.solve(
        graph, targets=[[9, 9, 9]], output=tmp_path, devices=(0, 1), max_seconds=1e-9,
    )
    assert result.status == "INCOMPLETE"
    assert result.report["stop_reason"] == "TIME_LIMIT"
    assert result.report["d_min"] is None
    assert list(result.iter_paths()) == []


def test_cpu_fallback_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    graph = _fixture()
    with pytest.raises(RuntimeError, match="PATH_SEARCH_REQUIRES_NVIDIA_CUDA"):
        path_search.solve(graph, targets=[graph.start], output=tmp_path)
    assert not (tmp_path / "report.json").exists()


def test_native_protocol_handoff_uses_the_public_parquet_contract(tmp_path, monkeypatch):
    graph = GraphDefinition.permutation([[1, 2, 3, 0]], [0, 1, 2, 3])
    targets = [[3, 0, 1, 2], [2, 3, 0, 1], [3, 0, 1, 2]]

    def call(command, **_):
        native, operation, query = command
        assert native == "mgbfs" and operation == "graph-path-nearest"
        request = json.loads(Path(query).read_text())
        assert request["graph"] == graph.to_dict()
        assert request["targets"] == targets[:2]
        assert request["device"] == 1
        receipt = dict(status="COMPLETE", stop_reason=None, d_min=2, target_index=1, generator_ids=[0, 0])
        return SimpleNamespace(returncode=0, stdout=json.dumps(receipt), stderr="")

    monkeypatch.setattr(path_search.subprocess, "run", call)
    result = path_search.solve(
        graph, targets=targets, output=tmp_path, native=("mgbfs", {}, 1),
        max_lineage_bytes=0,
    )
    assert result.report["backend"] == "native_cuda"
    assert result.report["lineage_storage"] == "native_predecessor_scan"
    assert result.report["lineage_peak_bytes"] == 0
    assert result.report["completed_through_length"] == 2
    assert _records(result) == {(1, (0, 0)): 1}
    assert not list(tmp_path.glob(".pq-*"))


@pytest.mark.parametrize("mixed_matrix_targets", [False, True])
def test_distributed_native_nearest_publishes_the_verified_witness(tmp_path, monkeypatch, mixed_matrix_targets):
    from multigpubfs import launch

    if mixed_matrix_targets:
        graph = GraphDefinition.matrix(1, 1, [([2], 5)], [1])
        targets, direction = [(6,), (4,), (-1,)], "bidirectional"
        target_index = 1
    else:
        graph = GraphDefinition.permutation([[1, 2, 3, 0]], [0, 1, 2, 3])
        targets, direction = [(2, 3, 0, 1)], "forward"
        target_index = 0

    def run_graph(definition, output, **options):
        assert definition == graph and options['devices'] == (0, 1)
        assert options['_path_targets'] == [targets[target_index]] and options['autotune'] is False
        return dict(status='COMPLETE', d_min=2, stop_reason=None, target_index=0, generator_ids=[0, 0])

    monkeypatch.setattr(launch, 'run_graph', run_graph)
    native = ('mgbfs', {}, (0, 1), True)
    result = path_search.solve(graph, targets=targets, direction=direction, output=tmp_path / "found", native=native)
    assert result.report['backend'] == 'native_cuda_nccl'
    assert result.report['devices'] == ['cuda:0', 'cuda:1']
    assert _records(result) == {(target_index, (0, 0)): 1}
    assert result.report["targets"] == [list(target) for target in targets]
    if mixed_matrix_targets:
        unreachable = path_search.solve(graph, targets=[(6,), (-1,)], direction=direction,
                                        output=tmp_path / "unreachable", native=native)
        assert unreachable.status == "UNREACHABLE"
        assert not list(unreachable.iter_paths())


@pytest.mark.skipif(not os.environ.get('MGBFS_TEST_NATIVE_DISTRIBUTED_PATH_BINARY'), reason='requires two CUDA GPUs')
def test_real_native_forward_nearest_on_two_gpus(tmp_path, monkeypatch):
    monkeypatch.setenv('MGBFS_NATIVE_DISTRIBUTED_NEAREST', '1')
    graph = GraphDefinition.permutation([[1, 2, 3, 0]], [0, 1, 2, 3])
    inverse_closed = GraphDefinition.permutation([[1, 0, 2], [0, 2, 1]], [0, 1, 2])
    binary = os.environ['MGBFS_TEST_NATIVE_DISTRIBUTED_PATH_BINARY']
    with open_session(gpus=2, executable=binary, output_root=tmp_path, tune='off') as session:
        found = session.solve(graph, targets=[[3, 0, 1, 2], [2, 3, 0, 1]])
        missing = session.solve(graph, targets=[[7, 7, 7, 7]])
        zero = session.solve(graph, targets=[graph.start])
        changed = session.solve(inverse_closed, targets=[[2, 1, 0]])
    assert all(result.report['backend'] == 'native_cuda_nccl' for result in (found, missing, zero, changed))
    assert _records(found) == {(1, (0, 0)): 1}
    assert missing.status == 'UNREACHABLE' and not _records(missing)
    assert _records(zero) == {(0, ()): 1}
    assert changed.report['d_min'] == 3


@pytest.mark.parametrize("moves", [[], [0]])
def test_native_witness_with_valid_shape_and_wrong_endpoint_is_rejected(tmp_path, monkeypatch, moves):
    graph = GraphDefinition.permutation([[1, 2, 3, 0]], [0, 1, 2, 3])
    native_result = dict(status="COMPLETE", stop_reason=None, d_min=len(moves),
                         target_index=1, generator_ids=moves)
    monkeypatch.setattr(path_search.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=json.dumps(native_result), stderr=""))

    with pytest.raises(RuntimeError, match="NATIVE_PATH_WITNESS_MISMATCH"):
        path_search.solve(graph, targets=[[1, 2, 3, 0], [2, 3, 0, 1]], output=tmp_path,
                          native=("mgbfs", {}, 0))

    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "paths.parquet").exists()


def test_native_resident_worker_keeps_traversal_generation_across_handoffs(tmp_path, monkeypatch):
    from multigpubfs import autotune, generic_session

    graph = GraphDefinition.permutation([[1, 2, 3, 0]], [0, 1, 2, 3])
    calls = []

    class Worker:
        def __init__(self, native, env, rank, device, root, kind):
            self.root = root
            self.root.mkdir()
            self.kind = kind
            self.job = None
            self.process = SimpleNamespace(poll=lambda: None)
            calls.append((kind, "start"))

        def submit(self, command, args, env):
            calls.append((self.kind, command))
            result = {}
            if command == "graph-path-nearest":
                request = json.loads(Path(args[0]).read_text())
                assert request["graph"] == graph.to_dict()
                result = dict(status="COMPLETE", stop_reason=None, d_min=0, target_index=0, generator_ids=[],
                              workspace=dict(worker_pid=5, arena_capacity=4, layout_reused=False,
                                             buffer_hits=0, cached_buffers=12))
            return SimpleNamespace(wait=lambda timeout=None: 0, response={"result": result})

        def close(self):
            calls.append((self.kind, "close"))

    monkeypatch.setattr(generic_session, "Worker", Worker)
    monkeypatch.setattr(autotune, "_dependency_identity", lambda *args: None)
    native = ("mgbfs", {}, 0, True)
    with generic_session.persistent_graph_workers() as pool:
        pool.select("mgbfs", (0,), {})
        traversal_worker = pool.workers[0]
        for index in range(2):
            result = path_search.solve(graph, targets=[graph.start], output=tmp_path / str(index), native=native)
            assert _records(result) == {(0, ()): 1}
            assert result.report["native_workspace"]["arena_capacity"] == 4
        nearest_worker = pool.nearest
        pool.select("mgbfs", (0,), {})
        assert pool.workers[0] is traversal_worker
        assert pool.select_nearest("mgbfs", {}, 0) is nearest_worker
    assert calls == [
        ("generic", "start"),
        ("generic", "release-memory"),
        ("path-nearest", "start"),
        ("path-nearest", "graph-path-nearest"),
        ("path-nearest", "graph-path-nearest"),
        ("path-nearest", "release-memory"),
        ("generic", "release-memory"),
        ("path-nearest", "close"),
        ("generic", "close"),
    ]


@pytest.mark.parametrize("native_status,native_reason,expected", [
    ("INCOMPLETE", "CANCELLED", "TIME_LIMIT"),
    ("COMPLETE", None, None),
])
def test_resident_deadline_only_relabels_actual_cancellation(
    monkeypatch, tmp_path, native_status, native_reason, expected,
):
    from multigpubfs import generic_session

    calls = []
    reply = dict(status=native_status, stop_reason=native_reason)

    def wait(timeout=None):
        calls.append(timeout)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired("job", timeout)
        return 0

    job = SimpleNamespace(wait=wait, request_cancel=lambda: calls.append("cancel"), response={"result": reply})
    worker = SimpleNamespace(submit=lambda *args: job)
    monkeypatch.setattr(generic_session, "_active", SimpleNamespace(select_nearest=lambda *args: worker))
    deadline = generic_session.time.perf_counter() + 60
    result = generic_session.resident_nearest("mgbfs", {}, 0, tmp_path / "query", deadline)
    assert result == dict(reply, stop_reason=expected)
    assert 0 < calls[0] <= 60 and calls[1:] == ["cancel", 5]


def test_resident_cancel_deadline_retires_unresponsive_worker(monkeypatch, tmp_path):
    from multigpubfs import generic_session

    calls = []

    def wait(timeout=None):
        calls.append(timeout)
        raise subprocess.TimeoutExpired("job", timeout)

    job = SimpleNamespace(
        wait=wait, kill=lambda: calls.append("kill"), request_cancel=lambda: calls.append("cancel"),
    )
    worker = SimpleNamespace(submit=lambda *args: job)
    monkeypatch.setattr(generic_session, "_active", SimpleNamespace(
        select_nearest=lambda *args: worker, close_nearest=lambda: calls.append("close"),
    ))
    deadline = generic_session.time.perf_counter() + 60
    result = generic_session.resident_nearest("mgbfs", {}, 0, tmp_path / "query", deadline)
    assert result == dict(status="INCOMPLETE", stop_reason="TIME_LIMIT",
                          d_min=None, target_index=None, generator_ids=[])
    assert 0 < calls[0] <= 60 and calls[1:] == ["cancel", 5, "kill", "close"]


def test_resident_submit_failure_retires_failed_worker(monkeypatch, tmp_path):
    from multigpubfs import generic_session

    closed = []

    def broken(*args):
        raise RuntimeError("GENERIC_SESSION_FAILED_NO_RESTART")

    monkeypatch.setattr(generic_session, "_active", SimpleNamespace(
        select_nearest=lambda *args: SimpleNamespace(submit=broken),
        close_nearest=lambda: closed.append(True),
    ))
    with pytest.raises(RuntimeError, match="GENERIC_SESSION_FAILED_NO_RESTART"):
        generic_session.resident_nearest("mgbfs", {}, 0, tmp_path / "query", None)
    assert closed == [True]


@pytest.mark.skipif(not os.environ.get("MGBFS_TEST_NATIVE_PATH_BINARY"), reason="requires rebuilt native CUDA executable")
def test_real_native_single_gpu_forward_nearest_session(tmp_path):
    executable = os.environ["MGBFS_TEST_NATIVE_PATH_BINARY"]
    directed = GraphDefinition.permutation([[1, 2, 3, 0]], [0, 1, 2, 3])
    modular = GraphDefinition.matrix(1, 1, [([1 << 62], 5)], [3])
    args = dict(gpus=1, devices=None, executable=executable, output_root=tmp_path)
    with open_session(**args) as session:
        nearest = session.solve(directed, targets=[[3, 0, 1, 2], [2, 3, 0, 1]])
        zero = session.solve(directed, targets=[directed.start])
        missing = session.solve(directed, targets=[[7, 7, 7, 7]])
        matrix = session.solve(modular, targets=[modular.successor(modular.start, 0)])
        across_tiles = session.solve(directed, targets=[[1000 + k] * 4 for k in range(8192)] + [directed.start])
    results = nearest, zero, missing, matrix, across_tiles
    assert all(result.report["backend"] == "native_cuda" for result in results)
    assert _records(nearest) == {(1, (0, 0)): 1}
    assert _records(zero) == {(0, ()): 1}
    assert missing.status == "UNREACHABLE" and not _records(missing)
    assert _records(matrix) == {(0, (0,)): 1}
    assert _records(across_tiles) == {(8192, ()): 1}

    cold, warm, again, changed, returned = [result.report["native_workspace"] for result in results]
    assert len({item["worker_pid"] for item in (cold, warm, again, changed, returned)}) == 1
    assert cold["arena_capacity"] == warm["arena_capacity"] == again["arena_capacity"] == returned["arena_capacity"]
    assert cold["cached_buffers"] == warm["cached_buffers"] == again["cached_buffers"]
    assert cold["layout_reused"] is False and cold["buffer_hits"] == 0
    assert warm["layout_reused"] and warm["buffer_hits"] > 0
    assert again["layout_reused"] and again["buffer_hits"] > 0
    assert changed["layout_reused"] is False and changed["buffer_hits"] == 0
    assert returned["layout_reused"] is False and returned["buffer_hits"] == 0


@pytest.mark.skipif(not os.environ.get("MGBFS_TEST_NATIVE_PATH_BINARY"), reason="requires rebuilt native CUDA executable")
def test_native_forward_nearest_recovers_exact_edge_when_destination_layer_exceeds_capacity(tmp_path):
    executable = os.environ["MGBFS_TEST_NATIVE_PATH_BINARY"]
    directed = GraphDefinition.permutation(
        [[1, 2, 3, 0]], [0, 1, 2, 3], expected_max_unique_states=2,
    )
    modular = GraphDefinition.matrix(1, 1, [([1 << 62], 5)], [3], expected_max_unique_states=1)
    near, far = [2, 3, 0, 1], [3, 0, 1, 2]
    with open_session(gpus=1, executable=executable, output_root=tmp_path) as session:
        recovered = session.solve(directed, targets=[far, near])
        beyond = session.solve(directed, targets=[far])
        repeated = session.solve(directed, targets=[near])
        matrix = session.solve(modular, targets=[[1]])

    assert recovered.report["native_workspace"]["arena_capacity"] == 2
    assert recovered.status == "COMPLETE" and recovered.report["d_min"] == 2
    assert _records(recovered) == {(1, (0, 0)): 1}
    assert beyond.status == "INCOMPLETE" and beyond.report["stop_reason"] == "CUDA_MEMORY"
    assert beyond.report["d_min"] is None and not _records(beyond)
    assert repeated.report["native_workspace"]["layout_reused"]
    assert repeated.report["native_workspace"]["buffer_hits"] > 0
    assert _records(repeated) == {(0, (0, 0)): 1}
    assert matrix.report["native_workspace"]["arena_capacity"] == 1
    assert matrix.status == "COMPLETE" and matrix.report["d_min"] == 1
    assert _records(matrix) == {(0, (0,)): 1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
def test_real_cuda_directed_bidirectional_window(tmp_path):
    graph = _fixture()
    targets = [graph.start, [-1, -1, 0]]
    devices = tuple(range(min(torch.cuda.device_count(), 2)))
    result = path_search.solve(
        graph, targets=targets, output=tmp_path, devices=devices,
        direction="bidirectional", goal_mode="window", extra_steps=4,
    )
    assert result.report["d_min"] == 0
    assert _records(result) == _oracle(graph, targets, 4)


def _run_rank_searches(rank, rendezvous, output, wait, *, persistent=False):
    option = {"rank": rank, "world_size": 2, "init_method": rendezvous, "timeout_seconds": 20}
    graph = _fixture()
    targets = [[-1, -1, 0], [-1, 0, -1]]
    base = dict(graph=graph, targets=targets, devices=(rank,), distributed=option)
    nearest = path_search.solve(
        **base, direction="bidirectional", output=output / "nearest",
    )
    wait()
    option["persistent"] = persistent
    for direction in ("forward", "bidirectional"):
        window = path_search.solve(
            **base, direction=direction, goal_mode="window", extra_steps=3, output=output / direction,
        )
        if rank == 0:
            assert window.status == "COMPLETE"
            assert window.report["world_size"] == 2
            assert window.report["process_group"] == ("persistent" if persistent else "per_query")
            assert window.report["d_min"] == 1
            assert _records(window) == _oracle(graph, targets, 4)
        else:
            assert window is None
        wait()
    stopped = path_search.solve(
        **base, direction="bidirectional", goal_mode="window", extra_steps=3,
        max_paths=1, output=output / "limited",
    )
    if rank == 0:
        assert nearest.status == "COMPLETE"
        assert nearest.report["d_min"] == 1
        assert sum(_records(nearest).values()) == 1
        assert _records(nearest) <= _oracle(graph, targets, 1)
        assert stopped.status == "INCOMPLETE"
        assert stopped.report["stop_reason"] == "PATH_LIMIT"
        assert stopped.report["path_count"] == 1
    else:
        assert nearest is None and stopped is None
    wait()
    limited = path_search.solve(
        graph, devices=(rank,), distributed=option, targets=[graph.start],
        direction="bidirectional", goal_mode="window", extra_steps=2,
        max_lineage_bytes=0, output=output / "lineage-limited",
        lineage_dir=output / f"scratch-is-local-to-{rank}",
    )
    if rank == 0:
        assert limited.report["stop_reason"] == "LINEAGE_LIMIT"
        assert limited.report["completed_through_length"] == 0
        assert _records(limited) == {(0, ()): 1}
        assert not list((output / "scratch-is-local-to-0").iterdir())
    else:
        assert limited is None
        assert not (output / "scratch-is-local-to-1").exists()


def _distributed_worker(rank, host, port, output, backend, coordinator):
    if backend == "gloo":
        # This backend is used by tests only; production requires NCCL.
        path_search._cuda_devices = lambda torch, ids: (torch.device("cpu"),)
        path_search.connect = lambda *args: connect(*args, backend="gloo")
    try:
        _run_rank_searches(rank, f"tcp://{host}:{port}?use_libuv=0", output, coordinator.wait, persistent=True)
    finally:
        close_path_group()


def _run_distributed(output, backend):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        host, port = listener.getsockname()
    coordinator = torch.multiprocessing.get_context("spawn").Barrier(2, timeout=30)
    torch.multiprocessing.spawn(_distributed_worker, args=(host, port, output, backend, coordinator), nprocs=2)


class _Collectives:
    """Test two rank programs simultaneously and check every collective edge."""

    def __init__(self):
        self.calls = [None, None]
        self.barrier = Barrier(2, timeout=15, action=self.execute)

    def execute(self):
        (kind, first), (other_kind, second) = self.calls
        assert kind == other_kind
        if kind == "all_to_all_single":
            for destination, (output, _, output_splits, _) in enumerate((first, second)):
                chunks = []
                for source, (_, send, _, input_splits) in enumerate((first, second)):
                    splits = input_splits if input_splits is not None else [len(send) // 2] * 2
                    chunk = send.split(splits)[destination]
                    if output_splits is not None:
                        assert len(chunk) == output_splits[source]
                    chunks.append(chunk)
                output.copy_(torch.cat(chunks))
        else:
            output, mode = first
            other, other_mode = second
            assert mode == other_mode
            if kind == "broadcast":
                value = (output if mode == 0 else other).clone()
            elif mode == torch.distributed.ReduceOp.MAX:
                value = torch.maximum(output, other)
            elif mode == torch.distributed.ReduceOp.MIN:
                value = torch.minimum(output, other)
            else:
                assert mode == torch.distributed.ReduceOp.SUM
                value = output + other
            output.copy_(value)
            other.copy_(value)

    def for_rank(self, rank):
        parent = self

        class Rank:
            ReduceOp = torch.distributed.ReduceOp

            def all_to_all_single(self, output, value, output_splits=None, input_splits=None):
                self.call("all_to_all_single", output, value, output_splits, input_splits)

            def broadcast(self, output, source):
                self.call("broadcast", output, source)

            def all_reduce(self, output, operation):
                self.call("all_reduce", output, operation)

            def call(self, name, *arguments):
                parent.calls[rank] = (name, arguments)
                parent.barrier.wait()

        return Rank()


def test_paired_ranks_follow_identical_collective_stages(monkeypatch, tmp_path):
    collectives = _Collectives()

    @contextmanager
    def connect_contract(torch, devices, options, check):
        rank = options["rank"]
        net = PathNetwork(torch, devices[0], rank, 2, check)
        net.dist = collectives.for_rank(rank)
        yield net

    monkeypatch.setattr(path_search, "_cuda_devices", lambda torch, ids: (torch.device("cpu"),))
    monkeypatch.setattr(path_search, "connect", connect_contract)
    coordinator = Barrier(2, timeout=15)
    with ThreadPoolExecutor(2) as pool:
        futures = [
            pool.submit(_run_rank_searches, rank, "ignored", tmp_path, coordinator.wait)
            for rank in range(2)
        ]
        for future in futures:
            future.result(timeout=40)


def test_path_group_keeps_only_matching_workspace_until_handoff():
    class Dist:
        def __init__(self):
            self.started = self.closed = 0

        def is_available(self):
            return True

        def is_initialized(self):
            return self.started > self.closed

        def init_process_group(self, *_args, **_kwargs):
            self.started += 1

        def destroy_process_group(self):
            self.closed += 1

    dist = Dist()
    driver = SimpleNamespace(distributed=dist, empty=torch.empty, int64=torch.int64)
    devices = (torch.device("cpu"),)
    options = dict(rank=0, world_size=2, init_method="tcp://example:3456", persistent=True)
    try:
        with connect(driver, devices, options, lambda: None, backend="gloo"):
            assert dist.started == 1
        with connect(driver, devices, options, lambda: None, backend="gloo"):
            assert dist.started == 1
        with pytest.raises(RuntimeError, match="PATH_DISTRIBUTED_PERSISTENT_GROUP_MISMATCH"):
            with connect(driver, devices, dict(options, rank=1), lambda: None, backend="gloo"):
                pass
        close_path_group()
        close_path_group()
        assert dist.closed == 1
        with pytest.raises(RuntimeError, match="transport error"):
            with connect(driver, devices, options, lambda: None, backend="gloo"):
                raise RuntimeError("transport error")
        assert dist.started == dist.closed == 2
    finally:
        close_path_group()


@pytest.mark.skipif(
    sys.platform == "win32" or not torch.distributed.is_gloo_available(),
    reason="requires a working non-Windows Gloo transport",
)
def test_two_process_exact_collectives_and_reopen(tmp_path):
    _run_distributed(tmp_path, "gloo")


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not torch.distributed.is_nccl_available(),
    reason="requires two NVIDIA GPUs and Torch NCCL",
)
def test_two_process_real_nccl(tmp_path):
    _run_distributed(tmp_path, "nccl")

def test_native_forward_window_streams_csr_and_tracks_partial_length(tmp_path, monkeypatch):
    from struct import pack
    from multigpubfs import launch

    graph = GraphDefinition.permutation([[1, 0], [1, 0]], [0, 1])
    targets = [(1, 0)]
    calls = []

    def fake_launch(graph, output, **config):
        calls.append(config["_path_window"])
        assert config["_path_targets"] == targets
        if config["_path_window"] is None:
            return dict(status="COMPLETE", stop_reason=None, d_min=1,
                        target_index=0, generator_ids=[0])
        assert config["_path_window"] == {"max_length": 1, "disk_budget": 4096}
        rank = output / "rank-0"
        csr = rank / "1.csr"
        csr.mkdir(parents=True)
        (csr / "offsets").write_bytes(pack("<QQ", 0, 2))
        (csr / "edges").write_bytes(pack("<QIIQII", 0, 0, 0, 0, 0, 1))
        (rank / "1.targets").write_bytes(pack("<QQ", 0, 0))
        return path_search._agree_window_receipts([
            dict(status="COMPLETE", stop_reason=None, graph_completed_through_depth=1,
                 layer_rows=[1, 1], lineage_peak_bytes=64, lineage_bytes=64, lineage_csr_bytes=48)
        ])

    monkeypatch.setattr(launch, "run_graph", fake_launch)
    options = dict(targets=targets, goal_mode="window", native=("mgbfs", {}, (0,), True),
                   max_lineage_bytes=4096, lineage_cache_bytes=0)
    full = path_search.solve(graph, output=tmp_path / "full", **options)
    assert full.report["completed_through_length"] == 1
    assert _records(full) == {(0, (0,)): 1, (0, (1,)): 1}
    partial = path_search.solve(graph, output=tmp_path / "partial", max_paths=1, **options)
    assert partial.report["completed_through_length"] is None
    assert partial.report["stop_reason"] == "PATH_LIMIT"
    assert _records(partial) == {(0, (0,)): 1}
    assert len(calls) == 4

def test_native_window_consensus_rejects_missing_cut_layers():
    rank = dict(status="COMPLETE", stop_reason=None, graph_completed_through_depth=4,
                first_cut_length=1, last_cut_length=4, side_layer_rows=[[1, 2, 3], [1, 2, 3]])
    assert path_search._agree_window_receipts([rank])["last_cut_length"] == 4
    missing = dict(rank, side_layer_rows=[[1, 2], [1, 2, 3]])
    with pytest.raises(RuntimeError, match="NATIVE_WINDOW_CONSENSUS_MISMATCH"):
        path_search._agree_window_receipts([missing])
    with pytest.raises(RuntimeError, match="NATIVE_WINDOW_CONSENSUS_MISMATCH"):
        path_search._agree_window_receipts([rank, missing])


def test_native_bidirectional_window_streams_unique_cuts_with_cycles(tmp_path, monkeypatch):
    from struct import pack
    from multigpubfs import launch

    graph = GraphDefinition.matrix(1, 1, [([1], 5), ([1], 5)], [2])
    targets = [(7,), (2,)]

    def fake_launch(graph, output, **config):
        assert config["_path_bidirectional"] is True
        assert config["_path_targets"] == [(2,)]
        if config["_path_window"] is None:
            return dict(status="COMPLETE", stop_reason=None, d_min=0, target_index=0, generator_ids=[])
        assert config["_path_window"] == {"min_length": 0, "max_length": 2, "disk_budget": 4096}
        rank = output / "rank-0"
        for side in ("forward", "reverse"):
            csr = rank / side / "1.csr"
            csr.mkdir(parents=True)
            (csr / "offsets").write_bytes(pack("<QQ", 0, 2))
            (csr / "edges").write_bytes(pack("<QIIQII", 0, 0, 0, 0, 0, 1))
        (rank / "reverse" / "0.targets").write_bytes(pack("<QQ", 0, 0))
        for length in range(3):
            (rank / f"{length}.matches").write_bytes(pack("<QQ", 0, 0))
        return path_search._agree_window_receipts([
            dict(status="COMPLETE", stop_reason=None, graph_completed_through_depth=2,
                 first_cut_length=0, last_cut_length=2, side_layer_rows=[[1, 1], [1, 1]],
                 lineage_peak_bytes=160, lineage_csr_bytes=96)
        ])

    monkeypatch.setattr(launch, "run_graph", fake_launch)
    options = dict(targets=targets, direction="bidirectional", goal_mode="window", extra_steps=2,
                   native=("mgbfs", {}, (0,), True), max_lineage_bytes=4096, lineage_cache_bytes=0)
    full = path_search.solve(graph, output=tmp_path / "full", **options)
    assert full.report["completed_through_length"] == 2
    assert full.report["nearest_witness"] == {"target_index": 1, "generator_ids": []}
    assert _records(full) == _oracle(graph, targets, 2)
    partial = path_search.solve(graph, output=tmp_path / "partial", max_paths=4, **options)
    assert partial.report["completed_through_length"] == 1
    assert partial.report["stop_reason"] == "PATH_LIMIT"
    assert len(_records(partial)) == 4

@pytest.mark.skipif(not os.environ.get("MGBFS_TEST_NATIVE_DISTRIBUTED_PATH_BINARY"),
                    reason="requires rebuilt native CUDA executable")
@pytest.mark.parametrize("bidirectional", [False, True])
def test_native_window_reuses_a_one_row_arena_across_cycle_depths(tmp_path, bidirectional):
    from multigpubfs.launch import run_graph

    graph = GraphDefinition.permutation([[0], [0]], [7])
    result_root = tmp_path / "window"
    receipt = run_graph(
        graph, result_root, devices=(0,), capacity=1, shards=1,
        executable=os.environ["MGBFS_TEST_NATIVE_DISTRIBUTED_PATH_BINARY"],
        backend="generic", autotune=False, _path_targets=[(7,)],
        _path_bidirectional=bidirectional,
        _path_window={"min_length": 0, "max_length": 8, "disk_budget": 4096},
    )
    assert receipt["status"] == "COMPLETE" and receipt["graph_completed_through_depth"] == 8
    report = {"direction": "bidirectional" if bidirectional else "forward", "lineage_cache_bytes": 128}
    found = Counter()
    path_search._read_native_window(
        result_root, receipt, 0, 8, [0], lambda: None,
        lambda target, labels: found.update([(target, labels)]), report,
    )
    assert report["completed_through_length"] == 8
    assert found == _oracle(graph, [(7,)], 8)


@pytest.mark.parametrize("modulus,state", [(8, [1, 6]), (0, [-(1 << 63), (1 << 63) - 1])])
def test_reverse_binary_ring_matrix_without_inverse_generator(modulus, state):
    from multigpubfs.path_gpu import inverse_action

    matrix = [2, 3, 1, 4]
    graph = GraphDefinition.matrix(2, 1, [(matrix, modulus)], [1, 6])
    assert not graph.inverse_closed()
    inverse = inverse_action(graph)["generators"][0]["matrix"]
    successor = graph._multiply(matrix, state, 2, 1, modulus)
    assert graph._multiply(inverse, successor, 2, 1, modulus) == state


def test_reverse_composite_modulus_without_a_unit_in_first_column():
    from multigpubfs.path_gpu import inverse_action

    matrix = [2, 3, 3, 4]
    graph = GraphDefinition.matrix(2, 1, [(matrix, 12)], [1, 6])
    assert not graph.inverse_closed()
    inverse = inverse_action(graph)["generators"][0]["matrix"]
    for x in range(12):
        for y in range(12):
            successor = graph._multiply(matrix, [x, y], 2, 1, 12)
            assert graph._multiply(inverse, successor, 2, 1, 12) == [x, y]


@pytest.mark.skipif(not os.environ.get("MGBFS_TEST_NATIVE_DISTRIBUTED_PATH_BINARY"),
                    reason="requires rebuilt native CUDA executable")
@pytest.mark.parametrize("bidirectional", [False, True])
def test_native_nearest_follows_stored_links_across_two_ranks(tmp_path, bidirectional):
    from multigpubfs.launch import run_graph

    graph = GraphDefinition.permutation([[1, 2, 3, 4, 5, 6, 0]], list(range(7)))
    near = tuple(range(4, 7)) + tuple(range(4))
    far = tuple(range(6, 7)) + tuple(range(6))
    receipt = run_graph(
        graph, tmp_path / "native", devices=(0, 1), capacity=16, shards=1,
        executable=os.environ["MGBFS_TEST_NATIVE_DISTRIBUTED_PATH_BINARY"],
        backend="generic", autotune=False, _path_targets=[far, near],
        _path_bidirectional=bidirectional,
    )
    assert receipt["status"] == "COMPLETE"
    assert receipt["target_index"] == 1
    assert receipt["d_min"] == 4
    assert receipt["generator_ids"] == [0] * 4


@pytest.mark.skipif(not os.environ.get("MGBFS_TEST_NATIVE_DISTRIBUTED_PATH_BINARY"),
                    reason="requires rebuilt native CUDA executable")
@pytest.mark.parametrize("bidirectional,include_unreachable,state_bytes", [
    (False, True, 1), (True, False, 1), (True, True, 8),
])
def test_native_nearest_keeps_full_parent_metadata(tmp_path, bidirectional, include_unreachable, state_bytes):
    from multigpubfs.launch import run_graph

    degree = 25
    graph = GraphDefinition.permutation([list(range(1, degree)) + [0]], list(range(degree)))
    goal = tuple(range(3, degree)) + (0, 1, 2)
    unreachable = (-1,) + goal[1:]
    targets = [unreachable, goal] if include_unreachable else [goal]
    receipt = run_graph(
        graph, tmp_path / "native", devices=(0, 1), capacity=32, shards=1,
        executable=os.environ["MGBFS_TEST_NATIVE_DISTRIBUTED_PATH_BINARY"],
        backend="generic", autotune=False, _path_targets=targets,
        _path_bidirectional=bidirectional,
    )
    assert receipt["status"] == "COMPLETE"
    assert receipt["native_state_bytes"] == state_bytes
    assert receipt["target_index"] == int(include_unreachable)
    assert receipt["d_min"] == 3
    assert receipt["generator_ids"] == [0, 0, 0]


@pytest.mark.skipif(not os.environ.get("MGBFS_TEST_NATIVE_DISTRIBUTED_PATH_BINARY"),
                    reason="requires rebuilt native CUDA executable and two GPUs")
@pytest.mark.parametrize("priority", [0, 1])
def test_native_forward_nearest_resolves_ties_in_request_order(tmp_path, priority):
    from multigpubfs.launch import run_graph

    moves = [[1, 0, 2], [0, 2, 1]]
    graph = GraphDefinition.permutation(moves, [0, 1, 2])
    targets = [tuple(moves[priority]), tuple(moves[1 - priority])]
    receipt = run_graph(
        graph, tmp_path / "native", devices=(0, 1), capacity=8, shards=1,
        executable=os.environ["MGBFS_TEST_NATIVE_DISTRIBUTED_PATH_BINARY"],
        backend="generic", autotune=False, _path_targets=targets,
        _path_bidirectional=False,
    )
    assert receipt["status"] == "COMPLETE"
    assert receipt["d_min"] == 1
    assert receipt["target_index"] == 0
    assert receipt["generator_ids"] == [priority]


@pytest.mark.skipif(not os.environ.get("MGBFS_TEST_NATIVE_DISTRIBUTED_PATH_BINARY"),
                    reason="requires rebuilt native CUDA executable")
@pytest.mark.parametrize("bidirectional", [False, True])
def test_native_window_byte_arena_preserves_parallel_labels(tmp_path, bidirectional):
    from multigpubfs.launch import run_graph

    degree = 25
    rotate = list(range(1, degree)) + [0]
    swap = [1, 0, *range(2, degree)]
    graph = GraphDefinition.permutation([rotate, rotate, swap], list(range(degree)))
    target = tuple(range(3, degree)) + (0, 1, 2)
    receipt = run_graph(
        graph, tmp_path / "native", devices=(0, 1), capacity=32, shards=1,
        executable=os.environ["MGBFS_TEST_NATIVE_DISTRIBUTED_PATH_BINARY"],
        backend="generic", autotune=False, _path_targets=[target],
        _path_bidirectional=bidirectional,
        _path_window={"min_length": 3, "max_length": 3, "disk_budget": 16384},
    )
    assert receipt["status"] == "COMPLETE"
    assert all(rank["native_state_bytes"] == 1 for rank in receipt["ranks"])
    report = {"direction": "bidirectional" if bidirectional else "forward", "lineage_cache_bytes": 128}
    found = Counter()
    path_search._read_native_window(
        tmp_path / "native", receipt, 3, 3, [0], lambda: None,
        lambda index, labels: found.update([(index, labels)]), report,
    )
    assert report["completed_through_length"] == 3
    assert found == _oracle(graph, [target], 3)


@pytest.mark.skipif(
    sys.platform != "linux" or os.environ.get("MGBFS_TEST_LARGE_NATIVE_WINDOW") != "1"
    or not os.environ.get("MGBFS_TEST_NATIVE_DISTRIBUTED_PATH_BINARY"),
    reason="requires two CUDA GPUs and disk space for the native CSR acceptance workload",
)
def test_native_forward_window_disk_graph_exceeds_rank_zero_ram_budget(tmp_path, record_property):
    from math import comb
    from resource import RUSAGE_SELF, getrusage
    import pyarrow as arrow
    import pyarrow.parquet as pq
    from multigpubfs.launch import run_graph

    # The cycle can only advance by generator zero. Independent involutions
    # generate a large exact-depth graph without more words reaching the goal.
    switches, length = 17, 25
    cycle = length + 1
    start = list(range(cycle + 2 * switches))
    turn = start.copy()
    turn[:cycle] = list(range(1, cycle)) + [0]
    generators = [turn]
    for index in range(cycle, len(start), 2):
        move = start.copy()
        move[index], move[index + 1] = move[index + 1], move[index]
        generators.append(move)
    target = tuple([length] + list(range(length)) + start[cycle:])
    graph = GraphDefinition.permutation(generators, start)
    receipt = run_graph(
        graph, tmp_path / "native", devices=(0, 1), capacity=1 << 21, shards=1,
        executable=os.environ["MGBFS_TEST_NATIVE_DISTRIBUTED_PATH_BINARY"],
        backend="generic", autotune=False, _path_targets=[target],
        _path_window={"min_length": 0, "max_length": length, "disk_budget": 8 << 30},
    )
    assert receipt["status"] == "COMPLETE"
    expected_rows = [
        sum(comb(switches, bits) * (1 + (depth - bits) // 2)
            for bits in range(min(depth, switches) + 1))
        for depth in range(length + 1)
    ]
    actual_rows = [sum(counts) for counts in zip(
        *(rank["layer_rows"] for rank in receipt["ranks"]), strict=True)]
    assert actual_rows == expected_rows

    report = {"direction": "forward", "lineage_cache_bytes": 8 << 20}
    writer = path_search._ParquetPaths(tmp_path, None, None, lambda: None, arrow, pq)
    try:
        path_search._read_native_window(
            tmp_path / "native", receipt, length, length, [0], lambda: None,
            writer.add, report, flush=writer.flush,
        )
    except BaseException:
        writer.abort()
        raise
    artifact = writer.finish()
    assert report["completed_through_length"] == length
    found = pq.read_table(tmp_path / artifact["path_file"]).to_pylist()
    assert found == [{"target_index": 0, "length": length, "generator_ids": [0] * length}]
    ram_budget = 1280 << 20
    assert report["lineage_committed_csr_bytes"] > 2 * ram_budget
    assert report["lineage_cache_peak_bytes"] <= report["lineage_cache_bytes"]
    assert 0 < report["native_process_peak_rss_bytes_by_rank"][0] < ram_budget
    reader_peak = getrusage(RUSAGE_SELF).ru_maxrss * 1024
    assert 0 < reader_peak < ram_budget
    for name, value in (
        ("rank_zero_ram_budget_bytes", ram_budget),
        ("rank_zero_peak_rss_bytes", report["native_process_peak_rss_bytes_by_rank"][0]),
        ("reader_peak_rss_bytes", reader_peak),
        ("committed_csr_bytes", report["lineage_committed_csr_bytes"]),
        ("lineage_peak_bytes", report["lineage_peak_bytes"]),
        ("cache_peak_bytes", report["lineage_cache_peak_bytes"]),
        ("completed_through_length", report["completed_through_length"]),
        ("verified_route_count", len(found)),
        ("parquet_bytes", artifact["path_bytes"]),
    ):
        record_property(name, value)


def test_native_lineage_rejects_offsets_outside_committed_edges(tmp_path):
    from struct import pack

    (tmp_path / "offsets").write_bytes(pack("<QQQ", 0, 3, 2))
    (tmp_path / "edges").write_bytes(pack("<QIIQII", 0, 0, 0, 0, 0, 1))
    links = path_lineage.DiskLinks.from_native_owner(tmp_path, path_lineage._Blocks(32), 2)
    for row in (0, 1):
        with pytest.raises(ValueError, match="PATH_LINEAGE_EDGE_BOUNDS"):
            links.bounds(row)


def test_native_walk_rejects_parent_outside_committed_roots(tmp_path):
    from struct import pack

    (tmp_path / "offsets").write_bytes(pack("<QQ", 0, 1))
    (tmp_path / "edges").write_bytes(pack("<QII", 0, 1, 0))
    links = path_lineage.DiskLinks.from_native_owner(tmp_path, path_lineage._Blocks(32), 1)
    history = [path_search.Layer([], []), path_search.Layer([], [links])]
    with pytest.raises(RuntimeError, match="NATIVE_WINDOW_ROOT_BOUNDS"):
        list(path_search._walks(history, 0, 0, lambda: None, root_counts=[1, 0]))


def test_native_lineage_rejects_missing_predecessors(tmp_path):
    from struct import pack

    (tmp_path / "offsets").write_bytes(pack("<QQQ", 0, 0, 2))
    (tmp_path / "edges").write_bytes(pack("<QIIQII", 0, 0, 0, 0, 0, 1))
    links = path_lineage.DiskLinks.from_native_owner(tmp_path, path_lineage._Blocks(32), 2)
    with pytest.raises(ValueError, match="PATH_LINEAGE_EDGE_BOUNDS"):
        links.bounds(0)


def test_native_lineage_rejects_unreferenced_leading_edge(tmp_path):
    from struct import pack

    (tmp_path / "offsets").write_bytes(pack("<QQ", 1, 2))
    (tmp_path / "edges").write_bytes(pack("<QIIQII", 0, 0, 0, 0, 0, 1))
    with pytest.raises(ValueError, match="PATH_LINEAGE_NATIVE_SIZE"):
        path_lineage.DiskLinks.from_native_owner(tmp_path, path_lineage._Blocks(32), 1)


def test_native_terminal_reads_share_small_lineage_cache(tmp_path):
    from multigpubfs.path_lineage import _Blocks

    left, right = tmp_path / "left.targets", tmp_path / "right.matches"
    left.write_bytes(b"".join(path_search._TERMINAL.pack(i, i + 1) for i in range(9)))
    right.write_bytes(path_search._TERMINAL.pack(5, 7))
    cache = _Blocks(63)
    rows = path_search._native_terminal_rows(left, cache)
    for i in range(9):
        assert next(rows) == (i, i + 1)
        assert list(path_search._native_terminal_rows(right, cache)) == [(5, 7)]
    assert 0 < cache.peak <= 63
    assert cache.used == sum(map(len, cache.cached.values()))
    right.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="NATIVE_WINDOW_TERMINAL_SIZE"):
        list(path_search._native_terminal_rows(right, cache))


def test_native_lineage_cache_bounds_short_file_entries(tmp_path):
    from multigpubfs.path_lineage import _Blocks

    cache = _Blocks(64 << 10, open_files=2)
    for i in range(12):
        path = tmp_path / f"{i}.targets"
        path.write_bytes(path_search._TERMINAL.pack(i, i + 1))
        assert cache.unpack(path, 0, path_search._TERMINAL) == (i, i + 1)
        assert len(cache.cached) == 1
    cache.close()
    assert cache.used == 0
    assert not cache.cached


def test_native_csr_file_handle_cache_evicts_and_closes(tmp_path):
    from multigpubfs.path_lineage import _Blocks

    cache = _Blocks(0, open_files=2)
    files = [tmp_path / f"{index}.targets" for index in range(3)]
    try:
        for index, path in enumerate(files):
            path.write_bytes(path_search._TERMINAL.pack(index, index + 1))
            assert cache.unpack(path, 0, path_search._TERMINAL) == (index, index + 1)
            if not index:
                first_stream = cache.streams[path]
            assert len(cache.streams) <= 2
        assert first_stream.closed
        live_streams = list(cache.streams.values())
        assert cache.peak == 0
    finally:
        cache.close()
    assert all(stream.closed for stream in live_streams)


def test_output_limit_is_reported_when_empty_parquet_cannot_fit(tmp_path):
    graph = GraphDefinition.permutation([[1, 0]], [0, 1])
    result = path_search.solve(
        graph, targets=[(2, 3)], devices=(0,), output=tmp_path, max_output_bytes=1,
        _native_receipt={"status": "UNREACHABLE", "stop_reason": None,
                         "d_min": None, "target_index": None, "generator_ids": []},
    )
    assert result.status == "INCOMPLETE"
    assert result.report["stop_reason"] == "OUTPUT_LIMIT"
    assert result.report["completed_through_length"] is None
    assert result.report["path_count"] == 0
    assert not (tmp_path / "paths.parquet").exists()


@pytest.mark.parametrize("relay_started", [False, True])
def test_native_window_timeout_keeps_nearest_proof_and_separates_coverage(tmp_path, relay_started):
    pytest.importorskip("pyarrow")
    from time import perf_counter

    graph = GraphDefinition.permutation([[1, 0]], [0, 1])
    nearest = dict(status="COMPLETE", stop_reason=None, d_min=1, target_index=0,
                   generator_ids=[0], search_confirmed_through_depth=1)
    receipt = None
    if relay_started:
        receipt = dict(status="INCOMPLETE", stop_reason="TIME_LIMIT",
                       graph_completed_through_depth=0, native_graph_completed_through_depth=3,
                       ranks=[dict(layer_rows=[1, 1, 1, 1], lineage_peak_bytes=256,
                                   lineage_csr_bytes=128)])
    result = path_search.solve(
        graph, targets=[[1, 0]], goal_mode="window", extra_steps=2, output=tmp_path,
        devices=("rank:0", "rank:1"), max_seconds=1, _started=perf_counter() - 2,
        _native_window_data=(nearest, receipt, tmp_path, "TIME_LIMIT"),
    )
    report = result.report
    assert report["status"] == "INCOMPLETE" and report["stop_reason"] == "TIME_LIMIT"
    assert report["d_min"] == 1 and report["max_length"] == 3
    assert report["nearest_witness"] == {"target_index": 0, "generator_ids": [0]}
    assert report["completed_through_length"] is None
    assert list(result.iter_paths()) == []
    if relay_started:
        assert report["window_graph_completed_through_length"] == 3
        assert report["window_lineage_available_through_length"] == 0
