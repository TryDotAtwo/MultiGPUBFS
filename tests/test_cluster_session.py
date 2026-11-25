import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from threading import Thread, current_thread

import pytest

from multigpubfs import GraphDefinition, open_session
from multigpubfs.cluster import Cluster, Placement
from multigpubfs.network_control import ControlStore
from multigpubfs.segmented_network import AgreementRelay


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_torchrun_and_slurm_rank_mapping(monkeypatch):
    torchrun = Placement.from_environment(dict(RANK="12", LOCAL_RANK="4", WORLD_SIZE="16",
                                                MASTER_ADDR="node01", MASTER_PORT="29500"))
    assert (torchrun.rank, torchrun.local_rank, torchrun.world) == (12, 4, 16)
    assert (torchrun.coordinator_port, torchrun.native_port, torchrun.path_port, torchrun.packet_port) == (
        29501, 29502, 29503, 29504,
    )
    monkeypatch.setattr("multigpubfs.cluster.subprocess.check_output", lambda *a, **k: "node01\nnode02\n")
    slurm = Placement.from_environment(dict(SLURM_PROCID="1", SLURM_LOCALID="0", SLURM_NTASKS="2",
                                              SLURM_JOB_ID="42", SLURM_JOB_NODELIST="node[01-02]"))
    assert (slurm.rank, slurm.local_rank, slurm.host) == (1, 0, "node01")


def test_cluster_session_does_not_allocate_a_second_local_pool(monkeypatch):
    from multigpubfs import cluster, generic_session

    monkeypatch.setattr(cluster, "current_cluster", lambda: object())
    with open_session(gpus=2):
        assert generic_session._active is None


def test_external_rank_failure_preserves_bounded_native_cause(tmp_path):
    from types import SimpleNamespace

    from multigpubfs.distributed_launch import _native_rank_failure

    log = tmp_path / 'rank-1.log'
    log.write_bytes(b'old context\n' + b'x' * 4096 + b'\nCUDA_JOB_CAUSE\n')
    job = SimpleNamespace(returncode=1, response=None)
    error = str(_native_rank_failure('EXTERNAL_NATIVE_RANK_FAILED', 1, job, log))
    assert error.startswith('EXTERNAL_NATIVE_RANK_FAILED 1 (exit 1)')
    assert error.endswith('CUDA_JOB_CAUSE') and 'old context' not in error

    job.response = {'error': 'RANK_CAPABILITY_MISMATCH'}
    assert str(_native_rank_failure('EXTERNAL_NATIVE_RANK_FAILED', 1, job, log)).endswith(
        'RANK_CAPABILITY_MISMATCH'
    )


@pytest.mark.parametrize("override,expected", [
    ({}, "127.0.0.1:29504"),
    ({"MGBFS_SEGMENT_PACKET_CONTROL_ADDR": "192.0.2.1:41000"}, "192.0.2.1:41000"),
])
def test_cluster_segmented_window_uses_its_assigned_packet_port(tmp_path, monkeypatch, override, expected):
    from multigpubfs import launch, native_distribution

    monkeypatch.delenv("MGBFS_SEGMENT_WINDOW_V1", raising=False)
    monkeypatch.delenv("MGBFS_GENERIC_PATH_COMPACT", raising=False)
    monkeypatch.setenv("MGBFS_NATIVE_DISTRIBUTED_NEAREST", "1")
    monkeypatch.setattr(native_distribution, "native_runtime", lambda _: ("mgbfs", override))
    monkeypatch.setattr(launch, "_require_memory_backend", lambda *_: {
        "graph_path_segmented_nearest_v1": True,
        "graph_path_segmented_window_v1": True,
        "segmented_network_agreement_v1": True,
    })
    monkeypatch.setattr(Cluster, "_dispatch", lambda self, job, output: job)
    placement = Placement.from_environment({
        "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2",
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500",
    })
    cluster = Cluster(placement, None, tmp_path)
    graph = GraphDefinition.permutation([[1, 0]], [0, 1])
    job = cluster.solve(open_session(gpus=2), graph, tmp_path / "routes", {
        "goal_mode": "window", "targets": [[1, 0]],
    })
    assert job["compact_path_packet_address"] == expected


def test_external_planner_uses_live_communicator_reserve():
    from multigpubfs.distributed_launch import _match_resident_reserve

    warm = {"additional_reserve_bytes": 1 << 28}
    cold = {"additional_reserve_bytes": 2 << 30}
    assert _match_resident_reserve({}, [warm, warm])["MGBFS_GENERIC_VRAM_RESERVE_BYTES"] == str(1 << 28)
    with pytest.raises(RuntimeError, match="EXTERNAL_RESIDENT_RESERVE_CONSENSUS"):
        _match_resident_reserve({}, [warm, cold])


def test_coordinator_dispatches_a_sequence_of_jobs(tmp_path, monkeypatch):
    port = _port()
    stores = [ControlStore("127.0.0.1", port, rank, "test-session", 5) for rank in range(2)]
    peers = [Cluster(Placement(rank, 0, 2, "127.0.0.1", port, port+1, port+2, "test-session"),
                     stores[rank], tmp_path) for rank in range(2)]
    worker_jobs = []

    def run(self, job, output):
        if self.placement.rank != 0:
            worker_jobs.append((job["mode"], self.sequence, Path(output)))
        return self.sequence

    monkeypatch.setattr(Cluster, "_run", run)
    thread = Thread(target=peers[1].serve)
    thread.start()
    graph = GraphDefinition.permutation([[1, 0]], [0, 1])
    session = open_session(gpus=2)
    try:
        assert peers[0].traverse(session, graph, tmp_path / "walk", "balanced", None) == 0
        with pytest.raises(ValueError, match="PATH_TARGET_SHAPE"):
            peers[0].solve(session, graph, tmp_path / "bad-path", {"targets": [[1]]})
        with pytest.raises(ValueError, match="PATH_DIRECTION"):
            peers[0].solve(session, graph, tmp_path / "bad-path", {"targets": [[1, 0]], "direction": "unknown"})
        with pytest.raises(ValueError, match="PATH_MAX_LINEAGE_BYTES"):
            peers[0].solve(session, graph, tmp_path / "bad-path", {"targets": [[1, 0]], "max_lineage_bytes": -1})
        occupied = tmp_path / "occupied"
        occupied.mkdir()
        (occupied / "keep").touch()
        with pytest.raises(FileExistsError, match="PATH_OUTPUT_NOT_EMPTY"):
            peers[0].solve(session, graph, occupied, {"targets": [[1, 0]]})
        targets = (goal for goal in [[1, 0], [1, 0]])
        assert peers[0].solve(session, graph, tmp_path / "path", {"targets": targets}) == 1
        peers[0].stop()
        thread.join(5)
        assert not thread.is_alive()
        assert [(mode, sequence) for mode, sequence, _ in worker_jobs] == [("traverse", 0), ("solve", 1)]
        assert worker_jobs[0][2] != worker_jobs[1][2]
    finally:
        for store in stores:
            store.close()


def test_caught_dispatch_error_closes_generation(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from multigpubfs import distributed_launch, launch, native_distribution

    port = _port()
    peers = [Cluster(Placement(rank, rank, 2, "127.0.0.1", port, port + 1, port + 2, "failure"),
                     ControlStore("127.0.0.1", port, rank, "failure", 2), tmp_path) for rank in range(2)]

    coordinator = current_thread()
    monkeypatch.setattr(native_distribution, "native_runtime", lambda _: ("mgbfs", {}))
    monkeypatch.setattr(launch, "_require_memory_backend", lambda *_: {
        "segmented_network_agreement_v1": True, "permutation_bitmap_v1": True,
    })
    monkeypatch.setattr(distributed_launch, "_native_artifact_hashes", lambda *_: {
        "native": "bin", "cuda": "sm80" if current_thread() is coordinator else "sm86",
    })

    def unexpected_inventory(*_):
        pytest.fail("launched a worker before checking the loaded CUDA library")

    monkeypatch.setattr(Cluster, "inventory", unexpected_inventory)

    def serve():
        try:
            peers[1].serve()
        except RuntimeError as error:
            peers[1]._fail(error)
            raise

    try:
        graph = GraphDefinition.permutation([[1, 0]], [0, 1])
        session = open_session(gpus=2)
        with ThreadPoolExecutor(1) as pool:
            worker = pool.submit(serve)
            with pytest.raises(RuntimeError, match="EXTERNAL_GRAPH_NATIVE_OR_PARAMETERS_MISMATCH"):
                peers[0].traverse(session, graph, tmp_path / "walk", "maxmemory", None)
            with pytest.raises(RuntimeError, match="DISTRIBUTED_PEER_ERROR"):
                worker.result(timeout=2)
            with pytest.raises(RuntimeError, match="CLUSTER_GENERATION_CLOSED"):
                peers[0].traverse(session, graph, tmp_path / "retry", "balanced", None)
    finally:
        peers[0].store.close()


def test_cluster_memory_policy_respects_reserve_and_features_on_every_rank(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    from multigpubfs import distributed_launch, launch, native_distribution

    port = _port()
    stores = [ControlStore("127.0.0.1", port, rank, "bitmap-policy", 5) for rank in range(2)]
    peers = [Cluster(Placement(rank, rank, 2, "127.0.0.1", port, port+1, port+2, "bitmap-policy", port+3),
                     stores[rank], tmp_path) for rank in range(2)]
    capabilities = {"segmented_network_agreement_v1": True, "permutation_bitmap_v1": True,
                    "bitmap_pack4_experimental_v1": True, "bitmap_pack4_adaptive_v2": True,
                    "permutation_bitmap_position_orbits_v1": True, "segmented_exact_exchange_v1": True,
                    "segmented_packet_transport_v1": True, "segmented_hash_series_v1": True,
                    "segmented_warp_hash_safe_v1": True, "segmented_strategies_v1": {"batch": True}}
    weak_adaptive_abi = False
    weak_batch_abi = False
    explicit_transport = None
    packet_keys = ("MGBFS_SEGMENT_PACKET_RING_SLOTS", "MGBFS_SEGMENT_PACKET_RX_SLOTS")

    def features(*_):
        if current_thread() is worker:
            return dict(capabilities, bitmap_pack4_adaptive_v2=not weak_adaptive_abi,
                        segmented_strategies_v1={"batch": not weak_batch_abi})
        return capabilities

    def runtime(_):
        if explicit_transport is None:
            return "native", {}
        return "native", dict.fromkeys(packet_keys, "1" if current_thread() is worker else "3")

    monkeypatch.setattr(native_distribution, "native_runtime", runtime)
    monkeypatch.setattr(launch, "_require_memory_backend", features)
    monkeypatch.setattr(distributed_launch, "_native_artifact_hashes", lambda *_: {"native": "bin", "cuda": "lib"})

    def inventory(self, native, env, device):
        assert env["MGBFS_EXPERIMENTAL_SEGMENTED24"] == "1"
        reserve = 2 << 30 if self.sequence == 1 and self.placement.rank == 1 else 0
        return {"free_bytes": (2 if self.sequence < 3 else 5) << 30,
                "additional_reserve_bytes": reserve}

    calls = []
    transport_calls = []

    def run_graph(graph, output, *, device, memory_backend, memory_preset, _native_env, **options):
        assert _native_env["MGBFS_SEGMENT_EXACT_EXCHANGE"] == "1"
        if explicit_transport is not None:
            address = _native_env.get("MGBFS_SEGMENT_PACKET_CONTROL_ADDR")
            if explicit_transport == "compact":
                assert address == f"127.0.0.1:{port + 3}"
                assert all(_native_env.get(key) == "3" for key in packet_keys)
            else:
                assert address is None
                assert all(key not in _native_env for key in packet_keys)
            transport_calls.append((device, explicit_transport))
            return {}
        assert ("MGBFS_SEGMENT_PACKET_CONTROL_ADDR" in _native_env) == (memory_preset == "maxmemory")
        calls.append((device, memory_backend, options.get("bitmap_pack4"), options["segment_strategy"]))
        return {}

    monkeypatch.setattr(Cluster, "inventory", inventory)
    monkeypatch.setattr(launch, "run_graph", run_graph)
    worker = Thread(target=peers[1].serve, daemon=True)
    worker.start()
    n = 13
    generators = [list(range(i)) + [i + 1, i] + list(range(i + 2, n)) for i in range(n - 1)]
    graph = GraphDefinition.permutation(generators, list(range(n)))
    try:
        for name, preset, expected in (
            ("fits", "maxmemory", "permutation_bitmap"),
            ("reserved", "maxmemory", "segmented24"),
            ("oversized-frontier", "balanced", "permutation_bitmap"),
        ):
            result = peers[0].traverse(open_session(gpus=2), graph, tmp_path / name, preset, None)
            assert result["cluster_memory_backend"] == expected
        n = 60
        wide = GraphDefinition.permutation(
            [[p ^ 1 if p // 2 == i else p for p in range(n)] for i in range(n // 2)],
            [-7 if p & 1 else 511 for p in range(n)],
        )
        for weak_adaptive_abi, expected in ((False, "permutation_bitmap"), (True, "segmented24")):
            weak_batch_abi = weak_adaptive_abi
            result = peers[0].traverse(open_session(gpus=2), wide, tmp_path / "wide", "maxmemory", None)
            assert result["cluster_memory_backend"] == expected
        weak_batch_abi = False
        unranked = GraphDefinition.permutation([list(range(65))], list(range(65)))
        result = peers[0].traverse(open_session(gpus=2), unranked, tmp_path / "wide-balanced", "balanced", None)
        assert result["cluster_memory_backend"] == "segmented24"
        for explicit_transport in ("compact", "exact"):
            options = {"transport": explicit_transport, "memory_backend": "segmented24"}
            if explicit_transport == "compact":
                options["packet_slots"] = 3
            peers[0].traverse(open_session(gpus=2), graph, tmp_path / explicit_transport,
                              "balanced", None, native_options=options)
        assert sorted(transport_calls) == [(0, "compact"), (0, "exact"), (1, "compact"), (1, "exact")]
        peers[0].stop()
        worker.join(5)
        assert not worker.is_alive()
        assert sorted(calls) == [(rank, backend, pack4, strategy) for rank in range(2)
                                 for backend, pack4, strategy in (
                                     ("permutation_bitmap", True, "atomic"),
                                     ("permutation_bitmap", True, "atomic"),
                                     ("permutation_bitmap", True, "atomic"),
                                     ("segmented24", None, "batch"),
                                     ("segmented24", None, "batch"),
                                     ("segmented24", None, "warp"),
                                 )]
    finally:
        for store in stores:
            store.close()


def test_segmented_agreement_relays_rank_local_commits(tmp_path):
    port = _port()
    stores = [ControlStore("127.0.0.1", port, rank, "test-relay", 5) for rank in range(2)]
    relays = [AgreementRelay(stores[rank], "production", tmp_path / str(rank) / "nccl-id", rank, 2) for rank in range(2)]
    records = [{"schema": 1, "phase": "comm-mode", "rank": rank, "world": 2,
                "descriptor": {"mode": "cold"}, "error": None} for rank in range(2)]
    try:
        for rank, relay in enumerate(relays):
            relay.root.mkdir(parents=True)
            (relay.root / f"comm-mode-rank-{rank}.json").write_text(json.dumps(records[rank]))
        relays[0].step()
        assert not (relays[0].root / "comm-mode-rank-1.json").exists()
        relays[1].step()
        relays[0].step()
        relays[1].step()
        for relay in relays:
            assert not relay.pending
            assert [json.loads((relay.root / f"comm-mode-rank-{rank}.json").read_text()) for rank in range(2)] == records
    finally:
        for store in stores:
            store.close()


@pytest.mark.parametrize("direction", ["forward", "bidirectional"])
def test_external_native_nearest_publishes_only_on_coordinator(tmp_path, monkeypatch, direction):
    from multigpubfs import launch, native_distribution

    pytest.importorskip("pyarrow")
    monkeypatch.setattr(native_distribution, "native_runtime", lambda value: (value, {}))
    monkeypatch.setattr(launch, "_require_memory_backend", lambda *_: {"graph_path_nearest_distributed_v1": True, "graph_path_nearest_bidirectional_v1": True})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    launched = []
    bidirectional = direction == "bidirectional"

    def run_graph(graph, output, **config):
        assert graph.start == [1]
        assert config["executable"] == "mgbfs" and config["_path_targets"] == ([(4,)] if bidirectional else [(6,), (4,)])
        assert config["_path_bidirectional"] is bidirectional
        assert config["max_seconds"] == 11 and config["autotune"] is False
        launched.append((config["device"], output))
        return dict(status="COMPLETE", stop_reason=None, d_min=2, target_index=0 if bidirectional else 1,
                    generator_ids=[0, 0], search_confirmed_through_depth=2)

    monkeypatch.setattr(launch, "run_graph", run_graph)
    graph = GraphDefinition.matrix(1, 1, [([2], 5)], [1])
    job = dict(mode="solve", executable="mgbfs", native_forward_nearest=True, graph=graph.to_dict(),
               options=dict(targets=[[6], [4], [4]], max_seconds=10.5, direction=direction))
    class SharedStore(dict):
        put = dict.__setitem__
        get = dict.__getitem__

    store = SharedStore()
    nodes = [Cluster(Placement(rank, rank, 2, "127.0.0.1", 33001, 33002, 33003, "test"), store, tmp_path)
             for rank in range(2)]
    result = nodes[0]._run(job, tmp_path / "coordinator")
    peer = tmp_path / "peer"
    assert nodes[1]._run(job, peer) is None
    assert not peer.exists()
    assert result.report["backend"] == "native_cuda_nccl"
    assert result.report["devices"] == ["rank:0", "rank:1"]
    assert result.report["search_confirmed_through_depth"] == 2
    assert result.report["targets"] == [[6], [4]]
    assert list(result.iter_paths()) == [{"target_index": 1, "length": 2, "generator_ids": [0, 0]}]
    assert [device for device, _ in launched] == [0, 1]
    assert launched[0][1] != launched[1][1]

    if bidirectional:
        empty = dict(job, options=dict(job["options"], targets=[[6], [-1]]))
        unreachable = nodes[0]._run(empty, tmp_path / "unreachable")
        assert unreachable.status == "UNREACHABLE" and not list(unreachable.iter_paths())
        assert len(launched) == 2

    def capacity_stop(*args, **kwargs):
        return dict(status="INCOMPLETE", stop_reason="CUDA_MEMORY", d_min=None,
                    target_index=None, generator_ids=[], search_confirmed_through_depth=1)

    monkeypatch.setattr(launch, "run_graph", capacity_stop)
    stopped = nodes[0]._run(job, tmp_path / "stopped")
    assert stopped.status == "INCOMPLETE"
    assert stopped.report["search_confirmed_through_depth"] == 1
    assert stopped.report["completed_through_length"] is None
    assert list(stopped.iter_paths()) == []


@pytest.mark.skipif(not os.environ.get("MGBFS_TEST_NATIVE_DISTRIBUTED_PATH_BINARY"), reason="requires two CUDA GPUs")
@pytest.mark.parametrize("direction", ["forward", "bidirectional"])
def test_torchrun_native_nearest(tmp_path, direction):
    pytest.importorskip("torch.distributed.run")
    script = tmp_path / "nearest.py"
    script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "from multigpubfs import GraphDefinition, open_session\n"
        "graph = GraphDefinition.permutation([[1, 2, 3, 0]], [0, 1, 2, 3])\n"
        "with open_session(gpus=2, executable=sys.argv[1], tune='off') as session:\n"
        "    results = [session.solve(graph, targets=[target], direction=sys.argv[3], output=Path(sys.argv[2]) / str(i))\n"
        "               for i, target in enumerate(([2, 3, 0, 1], [7, 7, 7, 7], graph.start))]\n"
        "assert [result.status for result in results] == ['COMPLETE', 'UNREACHABLE', 'COMPLETE']\n"
        "assert all(result.report['backend'] == 'native_cuda_nccl' for result in results)\n"
        "assert list(results[0].iter_paths())[0]['generator_ids'] == [0, 0]\n"
        "assert list(results[2].iter_paths())[0]['generator_ids'] == []\n"
        "cold, warm, again = [result.report['native_workspace']['ranks'] for result in results]\n"
        "assert len(cold) == len(warm) == len(again) == 2\n"
        "assert all(a['worker_pid'] == b['worker_pid'] == c['worker_pid']\n"
        "           and a['owned_raw_complete'] and b['owned_raw_complete'] and c['owned_raw_complete']\n"
        "           and b['buffer_hits'] > 0 and c['buffer_hits'] > 0\n"
        "           and b['comm_hits'] > 0 and c['comm_hits'] > 0\n"
        "           for a, b, c in zip(cold, warm, again))\n",
        encoding="utf-8",
    )
    run = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                          "--nproc-per-node=2", "-m", "multigpubfs", "cluster", str(script),
                          os.environ["MGBFS_TEST_NATIVE_DISTRIBUTED_PATH_BINARY"], str(tmp_path / "results"), direction],
                         env=dict(os.environ, MGBFS_NATIVE_DISTRIBUTED_NEAREST="1"),
                         capture_output=True, text=True, timeout=180)
    assert run.returncode == 0, run.stdout + run.stderr

@pytest.mark.parametrize("direction", ["forward", "bidirectional"])
def test_owner_csr_output(tmp_path, monkeypatch, direction):
    from concurrent.futures import ThreadPoolExecutor
    from itertools import product
    from struct import pack

    pytest.importorskip("pyarrow")
    from multigpubfs import launch, native_distribution
    from multigpubfs.path_search import _agree_window_receipts

    if os.name == "nt" and tmp_path.drive.endswith(":"):
        # The mocked native worker creates nested CSR directories under pytest's temp root.
        tmp_path = Path(f"\\\\?\\{tmp_path}")

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(native_distribution, "native_runtime", lambda _: ("native", {}))
    monkeypatch.setattr(launch, "_require_memory_backend",
                        lambda *args: {f"graph_path_window_{direction}_v1": True})

    def fake_launch(graph, output, *, device, _path_window, **options):
        if _path_window is None:
            return dict(status="COMPLETE", stop_reason=None, d_min=1,
                        target_index=0, generator_ids=[0])
        assert _path_window["max_length"] == 3
        root = output / f"rank-{device}"
        root.mkdir(parents=True)

        def layers(side, start, depth_limit):
            sizes = [(start + depth) % 2 == device for depth in range(depth_limit + 1)]
            for depth, owned in enumerate(sizes[1:], 1):
                folder = root / side / f"{depth}.csr"
                folder.mkdir(parents=True)
                (folder / "offsets").write_bytes(pack("<QQ", 0, 2) if owned else pack("<Q", 0))
                (folder / "edges").write_bytes(
                    pack("<QIIQII", 0, 1 - device, 0, 0, 1 - device, 1) if owned else b"")
            return list(map(int, sizes))

        if direction == "forward":
            layers("", 0, 3)
            for depth in range(4):
                (root / f"{depth}.targets").write_bytes(pack("<QQ", 0, 0) if depth % 2 == device == 1 else b"")
            receipts = [dict(layer_rows=[int(d % 2 == rank) for d in range(4)]) for rank in range(2)]
        else:
            layers("forward", 0, 1)
            layers("reverse", 1, 2)
            (root / "reverse" / "0.targets").write_bytes(pack("<QQ", 0, 0) if device == 1 else b"")
            for length in range(1, 4):
                (root / f"{length}.matches").write_bytes(
                    pack("<QQ", 0, 0) if length % 2 and length // 2 == device else b"")
            receipts = [dict(side_layer_rows=[[int(d % 2 == rank) for d in range(2)],
                                              [int((1 + d) % 2 == rank) for d in range(3)]],
                             first_cut_length=1, last_cut_length=3) for rank in range(2)]
        for receipt in receipts:
            receipt.update(status="COMPLETE", stop_reason=None, graph_completed_through_depth=3,
                           lineage_peak_bytes=256, lineage_csr_bytes=128)
        return _agree_window_receipts(receipts)

    monkeypatch.setattr(launch, "run_graph", fake_launch)
    port = _port()
    peers = []
    for rank in range(2):
        scratch = tmp_path / f"scratch-{rank}"
        scratch.mkdir()
        placement = Placement(rank, rank, 2, "127.0.0.1", port, port + 1, port + 2, "test-owner-files")
        store = ControlStore(placement.host, port, rank, placement.session_id, 5, token="owner-files")
        peers.append(Cluster(placement, store, scratch))
    graph = GraphDefinition.permutation([[1, 0], [1, 0]], [0, 1])
    session = open_session(gpus=2, executable="native")
    options = dict(targets=[[1, 0]], direction=direction, goal_mode="window", extra_steps=2,
                   lineage_cache_bytes=32, max_seconds=0)
    try:
        with ThreadPoolExecutor(1) as pool:
            worker = pool.submit(peers[1].serve)
            try:
                overflow = peers[0].solve(session, graph, tmp_path / "overflow",
                                          dict(options, extra_steps=0xffff_ffff))
                assert overflow.status == "INCOMPLETE" and overflow.report["stop_reason"] == "LENGTH_RANGE"
                assert overflow.report["d_min"] == 1 and overflow.report["max_length"] == 1 << 32
                assert overflow.report["nearest_witness"] == {"target_index": 0, "generator_ids": [0]}
                assert overflow.report["completed_through_length"] is None
                assert list(overflow.iter_paths()) == []

                output = tmp_path / "full"
                full = peers[0].solve(session, graph, output, dict(options, lineage_dir=output))
                assert full.report["completed_through_length"] == 3
                assert full.report["lineage_read_mode"] == "rank_owned"
                assert full.report["lineage_remote_read_bytes"] > 0
                assert full.report["lineage_cache_peak_bytes"] <= 32
                expected = {(0, labels) for length in (1, 3) for labels in product(range(2), repeat=length)}
                assert {(p["target_index"], tuple(p["generator_ids"])) for p in full.iter_paths()} == expected

                output = tmp_path / "partial"
                partial = peers[0].solve(session, graph, output, dict(options, max_paths=4, lineage_dir=output / "csr"))
                assert partial.report["stop_reason"] == "PATH_LIMIT"
                assert partial.report["completed_through_length"] == 2
                found = {(p["target_index"], tuple(p["generator_ids"])) for p in partial.iter_paths()}
                assert len(found) == 4 and found <= expected
            finally:
                if peers[0].failure is None:
                    peers[0].stop()
            worker.result(timeout=5)
        assert peers[0].sequence == peers[1].sequence == 4
    finally:
        peers[0].store.close()


def test_window_owner_reads_slices_without_copying_and_drains_on_deadline(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from time import perf_counter, sleep

    from multigpubfs.path_window_files import SearchStopped, WindowOwnerFiles, serve_window_files

    coordinator_root = tmp_path / "coordinator"
    owner_root = tmp_path / "owner"
    source = owner_root / "rank-1" / "1.csr"
    source.mkdir(parents=True)
    with (source / "edges").open("wb") as stream:
        stream.seek(2 << 20)
        stream.write(b"last")
    target = coordinator_root / "rank-1" / "1.csr" / "edges"
    receipt = {"ranks": [{"layer_rows": [0, 0]}] * 2}
    port = _port()
    coordinator, owner = [ControlStore("127.0.0.1", port, rank, "test-owner-slice", 3) for rank in range(2)]
    handler = coordinator.server.RequestHandlerClass
    block_request = handler.block_request
    truncated = []

    def read_with_one_early_eof(peer):
        if peer.path == "/get-block" and not truncated:
            truncated.append(True)
            peer.send_response(200)
            peer.send_header("Content-Length", "40")
            peer.end_headers()
            peer.wfile.write(b"short")
            return
        block_request(peer)

    monkeypatch.setattr(handler, "block_request", read_with_one_early_eof)
    try:
        with ThreadPoolExecutor(2) as pool:
            worker = pool.submit(serve_window_files, owner, "slices/", owner_root, 1, receipt)
            with WindowOwnerFiles(coordinator, "slices/", coordinator_root, 2, None) as files:
                assert files.size(target) == (2 << 20) + 4
                assert files.read(target, 2 << 20, 64 << 10) == b"last"
                assert files.bytes_read == 4
            worker.result(timeout=3)
            assert truncated
            assert not coordinator_root.exists()

            def delayed_owner():
                assert owner.await_value("limit/1/request/0", stop_key="limit/done") is not None
                sleep(.3)
                owner.put_block("limit/1/reply", 0, b"late")
                assert owner.await_value("limit/1/request/1", stop_key="limit/done") is None

            late = pool.submit(delayed_owner)
            with WindowOwnerFiles(coordinator, "limit/", coordinator_root, 2, perf_counter() + .2) as files:
                with pytest.raises(SearchStopped, match="TIME_LIMIT"):
                    files.read(target, 0, 8)
            late.result(timeout=3)
            coordinator.release_prefix("limit/")
    finally:
        coordinator.close()


def test_native_window_rejects_invalid_lineage_directory_before_dispatch(tmp_path, monkeypatch):
    from multigpubfs import launch, native_distribution
    from multigpubfs.cluster import Cluster

    monkeypatch.delenv("MGBFS_NATIVE_DISTRIBUTED_NEAREST", raising=False)
    monkeypatch.setattr(native_distribution, "native_runtime", lambda _: ("native", {}))
    monkeypatch.setattr(launch, "_require_memory_backend",
                        lambda *args: {"graph_path_window_forward_v1": True})
    invalid = tmp_path / "not-a-directory"
    invalid.write_bytes(b"existing file")
    placement = Placement(0, 0, 2, "127.0.0.1", 29501, 29502, 29503, "test", 29504)
    coordinator = Cluster(placement, None, tmp_path)
    session = open_session(gpus=2)
    graph = GraphDefinition.permutation([[1, 0]], [0, 1])
    options = dict(goal_mode="window", direction="forward", preset="balanced",
                   lineage_dir=invalid, targets=[[1, 0]])
    with pytest.raises(FileExistsError):
        Cluster.solve(coordinator, session, graph, tmp_path / "output", options)
