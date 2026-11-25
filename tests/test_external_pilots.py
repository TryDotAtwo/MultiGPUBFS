import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("stall", [True, False], ids=["reap-expired-pilot", "use-faster-batch"])
def test_cluster_reaps_pilots_and_admits_the_selected_batch(tmp_path, stall):
    native = tmp_path / "native_fixture.py"
    native.write_text(r'''
import hashlib, json, os, sys, time
from pathlib import Path

_, _, graph, config_path, rank, bootstrap, output, seconds = sys.argv
config = json.loads(Path(config_path).read_text(encoding="utf-8"))
rank = int(rank)
bootstrap, output = Path(bootstrap), Path(output)
trial = "profile_max_layers" in config
batch = config["plan"]["batch"]
assert os.environ["MGBFS_GENERIC_VRAM_RESERVE_BYTES"] == ("2097152" if trial else "1048576")
assert "online_size_tuning" not in config
root = Path(str(bootstrap) + ".segmented-host")
root.mkdir()
def agree(name):
    phase = root / f"{name}-rank-{rank}.json"
    phase.write_text(json.dumps({"schema": 1, "phase": name, "rank": rank, "world": 2,
                                 "error": None, "descriptor": {"batch": batch}}), encoding="utf-8")
    Path(os.environ["MGBFS_JOB_PHASE_PATH"]).write_text(json.dumps({"schema": 1, "pid": os.getpid(),
        "phase": f"segmented-host-{name}", "entered": {}}), encoding="utf-8")
    while not (root / f"{name}-rank-{1-rank}.json").is_file():
        time.sleep(0.005)
agree("arguments")
if rank == 0:
    pending = bootstrap.with_suffix(".tmp")
    pending.write_bytes(bytes(128))
    pending.replace(bootstrap)
else:
    while not bootstrap.is_file():
        time.sleep(0.005)
agree("entry")
if trial and rank == 1 and batch < 16 and os.environ["PILOT_FIXTURE_STALL"] == "1":
    time.sleep(20)
output.mkdir()
states = json.dumps({"current": [[1, 0]] if rank == 0 else [], "previous_small": []}).encode()
(output / "states.json").write_bytes(states)
sizes = [1, 2, 9000, 12000, 16000] if trial else [1, 1]
layer_time = 0.1 if batch == 4 else 0.2
report = {"status": "INCOMPLETE" if trial else "COMPLETE", "reason": "PROFILE_LAYER_LIMIT" if trial else "FRONTIER_EXHAUSTED",
    "rank": rank, "device": 0, "world": 2, "plan": config["rank_plans"][rank], "graph_digest": config["graph_digest"],
    "states_sha256": hashlib.sha256(states).hexdigest(), "layer_sizes": sizes, "layer_seconds": [layer_time] * (len(sizes)-1),
    "layer_committed_seconds": [layer_time*i for i in range(len(sizes))], "bfs_seconds": 0.8, "setup_seconds": 0.1,
    "backend": "EXPERIMENTAL_SEGMENTED24", "segment_strategy": "atomic",
    "segment_memory": {"admitted_batch": batch, "admitted_owner_cuts": config["owner_cuts"]}}
(output / "report.json").write_text(json.dumps(report), encoding="utf-8")
''', encoding="utf-8")
    runner = tmp_path / "rank_fixture.py"
    runner.write_text(r'''
import json, os, subprocess, sys
from pathlib import Path
from unittest.mock import patch
from multigpubfs import GraphDefinition, external_pilots, launch
from multigpubfs.cluster import Cluster, Placement, _active

rank = int(os.environ["RANK"])
port = int(os.environ["MGBFS_CONTROL_PORT"])
native, output = sys.argv[1:3]
worker = Cluster(Placement(rank, 0, 2, "127.0.0.1", port, port, port, "pilot-fixture"), None, Path(output).parent)
graph = GraphDefinition.permutation([[1,0]], [0,1])
inventory = {"device": 0, "free_bytes": 1<<30, "total_bytes": 2<<30, "additional_reserve_bytes": 1<<20,
    "physical_free_bytes": 1<<29, "cold_additional_reserve_bytes": 1<<21}
worker.inventory = lambda *_: inventory
children = []
popen = subprocess.Popen

def spawn(args, **kwargs):
    child = popen([sys.executable, native, *args[1:]], **kwargs)
    children.append(child)
    return child

def production(args, env, log):
    assert len(children) >= 2 and all(child.poll() is not None for child in children)
    config = json.loads(Path(args[3]).read_text())
    assert config["plan"]["batch"] == (16 if os.environ["PILOT_FIXTURE_STALL"] == "1" else 4)
    return popen([sys.executable, native, *args[1:]], env=env, stdout=log, stderr=subprocess.STDOUT)

def query(args, env, **kwargs):
    assert args[1] == "graph-plan"
    rows = json.loads(Path(args[3]).read_text())
    reserve = int(env["MGBFS_GENERIC_VRAM_RESERVE_BYTES"])
    assert all(row["free_bytes"] == (1<<29 if reserve == 1<<21 else 1<<30) for row in rows)
    plan = {"world": 2, "shards": int(args[4]), "batch": int(args[6]) if len(args) > 6 else 16,
            "history_layers": 3, "history_algorithm": "HASH", "state_bytes": 1, "queue_capacity": 128}
    result = {"graph_digest": graph.digest(), "devices": [0,0], "inventory": rows, "plan": plan, "rank_plans": [plan, plan],
              "owner_cuts": [0,1<<31,1<<32], "additional_reserve_bytes": reserve, "size_profile_capability": 1}
    return subprocess.CompletedProcess(args, 0, stdout=json.dumps(result), stderr="")

worker.launch_rank = production
_active.set(worker)
with patch.object(external_pilots.subprocess, "Popen", spawn), \
     patch("multigpubfs.distributed_launch.native_query", query), \
     patch("multigpubfs.distributed_launch._native_artifact_hashes", return_value={"native": "bin", "cuda": "so"}), \
     patch("multigpubfs.host_memory.admit", return_value={}), \
     patch("multigpubfs.shard_policy.choose_segmented_shards", return_value={"target_shards": 1}), \
     patch("multigpubfs.native_distribution.runtime_identity", return_value={}), \
     patch.object(launch, "native_runtime", return_value=(sys.executable, {})), \
     patch.object(launch, "_require_memory_backend", return_value={"segmented_network_agreement_v1": True}):
    report = launch.run_graph(graph, output, device=0, executable=sys.executable, memory_backend="segmented24",
                              _native_env=dict(os.environ), autotune=True)
assert report["layer_sizes"] == [1,1] and report["status"] == "COMPLETE"
profile = report["autotune"]
assert profile["pilots"][0]["completed_depth"] == 4
if os.environ["PILOT_FIXTURE_STALL"] == "1":
    assert profile["pilots"][1]["status"] == "UNAVAILABLE"
    assert profile["selected_batch"] == 16 and profile["measured"] is False
else:
    assert profile["common_depth"] == 4 and profile["selected_batch"] == 4 and profile["measured"] is True
''', encoding="utf-8")
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    jobs = []
    try:
        for rank in range(2):
            env = dict(os.environ, RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK="0", MASTER_ADDR="127.0.0.1",
                       MASTER_PORT=str(port - 1), MGBFS_CONTROL_PORT=str(port), MGBFS_RUN_ID="pilot-fixture",
                       PILOT_FIXTURE_STALL=str(int(stall)),
                       PYTHONPATH=os.pathsep.join((str(Path(__file__).resolve().parents[1]), os.environ.get("PYTHONPATH", ""))))
            jobs.append(subprocess.Popen([sys.executable, str(runner), str(native), str(tmp_path / f"rank-{rank}")],
                                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
        for job in jobs:
            output, _ = job.communicate(timeout=15)
            assert job.returncode == 0, output.decode(errors="replace")
        reports = [json.loads((tmp_path / f"rank-{rank}" / "report.json").read_text()) for rank in range(2)]
        assert reports[0]["autotune"] == reports[1]["autotune"]
    finally:
        for job in jobs:
            if job.poll() is None:
                job.kill()
            job.communicate(timeout=5)
