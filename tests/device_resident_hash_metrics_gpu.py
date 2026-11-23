"""Opt-in 4-GPU test of resident hash diagnostic timers and exact layers."""
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from multigpubfs import GraphDefinition, run_graph
from multigpubfs.generic_session import persistent_graph_workers

world = int(os.environ.get("RESIDENT_TEST_WORLD", "4"))
graph = GraphDefinition.permutation(
    [[1,0,2,3], [0,2,1,3], [0,1,3,2]], [0,1,2,3])
oracle = graph.exact_layers(1000)
root = Path(tempfile.mkdtemp(prefix="resident-hash-metrics-",
    dir=os.environ["MGBFS_TEST_TMP_ROOT"]))

def run(path, flag, limit, **extra):
    env = dict(os.environ, MGBFS_RESIDENT_HASH_METRICS=flag,
               MGBFS_RESIDENT_HASH_METRICS_LAYERS=str(limit),
               MGBFS_RESIDENT_HASH_REHASH="1", MGBFS_RESIDENT_HASH_VBUCKETS="4",
               MGBFS_SEGMENT_TEST_BATCH="1")
    return run_graph(graph, path, devices=list(range(world)), autotune=False,
        shards=2, _batch=2, segment_rows=2, memory_backend="segmented24",
        segment_strategy="atomic", execution_engine="device",
        device_scheduler="batch", device_batch_edges=33, device_receive_records=7,
        executable=os.environ["MGBFS_EXECUTABLE"], _native_env=env, **extra)

def check(path, result, flag, limit):
    completed = len(result["layer_sizes"])
    assert result["layer_sizes"] == [len(x) for x in oracle[:completed]]
    states = sum((Counter(tuple(s) for s in json.loads(
        (path/f"rank-{rank}"/"states.json").read_text())["current"])
        for rank in range(world)), Counter())
    assert states == Counter(tuple(s) for s in oracle[completed-1])
    assert all(v == 1 for v in states.values())
    send = remote = 0
    for rank in result["segment_memory_by_rank"]:
        resident = rank["resident"]
        metric = resident.get("hash_metrics")
        assert bool(metric) == bool(int(flag)), (flag, metric)
        if not metric:
            continue
        assert metric["recorded_expansions"] == min(completed,limit),metric
        assert metric["total_attempted_expansions"] == completed,metric
        assert metric["truncated"] == (completed > limit),metric
        for key in ("layer_seconds","generate_partition_seconds",
                    "publish_wait_ready_seconds","receive_dedup_commit_seconds","wait_ack_seconds"):
            assert len(metric[key]) == min(completed,limit), (key,metric)
            assert all(x >= 0 for x in metric[key]), (key,metric)
        for i, wall in enumerate(metric["layer_seconds"]):
            stages = sum(metric[k][i] for k in (
                "generate_partition_seconds","publish_wait_ready_seconds",
                "receive_dedup_commit_seconds","wait_ack_seconds"))
            assert stages <= wall + 1e-6, (i, stages, wall, metric)
        assert metric["local_records"] + metric["remote_records"] == resident["sent_packets"]
        assert metric["remote_payload_bytes"] == 16*metric["remote_records"]
        send += resident["sent_packets"]
        remote += metric["remote_records"]
    if flag == "1" and result["status"] == "COMPLETE":
        assert send == sum(map(len, oracle))*3, (send, oracle)
        assert remote > 0
    return completed

records=[]
with persistent_graph_workers() as pool:
    pids=None
    for flag in ("0", "1"):
        for depth in range(len(oracle)):
            path=root/f"flag-{flag}-depth-{depth}"
            result=run(path,flag,128,_profile_layers=depth)
            assert result["status"] == ("COMPLETE" if depth==len(oracle)-1 else "INCOMPLETE")
            check(path,result,flag,128)
            now=[w.process.pid for w in pool.workers]
            if pids is None:pids=now
            assert now == pids
            records.append((flag,depth,result["status"]))
    path=root/"truncated-telemetry"
    result=run(path,"1",2)
    assert result["status"]=="COMPLETE"
    check(path,result,"1",2)
    assert [w.process.pid for w in pool.workers] == pids
    path=root/"resource-limit"
    result=run(path,"1",128,capacity=1)
    assert result["status"]=="INCOMPLETE"
    check(path,result,"1",128)
    path=root/"healthy-after-limit"
    result=run(path,"1",128)
    assert result["status"]=="COMPLETE"
    check(path,result,"1",128)
    assert [w.process.pid for w in pool.workers] == pids
summary={"world":world,"graph":"S4 adjacent swaps","oracle_layers":len(oracle),
         "recorded_cases":records,"persistent_pids":pids,
         "truncated_diagnostics_do_not_stop_bfs":True,
         "resource_then_healthy":True}
(root/"acceptance.json").write_text(json.dumps(summary,indent=2))
print("PASS metrics",json.dumps(summary),"EVIDENCE",root,flush=True)
