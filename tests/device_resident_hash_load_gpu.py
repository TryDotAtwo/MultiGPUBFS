"""4-GPU oracle for changing resident packed hash load within persistent workers."""
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from multigpubfs import GraphDefinition, run_graph
from multigpubfs.generic_session import persistent_graph_workers

world = int(os.environ.get("RESIDENT_TEST_WORLD", "4"))
graph = GraphDefinition.permutation([[1,0,2,3], [0,2,1,3], [0,1,3,2]], [0,1,2,3])
oracle = graph.exact_layers(1000)
root = Path(tempfile.mkdtemp(prefix="resident-hash-load-", dir=os.environ["MGBFS_TEST_TMP_ROOT"]))

def run(path, load, **extra):
    env = dict(os.environ, MGBFS_RESIDENT_HASH_LOAD_PERCENT=str(load),
               MGBFS_RESIDENT_HASH_REHASH="1", MGBFS_RESIDENT_HASH_VBUCKETS="4",
               MGBFS_SEGMENT_TEST_BATCH="1", MGBFS_SEGMENT_COMPACT_POS="1")
    return run_graph(graph, path, devices=list(range(world)), autotune=False,
        shards=2, _batch=2, segment_rows=2, memory_backend="segmented24",
        segment_strategy="atomic", execution_engine="device", device_scheduler="batch",
        device_batch_edges=33, device_receive_records=7,
        executable=os.environ["MGBFS_EXECUTABLE"], _native_env=env, **extra)

def check(path, result, load):
    complete = len(result["layer_sizes"])
    assert result["layer_sizes"] == [len(x) for x in oracle[:complete]]
    states = sum((Counter(tuple(s) for s in json.loads(
        (path / f"rank-{rank}" / "states.json").read_text())["current"])
        for rank in range(world)), Counter())
    assert states == Counter(tuple(s) for s in oracle[complete-1])
    assert all(count == 1 for count in states.values())
    for memory in result["segment_memory_by_rank"]:
        assert memory["resident_hash_load_percent"] == load
        rows = memory["rows_per_bank"] * memory["banks"]
        needed = (rows * 100 + load - 1) // load
        assert memory["slots_per_segment"] == 1 << (needed - 1).bit_length()

with persistent_graph_workers() as pool:
    pids = None
    for i, load in enumerate((50,75,50)):
        for depth in range(len(oracle)):
            path = root / f"job-{i}-depth-{depth}"
            result = run(path, load, _profile_layers=depth)
            assert result["status"] == ("COMPLETE" if depth==len(oracle)-1 else "INCOMPLETE")
            check(path, result, load)
            now = [worker.process.pid for worker in pool.workers]
            if pids is None: pids=now
            assert now == pids
    path=root/"tight"
    limited=run(path,75,capacity=1)
    assert limited["status"]=="INCOMPLETE"
    check(path,limited,75)
    path=root/"healthy"
    healthy=run(path,75)
    assert healthy["status"]=="COMPLETE"
    check(path,healthy,75)
    assert [worker.process.pid for worker in pool.workers] == pids
summary={"world":world,"loads":[50,75,50],"full_multisets_all_layers":len(oracle),
         "persistent_pids":pids,"resource_then_healthy":True}
(root/"acceptance.json").write_text(json.dumps(summary,indent=2))
print("PASS resident hash load",json.dumps(summary),"EVIDENCE",root,flush=True)
