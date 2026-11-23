"""4-GPU layer-multiset oracle for exclusive independent hash bucket owners."""
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from multigpubfs import GraphDefinition, run_graph
from multigpubfs.generic_session import persistent_graph_workers

world = int(os.environ.get("RESIDENT_TEST_WORLD", "4"))
graph = GraphDefinition.permutation(
    [[1, 0, 2, 3], [0, 2, 1, 3], [0, 1, 3, 2]], [0, 1, 2, 3])
oracle = graph.exact_layers(1000)
root = Path(tempfile.mkdtemp(prefix="resident-hash-independent-",
                             dir=os.environ["MGBFS_TEST_TMP_ROOT"]))


def run(path, independent, collision=False, **extra):
    env = dict(os.environ, MGBFS_RESIDENT_HASH_LOAD_PERCENT="50",
        MGBFS_RESIDENT_HASH_REHASH="1", MGBFS_RESIDENT_HASH_VBUCKETS="4",
        MGBFS_RESIDENT_HASH_BUCKET_EXTENTS="1",
        MGBFS_RESIDENT_HASH_INDEPENDENT_BUCKETS=str(independent),
        MGBFS_RESIDENT_HASH_OWNER_STATS="1",
        MGBFS_SEGMENT_TEST_BATCH="1", MGBFS_SEGMENT_COMPACT_POS="1")
    if collision:
        env["MGBFS_SEGMENT_TEST_HASH_BITS"] = "0"
    else:
        env.pop("MGBFS_SEGMENT_TEST_HASH_BITS", None)
    return run_graph(graph, path, devices=list(range(world)), autotune=False,
        shards=2, _batch=2, segment_rows=2, memory_backend="segmented24",
        segment_strategy="atomic", execution_engine="device", device_scheduler="batch",
        device_batch_edges=33, device_receive_records=7,
        executable=os.environ["MGBFS_EXECUTABLE"], _native_env=env, **extra)


def check(path, result, independent):
    completed = len(result["layer_sizes"])
    assert completed > 0 and result["layer_sizes"] == [len(x) for x in oracle[:completed]]
    actual = sum((Counter(tuple(s) for s in json.loads(
        (path / f"rank-{rank}" / "states.json").read_text())["current"])
        for rank in range(world)), Counter())
    assert actual == Counter(tuple(s) for s in oracle[completed - 1]), (path, actual)
    assert all(count == 1 for count in actual.values())
    for m in result["segment_memory_by_rank"]:
        assert m["resident"]["hash_bucket_extents"] is True
        assert m["resident"]["hash_independent_buckets"] is bool(independent)


records = []
with persistent_graph_workers() as pool:
    pids = None
    for independent, collision in ((0, False), (1, False), (1, True), (0, True)):
        for depth in range(len(oracle)):
            path = root / f"mode-{independent}-collision-{int(collision)}-depth-{depth}"
            result = run(path, independent, collision, _profile_layers=depth)
            assert result["status"] == ("COMPLETE" if depth == len(oracle) - 1 else "INCOMPLETE")
            check(path, result, independent)
            now = [worker.process.pid for worker in pool.workers]
            if pids is None:
                pids = now
            assert now == pids
            records.append({"mode": independent, "collision": collision, "depth": depth})
    limited = run(root / "tight", 1, True, capacity=1)
    assert limited["status"] == "INCOMPLETE" and len(limited["layer_sizes"]) == 1
    check(root / "tight", limited, 1)
    healthy = run(root / "healthy", 1)
    assert healthy["status"] == "COMPLETE"
    check(root / "healthy", healthy, 1)
    assert [worker.process.pid for worker in pool.workers] == pids

summary = {"world": world, "mode_sequence": [0, 1, 1, 0],
           "forced_collision": True, "full_multisets_all_layers": len(oracle),
           "persistent_pids": pids, "resource_then_healthy": True,
           "records": records}
(root / "acceptance.json").write_text(json.dumps(summary, indent=2))
print("PASS resident hash independent buckets", json.dumps(summary),
      "EVIDENCE", root, flush=True)
