"""Physical multi-rank BFS oracle for opt-in device-side hash rebuild.

Run with MGBFS_RESIDENT_HASH_REHASH=1, then repeat with
MGBFS_SEGMENT_TEST_HASH_BITS=0 and MGBFS_SEGMENT_TEST_BATCH=1. Each full
canonical state multiset is compared independently on every completed layer.
"""
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from multigpubfs import GraphDefinition, run_graph
from multigpubfs.generic_session import persistent_graph_workers

assert os.environ.get("MGBFS_RESIDENT_HASH_REHASH") == "1"
world = int(os.environ.get("RESIDENT_TEST_WORLD", "4"))
n = 6
permutations = []
for i in range(n - 1):
    p = list(range(n))
    p[i], p[i + 1] = p[i + 1], p[i]
    permutations.append(p)
graph = GraphDefinition.permutation(permutations, list(range(n)))
oracle = graph.exact_layers(1000)
root = Path(tempfile.mkdtemp(prefix="resident-hash-rehash-",
                            dir=os.environ["MGBFS_TEST_TMP_ROOT"]))
workers = None
peak_rebuilds = 0
layer_rows = []

def execute(path, **extra):
    return run_graph(
        graph, path, devices=list(range(world)), autotune=False, shards=2,
        _batch=2, segment_rows=2, memory_backend="segmented24",
        segment_strategy="atomic", execution_engine="device",
        device_scheduler="batch", device_batch_edges=33,
        device_receive_records=7, executable=os.environ["MGBFS_EXECUTABLE"],
        **extra,
    )

def current(path):
    return [Counter(tuple(state) for state in json.loads(
        (path / f"rank-{rank}" / "states.json").read_text())["current"])
        for rank in range(world)]

def verify(path, result, depth):
    global peak_rebuilds
    assert result["execution_engine"] == "GPU_RESIDENT_EXPERIMENTAL"
    assert result["layer_sizes"] == list(map(len, oracle[:depth + 1]))
    ranked = current(path)
    observed = sum(ranked, Counter())
    assert observed == Counter(tuple(x) for x in oracle[depth])
    assert all(count == 1 for count in observed.values())
    for memory in result["segment_memory_by_rank"]:
        resident = memory["resident"]
        load = int(os.environ.get("MGBFS_RESIDENT_HASH_LOAD_PERCENT", "50"))
        assert memory["resident_hash_load_percent"] == load
        logical_rows = memory["rows_per_bank"] * memory["banks"]
        needed = (logical_rows * 100 + load - 1) // load
        assert memory["slots_per_segment"] == 1 << (needed - 1).bit_length()
        assert resident["hash_rehash_enabled"] is True
        assert resident["hash_rehash_count"] is not None
        peak_rebuilds = max(peak_rebuilds, resident["hash_rehash_count"])
    return ranked

with persistent_graph_workers() as pool:
    for depth in range(len(oracle)):
        path = root / f"depth-{depth}"
        result = execute(path, _profile_layers=depth)
        assert result["status"] == ("COMPLETE" if depth == len(oracle) - 1 else "INCOMPLETE")
        rows = verify(path, result, depth)
        layer_rows.append(rows)
        pids = [worker.process.pid for worker in pool.workers]
        if workers is None:
            workers = pids
        assert pids == workers
    assert peak_rebuilds > 0, "the physical run never reached a rebuild"
    need = max(sum(counter.values()) for ranks in layer_rows for counter in ranks)
    for cap in (need, need - 1):
        first = next((d for d, ranks in enumerate(layer_rows)
                      if max(sum(counter.values()) for counter in ranks) > cap), None)
        last = len(oracle) - 1 if first is None else first - 1
        path = root / f"cap-{cap}"
        result = execute(path, capacity=cap)
        assert result["status"] == ("COMPLETE" if first is None else "INCOMPLETE")
        if first is not None:
            assert result["reason"] == "RESIDENT_RESOURCE_CAPACITY"
        verify(path, result, last)
        assert [worker.process.pid for worker in pool.workers] == workers
    path = root / "healthy-after-cap"
    result = execute(path)
    assert result["status"] == "COMPLETE"
    verify(path, result, len(oracle) - 1)
    assert [worker.process.pid for worker in pool.workers] == workers

record = {
    "world": world, "graph": "S6 adjacent transpositions", "layers": len(oracle),
    "full_multiset_oracle": True, "max_rebuilds_per_rank": peak_rebuilds,
    "hash_bits": os.environ.get("MGBFS_SEGMENT_TEST_HASH_BITS", "native"),
    "sufficient_cap": need, "insufficient_cap": need - 1,
    "persistent_workers": workers,
}
(root / "acceptance.json").write_text(json.dumps(record, indent=2))
print("PASS resident hash rebuild", json.dumps(record), "EVIDENCE", root, flush=True)
