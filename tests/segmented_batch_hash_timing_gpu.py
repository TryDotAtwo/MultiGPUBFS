"""Opt-in hash timing: complete correctness and persistent worker isolation on two GPUs."""
import json
import os
import tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

root = Path(tempfile.mkdtemp(prefix="batch-hash-timing-", dir="/workspace"))
graph = GraphDefinition.permutation(
    [[1,0,2,3,4], [0,2,1,3,4], [0,1,3,2,4], [0,1,2,4,3]], list(range(5)))
expected = graph.exact_layers(1000)
base = dict(os.environ, MGBFS_SEGMENT_TEST_BATCH="1", MGBFS_SEGMENT_BATCH_TILE="4")
for key in ("MGBFS_SEGMENT_HASH_TIMING", "MGBFS_SEGMENT_HASH_TIMING_MIN_DEPTH",
            "MGBFS_SEGMENT_HASH_TIMING_MAX_DEPTH"):
    base.pop(key, None)
timed = dict(base, MGBFS_SEGMENT_HASH_TIMING="1",
             MGBFS_SEGMENT_HASH_TIMING_MIN_DEPTH="2",
             MGBFS_SEGMENT_HASH_TIMING_MAX_DEPTH="6")
with persistent_graph_workers() as workers:
    worker_ids = None
    for name, env in (("timed", timed), ("untimed", base), ("timed-again", timed)):
        result = run_graph(graph, root/name, devices=[0,1], shards=2, autotune=False,
            executable=os.environ["MGBFS_EXECUTABLE"], memory_backend="segmented24",
            segment_rows=32, segment_strategy="batch", _batch=4, _native_env=env)
        assert result["status"] == "COMPLETE"
        assert result["layer_sizes"] == [len(layer) for layer in expected]
        states = json.loads((root/name/"states.json").read_text())
        assert sorted(states["current"]) == expected[-1]
        assert sorted(states["previous_small"]) == expected[-2]
        current_ids = [worker.process.pid for worker in workers.workers]
        if worker_ids is None:
            worker_ids = current_ids
        assert worker_ids == current_ids
        stats = [rank["batch_hash_timing"] for rank in result["segment_memory_by_rank"]]
        if name == "untimed":
            assert stats == [None, None], stats
        else:
            assert all(x is not None and x["tiles"] > 0 and x["owner_lane_ms"] > 0
                and x["unique_probe_lane_ms"] > 0
                and 0 <= x["unique_probe_plus_merge_fraction_of_owner_and_merge"] <= 1.05
                and x["target_depth_min"] == 2 and x["target_depth_max"] == 6
                for x in stats), stats
            runs = [entry for rank in result["segment_memory_by_rank"]
                    for entry in rank["batch_hash_runs_by_shard"] if entry]
            assert any(entry["timed_window_observations"] > 0 for entry in runs), runs
            assert all(entry["timed_window_median_observed"] is not None
                       for entry in runs if entry["timed_window_observations"] > 0), runs
        print("PASS", name, json.dumps(stats), flush=True)
print("TIMING_VERIFY_TWO_GPU_PASS", root, flush=True)
