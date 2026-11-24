
"""Batch hash: exact collision handling and deferred compaction on two GPUs."""
import json, os, tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

os.environ["MGBFS_SEGMENT_TEST_BATCH"]="1"
os.environ["MGBFS_SEGMENT_BATCH_TILE"]="2"
os.environ["MGBFS_SEGMENT_TEST_HASH_RUNS"]="1"
root=Path(tempfile.mkdtemp(prefix="batch-hash-pressure-",dir="/workspace"))
native=os.environ["MGBFS_EXECUTABLE"]
graph=GraphDefinition.permutation([[1,0,2,3,4],[0,2,1,3,4],[0,1,3,2,4],[0,1,2,4,3]],list(range(5)))
layers=graph.exact_layers(1000)
records=[]
with persistent_graph_workers() as pool:
 for name,bits,merge_limit,prefix in [("normal",64,100000,0),("all-hashes-equal-deferred",0,0,0),
                                      ("prefix-all-hashes-equal",0,0,1),("restored",64,100000,1)]:
  out=root/name
  env=dict(os.environ,MGBFS_SEGMENT_TEST_HASH_BITS=str(bits),MGBFS_SEGMENT_TEST_MERGE_ROWS=str(merge_limit),
           MGBFS_SEGMENT_HASH_PREFIX_PROBE=str(prefix))
  if bits==0 or prefix:
   for depth,expected in enumerate(layers):
    partial=root/f"{name}-depth-{depth}"
    part=run_graph(graph,partial,devices=[0,1],autotune=False,shards=1,_batch=4,
        memory_backend="segmented24",segment_rows=128,segment_strategy="batch",
        executable=native,_native_env=env,_profile_layers=depth)
    assert part["layer_sizes"]==list(map(len,layers[:depth+1])) and part["reason"]=="PROFILE_LAYER_LIMIT",(depth,part)
    saved=json.loads((partial/"states.json").read_text())
    assert sorted(saved["current"])==expected,(depth,saved["current"],expected)
    assert len(saved["current"])==len({tuple(x) for x in saved["current"]})
  r=run_graph(graph,out,devices=[0,1],autotune=False,shards=1,_batch=4,
      memory_backend="segmented24",segment_rows=128,segment_strategy="batch",
      executable=native,_native_env=env)
  assert r["status"]=="COMPLETE" and r["layer_sizes"]==[len(x) for x in layers],r
  state=json.loads((out/"states.json").read_text())
  assert sorted(state["current"])==layers[-1] and sorted(state["previous_small"])==layers[-2],name
  stats=[x for rank in r["segment_memory_by_rank"] for x in rank["batch_hash_runs_by_shard"] if x]
  merged=sum(x["merged"] for x in stats);deferred=sum(x["deferred"] for x in stats)
  if name in ("normal","restored"):assert merged>0,(name,stats)
  if bits==0:assert deferred>0 and max(x["max_observed"] for x in stats)>1,(name,stats)
  item={"case":name,"hash_bits":bits,"hash_prefix_probe":bool(prefix),"layer_sizes":r["layer_sizes"],"merged":merged,"deferred":deferred,
        "runs":stats,"workers":[w.process.pid for w in pool.workers]}
  if records:assert item["workers"]==records[0]["workers"]
  records.append(item);print("PASS",json.dumps(item),flush=True)
(root/"evidence.json").write_text(json.dumps(records,indent=2))
print("BATCH_HASH_COLLISION_PRESSURE_TWO_GPU_PASS",root,flush=True)
