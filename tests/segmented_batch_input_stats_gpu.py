"""Measure raw and retry work for both batch owners on two physical GPUs."""
import os,json,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.launch import run_graph

native=os.environ["MGBFS_EXECUTABLE"]
env=dict(os.environ,MGBFS_SEGMENT_TEST_INPUT_STATS="1",MGBFS_SEGMENT_TEST_INPUT_MIN_DEPTH="2",
 MGBFS_SEGMENT_TEST_INPUT_MAX_DEPTH="6",MGBFS_SEGMENT_ROWS="4",MGBFS_SEGMENT_BATCH_TILE="2")
gens=[]
for i in range(4):
 p=list(range(5));p[i],p[i+1]=p[i+1],p[i];gens.append(p)
g=GraphDefinition.permutation(gens,list(range(5)))
expected=list(map(len,g.exact_layers(1000)))
out=Path(tempfile.mkdtemp(prefix="batch-input-stats-",dir="/workspace"))
for backend in ("segmented24","permutation_bitmap"):
 r=run_graph(g,out/backend,devices=[0,1],shards=4,autotune=False,executable=native,
  memory_backend=backend,segment_strategy="batch",segment_rows=4,_batch=4,_profile_layers=30,_native_env=env)
 assert r["status"]=="COMPLETE" and r["layer_sizes"]==expected,r
 stats=[x["batch_input_stats"] for x in r["segment_memory_by_rank"]]
 for x in stats:
  assert x["target_depth_min"]==2 and x["target_depth_max"]==6
  assert 0<x["valid_received"]<=x["raw_sort_scan_elements"],x
  assert x["raw_tiles"]>0 and x["raw_sort_scan_elements"]<=2*x["raw_tiles"],x
  assert x["spill_sort_scan_elements"]<=2*x["spill_tiles"],x
 print("PASS",backend,json.dumps(stats),flush=True)
print("INPUT_STATS_TWO_GPU_PASS",out,flush=True)
