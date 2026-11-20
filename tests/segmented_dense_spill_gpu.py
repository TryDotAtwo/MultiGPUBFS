"""Force spill/replay and a resource stop across dense tiled two-GPU jobs."""
import json,os,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

native=os.environ["MGBFS_EXECUTABLE"]
gens=[]
for i in range(4):
 v=list(range(5));v[i],v[i+1]=v[i+1],v[i];gens.append(v)
g=GraphDefinition.permutation(gens,list(range(5)))
layers=g.exact_layers(1000);expected=list(map(len,layers))
root=Path(tempfile.mkdtemp(prefix="dense-spill-",dir="/workspace"))
with persistent_graph_workers() as pool:
 ids=None
 for mode in ("exact","padded"):
  for backend in ("segmented24","permutation_bitmap"):
   env=dict(os.environ,MGBFS_SEGMENT_BATCH_TILE="2",MGBFS_SEGMENT_ROWS="1",
    MGBFS_SEGMENT_EXACT_EXCHANGE="1" if mode=="exact" else "0",MGBFS_SEGMENT_TEST_INPUT_STATS="1")
   env.pop("MGBFS_SEGMENT_TEST_RECTANGULAR",None)
   kw=dict(devices=[0,1],shards=1,autotune=False,executable=native,memory_backend=backend,
     segment_strategy="batch",segment_rows=1,_batch=2,_native_env=env)
   out=root/f"{mode}-{backend}-full";r=run_graph(g,out,**kw)
   assert r["status"]=="COMPLETE" and r["layer_sizes"]==expected,r
   states=json.loads((out/"states.json").read_text())
   assert sorted(states["current"])==layers[-1] and sorted(states["previous_small"])==layers[-2]
   stats=[x["batch_input_stats"] for x in r["segment_memory_by_rank"]]
   assert sum(x["spill_tiles"] for x in stats)>0,(mode,backend,stats)
   assert all(x["raw_sort_scan_elements"]==x["valid_received"] for x in stats),stats
   stop=root/f"{mode}-{backend}-stop";q=run_graph(g,stop,capacity=3,**kw)
   assert q["status"]=="INCOMPLETE" and q["reason"]=="RESOURCE_10",q
   depth=len(q["layer_sizes"])-1;state=json.loads((stop/"states.json").read_text())
   assert q["layer_sizes"]==expected[:depth+1]
   assert sorted(state["current"])==layers[depth]
   if depth:assert sorted(state["previous_small"])==layers[depth-1]
   again=run_graph(g,root/f"{mode}-{backend}-again",**kw)
   assert again["status"]=="COMPLETE" and again["layer_sizes"]==expected
   workers=[w.process.pid for w in pool.workers]
   if ids is None:ids=workers
   assert workers==ids,(ids,workers)
   print("PASS",mode,backend,"spill",sum(x["spill_tiles"] for x in stats),"stopped-depth",depth,flush=True)
print("DENSE_SPILL_TWO_GPU_PASS",root,flush=True)
