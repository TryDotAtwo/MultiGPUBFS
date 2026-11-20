"""Two GPU exact per-layer sets and persistent strategy switching, small oracles."""
import json, os, tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

root=Path(tempfile.mkdtemp(prefix="strategy-layer-oracle-",dir="/workspace"))
native=os.environ["MGBFS_EXECUTABLE"]
def rotation(k): return [(i+k)%6 for i in range(6)]
cases=[
 ("s4",GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],list(range(4)))),
 ("directed-c6",GraphDefinition.permutation([rotation(1),rotation(4)],list(range(6)))),
 ("repeated",GraphDefinition.permutation([[1,0,2,3,4],[0,2,1,3,4],[0,1,3,2,4],[0,1,2,4,3]],[0,0,1,2,3])),
]
sequence=[("segmented24","atomic"),("permutation_bitmap","warp"),
          ("segmented24","warp"),("permutation_bitmap","atomic")]
records=[]; worker_ids=None
with persistent_graph_workers() as pool:
 for backend,strategy in sequence:
  for name,graph in cases:
   layers=graph.exact_layers(1000)
   for depth in range(len(layers)):
    out=root/f"{backend}-{strategy}-{name}-{depth}"
    report=run_graph(graph,out,devices=[0,1],autotune=False,shards=4,_batch=2,
      memory_backend=backend,segment_rows=3,segment_strategy=strategy,
      executable=native,_profile_layers=depth)
    assert report["layer_sizes"]==[len(x) for x in layers[:depth+1]],(backend,strategy,name,depth,report)
    assert report["status"]=="INCOMPLETE" and report["reason"]=="PROFILE_LAYER_LIMIT",report
    state=json.loads((out/"states.json").read_text())
    assert sorted(state["current"])==layers[depth],(backend,strategy,name,depth,state["current"],layers[depth])
    assert len(state["current"])==len({tuple(v) for v in state["current"]})
    if depth:assert sorted(state["previous_small"])==layers[depth-1]
    assert report["segment_strategy"]==strategy
    assert all(r["segment_strategy"]==strategy for r in report["segment_memory_by_rank"])
    ids=[w.process.pid for w in pool.workers]
    if worker_ids is None:worker_ids=ids
    assert ids==worker_ids,(ids,worker_ids)
   out=root/f"{backend}-{strategy}-{name}-complete"
   report=run_graph(graph,out,devices=[0,1],autotune=False,shards=4,_batch=2,
      memory_backend=backend,segment_rows=3,segment_strategy=strategy,executable=native)
   assert report["status"]=="COMPLETE" and report["layer_sizes"]==[len(x) for x in layers],report
   records.append({"backend":backend,"strategy":strategy,"case":name,"depths":len(layers),
                   "states":sum(map(len,layers)),"workers":worker_ids})
   print("PASS exact per-layer",json.dumps(records[-1]),flush=True)
(root/"evidence.json").write_text(json.dumps(records,indent=2))
print("STRATEGY_EXACT_LAYERS_TWO_GPU_PASS",root,flush=True)
