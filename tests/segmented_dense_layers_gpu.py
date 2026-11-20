"""Exact/padded two-rank transport, dense origins, spills and persistent reuse."""
import json,os,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

native=os.environ["MGBFS_EXECUTABLE"]
root=Path(tempfile.mkdtemp(prefix="dense-layers-",dir="/workspace"))
def rotation(k):return [(i+k)%6 for i in range(6)]
cases=[
 ("s4",GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],list(range(4)))),
 ("directed-c6",GraphDefinition.permutation([rotation(1),rotation(4)],list(range(6)))),
 ("repeated",GraphDefinition.permutation([[1,0,2,3,4],[0,2,1,3,4],[0,1,3,2,4],[0,1,2,4,3]],[0,0,1,2,3]))]
ids=None;records=[]
with persistent_graph_workers() as pool:
 for mode in ("exact","padded"):
  for backend in ("permutation_bitmap","segmented24"):
   env=dict(os.environ,MGBFS_SEGMENT_BATCH_TILE="2",MGBFS_SEGMENT_ROWS="3",
            MGBFS_SEGMENT_EXACT_EXCHANGE="1" if mode=="exact" else "0",
            MGBFS_SEGMENT_TEST_INPUT_STATS="1")
   env.pop("MGBFS_SEGMENT_TEST_RECTANGULAR",None)
   for name,g in cases:
    layers=g.exact_layers(1000)
    for depth in range(len(layers)):
     out=root/f"{mode}-{backend}-{name}-{depth}"
     r=run_graph(g,out,devices=[0,1],shards=4,autotune=False,executable=native,
      memory_backend=backend,segment_strategy="batch",segment_rows=3,_batch=2,
      _profile_layers=depth,_native_env=env)
     assert r["status"]=="INCOMPLETE" and r["reason"]=="PROFILE_LAYER_LIMIT",r
     assert r["layer_sizes"]==[len(x) for x in layers[:depth+1]],(mode,backend,name,depth,r)
     state=json.loads((out/"states.json").read_text())
     assert sorted(state["current"])==layers[depth],(mode,backend,name,depth)
     assert len(state["current"])==len({tuple(v) for v in state["current"]})
     if depth:assert sorted(state["previous_small"])==layers[depth-1]
     current=[w.process.pid for w in pool.workers]
     if ids is None:ids=current
     assert current==ids,(ids,current)
    out=root/f"{mode}-{backend}-{name}-full"
    r=run_graph(g,out,devices=[0,1],shards=4,autotune=False,executable=native,
      memory_backend=backend,segment_strategy="batch",segment_rows=3,_batch=2,_native_env=env)
    assert r["status"]=="COMPLETE" and r["layer_sizes"]==list(map(len,layers)),r
    stats=[]
    for rank in r["segment_memory_by_rank"]:
     assert rank["batch_input_layout"]=="dense",rank
     data=rank["batch_input_stats"]
     assert data["raw_sort_scan_elements"]==data["valid_received"],data
     stats.append(data)
    rec={"mode":mode,"backend":backend,"case":name,"layers":len(layers),
         "spill_tiles":sum(x["spill_tiles"] for x in stats),
         "empty_self":sum(x["empty_self_source_queues"] for x in stats),
         "empty_peer":sum(x["empty_peer_source_queues"] for x in stats)}
    records.append(rec);print("PASS",json.dumps(rec),flush=True)
  # Shared worker pool is still live after switching transport and backends.
assert any(x["spill_tiles"]>0 for x in records),records
for mode in ("exact","padded"):
 for backend in ("segmented24","permutation_bitmap"):
  matching=[x for x in records if x["mode"]==mode and x["backend"]==backend]
  assert sum(x["empty_self"] for x in matching)>0,matching
  assert sum(x["empty_peer"] for x in matching)>0,matching
(root/"evidence.json").write_text(json.dumps(records,indent=2))
print("DENSE_EXACT_PADDED_TWO_GPU_PASS",root,flush=True)
