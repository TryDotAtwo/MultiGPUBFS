"""Opt-in GPU oracle for producer-derived IDs under remapped bitmap ownership."""
import json,os,tempfile
from collections import Counter
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
from multigpubfs.generic_session import persistent_graph_workers
root=Path(tempfile.mkdtemp(prefix="resident-payload-page-",dir=os.environ.get("MGBFS_TEST_TMP_ROOT")))
cases=[
 ("s4",GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],[0,1,2,3])),
 ("repeated",GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],[0,0,1,2])),
]
records=[]
with persistent_graph_workers() as pool:
 pids=None
 for mode in ("carry","verify"):
  os.environ["MGBFS_RESIDENT_PAYLOAD_ID"]=mode
  for page in (1,2):
   for name,graph in cases:
    layers=graph.exact_layers(1000)
    for depth,layer in enumerate(layers):
     out=root/f"{mode}-page{page}-{name}-{depth}"
     r=run_graph(graph,out,devices=[0,1],autotune=False,shards=2,_batch=2,segment_rows=4,
      memory_backend="permutation_bitmap",segment_strategy="atomic",execution_engine="device",
      bitmap_owner_page=page,executable=os.environ["MGBFS_EXECUTABLE"],_profile_layers=depth)
     assert r["layer_sizes"]==[len(x) for x in layers[:depth+1]],r
     assert r["status"]==("COMPLETE" if depth+1==len(layers) else "INCOMPLETE"),r
     actual=json.loads((out/"states.json").read_text())["current"]
     assert Counter(map(tuple,actual))==Counter(map(tuple,layer)),(mode,page,name,depth)
     assert all(m["resident"]["payload_identity"]==mode for m in r["segment_memory_by_rank"])
     now=[w.process.pid for w in pool.workers]
     if pids is None:pids=now
     assert now==pids,(pids,now)
    records.append(dict(mode=mode,page=page,case=name,layer_count=len(layers)))
    print("PASS remapped exact layer multisets",mode,page,name,len(layers),flush=True)
(root/"acceptance.json").write_text(json.dumps({"world":2,"pids":pids,"cases":records},indent=2))
print("EVIDENCE",root,flush=True)
