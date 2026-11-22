"""Opt-in 4-GPU small-graph A/B; no large-layer performance claim."""
import json,os,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
from multigpubfs.generic_session import persistent_graph_workers
n=7
generators=[]
for i in range(n-1):
 p=list(range(n));p[i],p[i+1]=p[i+1],p[i];generators.append(p)
graph=GraphDefinition.permutation(generators,list(range(n)))
sizes=[1]
for k in range(1,n+1):
 nxt=[0]*(len(sizes)+k-1)
 for d,count in enumerate(sizes):
  for j in range(k):nxt[d+j]+=count
 sizes=nxt
root=Path(tempfile.mkdtemp(prefix="resident-s7-ab-",dir=os.environ.get("MGBFS_TEST_TMP_ROOT")))
libs={"old":os.environ["RESIDENT_OLD_LIB"],"new":os.environ["RESIDENT_NEW_LIB"]}
hashes={"old":os.environ["RESIDENT_OLD_SHA"],"new":os.environ["RESIDENT_NEW_SHA"]}
tail=os.environ["RESIDENT_LD_TAIL"]
records=[]
with persistent_graph_workers() as pool:
 for i,variant in enumerate(("old","new","new","old","old","new")):
  os.environ["LD_LIBRARY_PATH"]=libs[variant]+":"+tail
  out=root/f"{i}-{variant}"
  r=run_graph(graph,out,devices=[0,1,2,3],autotune=False,shards=2,_batch=128,segment_rows=4096,
   memory_backend="permutation_bitmap",segment_strategy="atomic",execution_engine="device",
   executable=os.environ["MGBFS_EXECUTABLE"])
  assert r["status"]=="COMPLETE" and r["layer_sizes"]==sizes,r
  assert r["runtime_identity"]["cuda_library_sha256"]==hashes[variant],r["runtime_identity"]
  snapshot=json.loads((out/"states.json").read_text())
  assert snapshot["current"]==[list(reversed(range(n)))]
  row={"variant":variant,"bfs_seconds":r["bfs_seconds"],"setup_seconds":r["setup_seconds"],
       "wall_seconds":r["launch_wall_seconds"],"cuda_library_sha256":hashes[variant]}
  records.append(row);print("PASS",row,flush=True)
(root/"comparison.json").write_text(json.dumps(records,indent=2))
print("EVIDENCE",root,flush=True)
