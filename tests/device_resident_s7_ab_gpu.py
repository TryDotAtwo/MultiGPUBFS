"""Opt-in 4-GPU Coxeter A/B; set RESIDENT_AB_N and order for larger cases."""
import json,os,tempfile,time
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
from multigpubfs.generic_session import persistent_graph_workers
n=int(os.environ.get("RESIDENT_AB_N","7"))
if n<2 or n>16:raise ValueError("RESIDENT_AB_N must be between 2 and 16")
rows=int(os.environ.get("RESIDENT_AB_SEGMENT_ROWS","4096"))
if rows<1:raise ValueError("RESIDENT_AB_SEGMENT_ROWS must be positive")
order=os.environ.get("RESIDENT_AB_ORDER","old,new,new,old,old,new").split(",")
if not order or any(v not in ("old","new","host") for v in order):raise ValueError("RESIDENT_AB_ORDER must contain old/new/host")
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
root=Path(tempfile.mkdtemp(prefix=f"resident-s{n}-ab-",dir=os.environ.get("MGBFS_TEST_TMP_ROOT")))
libs={"old":os.environ["RESIDENT_OLD_LIB"],"new":os.environ["RESIDENT_NEW_LIB"],"host":os.environ["RESIDENT_NEW_LIB"]}
hashes={"old":os.environ["RESIDENT_OLD_SHA"],"new":os.environ["RESIDENT_NEW_SHA"],"host":os.environ["RESIDENT_NEW_SHA"]}
tail=os.environ["RESIDENT_LD_TAIL"]
records=[]
with persistent_graph_workers() as pool:
 for i,variant in enumerate(order):
  os.environ["LD_LIBRARY_PATH"]=libs[variant]+":"+tail
  out=root/f"{i}-{variant}"
  started=time.time()
  r=run_graph(graph,out,devices=[0,1,2,3],autotune=False,shards=2,_batch=128,segment_rows=rows,
   memory_backend="permutation_bitmap",segment_strategy="atomic",execution_engine="host" if variant=="host" else "device",
   executable=os.environ["MGBFS_EXECUTABLE"])
  assert r["status"]=="COMPLETE" and r["layer_sizes"]==sizes,r
  assert r["runtime_identity"]["cuda_library_sha256"]==hashes[variant],r["runtime_identity"]
  snapshot=json.loads((out/"states.json").read_text())
  assert snapshot["current"]==[list(reversed(range(n)))]
  ended=time.time()
  row={"start_unix":started,"end_unix":ended,"rank_geometry":[{k:m.get(k) for k in ("rank","device","rows_per_bank","segment_count","storage_allocated_bytes","bitmap_bytes","queues_allocated_bytes","resident")} for m in r["segment_memory_by_rank"]],"graph_n":n,"segment_rows":rows,"world":4,"shards":2,"variant":variant,"execution_engine":r["execution_engine"],"bfs_seconds":r["bfs_seconds"],"setup_seconds":r["setup_seconds"],
       "wall_seconds":r["launch_wall_seconds"],"cuda_library_sha256":hashes[variant]}
  records.append(row);print("PASS",row,flush=True)
(root/"comparison.json").write_text(json.dumps(records,indent=2))
print("EVIDENCE",root,flush=True)
