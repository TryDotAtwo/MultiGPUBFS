"""Opt-in, sequential host/device comparison with exact output checks.

No VRAM sampler is started by default. Native allocation receipts are reported
as allocated bytes, not as an observed device-wide peak.
"""
import hashlib,json,math,os,tempfile,time
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
from multigpubfs.generic_session import persistent_graph_workers

world=int(os.environ.get("RESIDENT_AB_WORLD","4"))
n=int(os.environ.get("RESIDENT_AB_N","8"))
rows=int(os.environ.get("RESIDENT_AB_SEGMENT_ROWS","8192"))
order=os.environ.get("RESIDENT_AB_ORDER","hash-host,hash-device,bitmap-host,bitmap-device").split(",")
allowed={"hash-host","hash-device","bitmap-host","bitmap-device"}
if not order or set(order)-allowed or not 2<=n<=16:raise ValueError("invalid comparison config")
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
assert sum(sizes)==math.factorial(n)
root=Path(tempfile.mkdtemp(prefix=f"resident-hash-s{n}-ab-",dir=os.environ.get("MGBFS_TEST_TMP_ROOT")))
exe=Path(os.environ["MGBFS_EXECUTABLE"])
lib=Path(os.environ["MGBFS_CUDA_LIB_FILE"])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
exe_sha,lib_sha=sha(exe),sha(lib)
print("IDENTITY",exe_sha,lib_sha,"root",root,flush=True)
records=[]
with persistent_graph_workers() as pool:
 for i,variant in enumerate(order):
  backend="segmented24" if variant.startswith("hash") else "permutation_bitmap"
  engine="device" if variant.endswith("device") else "host"
  out=root/f"{i}-{variant}"
  t=time.perf_counter()
  r=run_graph(graph,out,devices=list(range(world)),autotune=False,shards=2,_batch=128,segment_rows=rows,
   memory_backend=backend,segment_strategy="atomic",execution_engine=engine,executable=str(exe))
  wall=time.perf_counter()-t
  assert r["status"]=="COMPLETE" and r["layer_sizes"]==sizes,(variant,r)
  state=json.loads((out/"states.json").read_text())["current"]
  assert state==[list(reversed(range(n)))],(variant,state)
  assert r["runtime_identity"]["cuda_library_sha256"]==lib_sha,r["runtime_identity"]
  by_rank=[]
  for m in r["segment_memory_by_rank"]:
   by_rank.append({k:m.get(k) for k in (
    "rank","device","rows_per_bank","segment_count","storage_allocated_bytes",
    "segment_state_allocated_bytes","segment_index_allocated_bytes","bitmap_bytes",
    "queues_allocated_bytes","reserve_bytes","resident")})
  rec={"variant":variant,"world":world,"graph":f"S{n} adjacent transpositions",
   "vertices":sum(sizes),"layers":len(sizes),"generated_edges":sum(sizes)*(n-1),
   "segment_rows":rows,"shards":2,"batch":128,"execution_engine":r["execution_engine"],
   "bfs_seconds":r["bfs_seconds"],"setup_seconds":r["setup_seconds"],
   "launch_wall_seconds":r["launch_wall_seconds"],"observed_wall_seconds":wall,
   "rank_worker_pids":r["rank_worker_pids"],"executable_sha256":exe_sha,"cuda_sha256":lib_sha,
   "rank_allocations":by_rank,"exact_check":"all Mahonian counts and final state"}
  records.append(rec)
  (root/"comparison.json").write_text(json.dumps(records,indent=2))
  print("PASS",variant,"bfs_s",r["bfs_seconds"],"wall_s",wall,"total",sum(sizes),flush=True)
print("EVIDENCE",root,flush=True)
