import hashlib,json,math,os,tempfile,time
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
from multigpubfs.generic_session import persistent_graph_workers

world=int(os.environ.get("RESIDENT_AB_WORLD","4"))
n=int(os.environ.get("RESIDENT_AB_N","9"))
backend=os.environ.get("RESIDENT_AB_BACKEND","segmented24")
rows=int(os.environ.get("RESIDENT_AB_SEGMENT_ROWS","8192"))
order=[int(x) for x in os.environ.get("RESIDENT_AB_SHARDS_ORDER","2,4,8,8,4,2").split(",")]
assert all(v in (2,4,8) for v in order)
gens=[]
for i in range(n-1):
 p=list(range(n));p[i],p[i+1]=p[i+1],p[i];gens.append(p)
graph=GraphDefinition.permutation(gens,list(range(n)))
sizes=[1]
for k in range(1,n+1):
 nxt=[0]*(len(sizes)+k-1)
 for d,count in enumerate(sizes):
  for j in range(k):nxt[d+j]+=count
 sizes=nxt
assert sum(sizes)==math.factorial(n)
root=Path(tempfile.mkdtemp(prefix=f"shards-{backend}-s{n}-",dir=os.environ["MGBFS_TEST_TMP_ROOT"]))
exe=Path(os.environ["MGBFS_EXECUTABLE"]);lib=Path(os.environ["MGBFS_CUDA_LIB_FILE"])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
exe_sha,lib_sha=sha(exe),sha(lib)
print("IDENTITY",exe_sha,lib_sha,"root",root,flush=True)
records=[];pids=None
with persistent_graph_workers() as pool:
 for i,shards in enumerate(order):
  slots=4
  os.environ["MGBFS_RESIDENT_MAILBOX_SLOTS"]=str(slots)
  out=root/f"{i}-shards-{shards}"
  t=time.perf_counter()
  r=run_graph(graph,out,devices=list(range(world)),autotune=False,shards=shards,_batch=128,
    segment_rows=rows,memory_backend=backend,segment_strategy="atomic",
    execution_engine="device",executable=str(exe))
  wall=time.perf_counter()-t
  assert r["status"]=="COMPLETE" and r["layer_sizes"]==sizes,(slots,r)
  assert json.loads((out/"states.json").read_text())["current"]==[list(reversed(range(n)))]
  assert r["runtime_identity"]["cuda_library_sha256"]==lib_sha
  current=[w.process.pid for w in pool.workers]
  if pids is None:pids=current
  assert current==pids,(pids,current)
  assert all(m["resident"]["slots_per_ring"]==slots for m in r["segment_memory_by_rank"])
  rec={"backend":backend,"slots":slots,"world":world,"graph":f"S{n} adjacent transpositions",
    "vertices":sum(sizes),"layers":len(sizes),"layer_sizes":sizes,"generated_edges":sum(sizes)*(n-1),
    "segment_rows":rows,"shards":shards,"batch":128,"execution_engine":r["execution_engine"],
    "bfs_seconds":r["bfs_seconds"],"setup_seconds":r["setup_seconds"],
    "launch_wall_seconds":r["launch_wall_seconds"],"observed_wall_seconds":wall,
    "rank_worker_pids":pids,"executable_sha256":exe_sha,"cuda_sha256":lib_sha,
    "rank_allocations":r["segment_memory_by_rank"],
    "exact_check":"all Mahonian layer counts and exact final state"}
  records.append(rec)
  (root/"comparison.json").write_text(json.dumps(records,indent=2))
  print("PASS",backend,"shards",shards,"bfs_s",r["bfs_seconds"],"wall_s",wall,"total",sum(sizes),flush=True)
print("EVIDENCE",root,flush=True)
