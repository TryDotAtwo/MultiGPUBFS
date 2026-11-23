"""Explicit physical-GPU A/B; no full workload is run by default CI."""
import hashlib, json, math, os, subprocess, sys, tempfile, time
from pathlib import Path
from bitmap_pack4_sampling import collect_vram_samples, vram_summary
from multigpubfs import GraphDefinition, run_graph
from multigpubfs.generic_session import persistent_graph_workers

n=int(sys.argv[1]);mode=sys.argv[2]
backend=os.environ.get("RESIDENT_AB_BACKEND","permutation_bitmap")
if backend not in ("permutation_bitmap","segmented24"): raise ValueError("unsupported benchmark backend")
if mode not in ("host","batch"): raise ValueError("mode must be host or batch")
world=int(os.environ.get("RESIDENT_AB_WORLD","4"))
shards=int(os.environ.get("RESIDENT_AB_SHARDS","4"))
segment_rows=int(os.environ.get("RESIDENT_AB_SEGMENT_ROWS",str(1<<20)))
cap=int(os.environ.get("RESIDENT_AB_RANK_CAP","100000000"))
batch_edges=int(os.environ.get("RESIDENT_AB_BATCH_EDGES",str(1<<18)))
receive_records=int(os.environ.get("RESIDENT_AB_RECEIVE_RECORDS",str(1<<18)))
depth_env=os.environ.get("RESIDENT_AB_PROFILE_DEPTH")
depth=None if depth_env is None else int(depth_env)
exe=Path(os.environ["MGBFS_EXECUTABLE"])
lib=Path(os.environ["MGBFS_CUDA_LIB_FILE"])
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
exe_sha,lib_sha=digest(exe),digest(lib)
gens=[]
for i in range(n-1):
 x=list(range(n));x[i],x[i+1]=x[i+1],x[i];gens.append(x)
graph=GraphDefinition.permutation(gens,list(range(n)))
sizes=[1]
for k in range(1,n+1):
 nxt=[0]*(len(sizes)+k-1)
 for d,value in enumerate(sizes):
  for offset in range(k):nxt[d+offset]+=value
 sizes=nxt
assert sum(sizes)==math.factorial(n)
root=Path(tempfile.mkdtemp(prefix=f"multicta-{backend}-{mode}-s{n}-{world}gpu-",dir=os.environ["MGBFS_TEST_TMP_ROOT"]))
devices=list(range(world))
other=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,used_gpu_memory","--format=csv,noheader"],text=True).strip()
assert not other,("foreign GPU processes before benchmark",other)
topology=subprocess.check_output(["nvidia-smi","topo","-m"],text=True)
inventory=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name,memory.total,driver_version","--format=csv,noheader"],text=True)
(root/"topology.txt").write_text(topology);(root/"gpu-inventory.csv").write_text(inventory)
def sample(stop,samples):
 while not stop.is_set():
  try:
   raw=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True,timeout=3)
   samples.append({"monotonic":time.monotonic(),"gpus":{int(a):int(b) for a,b in (x.split(",") for x in raw.splitlines())}})
  except Exception as e:samples.append({"error":str(e)})
  stop.wait(.5)
env=dict(os.environ,MGBFS_SEGMENT_PACK4="1",MGBFS_SEGMENT_EXACT_EXCHANGE="1",
 MGBFS_SEGMENT_QUEUE_CAP="8192",MGBFS_SEGMENT_OWNER_LANES="4",
 MGBFS_SEGMENT_BATCH_RECOVERY="1")
if backend=="segmented24":env.pop("MGBFS_SEGMENT_PACK4",None)
if depth is not None:env["MGBFS_SEGMENT_TEST_BATCH"]="1"
base=dict(devices=devices,autotune=False,shards=shards,_batch=4096,
 segment_rows=segment_rows,capacity=cap,memory_backend=backend,
 memory_preset="balanced",segment_strategy="atomic",executable=str(exe),_native_env=env)
if mode=="batch":
 base.update(execution_engine="device",device_scheduler="batch",
  device_batch_edges=batch_edges,device_receive_records=receive_records)
else:base.update(execution_engine="host")
print("START",mode,"S",n,"depth",depth,"root",root,flush=True)
with persistent_graph_workers() as pool:
 warm=GraphDefinition.permutation([[1,0,2],[0,2,1]],[0,1,2])
 rw=run_graph(warm,root/"warm",**dict(base,segment_rows=32,capacity=16,_batch=2))
 assert rw["status"]=="COMPLETE"
 pids=[w.process.pid for w in pool.workers]
 kwargs=dict(base)
 if depth is not None:kwargs["_profile_layers"]=depth
 with collect_vram_samples(True,sample) as samples:
  t=time.perf_counter();r=run_graph(graph,root/"result",**kwargs);wall=time.perf_counter()-t
 assert [w.process.pid for w in pool.workers]==pids
 assert r["runtime_identity"]["cuda_library_sha256"]==lib_sha,r["runtime_identity"]
 actual=r["layer_sizes"]
 assert actual==sizes[:len(actual)],(actual,sizes[:len(actual)])
 complete=r["status"]=="COMPLETE"
 if complete:
  assert len(actual)==len(sizes)
  assert json.loads((root/"result"/"states.json").read_text())["current"]==[list(reversed(range(n)))]
 if mode=="batch":
  assert all(m["resident"]["scheduler"]=="multi_cta" for m in r["segment_memory_by_rank"])
 states=sum(actual);expanded=sum(actual if complete else actual[:-1])
 result={"mode":mode,"memory_backend":backend,"graph":f"S{n} adjacent transpositions identity","world":world,
  "status":r["status"],"reason":r["reason"],"completed_depth":len(actual)-1,
  "last_depth_verified":len(actual)-1,"all_layer_counts_mahonian":True,"terminal_reverse_exact":complete,
  "layer_sizes":actual,"complete_states":states,"guaranteed_expanded_edges":expanded*(n-1),
  "bfs_seconds":r["bfs_seconds"],"setup_seconds":r["setup_seconds"],
  "launch_wall_seconds":r["launch_wall_seconds"],"observed_wall_seconds":wall,
  "states_per_s":states/r["bfs_seconds"] if r["bfs_seconds"] else None,
  "guaranteed_edges_per_s":expanded*(n-1)/r["bfs_seconds"] if r["bfs_seconds"] else None,
  "shards":shards,"segment_rows":segment_rows,"rank_capacity":cap,
  "batch_edges":batch_edges if mode=="batch" else None,
  "receive_records":receive_records if mode=="batch" else None,
  "sample_period_seconds":0.5,**vram_summary(True,samples,devices),
  "rank_allocations":r["segment_memory_by_rank"],"rank_worker_pids":pids,
  "executable_sha256":exe_sha,"cuda_sha256":lib_sha,
  "timing_scope":"whole BFS; stage and individual layer times not instrumented in multi-CTA path"}
 (root/"samples.json").write_text(json.dumps(samples))
 (root/"comparison.json").write_text(json.dumps(result,indent=2))
 print("DONE",json.dumps(result),flush=True)
print("EVIDENCE",root,flush=True)
