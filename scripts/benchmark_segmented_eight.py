"""Segmented S_n geometry comparison; raw layer states stay on this GPU host."""
import argparse,hashlib,json,math,os,subprocess,threading,time,traceback
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.launch import run_graph
from multigpubfs.generic_session import persistent_graph_workers

ap=argparse.ArgumentParser()
ap.add_argument("--backend",choices=["segmented24","permutation_bitmap"],required=True)
ap.add_argument("--strategy",choices=["atomic","warp","batch"],required=True)
ap.add_argument("--repeat",type=int,default=1)
ap.add_argument("--n",type=int,default=12)
ap.add_argument("--batch",type=int,default=8192,help="requested batch; admission may lower it")
ap.add_argument("--max-depth",type=int)
ap.add_argument("--base",default="/workspace/bench-segmented-snapshot")
a=ap.parse_args()
if a.batch <= 0:
 ap.error("--batch must be positive")
n=a.n
base=Path(a.base)
base.mkdir(exist_ok=True)
out=base/("S%d-%s-%s-%d"%(n,a.backend,a.strategy,a.repeat))
out.mkdir(exist_ok=False)
os.environ["MGBFS_SEGMENT_ROWS"]="1048576"
os.environ["MGBFS_SEGMENT_OWNER_LANES"]="4"
os.environ["MGBFS_SEGMENT_EXACT_EXCHANGE"]="1"
os.environ["MGBFS_GENERIC_GENERATOR"]="cuda"
os.environ["MGBFS_GENERIC_HISTORY"]="hash"
os.environ["MGBFS_GENERIC_TRANSPORT"]="full"
os.environ["MGBFS_GENERIC_SORT"]="none"
native=os.environ["MGBFS_EXECUTABLE"]
stop=threading.Event();samples=[]
def sample():
 while not stop.is_set():
  try:
   q=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=8)
   values={}
   for row in q.stdout.strip().splitlines():
    idx,mem=row.split(",",1);values[idx.strip()]=int(mem.strip())
   samples.append({"monotonic":time.monotonic(),"mib_by_gpu":values})
  except Exception as ex:samples.append({"monotonic":time.monotonic(),"error":repr(ex)})
  stop.wait(1)
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def untracked_sources():
 root=Path(os.environ.get("MGBFS_BENCH_REPO","/workspace/mgbfs"))
 result=subprocess.run(["git","ls-files","--others","--exclude-standard","-z"],cwd=root,capture_output=True,check=True).stdout
 return {p:sha(root/p) for p in result.decode().split("\0") if p and (root/p).is_file()}
def mahonian(n):
 sizes=[1]
 for k in range(2,n+1):
  nxt=[0]*(len(sizes)+k-1)
  for j,count in enumerate(sizes):
   for delta in range(k):nxt[j+delta]+=count
  sizes=nxt
 return sizes
perms=[]
for i in range(n-1):
 s=list(range(n));s[i],s[i+1]=s[i+1],s[i];perms.append(s)
graph=GraphDefinition.permutation(perms,list(range(n)))
initial={"backend":a.backend,"strategy":a.strategy,"repeat":a.repeat,
 "gpu_info":subprocess.run(["nvidia-smi","--query-gpu=index,name,uuid,memory.total,driver_version","--format=csv,noheader"],capture_output=True,text=True).stdout,
 "topology":subprocess.run(["nvidia-smi","topo","-m"],capture_output=True,text=True).stdout,
 "source_commit":subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,cwd=os.environ.get("MGBFS_BENCH_REPO","/workspace/mgbfs")).stdout.strip(),
 "source_tree":subprocess.run(["git","rev-parse","HEAD^{tree}"],capture_output=True,text=True,cwd=os.environ.get("MGBFS_BENCH_REPO","/workspace/mgbfs")).stdout.strip(),
 "native_sha256":sha(native),"cuda_lib_path":os.environ.get("MGBFS_BENCH_CUDA_LIBRARY","/workspace/build/libmgbfs_cuda.so"),
 "cuda_lib_sha256":sha(os.environ.get("MGBFS_BENCH_CUDA_LIBRARY","/workspace/build/libmgbfs_cuda.so")),
 "source_patch_sha256":hashlib.sha256(subprocess.run(["git","diff","--binary"],capture_output=True,cwd=os.environ.get("MGBFS_BENCH_REPO","/workspace/mgbfs"),check=True).stdout).hexdigest(),
 "untracked_source_sha256":untracked_sources(),
 "batch_recovery":os.environ.get("MGBFS_SEGMENT_BATCH_RECOVERY","0"),
 "layer_stats":os.environ.get("MGBFS_SEGMENT_LAYER_STATS","0"),
 "benchmark_script_sha256":sha(__file__),
 "bitmap_owner_page":os.environ.get("MGBFS_SEGMENT_BITMAP_OWNER_PAGE","0"),
 "telemetry_interval_seconds":1,"telemetry_scope":"sampled maximum, can miss short peaks",
 "graph":"adjacent transpositions S%d, initial identity"%n,"devices":list(range(8)),"shards":8,
 "owner_lanes":4,"configured_segment_rows":2**32,"allocation_segment_rows":1048576,
 "batch":a.batch,"preset":"balanced","autotune":False,"exact_exchange":True,"max_depth":a.max_depth}
(out/"invocation.json").write_text(json.dumps(initial,indent=2))
thread=threading.Thread(target=sample,daemon=True);thread.start()
try:
 context_started=time.monotonic()
 with persistent_graph_workers():
  run_started=time.monotonic()
  r=run_graph(graph,out/"result",devices=list(range(8)),shards=8,autotune=False,
   executable=native,memory_backend=a.backend,memory_preset="balanced",
   segment_rows=2**32,segment_strategy=a.strategy,_batch=a.batch,_profile_layers=a.max_depth)
  run_finished=time.monotonic()
 context_finished=time.monotonic()
 expected=mahonian(n); sizes=r["layer_sizes"]; times=r["layer_seconds"]
 assert sizes==expected[:len(sizes)],"Mahonian layer mismatch"
 assert all(x["segment_strategy"]==a.strategy for x in r["segment_memory_by_rank"])
 checks={"all_completed_layer_counts_match_mahonian":True,"full_result":r["status"]=="COMPLETE"}
 if r["status"]=="COMPLETE":
  assert len(sizes)==len(expected) and sum(sizes)==math.factorial(n)
  states=json.loads((out/"result"/"states.json").read_text())
  last=tuple(reversed(range(n)));previous=set()
  for i in range(n-1):
   x=list(last);x[i],x[i+1]=x[i+1],x[i];previous.add(tuple(x))
  current=[tuple(x) for x in states["current"]]
  prev=[tuple(x) for x in states["previous_small"]]
  assert len(current)==len(set(current))==1 and set(current)=={last}
  assert len(prev)==len(set(prev))==len(previous) and set(prev)==previous
  checks["final_two_layers_full_exact_sets_no_duplicates"]=True
 else:checks["final_two_layers_full_exact_sets_no_duplicates"]=False
 lo,hi=({12:28,13:34}.get(n,max(1,n)),min({12:38,13:44}.get(n,len(sizes)-1),len(sizes)-1))
 window=None
 if hi>=lo:
  secs=sum(times[lo-1:hi]); window={"depth_first":lo,"depth_last":hi,"seconds":secs,
   "new_states":sum(sizes[lo:hi+1]),"generated_edges":sum(sizes[lo-1:hi])*(n-1)}
  window["states_per_second"]=window["new_states"]/secs
  window["edges_per_second"]=window["generated_edges"]/secs
 result={"status":r["status"],"reason":r["reason"],"complete_depth":len(sizes)-1,
  "python_run_graph_wall_seconds":run_finished-run_started,"persistent_context_exit_seconds":context_finished-run_finished,"persistent_context_enter_seconds":run_started-context_started,
  "bfs_seconds":r["bfs_seconds"],"setup_seconds":r["setup_seconds"],
  "launch_wall_seconds":r["launch_wall_seconds"],"source_retries":r["source_retries"],
  "window":window,"checks":checks,"plan":r["plan"],"ranks":r["segment_memory_by_rank"],
  "layer_sizes":sizes,"layer_seconds":times,"runtime_identity":r["runtime_identity"]}
 (out/"summary.json").write_text(json.dumps(result,indent=2))
 print("BENCH_RESULT",json.dumps(result),flush=True)
except BaseException as ex:
 (out/"failure.txt").write_text(traceback.format_exc())
 print("BENCH_FAILURE",repr(ex),flush=True)
 raise
finally:
 stop.set();thread.join(timeout=10)
 (out/"telemetry.json").write_text(json.dumps(samples))
 maxes={}
 for sample in samples:
  for gpu,used in sample.get("mib_by_gpu",{}).items():maxes[gpu]=max(maxes.get(gpu,0),used)
 (out/"sampled_peak_mib.json").write_text(json.dumps({"mib_by_gpu":maxes,"samples":len(samples),
  "scope":"1-second nvidia-smi samples, may miss short peaks"},indent=2))
 print("BENCH_TELEMETRY",json.dumps(maxes),"samples",len(samples),flush=True)
