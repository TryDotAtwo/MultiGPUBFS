"""Opt-in physical GPU fault and poisoned persistent-generation regression."""
import json, os, threading, time, tempfile, traceback
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
from multigpubfs.generic_session import persistent_graph_workers

root=Path(tempfile.mkdtemp(prefix="resident-hash-kill-",dir=os.environ.get("MGBFS_TEST_TMP_ROOT")))
n=9
generators=[]
for i in range(n-1):
 p=list(range(n));p[i],p[i+1]=p[i+1],p[i];generators.append(p)
graph=GraphDefinition.permutation(generators,list(range(n)))
def run(g,out):
 return run_graph(g,out,devices=[0,1],autotune=False,shards=2,_batch=8,segment_rows=65536,
  memory_backend="segmented24",segment_strategy="atomic",execution_engine="device",executable=os.environ["MGBFS_EXECUTABLE"])
got={}
with persistent_graph_workers() as pool:
 def work():
  try:got["result"]=run(graph,root/"killed")
  except BaseException as e:got["error"]=repr(e);got["trace"]=traceback.format_exc()
 thread=threading.Thread(target=work);thread.start()
 deadline=time.monotonic()+120
 while time.monotonic()<deadline:
  if not thread.is_alive():break
  workers=pool.workers
  if len(workers)==2 and all(w.job is not None for w in workers):
   phases=[w.job.phase() for w in workers]
   if all(p is not None and p.get("phase")=="resident-bfs" for p in phases):break
  time.sleep(.05)
 else:raise AssertionError("resident BFS phase not reached")
 assert thread.is_alive(),got
 workers=list(pool.workers);pids=[w.process.pid for w in workers]
 print("PHASE",phases,"PIDS",pids,flush=True)
 time.sleep(.2)
 assert thread.is_alive(),"BFS finished before injected kill"
 workers[1].process.kill()
 killed=time.monotonic()
 thread.join(35)
 assert not thread.is_alive(),"supervisor failed to terminate after killing rank"
 assert "NATIVE_DISTRIBUTED_RANK_FAILED" in got.get("error",""),got
 assert not (root/"killed"/"report.json").exists()
 assert (root/"killed"/"failure-evidence").exists()
 assert pool.poisoned,("pool not poisoned",pool.identity,pool.workers)
 assert all(w.process.poll() is not None for w in workers)
 try:run(graph,root/"forbidden-reuse")
 except RuntimeError as e:
  assert "GENERIC_SESSION_FAILED_GROUP_NO_RESTART" in str(e),e
 else:raise AssertionError("poisoned group restarted silently")
 print("PASS rank kill",round(time.monotonic()-killed,3),"seconds",got["error"],flush=True)

small=GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],[0,1,2,3])
layers=small.exact_layers(1000)
with persistent_graph_workers() as nextpool:
 r=run(small,root/"healthy-next")
 assert r["status"]=="COMPLETE" and r["layer_sizes"]==[len(l) for l in layers],r
 assert r["execution_engine"]=="GPU_RESIDENT_EXPERIMENTAL",r
 assert r["segment_memory_by_rank"][0]["segment_rows_per_bank_by_shard"] is None
 assert r["segment_memory_by_rank"][0]["resident_extent_pool_rows_per_bank"]
 print("PASS explicit new generation",r["layer_sizes"],"workers",[w.process.pid for w in nextpool.workers],flush=True)
(root/"acceptance.json").write_text(json.dumps({"workers_killed":pids,"error":got["error"],"next_layers":r["layer_sizes"]},indent=2))
print("EVIDENCE",root,flush=True)
