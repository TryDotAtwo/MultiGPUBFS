"""Two-GPU reserve-pressure regression with real CUDA allocations.

LD_PRELOAD injects one low-free-memory cudaMemGetInfo immediately after an
allocated 4-row segment. The original allocation is discarded; a smaller
segment must be retried. Check complete S5 counts and exact last two layers.
The injected query is not a measurement of physical CUDA allocation pressure.
"""
import argparse,json,os,subprocess,tempfile,time
from pathlib import Path
from multigpubfs import GraphDefinition

ap=argparse.ArgumentParser()
ap.add_argument("--backend",choices=("hash","bitmap"),default="hash")
ap.add_argument("--transport",choices=("exact","padded"),default="exact")
ap.add_argument("--event",type=int,choices=(1,2),default=2)
ap.add_argument("--persistent-pressure",action="store_true")
a=ap.parse_args()
native=os.environ["MGBFS_EXECUTABLE"]
root=Path(tempfile.mkdtemp(prefix="postalloc-regression-",dir="/workspace"))
def swap(i):
 p=list(range(5));p[i],p[i+1]=p[i+1],p[i];return p
g=GraphDefinition.permutation([swap(i) for i in range(4)],list(range(5)))
expected=g.exact_layers(1000)
definition=root/"graph.json";definition.write_text(g.to_json())
preload=root/"postalloc_inject.so"
subprocess.run(["cc","-shared","-fPIC","-O2",str(Path(__file__).with_name("segmented_postalloc_guard_inject.c")),"-o",str(preload),"-ldl"],check=True)
env=dict(os.environ,LD_PRELOAD=str(preload),MGBFS_TEST_POSTALLOC_BACKEND=a.backend,
 MGBFS_TEST_POSTALLOC_EVENT=str(a.event),MGBFS_EXPERIMENTAL_SEGMENTED24="1",
 MGBFS_EXPERIMENTAL_BITMAP=str(int(a.backend=="bitmap")),MGBFS_SEGMENT_STRATEGY="batch",
 MGBFS_SEGMENT_ROWS="4",MGBFS_SEGMENT_OWNER_LANES="1",MGBFS_SEGMENT_TEST_QUEUE_CAP="4",
 MGBFS_SEGMENT_EXACT_EXCHANGE=str(int(a.transport=="exact")),
 MGBFS_GENERIC_GENERATOR="cuda",MGBFS_GENERIC_HISTORY="hash")
if a.persistent_pressure:env["MGBFS_TEST_POSTALLOC_REPEAT"]="1"
q=subprocess.run([native,"graph-info",str(definition),"0,1","1","1","1"],env=env,text=True,capture_output=True,timeout=90,check=True)
plan=json.loads(q.stdout);plan["owner_cuts"]=None
config=root/"launch.json";config.write_text(json.dumps(plan));jobs=[];logs=[]
try:
 for rank in range(2):
  f=(root/f"rank-{rank}.log").open("w");logs.append(f)
  jobs.append(subprocess.Popen([native,"graph-rank",str(definition),str(config),str(rank),str(root/"id"),str(root/f"rank-{rank}"),"0"],env=env,stdout=f,stderr=subprocess.STDOUT))
 deadline=time.monotonic()+150
 while any(p.poll() is None for p in jobs) and time.monotonic()<deadline:
  if any(p.poll() not in (None,0) for p in jobs):break
  time.sleep(.05)
 for f in logs:f.flush()
 codes=[p.poll() for p in jobs]
 logtext=[(root/f"rank-{i}.log").read_text(errors="replace") for i in range(2)]
 print("EVIDENCE",root,"CODES",codes,"INJECTED",[("TEST_POSTALLOC_RESERVE_PRESSURE" in x) for x in logtext],flush=True)
 assert any("TEST_POSTALLOC_RESERVE_PRESSURE" in x for x in logtext),("injection not reached",root,[x[-2000:] for x in logtext])
 assert all(code==0 for code in codes),("rank failure after injected pressure",root,codes,[x[-2000:] for x in logtext])
 parts=[json.loads((root/f"rank-{i}"/"report.json").read_text()) for i in range(2)]
 if a.persistent_pressure:
  assert all(p["status"]=="INCOMPLETE" and p["reason"]=="RESOURCE_10" for p in parts),("expected resource stop",root,[(p["status"],p.get("reason")) for p in parts])
 else:
  assert all(p["status"]=="COMPLETE" for p in parts),("premature resource stop",root,[(p["status"],p.get("reason")) for p in parts])
 sizes=parts[0]["layer_sizes"]
 assert all(p["layer_sizes"]==sizes for p in parts) and sizes==list(map(len,expected))[:len(sizes)],("layer mismatch",root)
 if not a.persistent_pressure:assert sizes==list(map(len,expected))
 depth=len(sizes)-1
 snaps=[json.loads((root/f"rank-{i}"/"states.json").read_text()) for i in range(2)]
 rows=lambda xs:sorted(map(tuple,xs))
 assert rows(sum((s["current"] for s in snaps),[]))==rows(expected[depth])
 assert rows(sum((s["previous_small"] for s in snaps),[]))==rows(expected[depth-1] if depth else [])
 actual=[[n for bucket in p["segment_memory"]["segment_rows_per_bank_by_shard"] for n in bucket] for p in parts]
 if not a.persistent_pressure:
  assert all(not ("TEST_POSTALLOC_RESERVE_PRESSURE" in logtext[i]) or any(n<4 for n in actual[i]) for i in range(2)),("smaller segment retry not seen",root,actual)
 print("POSTALLOC_RETRY_TWO_GPU_PASS",a.backend,a.transport,"event",a.event,"persistent_pressure",a.persistent_pressure,"depth",depth,"segments",actual,"evidence",root,flush=True)
finally:
 for p in jobs:
  if p.poll() is None:p.terminate()
 for p in jobs:
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:p.kill();p.wait()
 for f in logs:f.close()
