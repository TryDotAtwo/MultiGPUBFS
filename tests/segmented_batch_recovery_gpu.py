"""GPU regression: collective recovery, layer telemetry and persistent isolation."""
import json, os, subprocess, tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

root = Path(tempfile.mkdtemp(prefix="segment-recovery-", dir="/workspace"))
native = os.environ["MGBFS_EXECUTABLE"]
gens=[]
for i in range(4):
    v=list(range(5));v[i],v[i+1]=v[i+1],v[i];gens.append(v)
graph=GraphDefinition.permutation(gens,list(range(5)))
expected=graph.exact_layers(1000)
sequence=[("segmented24","atomic",False),("permutation_bitmap","atomic",True),
          ("segmented24","warp",True),("permutation_bitmap","warp",False),
          ("segmented24","batch",True),("permutation_bitmap","batch",True)]
baseline_ids=None
with persistent_graph_workers() as pool:
 for backend,strategy,enable in sequence:
  out=root/f"{backend}-{strategy}-{int(enable)}"
  env=dict(os.environ,MGBFS_SEGMENT_BATCH_RECOVERY=str(int(enable)),MGBFS_SEGMENT_LAYER_STATS="1",
           MGBFS_SEGMENT_TEST_QUEUE_CAP="4",MGBFS_SEGMENT_ROWS="3",MGBFS_SEGMENT_TEST_BATCH="1")
  report=run_graph(graph,out,devices=[0,1],shards=2,autotune=False,_batch=16,
      memory_backend=backend,segment_strategy=strategy,segment_rows=3,
      executable=native,_native_env=env)
  assert report["status"]=="COMPLETE",report
  assert report["layer_sizes"]==list(map(len,expected)),report
  assert all(x["batch_recovery"]==enable for x in report["segment_memory_by_rank"]),report
  state=json.loads((out/"states.json").read_text())
  assert sorted(state["current"])==expected[-1]
  assert sorted(state["previous_small"])==expected[-2]
  ids=[w.process.pid for w in pool.workers]
  if baseline_ids is None:baseline_ids=ids
  assert baseline_ids==ids
  per_rank=[]
  for rank in range(2):
   lines=(out/f"rank-{rank}.log").read_text().splitlines()
   values=[json.loads(line.split("SEGMENT_LAYER_STATS ",1)[1]) for line in lines if "SEGMENT_LAYER_STATS " in line]
   per_rank.append({v["depth"]:v for v in values})
  assert per_rank[0].keys()==per_rank[1].keys()
  for depth in per_rank[0]:
   a,b=per_rank[0][depth],per_rank[1][depth]
   assert a["source_parents"]==a["source_frontier"] and b["source_parents"]==b["source_frontier"],(depth,a,b)
   assert a["generated_edges"]==4*a["source_parents"] and b["generated_edges"]==4*b["source_parents"]
   assert a["accepted"]+b["accepted"]==a["global_accepted"]==b["global_accepted"]
   assert sum(a["send_by_peer_shard"])+sum(b["send_by_peer_shard"])==sum(a["receive_by_peer_shard"])+sum(b["receive_by_peer_shard"])
  print("PASS recovery persistent/telemetry",backend,strategy,enable,"retries",report["source_retries"],flush=True)

# Force a real collective queue retry, then observe the later, collectively
# agreed recovery of the batch size. Queue 8 for hash admits all five
# children of one parent even during the one-shard communication preflight.
larger=[]
for i in range(5):
 a=list(range(6));a[i],a[i+1]=a[i+1],a[i];larger.append(a)
graph6=GraphDefinition.permutation(larger,list(range(6)))
expected6=graph6.exact_layers(10000)
with persistent_graph_workers():
 for backend,q,shards in (("permutation_bitmap",4,16),("segmented24",8,4)):
  out=root/f"actual-recovery-{backend}"
  env=dict(os.environ,MGBFS_SEGMENT_BATCH_RECOVERY="1",MGBFS_SEGMENT_TEST_QUEUE_CAP=str(q),
           MGBFS_SEGMENT_ROWS="64")
  report=run_graph(graph6,out,devices=[0,1],shards=shards,autotune=False,_batch=128,
       memory_backend=backend,segment_strategy="atomic",segment_rows=1<<32,
       executable=native,_native_env=env)
  assert report["status"]=="COMPLETE" and report["layer_sizes"]==list(map(len,expected6)),report
  recovered=[v["batch_recoveries"] for v in report["segment_memory_by_rank"]]
  assert report["source_retries"]>0 and len(set(recovered))==1 and recovered[0]>0,report
  state=json.loads((out/"states.json").read_text())
  assert sorted(state["current"])==expected6[-1] and sorted(state["previous_small"])==expected6[-2]
  print("PASS real recovery",backend,"retries",report["source_retries"],"recoveries",recovered,flush=True)

# A different setting on one rank must fail all participants before data transfer.
definition=root/"graph.json";definition.write_text(graph.to_json())
config=root/"segmented24-atomic-0"/"launch.json"
for name,settings,caps,expected_error in [
 ("mismatch",("0","1"),(None,None),"SEGMENT_BATCH_RECOVERY_RANK_MISMATCH"),
 ("invalid",("bad","bad"),(None,None),"SEGMENT_BATCH_RECOVERY_INVALID"),
 ("queue-mismatch",("1","1"),("4","8"),"SEGMENTED_QUEUE_CAPACITY_RANK_MISMATCH")]:
 parent=root/name;parent.mkdir();procs=[]
 try:
  for rank,setting in enumerate(settings):
   env=dict(os.environ,MGBFS_SEGMENT_BATCH_RECOVERY=setting,MGBFS_SEGMENT_STRATEGY="atomic",
            MGBFS_EXPERIMENTAL_SEGMENTED24="1",RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE="2")
   env.pop("MGBFS_EXPERIMENTAL_BITMAP",None)
   if caps[rank] is None:env.pop("MGBFS_SEGMENT_TEST_QUEUE_CAP",None)
   else:env["MGBFS_SEGMENT_TEST_QUEUE_CAP"]=caps[rank]
   procs.append(subprocess.Popen([native,"graph-rank",str(definition),str(config),str(rank),
     str(parent/"id"),str(parent/f"rank-{rank}"),"0"],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
  for rank,proc in enumerate(procs):
   out,err=proc.communicate(timeout=75)
   (parent/f"rank-{rank}.log").write_text(out+"\n"+err)
   assert proc.returncode!=0 and expected_error in out+err,(name,rank,proc.returncode,out,err)
   assert not (parent/f"rank-{rank}"/"report.json").exists()
 finally:
  for proc in procs:
   if proc.poll() is None:proc.kill();proc.wait()
 print("PASS recovery rank agreement",name,expected_error,flush=True)
print("SEGMENT_RECOVERY_TWO_GPU_PASS",root,flush=True)
