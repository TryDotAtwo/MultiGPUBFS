"""Physical bounded-pool pressure: exact prefix, full-pool recheck, clean next job."""
import json,os,tempfile
from collections import Counter
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
from multigpubfs.generic_session import persistent_graph_workers

world=int(os.environ.get("RESIDENT_TEST_WORLD","2"))
root=Path(tempfile.mkdtemp(prefix=f"resident-hash-pressure-{world}-",dir=os.environ.get("MGBFS_TEST_TMP_ROOT")))
graph=GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],[0,1,2,3])
oracle=graph.exact_layers(1000)
def run(g,out):
 return run_graph(g,out,devices=list(range(world)),autotune=False,shards=2,_batch=2,segment_rows=2,
  memory_backend="segmented24",segment_strategy="atomic",execution_engine="device",
  executable=os.environ["MGBFS_EXECUTABLE"])
with persistent_graph_workers() as pool:
 r=run(graph,root/"bounded")
 assert r["status"]=="INCOMPLETE" and r["reason"]=="RESIDENT_RESOURCE_CAPACITY",r
 d=len(r["layer_sizes"])-1
 assert r["layer_sizes"]==[len(v) for v in oracle[:d+1]],r["layer_sizes"]
 got=Counter(map(tuple,json.loads((root/"bounded"/"states.json").read_text())["current"]))
 want=Counter(map(tuple,oracle[d]))
 assert got==want and all(n==1 for n in got.values()),(got,want)
 stats=[m["resident"] for m in r["segment_memory_by_rank"]]
 assert all(s["hash_owner_execution"]=="parallel_per_shard" for s in stats),stats
 seen=sum(s["hash_extent_full_seen"] for s in stats)
 retries=sum(s["hash_freeze_retry"] for s in stats)
 checks=sum(s["hash_extent_full_recheck"] for s in stats)
 confirmed=sum(s["hash_extent_full_confirmed"] for s in stats)
 assert seen>=checks>=confirmed>0,(seen,retries,checks,confirmed)
 pids=[w.process.pid for w in pool.workers]
 print("PRESSURE_COUNTS",seen,retries,checks,confirmed,flush=True)
 limit=os.environ.pop("MGBFS_SEGMENT_TEST_LIMIT")
 try:result=run(graph,root/"healthy-next")
 finally:os.environ["MGBFS_SEGMENT_TEST_LIMIT"]=limit
 assert result["status"]=="COMPLETE" and result["layer_sizes"]==[len(v) for v in oracle],result
 assert pids==[w.process.pid for w in pool.workers]
 print("PASS pool pressure",r["layer_sizes"],"seen",seen,"freeze_retries",retries,
       "rechecks",checks,"confirmed",confirmed,"workers",pids,flush=True)
(root/"acceptance.json").write_text(json.dumps({"world":world,"prefix":r["layer_sizes"],
 "extent_full_seen":seen,"freeze_retries":retries,"rechecks":checks,
 "confirmed":confirmed,"workers":pids},indent=2))
print("EVIDENCE",root,flush=True)
