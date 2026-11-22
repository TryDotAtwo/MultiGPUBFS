"""Bounded physical search for an overlapping full-pool retry."""
import json,os,tempfile
from collections import Counter
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
from multigpubfs.generic_session import persistent_graph_workers
world=int(os.environ.get("RESIDENT_TEST_WORLD","4"))
n=6
gens=[]
for i in range(n-1):
 p=list(range(n));p[i],p[i+1]=p[i+1],p[i];gens.append(p)
g=GraphDefinition.permutation(gens,list(range(n)))
oracle=g.exact_layers(1000)
root=Path(tempfile.mkdtemp(prefix="resident-hash-contention-",dir=os.environ["MGBFS_TEST_TMP_ROOT"]))
records=[]
with persistent_graph_workers() as pool:
 for shards in (2,4,8):
  for rows in (1,2,4):
   for limit in (2,4,8,16):
    os.environ["MGBFS_SEGMENT_TEST_LIMIT"]=str(limit)
    out=root/("s%d-r%d-l%d"%(shards,rows,limit))
    r=run_graph(g,out,devices=list(range(world)),autotune=False,shards=shards,_batch=8,segment_rows=rows,
      memory_backend="segmented24",segment_strategy="atomic",execution_engine="device",executable=os.environ["MGBFS_EXECUTABLE"])
    assert r["status"] in ("COMPLETE","INCOMPLETE"),r
    d=len(r["layer_sizes"])-1
    assert r["layer_sizes"]==[len(v) for v in oracle[:d+1]],(shards,rows,limit,r["layer_sizes"])
    got=Counter(map(tuple,json.loads((out/"states.json").read_text())["current"]))
    assert got==Counter(map(tuple,oracle[d])),(shards,rows,limit,d)
    if r["status"]=="COMPLETE":assert d==len(oracle)-1
    else:assert r["reason"]=="RESIDENT_RESOURCE_CAPACITY",r["reason"]
    rr=[m["resident"] for m in r["segment_memory_by_rank"]]
    s=lambda k:sum(v[k] or 0 for v in rr)
    item={"shards":shards,"segment_rows":rows,"pool_limit":limit,"status":r["status"],"last_depth":d,
          "peak_owners":max(v["hash_owner_peak_active"] or 0 for v in rr),
          "extent_full_seen":s("hash_extent_full_seen"),"freeze_retry":s("hash_freeze_retry"),
          "recheck":s("hash_extent_full_recheck"),"confirmed":s("hash_extent_full_confirmed")}
    records.append(item)
    (root/"search.json").write_text(json.dumps(records,indent=2))
    print(json.dumps(item),flush=True)
    if item["freeze_retry"]:
     print("PASS physical lock-contention retry and exact completed prefix",root,flush=True)
     raise SystemExit(0)
print("NO_CONTENTION_RETRY_OBSERVED",root,flush=True)
