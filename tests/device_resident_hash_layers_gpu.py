"""Opt-in physical GPU oracle for experimental device-resident segmented hash.

Run twice: normal hashes and MGBFS_SEGMENT_TEST_HASH_BITS=0 (with
MGBFS_SEGMENT_TEST_BATCH=1). Each layer is compared as a Counter of complete
states from every rank, not just a count or a set.
"""
import json, os, tempfile
from collections import Counter
from pathlib import Path

from multigpubfs import GraphDefinition, run_graph
from multigpubfs.generic_session import persistent_graph_workers

world=int(os.environ.get("RESIDENT_TEST_WORLD","2"))
devices=list(range(world))
root=Path(tempfile.mkdtemp(prefix=f"resident-hash-layers-{world}-",
                           dir=os.environ.get("MGBFS_TEST_TMP_ROOT")))
collisions=os.environ.get("MGBFS_SEGMENT_TEST_HASH_BITS")=="0"
cases=[
 ("s4",GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],[0,1,2,3])),
 ("repeated",GraphDefinition.permutation([[1,0,2,3,4],[0,2,1,3,4],[0,1,3,2,4],[0,1,2,4,3]],[0,0,1,2,3])),
 ("wide",GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],[1000,-5,7,1000])),
]
duplicate=("duplicate_at_full",GraphDefinition.permutation([[1,0,2],[1,0,2],[0,1,2]],[0,1,2]))

def execute(g,path,**kw):
 return run_graph(g,path,devices=devices,autotune=False,shards=2,_batch=2,segment_rows=2,
   memory_backend="segmented24",segment_strategy="atomic",execution_engine="device",
   executable=os.environ["MGBFS_EXECUTABLE"],**kw)

def rank_counters(path):
 states=[]
 for rank in range(world):
  data=json.loads((path/f"rank-{rank}"/"states.json").read_text())
  states.append(Counter(tuple(row) for row in data["current"]))
 return states

def assert_layer(path,want):
 ranked=rank_counters(path)
 got=sum(ranked,Counter())
 expected=Counter(tuple(v) for v in want)
 assert got==expected and all(n==1 for n in got.values()),(str(path),got,expected)
 return [sum(c.values()) for c in ranked]

def check_receipt(r):
 assert r["execution_engine"]=="GPU_RESIDENT_EXPERIMENTAL",r
 assert len(r["segment_memory_by_rank"])==world
 for m in r["segment_memory_by_rank"]:
  assert m["resident"]["storage"]=="segmented_hash"
  assert m["resident"]["transport"]=="NCCL_LSA_DEVICE_RING"
  assert m["resident"]["hash_owner_execution"]=="serialized_per_rank"

records=[];workers=None;seen_shards=0
with persistent_graph_workers() as pool:
 for name,g in cases:
  layers=g.exact_layers(1000)
  per_rank=[]
  for d,layer in enumerate(layers):
   path=root/f"{name}-depth-{d}"
   r=execute(g,path,_profile_layers=d)
   check_receipt(r)
   assert r["layer_sizes"]==list(map(len,layers[:d+1])),(name,d,r["layer_sizes"])
   assert r["status"]==("COMPLETE" if d==len(layers)-1 else "INCOMPLETE"),(name,d,r)
   if d<len(layers)-1:assert r["reason"]=="PROFILE_LAYER_LIMIT",(name,d,r["reason"])
   per_rank.append(assert_layer(path,layer))
   pids=[w.process.pid for w in pool.workers]
   if workers is None:workers=pids
   assert pids==workers,(workers,pids)
   seen_shards=max(seen_shards,*(m["resident"]["max_active_shards_first64"] for m in r["segment_memory_by_rank"]))
  need=max(max(row) for row in per_rank)
  for cap in sorted({need,max(1,need-1)}):
   first=next((d for d,row in enumerate(per_rank) if max(row)>cap),None)
   d=len(layers)-1 if first is None else first-1
   path=root/f"{name}-cap-{cap}"
   r=execute(g,path,capacity=cap)
   check_receipt(r)
   assert r["layer_sizes"]==list(map(len,layers[:d+1])),(name,cap,first,per_rank,r["layer_sizes"])
   assert r["status"]==("COMPLETE" if first is None else "INCOMPLETE"),(name,cap,r["status"])
   assert_layer(path,layers[d])
   assert [w.process.pid for w in pool.workers]==workers
   records.append({"case":name,"cap":cap,"status":r["status"],"last_depth":d,"first_incomplete":first,"per_rank":per_rank})
  print("PASS",name,"layers",len(layers),"cap",need,"forced_collision",collisions,flush=True)
 name,g=duplicate;layers=g.exact_layers(1000);path=root/name
 r=execute(g,path,capacity=1);check_receipt(r)
 assert r["status"]=="COMPLETE" and r["layer_sizes"]==list(map(len,layers)),r
 assert_layer(path,layers[-1])
 full=sum(m["resident"]["duplicates_at_rank_cap"] for m in r["segment_memory_by_rank"])
 assert full>0,full
 assert [w.process.pid for w in pool.workers]==workers
 if not collisions:assert seen_shards>=2,seen_shards
 records.append({"case":name,"duplicates_at_full":full})
print("PASS exact resident hash layers",world,root,"workers",workers,flush=True)
(root/"acceptance.json").write_text(json.dumps({"world":world,"forced_collision":collisions,
 "workers":workers,"max_active_shards":seen_shards,"records":records},indent=2))
