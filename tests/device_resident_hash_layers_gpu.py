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
shards=int(os.environ.get("RESIDENT_TEST_SHARDS","2"))
devices=list(range(world))
root=Path(tempfile.mkdtemp(prefix=f"resident-hash-layers-{world}-",
                           dir=os.environ.get("MGBFS_TEST_TMP_ROOT")))
collisions=os.environ.get("MGBFS_SEGMENT_TEST_HASH_BITS")=="0"
scheduler=os.environ.get("RESIDENT_TEST_SCHEDULER","single")
batch_options={"device_scheduler":"batch","device_batch_edges":33,"device_receive_records":7} if scheduler=="batch" else {}
cases=[
 ("s4",GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],[0,1,2,3])),
 ("repeated",GraphDefinition.permutation([[1,0,2,3,4],[0,2,1,3,4],[0,1,3,2,4],[0,1,2,4,3]],[0,0,1,2,3])),
 ("wide",GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],[1000,-5,7,1000])),
]
duplicate=("duplicate_at_full",GraphDefinition.permutation([[1,0,2],[1,0,2],[0,1,2]],[0,1,2]))

def execute(g,path,**kw):
 return run_graph(g,path,devices=devices,autotune=False,shards=shards,_batch=2,segment_rows=2,
   memory_backend="segmented24",segment_strategy="atomic",execution_engine="device",
   executable=os.environ["MGBFS_EXECUTABLE"],**batch_options,**kw)

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
  indexed=os.environ.get("MGBFS_RESIDENT_HASH_PARENT_INDEX","0")=="1"
  assert m["resident"]["hash_parent_index_enabled"] is indexed
  assert m["resident"]["hash_parent_index_bytes"]==((m["resident"]["extent_count"]+1)*8 if indexed else 0)
  assert m["resident"]["slots_per_ring"]==int(os.environ.get("MGBFS_RESIDENT_MAILBOX_SLOTS","1"))
  assert m["resident"]["transport"]==("NCCL_LSA_DEVICE_BATCH" if scheduler=="batch" else "NCCL_LSA_DEVICE_RING")
  assert m["resident"]["scheduler"]==("multi_cta" if scheduler=="batch" else "single_cta")
  mode=("batched_parallel_per_shard" if scheduler=="batch" else
        "parallel_per_shard" if os.environ.get("MGBFS_RESIDENT_HASH_SHARD_OWNERS")=="1" else "serialized_per_rank")
  assert m["resident"]["hash_owner_execution"]==mode
  assert m["resident"]["hash_indexed_extents"]==(scheduler=="batch" or os.environ.get("MGBFS_RESIDENT_HASH_INDEXED_EXTENTS")=="1")

records=[];workers=None;seen_shards=0;peak_owners=0
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
   peak_owners=max(peak_owners,*(m["resident"]["hash_owner_peak_active"] or 0 for m in r["segment_memory_by_rank"]))
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
 if not collisions and os.environ.get("MGBFS_RESIDENT_HASH_SHARD_OWNERS")=="1" and os.environ.get("MGBFS_RESIDENT_HASH_OWNER_STATS")=="1":
  assert peak_owners>=2,("no measured simultaneous shard owners",peak_owners)
 records.append({"case":name,"duplicates_at_full":full})
print("PASS exact resident hash layers",world,root,"workers",workers,flush=True)
(root/"acceptance.json").write_text(json.dumps({"world":world,"forced_collision":collisions,
 "workers":workers,"shards":shards,"max_active_shards":seen_shards,"peak_concurrent_owners":peak_owners,"records":records},indent=2))
