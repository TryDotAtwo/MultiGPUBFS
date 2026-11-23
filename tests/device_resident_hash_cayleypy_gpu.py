"""Opt-in public CayleyPy resident hash + persistent acceptance on physical GPUs."""
import json,os,tempfile
from collections import Counter
from pathlib import Path
from cayleypy import CayleyGraph,CayleyGraphDef
import multigpubfs
from multigpubfs import NativeOptions

world=int(os.environ.get("RESIDENT_TEST_WORLD","2"))
scheduler=os.environ.get("RESIDENT_TEST_SCHEDULER","single")
batch_options={"device_scheduler":"batch","device_batch_edges":33,"device_receive_records":7} if scheduler=="batch" else {}
root=Path(tempfile.mkdtemp(prefix=f"resident-hash-cayley-{world}-",dir=os.environ.get("MGBFS_TEST_TMP_ROOT")))
opts=NativeOptions(devices=tuple(range(world)),autotune=False,memory_backend="segmented24",
                   segment_strategy="atomic",segment_rows=2,execution_engine="device",**batch_options)
cases=[
 ("s4",CayleyGraphDef.create([[1,0,2,3],[0,2,1,3],[0,1,3,2]])),
 ("repeated",CayleyGraphDef.create([[1,0,2,3,4],[0,2,1,3,4],[0,1,3,2,4],[0,1,2,4,3]],central_state=[0,0,1,2,3])),
 ("duplicate-full",CayleyGraphDef.create([[1,0,2],[1,0,2],[0,1,2]])),
]
def verify(r,layers,completed):
 assert r.report["execution_engine"]=="GPU_RESIDENT_EXPERIMENTAL"
 assert len(r.report["segment_memory_by_rank"])==world
 assert all(m["resident"]["storage"]=="segmented_hash" for m in r.report["segment_memory_by_rank"])
 assert all(m["resident"]["hash_packed_states"] is (scheduler=="batch") for m in r.report["segment_memory_by_rank"])
 assert all(m["resident"]["scheduler"]==("multi_cta" if scheduler=="batch" else "single_cta") for m in r.report["segment_memory_by_rank"])
 indexed=os.environ.get("MGBFS_RESIDENT_HASH_PARENT_INDEX","0")=="1"
 assert all(m["resident"]["hash_parent_index_enabled"] is indexed for m in r.report["segment_memory_by_rank"])
 assert all(m["resident"]["hash_parent_index_bytes"]==((m["resident"]["extent_count"]+1)*8 if indexed else 0) for m in r.report["segment_memory_by_rank"])
 assert all(m["resident"]["slots_per_ring"]==int(os.environ.get("MGBFS_RESIDENT_MAILBOX_SLOTS","1")) for m in r.report["segment_memory_by_rank"])
 assert r.layer_sizes==list(map(len,layers[:len(r.layer_sizes)])),r.layer_sizes
 assert r.bfs_completed is completed,r.report
 got=Counter(map(tuple,r.last_layer().tolist()))
 want=Counter(map(tuple,layers[len(r.layer_sizes)-1]))
 assert got==want and all(v==1 for v in got.values()),(got,want)

records=[];pids=None
try:
 multigpubfs.enable_native(opts,recovery=False)
 with multigpubfs.persistent_graph_workers() as pool:
  for name,definition in cases:
   graph=CayleyGraph(definition,device="cpu")
   oracle=multigpubfs.from_cayleypy(graph).exact_layers(1000)
   if name=="duplicate-full":os.environ["MGBFS_SEGMENT_MAX_RANK_STATES"]="1"
   try:result=graph.bfs(output=root/name)
   finally:
    if name=="duplicate-full":del os.environ["MGBFS_SEGMENT_MAX_RANK_STATES"]
   verify(result,oracle,True)
   now=[w.process.pid for w in pool.workers]
   if pids is None:pids=now
   assert pids==now,(pids,now)
   if name=="duplicate-full":
    count=sum(m["resident"]["duplicates_at_rank_cap"] for m in result.report["segment_memory_by_rank"])
    assert count>0,count
   else:count=None
   records.append({"case":name,"status":"COMPLETE","sizes":result.layer_sizes,"duplicates_at_full":count})
   print("PASS CayleyPy",name,"layers",result.layer_sizes,"pids",pids,flush=True)
  graph=CayleyGraph(cases[0][1],device="cpu")
  oracle=multigpubfs.from_cayleypy(graph).exact_layers(1000)
  os.environ["MGBFS_SEGMENT_MAX_RANK_STATES"]="1"
  try:result=graph.bfs(output=root/"resource-cap")
  finally:del os.environ["MGBFS_SEGMENT_MAX_RANK_STATES"]
  verify(result,oracle,False)
  assert result.report["reason"]=="RESIDENT_RESOURCE_CAPACITY",result.report
  assert len(result.layer_sizes)<len(oracle)
  assert pids==[w.process.pid for w in pool.workers]
  records.append({"case":"resource-cap","status":"INCOMPLETE","sizes":result.layer_sizes})
  print("PASS CayleyPy resource prefix",result.layer_sizes,flush=True)
  result=graph.bfs(output=root/"healthy-after-cap")
  verify(result,oracle,True)
  assert pids==[w.process.pid for w in pool.workers]
  print("PASS CayleyPy healthy next job",pids,flush=True)
  graph=CayleyGraph(cases[0][1],device="cpu")
  oracle=multigpubfs.from_cayleypy(graph).exact_layers(1000)
  for i,slots in enumerate(("1","2","4","1")):
   os.environ["MGBFS_RESIDENT_MAILBOX_SLOTS"]=slots
   result=graph.bfs(output=root/f"slots-switch-{i}")
   verify(result,oracle,True)
   assert pids==[w.process.pid for w in pool.workers]
   print("PASS CayleyPy persistent slots",slots,pids,flush=True)
  for i,indexed in enumerate(("0","1","0")):
   os.environ["MGBFS_RESIDENT_HASH_PARENT_INDEX"]=indexed
   result=graph.bfs(output=root/f"parent-index-switch-{i}")
   verify(result,oracle,True)
   assert pids==[w.process.pid for w in pool.workers]
   print("PASS CayleyPy persistent parent index",indexed,pids,flush=True)
finally:multigpubfs.disable_native()
(root/"acceptance.json").write_text(json.dumps({"world":world,"workers":pids,"records":records},indent=2))
print("EVIDENCE",root,flush=True)
