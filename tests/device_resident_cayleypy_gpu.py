"""Opt-in CayleyPy API acceptance on physical GPUs and persistent workers."""
import itertools, json, os, tempfile
from collections import Counter
from pathlib import Path
from cayleypy import CayleyGraph, CayleyGraphDef, PermutationGroups
import multigpubfs
from multigpubfs import NativeOptions

world=int(os.environ.get("RESIDENT_TEST_WORLD","2"))
root=Path(tempfile.mkdtemp(prefix=f"resident-cayley-{world}-",dir=os.environ.get("MGBFS_TEST_TMP_ROOT")))
devices=tuple(range(world))
opts=NativeOptions(devices=devices,autotune=False,memory_backend="permutation_bitmap",
                   segment_strategy="atomic",segment_rows=2,execution_engine="device")
def rotation(k):return [(i+k)%6 for i in range(6)]
cases=[
 ("s4",CayleyGraphDef.create([[1,0,2,3],[0,2,1,3],[0,1,3,2]])),
 ("directed-c6",CayleyGraphDef.create([rotation(1),rotation(4)])),
 ("repeated",CayleyGraphDef.create([[1,0,2,3,4],[0,2,1,3,4],[0,1,3,2,4],[0,1,2,4,3]],central_state=[0,0,1,2,3])),
 ("lrx3",PermutationGroups.lrx(3)),
]
def rank_caps(g,layers):
 domain=sorted(set(itertools.permutations(g.start)))
 ids={state:i for i,state in enumerate(domain)}
 mapping=list(range(world));owner=ids[tuple(g.start)]%world
 mapping[0],mapping[owner]=mapping[owner],mapping[0]
 return [[sum(mapping[ids[tuple(s)]%world]==rank for s in layer) for rank in range(world)] for layer in layers]
def verify(result,layers,completed=True):
 assert result.report["execution_engine"]=="GPU_RESIDENT_EXPERIMENTAL",result.report
 expected=int(os.environ.get("MGBFS_RESIDENT_PACKET_RECORDS","1"))
 expected_lanes=int(os.environ.get("MGBFS_RESIDENT_LANES","4"))
 expected_slots=int(os.environ.get("MGBFS_RESIDENT_MAILBOX_SLOTS","1"))
 assert [m["resident"]["packet_records"] for m in result.report["segment_memory_by_rank"]]==[expected]*world
 assert [m["resident"]["lanes"] for m in result.report["segment_memory_by_rank"]]==[expected_lanes]*world
 assert [m["resident"]["slots_per_ring"] for m in result.report["segment_memory_by_rank"]]==[expected_slots]*world
 assert result.layer_sizes==list(map(len,layers[:len(result.layer_sizes)])),result.layer_sizes
 assert result.bfs_completed is completed,result.report
 last=result.last_layer().tolist()
 assert Counter(map(tuple,last))==Counter(map(tuple,layers[len(result.layer_sizes)-1]))
 assert len(last)==len({tuple(v) for v in last})
pids=None;records=[]
try:
 multigpubfs.enable_native(opts,recovery=False)
 with multigpubfs.persistent_graph_workers() as pool:
  for name,definition in cases:
   graph=CayleyGraph(definition,device="cpu")
   oracle=multigpubfs.from_cayleypy(graph).exact_layers(1000)
   result=graph.bfs(output=root/name)
   verify(result,oracle)
   now=[w.process.pid for w in pool.workers]
   if pids is None:pids=now
   assert now==pids
   records.append({"case":name,"status":result.report["status"],"layer_sizes":result.layer_sizes,
    "last_layer_full":True,"active_shards":max(m["resident"]["max_active_shards_first64"] for m in result.report["segment_memory_by_rank"])})
   print("PASS CayleyPy",name,"layers",result.layer_sizes,"workers",pids,flush=True)
  name,definition=cases[0]
  graph=CayleyGraph(definition,device="cpu")
  g=multigpubfs.from_cayleypy(graph);oracle=g.exact_layers(1000)
  dist=rank_caps(g,oracle);cap=1
  first=next(d for d,counts in enumerate(dist) if max(counts)>cap)
  os.environ["MGBFS_SEGMENT_MAX_RANK_STATES"]=str(cap)
  try: result=graph.bfs(output=root/"resource-cap")
  finally:del os.environ["MGBFS_SEGMENT_MAX_RANK_STATES"]
  verify(result,oracle,completed=False)
  assert result.report["reason"]=="RESIDENT_RESOURCE_CAPACITY",result.report
  assert result.completed_depth==first-1,(result.completed_depth,first,dist)
  assert pids==[w.process.pid for w in pool.workers]
  records.append({"case":"resource-cap","cap":cap,"first_incomplete":first,"layer_sizes":result.layer_sizes,"last_layer_full":True})
  print("PASS CayleyPy resource cap",result.layer_sizes,"first incomplete",first,flush=True)
  definition=CayleyGraphDef.create([[1,0,2],[1,0,2],[0,1,2]])
  graph=CayleyGraph(definition,device="cpu")
  oracle=multigpubfs.from_cayleypy(graph).exact_layers(1000)
  os.environ["MGBFS_SEGMENT_MAX_RANK_STATES"]="1"
  try:result=graph.bfs(output=root/"duplicate-full")
  finally:del os.environ["MGBFS_SEGMENT_MAX_RANK_STATES"]
  verify(result,oracle)
  duplicates=sum(m["resident"]["duplicates_at_rank_cap"] for m in result.report["segment_memory_by_rank"])
  assert duplicates>0,duplicates
  assert pids==[w.process.pid for w in pool.workers]
  records.append({"case":"duplicate-at-full","duplicates":duplicates,"layer_sizes":result.layer_sizes})
  print("PASS CayleyPy duplicate at cap",duplicates,flush=True)
  graph=CayleyGraph(cases[0][1],device="cpu")
  oracle=multigpubfs.from_cayleypy(graph).exact_layers(1000)
  for i,slots in enumerate(("1","2","4","1")):
   os.environ["MGBFS_RESIDENT_MAILBOX_SLOTS"]=slots
   result=graph.bfs(output=root/f"slots-switch-{i}")
   verify(result,oracle,True)
   assert pids==[w.process.pid for w in pool.workers]
   print("PASS CayleyPy persistent slots",slots,pids,flush=True)
finally:
 multigpubfs.disable_native()
(root/"acceptance.json").write_text(json.dumps({"world":world,"devices":devices,"workers":pids,"cases":records},indent=2))
print("PASS CayleyPy persistent",world,"EVIDENCE",root,flush=True)
