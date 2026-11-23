"""Physical multi-GPU full-layer oracle; opt-in, not part of the CPU suite."""
import itertools,json,os,tempfile
from collections import Counter
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
from multigpubfs.generic_session import persistent_graph_workers

world=int(os.environ.get("RESIDENT_TEST_WORLD","2"))
tmp=os.environ.get("MGBFS_TEST_TMP_ROOT")
root=Path(tempfile.mkdtemp(prefix=f"resident-native-layers-{world}-",dir=tmp))
devices=list(range(world))
scheduler=os.environ.get("RESIDENT_TEST_SCHEDULER","single")
batch_options={"device_scheduler":"batch","device_batch_edges":33,"device_receive_records":7} if scheduler=="batch" else {}
def rotation(k):return [(i+k)%6 for i in range(6)]
cases=[
 ("s4",GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],[0,1,2,3])),
 ("directed-c6",GraphDefinition.permutation([rotation(1),rotation(4)],list(range(6)))),
 ("repeated",GraphDefinition.permutation([[1,0,2,3,4],[0,2,1,3,4],[0,1,3,2,4],[0,1,2,4,3]],[0,0,1,2,3])),
]
dup=("duplicate_at_full",GraphDefinition.permutation([[1,0,2],[1,0,2],[0,1,2]],[0,1,2]))
def rank_cap(g,layers):
 domain=sorted(set(itertools.permutations(g.start)))
 idx={state:i for i,state in enumerate(domain)}
 owner=[i for i in range(world)]
 root_rank=idx[tuple(g.start)]%world
 owner[0],owner[root_rank]=owner[root_rank],owner[0]
 dist=[[0]*world for _ in layers]
 for d,layer in enumerate(layers):
  for state in layer:dist[d][owner[idx[tuple(state)]%world]]+=1
 return max(max(row) for row in dist),dist
def check_packet(r):
 assert all(m["resident"]["scheduler"]==("multi_cta" if scheduler=="batch" else "single_cta") for m in r["segment_memory_by_rank"])
 expected=int(os.environ.get("MGBFS_RESIDENT_PACKET_RECORDS","1"))
 expected_lanes=int(os.environ.get("MGBFS_RESIDENT_LANES","4"))
 expected_slots=int(os.environ.get("MGBFS_RESIDENT_MAILBOX_SLOTS","1"))
 assert [m["resident"]["packet_records"] for m in r["segment_memory_by_rank"]]==[expected]*world
 assert [m["resident"]["lanes"] for m in r["segment_memory_by_rank"]]==[expected_lanes]*world
 assert [m["resident"]["slots_per_ring"] for m in r["segment_memory_by_rank"]]==[expected_slots]*world

def checkstate(out,layer):
 actual=json.loads((out/"states.json").read_text())["current"]
 got=Counter(tuple(v) for v in actual);want=Counter(tuple(v) for v in layer)
 assert got==want and all(v==1 for v in got.values()),(out,got,want)

pids=None;seen_shards=0;records=[]
with persistent_graph_workers() as pool:
 for name,g in cases:
  layers=g.exact_layers(1000)
  for depth in range(len(layers)):
   out=root/f"{name}-depth-{depth}"
   r=run_graph(g,out,devices=devices,autotune=False,shards=2,_batch=2,segment_rows=2,
      memory_backend="permutation_bitmap",segment_strategy="atomic",execution_engine="device",executable=os.environ["MGBFS_EXECUTABLE"],_profile_layers=depth,**batch_options)
   assert r["execution_engine"]=="GPU_RESIDENT_EXPERIMENTAL",r
   check_packet(r)
   assert r["layer_sizes"]==[len(x) for x in layers[:depth+1]],(name,depth,r["layer_sizes"])
   if depth+1<len(layers):assert (r["status"],r["reason"])==("INCOMPLETE","PROFILE_LAYER_LIMIT"),r
   else:assert r["status"]=="COMPLETE",r
   checkstate(out,layers[depth])
   current=[w.process.pid for w in pool.workers]
   if pids is None:pids=current
   assert pids==current,(pids,current)
   seen_shards=max(seen_shards,*(x["resident"]["max_active_shards_first64"] for x in r["segment_memory_by_rank"]))
  needed,dist=rank_cap(g,layers)
  for cap in [needed,max(1,needed-1)]:
   first=next((d for d,row in enumerate(dist) if max(row)>cap),None)
   out=root/f"{name}-cap-{cap}"
   r=run_graph(g,out,devices=devices,autotune=False,shards=2,_batch=2,segment_rows=2,capacity=cap,
      memory_backend="permutation_bitmap",segment_strategy="atomic",execution_engine="device",executable=os.environ["MGBFS_EXECUTABLE"],**batch_options)
   check_packet(r)
   done=len(layers)-1 if first is None else first-1
   assert r["layer_sizes"]==[len(x) for x in layers[:done+1]],(name,cap,first,dist,r["layer_sizes"])
   assert r["status"]==("COMPLETE" if first is None else "INCOMPLETE"),(name,cap,r["status"])
   checkstate(out,layers[done])
   records.append({"case":name,"cap":cap,"status":r["status"],"completed":done,"first_incomplete":first,
       "peak_shards":max(x["resident"]["max_active_shards_first64"] for x in r["segment_memory_by_rank"])})
  print("PASS full-layer oracle",name,"layers",len(layers),"needed",needed,flush=True)
 name,g=dup;layers=g.exact_layers(1000);out=root/name
 r=run_graph(g,out,devices=devices,autotune=False,shards=2,_batch=2,segment_rows=2,capacity=1,
   memory_backend="permutation_bitmap",segment_strategy="atomic",execution_engine="device",executable=os.environ["MGBFS_EXECUTABLE"],**batch_options)
 check_packet(r)
 assert r["status"]=="COMPLETE" and r["layer_sizes"]==[len(x) for x in layers],r
 checkstate(out,layers[-1])
 duplicate_full=sum(x["resident"]["duplicates_at_rank_cap"] for x in r["segment_memory_by_rank"])
 assert duplicate_full>0,duplicate_full
 assert pids==[w.process.pid for w in pool.workers]
 assert seen_shards>=2,seen_shards
 records.append({"case":name,"duplicate_full":duplicate_full})
print("PASS native exact persistent layer multisets",world,root,pids,flush=True)
(root/"acceptance.json").write_text(json.dumps({"devices":devices,"workers":pids,"cases":records,"max_active_shards":seen_shards},indent=2))
