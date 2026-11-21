"""Exact owner-map BFS on odd GPU counts, including persistent job reuse."""
import json,os,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
root=Path(tempfile.mkdtemp(prefix='bitmap-owner-world-',dir='/workspace'))
native=os.environ['MGBFS_EXECUTABLE']
gens=[]
for i in range(3):
 a=list(range(4));a[i],a[i+1]=a[i+1],a[i];gens.append(a)
cases=[('distinct',GraphDefinition.permutation(gens,[0,1,2,3])),('repeat',GraphDefinition.permutation(gens,[0,0,1,2]))]
for world in (3,5,7):
 devices=[7,1,5,0,6,2,4][:world]
 with persistent_graph_workers() as pool:
  pids=None
  for strategy,page in [('atomic',64),('warp',8),('batch',64),('atomic',0)]:
   for name,graph in cases:
    layers=graph.exact_layers(1000)
    for depth in range(len(layers)):
     out=root/f'{world}-{strategy}-{page}-{name}-{depth}'
     env=dict(os.environ,MGBFS_SEGMENT_BITMAP_OWNER_PAGE=str(page),MGBFS_SEGMENT_TEST_BATCH='1',MGBFS_SEGMENT_BATCH_TILE='2')
     r=run_graph(graph,out,devices=devices,autotune=False,shards=4,_batch=2,memory_backend='permutation_bitmap',segment_rows=3,segment_strategy=strategy,executable=native,_profile_layers=depth,_native_env=env)
     assert r['status']=='INCOMPLETE' and r['reason']=='PROFILE_LAYER_LIMIT',r
     assert r['layer_sizes']==list(map(len,layers[:depth+1])),(world,strategy,page,name,depth)
     saved=json.loads((out/'states.json').read_text())
     assert sorted(saved['current'])==layers[depth],(world,strategy,page,name,depth)
     assert len(saved['current'])==len({tuple(x) for x in saved['current']})
     if depth:assert sorted(saved['previous_small'])==layers[depth-1]
     assert all(x['bitmap_owner_page']==page for x in r['segment_memory_by_rank'])
     now=[w.process.pid for w in pool.workers]
     if pids is None:pids=now
     assert pids==now
    print('PASS',world,strategy,page,name,len(layers),flush=True)
print('BITMAP_OWNER_PAGE_ODD_WORLDS_PASS',root,flush=True)
