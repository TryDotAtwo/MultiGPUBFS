"""Compare physical GPU BFS multisets to an independent Python BFS oracle."""
import json,os,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
root=Path(tempfile.mkdtemp(prefix='bitmap-product-domain-',dir='/workspace'))
native=os.environ['MGBFS_EXECUTABLE']
cases=[
 ('coupled',[[1,0,3,2,4,5]],[-7,8,10,12,90,91],4),
 ('repeated',[[2,1,0,4,3],[1,0,2,3,4]],[1,1,2,3,3],3),
 ('connected',[[1,0,2],[0,2,1]],[2,0,1],6),
 ('fixed',[[1,0,2,3,4],[0,2,1,3,4]],[0,1,2,3,4],6),
 ('lrx-repeat',[[1,2,3,4,0],[4,0,1,2,3],[1,0,2,3,4]],[0,1,2,3,3],60),
 ('rapaport-m2-repeat',[[1,0,2,3,4],[1,0,3,2,4],[0,2,1,4,3]],[0,1,2,3,3],60),
]
def independent_layers(gens,start):
 current={tuple(start)};visited=set(current);layers=[]
 while current:
  layers.append([list(s) for s in sorted(current)])
  nxt={tuple(parent[i] for i in gen) for parent in current for gen in gens}-visited
  visited.update(nxt);current=nxt
 return layers

for world in (2,8):
 devices=[7,1,5,0,6,2,4,3][:world]
 with persistent_graph_workers() as pool:
  pids=None
  for strategy,page,preset in [('atomic',0,'balanced'),('warp',64,'max_capacity'),('batch',8,'balanced')]:
   for name,gens,start,domain in cases:
    graph=GraphDefinition.permutation(gens,start)
    layers=independent_layers(gens,start)
    for depth in range(len(layers)):
     out=root/f'{world}-{strategy}-{page}-{preset}-{name}-{depth}'
     env=dict(os.environ,MGBFS_SEGMENT_BITMAP_OWNER_PAGE=str(page),MGBFS_SEGMENT_TEST_BATCH='1',MGBFS_SEGMENT_BATCH_TILE='2')
     r=run_graph(graph,out,devices=devices,autotune=False,shards=4,_batch=2,memory_backend='permutation_bitmap',memory_preset=preset,segment_rows=3,segment_strategy=strategy,executable=native,_profile_layers=depth,_native_env=env)
     assert r['status']=='INCOMPLETE' and r['reason']=='PROFILE_LAYER_LIMIT',r
     assert r['layer_sizes']==list(map(len,layers[:depth+1]))
     saved=json.loads((out/'states.json').read_text());assert sorted(saved['current'])==layers[depth]
     if depth:assert sorted(saved['previous_small'])==layers[depth-1]
     for rr in r['segment_memory_by_rank']:
      assert rr['bitmap_domain']==domain,(name,rr['bitmap_domain'],domain)
      assert rr['bitmap_owner_page']==page
      assert rr['banks']==(2 if preset=='max_capacity' else 3)
      assert rr['bitmap_bytes']==max(8,((rr['bitmap_owner_rows']+63)//64)*8)
     now=[w.process.pid for w in pool.workers]
     if pids is None:pids=now
     assert pids==now
    print('PASS',world,strategy,page,preset,name,domain,flush=True)
print('BITMAP_PRODUCT_DOMAIN_ALL_PASS',root,flush=True)
