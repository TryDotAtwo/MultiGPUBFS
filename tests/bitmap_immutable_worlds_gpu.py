"""Per-layer bitmap multiset oracle across real non-cyclic owner layouts."""
import json, os, tempfile
from pathlib import Path
from collections import Counter
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

def independent(start,gens):
 current={tuple(start)};seen=set(current);layers=[]
 while current:
  layers.append(current)
  nxt={tuple(state[i] for i in g) for state in current for g in gens}-seen
  seen.update(nxt);current=nxt
 return layers

start=list(range(6));gens=[]
for i in range(5):
 g=list(range(6));g[i],g[i+1]=g[i+1],g[i];gens.append(g)
layers=independent(start,gens);graph=GraphDefinition.permutation(gens,start)
root=Path(tempfile.mkdtemp(prefix='bitmap-immutable-worlds-',dir='/workspace'))
native=os.environ['MGBFS_EXECUTABLE']
for world in (2,3,5,7,8):
 devices=[7,1,5,0,6,2,4,3][:world]
 with persistent_graph_workers() as pool:
  pids=None
  for strategy,page in [('atomic',64),('batch',8),('warp',0)]:
   if page:assert 0<720-(720%(page*world))<=720
   for depth in range(len(layers)):
    out=root/f'{world}-{strategy}-{page}-{depth}'
    env=dict(os.environ,MGBFS_SEGMENT_BITMAP_OWNER_PAGE=str(page),MGBFS_SEGMENT_TEST_BATCH='1',MGBFS_SEGMENT_BATCH_TILE='2')
    r=run_graph(graph,out,devices=devices,shards=4,autotune=False,_batch=8,memory_backend='permutation_bitmap',segment_rows=64,segment_strategy=strategy,executable=native,_native_env=env,_profile_layers=depth)
    assert r['status']=='INCOMPLETE' and r['reason']=='PROFILE_LAYER_LIMIT',(world,strategy,page,depth,r)
    assert r['layer_sizes']==list(map(len,layers[:depth+1]))
    data=json.loads((out/'states.json').read_text())
    assert Counter(tuple(x) for x in data['current'])==Counter({s:1 for s in layers[depth]}),(world,strategy,page,depth)
    if depth:assert Counter(tuple(x) for x in data['previous_small'])==Counter({s:1 for s in layers[depth-1]})
    rankinfo=r['segment_memory_by_rank'];assert sum(x['bitmap_owner_rows'] for x in rankinfo)==720
    assert all(x['bitmap_owner_page']==page and x['bitmap_domain']==720 for x in rankinfo)
    now=[w.process.pid for w in pool.workers]
    if pids is None:pids=now
    assert now==pids,(world,strategy,page)
   out=root/f'{world}-{strategy}-{page}-terminal'
   r=run_graph(graph,out,devices=devices,shards=4,autotune=False,_batch=8,memory_backend='permutation_bitmap',segment_rows=64,segment_strategy=strategy,executable=native,_native_env=env,_profile_layers=999)
   assert r['status']=='COMPLETE' and r['layer_sizes']==list(map(len,layers)),r
   print('PASS WORLD',world,strategy,'page',page,'boundary',720-(720%(page*world)) if page else 0,'layers',len(layers),'pids',pids,flush=True)
print('BITMAP_IMMUTABLE_WORLDS_PASS',root,flush=True)
