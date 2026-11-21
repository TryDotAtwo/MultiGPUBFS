"""Public CayleyPy API and persistent switching of the optional bitmap map."""
import os,tempfile
from pathlib import Path
from cayleypy import CayleyGraph,PermutationGroups
import multigpubfs
from multigpubfs.cayleypy_backend import NativeOptions
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.graph_definition import from_cayleypy
root=Path(tempfile.mkdtemp(prefix='bitmap-owner-api-',dir='/workspace'))
base=NativeOptions(output_root=root,devices=(0,1),autotune=False,memory_backend='permutation_bitmap')
multigpubfs.enable_native(base,recovery=False)
try:
 with persistent_graph_workers() as pool:
  original=None
  for page in (64,0,8):
   graph=CayleyGraph(PermutationGroups.coxeter(5),device='cpu')
   opts=NativeOptions(output_root=root,devices=(0,1),autotune=False,memory_backend='permutation_bitmap',bitmap_owner_page=page)
   result=graph.bfs(native_options=opts);expected=from_cayleypy(graph).exact_layers(1000)
   assert result.bfs_completed and result.layer_sizes==list(map(len,expected)),result.report
   assert all(x['bitmap_owner_page']==page for x in result.report['segment_memory_by_rank'])
   ids=[w.process.pid for w in pool.workers]
   if original is None:original=ids
   assert ids==original
   print('PASS CayleyPy owner page',page,ids,flush=True)
finally:multigpubfs.disable_native()
print('BITMAP_OWNER_PAGE_CAYLEYPY_PASS',root,flush=True)

# The mapping is also exercised on all eight physical ranks, with shuffled
# devices, exact and padded transport, all three owner strategies and a real
# collective resource stop before another persistent job.
import json
from multigpubfs import GraphDefinition
from multigpubfs.launch import run_graph
generators=[]
for i in range(4):
 a=list(range(5));a[i],a[i+1]=a[i+1],a[i];generators.append(a)
graph=GraphDefinition.permutation(generators,list(range(5)));layers=graph.exact_layers(1000)
devices=[7,3,1,5,0,6,2,4];ids=None
with persistent_graph_workers() as pool:
 for strategy in ('atomic','warp','batch'):
  for page in (8,0):
   for exact in (True,False):
    env=dict(os.environ,MGBFS_SEGMENT_TEST_BATCH='1',MGBFS_SEGMENT_EXACT_EXCHANGE=str(int(exact)),
      MGBFS_SEGMENT_BITMAP_OWNER_PAGE='0')
    out=root/f'eight-{strategy}-{page}-{int(exact)}'
    r=run_graph(graph,out,devices=devices,shards=8,autotune=False,memory_backend='permutation_bitmap',
      segment_strategy=strategy,segment_rows=3,bitmap_owner_page=page,_batch=16,executable=os.environ['MGBFS_EXECUTABLE'],_native_env=env)
    assert r['status']=='COMPLETE' and r['layer_sizes']==list(map(len,layers)),r
    state=json.loads((out/'states.json').read_text());assert sorted(state['current'])==layers[-1] and sorted(state['previous_small'])==layers[-2]
    assert all(x['bitmap_owner_page']==page for x in r['segment_memory_by_rank'])
    now=[w.process.pid for w in pool.workers]
    if ids is None:ids=now
    assert ids==now
    print('PASS eight shuffled',strategy,page,'exact' if exact else 'padded',flush=True)
 out=root/'eight-capacity-stop'
 r=run_graph(graph,out,devices=devices,shards=8,autotune=False,memory_backend='permutation_bitmap',
    segment_rows=3,bitmap_owner_page=8,_batch=16,capacity=1,executable=os.environ['MGBFS_EXECUTABLE'])
 assert r['status']=='INCOMPLETE' and r['reason']=='RESOURCE_10',r
 out=root/'eight-after-capacity-stop'
 r=run_graph(graph,out,devices=devices,shards=8,autotune=False,memory_backend='permutation_bitmap',
    segment_rows=3,bitmap_owner_page=0,_batch=16,executable=os.environ['MGBFS_EXECUTABLE'])
 assert r['status']=='COMPLETE' and r['layer_sizes']==list(map(len,layers))
 assert [w.process.pid for w in pool.workers]==ids
 print('PASS eight capacity stop and next persistent job',flush=True)
print('BITMAP_OWNER_PAGE_EIGHT_GPU_PASS',root,flush=True)
