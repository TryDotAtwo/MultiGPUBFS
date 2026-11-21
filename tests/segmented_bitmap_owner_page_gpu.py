"""Page ownership: exact layers, persistent resets, and rank/ABI agreement."""
import json,os,subprocess,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
root=Path(tempfile.mkdtemp(prefix='bitmap-owner-page-',dir='/workspace'))
native=os.environ['MGBFS_EXECUTABLE']
def coxeter(n):
 gens=[]
 for i in range(n-1):
  a=list(range(n));a[i],a[i+1]=a[i+1],a[i];gens.append(a)
 return gens
s5=GraphDefinition.permutation(coxeter(5),list(range(5)))
repeat=GraphDefinition.permutation(coxeter(5),[0,0,1,2,3])
directed=GraphDefinition.permutation([[(i+1)%6 for i in range(6)],[(i+4)%6 for i in range(6)]],list(range(6)))
lrx=GraphDefinition.permutation([[1,2,3,4,0],[4,0,1,2,3],[1,0,2,3,4]],[0,1,2,3,3])
cases=[('s5',s5),('repeat',repeat),('directed',directed),('lrx',lrx)]
sequence=[('atomic',0),('warp',8),('batch',8),('atomic',64),('batch',0),('warp',1),('atomic',256)]
ids=None
with persistent_graph_workers() as pool:
 for attempt,(strategy,page) in enumerate(sequence):
  for name,graph in cases:
   layers=graph.exact_layers(1000)
   for depth in range(len(layers)):
    out=root/f'{attempt}-{strategy}-{page}-{name}-{depth}'
    env=dict(os.environ,MGBFS_SEGMENT_BITMAP_OWNER_PAGE=str(page),MGBFS_SEGMENT_TEST_BATCH='1',MGBFS_SEGMENT_BATCH_TILE='2')
    report=run_graph(graph,out,devices=[0,1],autotune=False,shards=4,_batch=2,memory_backend='permutation_bitmap',
          segment_rows=3,segment_strategy=strategy,executable=native,_profile_layers=depth,_native_env=env)
    assert report['status']=='INCOMPLETE' and report['reason']=='PROFILE_LAYER_LIMIT',report
    assert report['layer_sizes']==list(map(len,layers[:depth+1])),(name,strategy,page,depth,report)
    saved=json.loads((out/'states.json').read_text())
    assert sorted(saved['current'])==layers[depth],(name,strategy,page,depth)
    assert len(saved['current'])==len({tuple(x) for x in saved['current']})
    if depth:assert sorted(saved['previous_small'])==layers[depth-1]
    assert all(x['bitmap_owner_page']==page for x in report['segment_memory_by_rank'])
    now=[w.process.pid for w in pool.workers]
    if ids is None:ids=now
    assert ids==now
   print('PASS owner page exact',name,strategy,page,'layers',len(layers),flush=True)
config=root/'0-atomic-0-s5-1'/'launch.json'
definition=root/'graph.json';definition.write_text(s5.to_json())
for name,settings,oldlib,expected in [
 ('rank-mismatch',('8','64'),False,'BITMAP_OWNER_PAGE_RANK_MISMATCH'),
 ('invalid',('3','3'),False,'BITMAP_OWNER_PAGE_INVALID'),
 ('old-bundle',('8','8'),True,'BITMAP_OWNER_PAGE_NATIVE_ABI_MISSING')]:
 parent=root/name;parent.mkdir();ps=[]
 try:
  for rank,page in enumerate(settings):
   env=dict(os.environ,MGBFS_SEGMENT_BITMAP_OWNER_PAGE=page,MGBFS_SEGMENT_STRATEGY='atomic',
            MGBFS_EXPERIMENTAL_SEGMENTED24='1',MGBFS_EXPERIMENTAL_BITMAP='1',RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='2')
   if oldlib:env['LD_LIBRARY_PATH']='/workspace/libs/fast:/workspace/assets/common/lib'
   ps.append(subprocess.Popen([native,'graph-rank',str(definition),str(config),str(rank),str(parent/'id'),str(parent/f'rank-{rank}'),'0'],
       env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
  for rank,p in enumerate(ps):
   out,err=p.communicate(timeout=75);(parent/f'rank-{rank}.log').write_text(out+'\n'+err)
   assert p.returncode!=0 and expected in out+err,(name,rank,p.returncode,out,err)
 finally:
  for p in ps:
   if p.poll() is None:p.kill();p.wait()
 print('PASS owner page',name,expected,flush=True)
print('BITMAP_OWNER_PAGE_EXACT_GPU_PASS',root,flush=True)
