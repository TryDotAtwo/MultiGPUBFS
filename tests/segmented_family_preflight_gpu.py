import os,json,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
root=Path(tempfile.mkdtemp(prefix='family-preflight-switch-'));native=os.environ['MGBFS_EXECUTABLE']
g=GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],[0,1,2,3]);layers=g.exact_layers(100)
with persistent_graph_workers() as pool:
 last_ids=None
 for i,mode in enumerate(('legacy','segmented24','permutation_bitmap','legacy')):
  args=dict(devices=[0,1],shards=2,autotune=False,_batch=4,memory_backend=mode,executable=native)
  if mode!='legacy':args.update(segment_rows=3,segment_strategy='atomic')
  out=root/f'{i}-{mode}';r=run_graph(g,out,**args);c=r['communication_preflight'];ids=[w.process.pid for w in pool.workers]
  assert r['status']=='COMPLETE' and r['layer_sizes']==list(map(len,layers)),r
  assert c['status']=='VERIFIED',c
  if i in (1,3):assert not c.get('cache_hit') and not c.get('transport_cache_hit') and ids!=last_ids,(mode,ids,last_ids,c)
  if i==2:assert c.get('transport_cache_hit') and ids==last_ids,(mode,ids,last_ids,c)
  print('PASS_FAMILY_PREFLIGHT',mode,ids,json.dumps({'cache_hit':c.get('cache_hit'),'transport_cache_hit':c.get('transport_cache_hit'),'seconds':c.get('seconds')}),flush=True);last_ids=ids
print('FAMILY_PREFLIGHT_PASS',root,flush=True)
