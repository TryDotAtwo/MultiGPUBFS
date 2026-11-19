"""A large parked legacy pool must be available to a later segmented job."""
import json,os,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.launch import run_graph
from multigpubfs.generic_session import persistent_graph_workers
root=Path(tempfile.mkdtemp(prefix='seg24-pool-',dir='/workspace'));native=os.environ['MGBFS_EXECUTABLE']
g=GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],list(range(4)))
env=dict(os.environ,MGBFS_SEGMENT_ROWS='8',MGBFS_GENERIC_GENERATOR='cuda',MGBFS_GENERIC_HISTORY='hash',MGBFS_GENERIC_TRANSPORT='full',MGBFS_GENERIC_SORT='none')
with persistent_graph_workers() as pool:
 old=run_graph(g,root/'legacy',devices=[0,1],capacity=2_000_000,shards=4,autotune=False,executable=native,memory_backend='legacy',_native_env=env)
 assert old['status']=='COMPLETE';ids=[w.process.pid for w in pool.workers]
 inv=pool.inventory(native,[0,1],env,60);cached=[x['cached_reusable_bytes']+x['pool_reusable_bytes'] for x in inv]
 assert min(cached)>128*1024*1024,cached
 new=run_graph(g,root/'segmented',devices=[0,1],shards=4,autotune=False,executable=native,memory_backend='segmented24',_native_env=env)
 assert new['status']=='COMPLETE' and new['layer_sizes']==old['layer_sizes']
 assert [w.process.pid for w in pool.workers]==ids
 parts=[json.loads((root/'segmented'/f'rank-{i}'/'report.json').read_text())['segment_memory'] for i in range(2)]
 for i,m in enumerate(parts):
  assert m['pool_idle_before_trim_bytes']>128*1024*1024,(i,m,cached)
  assert m['pool_idle_before_trim_bytes']-m['pool_idle_after_trim_bytes']>128*1024*1024,(i,m,cached)
  print('PASS pool trim rank',i,'released',m['pool_idle_before_trim_bytes']-m['pool_idle_after_trim_bytes'],'budget',m['storage_budget_bytes'],flush=True)
print('EVIDENCE',root)
