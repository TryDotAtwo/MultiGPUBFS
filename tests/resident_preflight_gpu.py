"""Real 8-GPU communication rehearsal reuse, failure isolation, and BFS."""
import json,os,tempfile,time
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
from multigpubfs.nccl_preflight import _cached_prefix
from multigpubfs.native_distribution import native_runtime
root=Path(tempfile.mkdtemp(prefix='resident-preflight-eight-',dir='/workspace'))
native=os.environ['MGBFS_EXECUTABLE'];devices=list(range(8));records=[]
def graph(n):
 g=[]
 for i in range(n-1):
  p=list(range(n));p[i],p[i+1]=p[i+1],p[i];g.append(p)
 return GraphDefinition.permutation(g,list(range(n)))
g1,g2=graph(5),graph(6)
def run(label,g,page,strategy):
 start=time.monotonic();out=root/label
 r=run_graph(g,out,devices=devices,shards=4,autotune=False,executable=native,memory_backend='permutation_bitmap',segment_rows=64,segment_strategy=strategy,bitmap_owner_page=page,_batch=16)
 elapsed=time.monotonic()-start
 oracle=g.exact_layers(1000)
 assert r['status']=='COMPLETE' and r['layer_sizes']==list(map(len,oracle)),(label,r)
 s=json.loads((out/'states.json').read_text())
 assert sorted(s['current'])==oracle[-1] and sorted(s['previous_small'])==oracle[-2]
 c=r['communication_preflight'];assert c['rehearsal_graph']==g.digest() and c['rehearsal_backend']=='permutation_bitmap',c
 assert c['bfs_prefix'] is not None,c
 records.append({'label':label,'elapsed':elapsed,'preflight_seconds':c['seconds'],'cache_hit':c['cache_hit'],'transport_cache_hit':c.get('transport_cache_hit'),'bfs_seconds':r['bfs_seconds'],'prefix':c['bfs_prefix']})
 print('PASS',label,json.dumps(records[-1]),flush=True)
 return c
with persistent_graph_workers() as pool:
 c1=run('first',g1,0,'atomic');ids=[w.process.pid for w in pool.workers];assert len(ids)==8
 c2=run('new-graph',g2,0,'atomic');assert c2['transport_cache_hit'] and c2['cache_source_graph']==g1.digest()
 assert [w.process.pid for w in pool.workers]==ids
 detail=json.loads((root/'new-graph-communication-evidence'/'cached-transport'/'prefix-execution.json').read_text())
 assert detail['mode']=='resident' and detail['status']=='VERIFIED',detail
 logs=[json.loads((root/'new-graph-communication-evidence'/'cached-transport'/'prefix'/f'rank-{i}.log').read_text().splitlines()[0]) for i in range(8)]
 assert [x['resident_pid'] for x in logs]==ids
 c3=run('repeat-graph',g2,0,'atomic');assert c3['cache_hit'] and c3['seconds']==0.0
 c4=run('new-mode',g2,8,'warp');assert c4['transport_cache_hit'];assert [w.process.pid for w in pool.workers]==ids
 # A cleanly retired pool is a new communicator epoch.  Even an identical
 # graph must perform a cold transport probe before its first new group.
 pool.close_workers()
 c5=run('new-generation',g2,8,'warp')
 assert not c5['cache_hit'] and not c5.get('transport_cache_hit',False),c5
 next_ids=[w.process.pid for w in pool.workers]
 assert next_ids!=ids and len(next_ids)==8
 # Force invalid mapping only inside a bounded diagnostic prefix. All ranks
 # must fail, the pool/cache must be discarded, and no result can be accepted.
 _,env=native_runtime(native)
 env.update(MGBFS_EXPERIMENTAL_SEGMENTED24='1',MGBFS_EXPERIMENTAL_BITMAP='1',MGBFS_GENERIC_HISTORY='hash',MGBFS_GENERIC_GENERATOR='cuda',MGBFS_GENERIC_TRANSPORT='full',MGBFS_GENERIC_SORT='none',MGBFS_SEGMENT_STRATEGY='atomic',MGBFS_SEGMENT_BITMAP_OWNER_PAGE='3')
 (root/'injected-failure').mkdir()
 try:
  _cached_prefix(root/'injected-failure',native,devices,env,{}, {'policy':{}},g1,'permutation_bitmap')
 except Exception as e:
  print('PASS expected resident prefix failure',repr(e),flush=True)
 else:raise AssertionError('invalid mapping accepted')
 assert not pool.workers and not pool.communication_profiles
 detail=json.loads((root/'injected-failure'/'cached-transport'/'prefix-execution.json').read_text());assert detail['status']=='FAILED',detail
 # An explicit new request after the failed diagnostic must be able to start
 # cleanly; no production BFS is restarted by the library.
 run('explicit-next-job',g1,0,'atomic')
(root/'summary.json').write_text(json.dumps({'records':records,'initial_pids':ids},indent=2))
print('RESIDENT_PREFLIGHT_EIGHT_GPU_PASS',root,flush=True)
