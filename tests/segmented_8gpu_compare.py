import json,os,time,tempfile,hashlib
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

def graph_and_oracle(n):
 generators=[]
 for i in range(n-1):
  g=list(range(n));g[i],g[i+1]=g[i+1],g[i];generators.append(g)
 layers=[1]
 for degree in range(1,n+1):
  nxt=[0]*(len(layers)+degree-1)
  for i,v in enumerate(layers):
   for j in range(degree):nxt[i+j]+=v
  layers=nxt
 return GraphDefinition.permutation(generators,list(range(n))),layers

os.environ['MGBFS_SEGMENT_ROWS']='1048576'
os.environ['MGBFS_SEGMENT_OWNER_LANES']='8'
os.environ.pop('MGBFS_SEGMENT_SHARD_TARGET_ROWS',None)
root=Path(tempfile.mkdtemp(prefix='segmented32-8x3090-',dir='/workspace'))
records=[]
with persistent_graph_workers():
 for n in (8,13):
  graph,oracle=graph_and_oracle(n)
  for mode in ('segmented','permutation_bitmap','legacy'):
   start=time.monotonic();print('START',n,mode,flush=True)
   result=run_graph(graph,root/f'{n}-{mode}',devices=list(range(8)),autotune=False,memory_backend=mode,shards=8,_batch=(512 if n==8 else 8192),segment_rows=1<<32)
   assert result['status']=='COMPLETE',(n,mode,result)
   assert result['layer_sizes']==oracle,(n,mode,result['layer_sizes'],oracle)
   assert result['devices']==list(range(8)),result['devices']
   assert len(result['rank_worker_pids'])==8,result['rank_worker_pids']
   rec={'n':n,'mode':mode,'status':result['status'],'backend':result.get('backend'),'bfs_seconds':result['bfs_seconds'],'wall_seconds':time.monotonic()-start,'states':sum(oracle),'layer_sizes':result['layer_sizes'],'shards':8,'batch':(512 if n==8 else 8192),'segmented_lanes':8,'segment_rows':1048576,'config_max':1<<32,'devices':result['devices'],'workers':result['rank_worker_pids'],'segment_memory_by_rank':result.get('segment_memory_by_rank')}
   if mode!='legacy':
    assert len(rec['segment_memory_by_rank'])==8
    assert all(x['configured_segment_rows']==1<<32 for x in rec['segment_memory_by_rank'])
   records.append(rec);(root/'summary.json').write_text(json.dumps(records,indent=2))
   print(json.dumps({'n':n,'mode':mode,'bfs_seconds':rec['bfs_seconds'],'wall_seconds':rec['wall_seconds'],'states':rec['states'],'layers':len(oracle)}),flush=True)
print('EVIDENCE',root,flush=True)
