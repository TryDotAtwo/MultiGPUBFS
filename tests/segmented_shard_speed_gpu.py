"""Matched physical GPU shard-count comparison for one exact S12 graph."""
import json,os,tempfile,time
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
n=12; gens=[]
for i in range(n-1):
 g=list(range(n));g[i],g[i+1]=g[i+1],g[i];gens.append(g)
graph=GraphDefinition.permutation(gens,list(range(n)))
oracle=[1]
for degree in range(1,n+1):
 nxt=[0]*(len(oracle)+degree-1)
 for i,v in enumerate(oracle):
  for j in range(degree):nxt[i+j]+=v
 oracle=nxt
os.environ['MGBFS_SEGMENT_ROWS']='1048576'
os.environ['MGBFS_SEGMENT_OWNER_LANES']='8'
os.environ.pop('MGBFS_SEGMENT_SHARD_TARGET_ROWS',None)
root=Path(tempfile.mkdtemp(prefix='shard-comparison-s12-',dir='/workspace'))
records=[]
with persistent_graph_workers() as pool:
 for index,shards in enumerate((8,32,32,8)):
  t=time.monotonic()
  result=run_graph(graph,root/str(index),devices=[0,1],autotune=False,memory_backend='segmented24',shards=shards,_batch=8192)
  assert result['status']=='COMPLETE' and result['layer_sizes']==oracle,result
  r=dict(shards=shards,lanes=8,batch=8192,status=result['status'],bfs_seconds=result['bfs_seconds'],
         wall_seconds=time.monotonic()-t,segments=[v['segment_count'] for v in result['segment_memory_by_rank']])
  records.append(r);print(json.dumps(r),flush=True)
(root/'summary.json').write_text(json.dumps(records,indent=2))
print('EVIDENCE',root,flush=True)
