import sys,os,time,json,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
def graph_and_oracle(n):
 gens=[]
 for i in range(n-1):
  g=list(range(n));g[i],g[i+1]=g[i+1],g[i];gens.append(g)
 oracle=[1]
 for degree in range(1,n+1):
  nxt=[0]*(len(oracle)+degree-1)
  for i,v in enumerate(oracle):
   for j in range(degree):nxt[i+j]+=v
  oracle=nxt
 return GraphDefinition.permutation(gens,list(range(n))),oracle
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
os.environ.pop('MGBFS_SEGMENT_ROWS',None)
os.environ.pop('MGBFS_SEGMENT_OWNER_LANES',None)
os.environ.pop('MGBFS_SEGMENT_SHARD_TARGET_ROWS',None)
root=Path(tempfile.mkdtemp(prefix='legacy-segmented-auto-',dir='/workspace'))
records=[]
with persistent_graph_workers():
 for n,modes in ((11,('legacy','segmented24','segmented24','legacy')),(12,('segmented24','legacy','legacy','segmented24'))):
  if os.environ.get('MGBFS_BENCH_N') and os.environ['MGBFS_BENCH_N'] != str(n): continue
  graph,oracle=graph_and_oracle(n)
  for mode in modes:
   started=time.monotonic()
   result=run_graph(graph,root/str(len(records)),devices=[0,1],autotune=False,memory_backend=mode,shards=None,_batch=8192,segment_rows=1<<32)
   assert result['status']=='COMPLETE',(n,mode,result)
   assert result['layer_sizes']==oracle,(n,mode,result['layer_sizes'])
   rec={'n':n,'mode':mode,'status':result['status'],'bfs_seconds':result['bfs_seconds'],'wall_seconds':time.monotonic()-started,'states':sum(oracle),'shards':result['plan']['shards'],'batch':8192,'segment_geometry':result.get('segment_memory_by_rank'),'config_max':1<<32}
   records.append(rec)
   (root/'summary.json').write_text(json.dumps(records,indent=2))
   print(json.dumps(rec),flush=True)
print('EVIDENCE',root,flush=True)
