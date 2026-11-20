import json,os,tempfile,time
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

def graph_and_oracle(n):
 gens=[]
 for i in range(n-1):
  g=list(range(n));g[i],g[i+1]=g[i+1],g[i];gens.append(g)
 layers=[1]
 for degree in range(1,n+1):
  nxt=[0]*(len(layers)+degree-1)
  for i,v in enumerate(layers):
   for j in range(degree):nxt[i+j]+=v
  layers=nxt
 return GraphDefinition.permutation(gens,list(range(n))),layers

for key in ('MGBFS_SEGMENT_ROWS','MGBFS_SEGMENT_OWNER_LANES','MGBFS_SEGMENT_SHARD_TARGET_ROWS'):
 os.environ.pop(key,None)
root=Path(tempfile.mkdtemp(prefix='segmented-tuned-',dir='/workspace'))
os.environ['MGBFS_PROFILE_CACHE']=str(root/'profiles')
records=[]
with persistent_graph_workers():
 for n in (11,12):
  graph,oracle=graph_and_oracle(n)
  workers=None
  for rep in range(2):
   t=time.monotonic()
   result=run_graph(graph,root/f'{n}-{rep}',devices=[0,1],autotune=True,memory_backend='segmented',segment_rows=1<<32)
   assert result['status']=='COMPLETE',(n,rep,result)
   assert result['layer_sizes']==oracle,(n,rep)
   if workers is None: workers=result['rank_worker_pids']
   assert result['rank_worker_pids']==workers,(n,rep,result['rank_worker_pids'],workers)
   record={'n':n,'repeat':rep,'status':result['status'],'bfs_seconds':result['bfs_seconds'],'wall_seconds':time.monotonic()-t,'states':sum(oracle),'plan':result['plan'],'segment_memory_by_rank':result['segment_memory_by_rank'],'autotune':result.get('autotune'),'workers':workers}
   records.append(record)
   (root/'summary.json').write_text(json.dumps(records,indent=2))
   p=record['autotune'] or {}
   print(json.dumps({'n':n,'repeat':rep,'bfs_seconds':record['bfs_seconds'],'wall_seconds':record['wall_seconds'],'shards':record['plan']['shards'],'profile_status':p.get('status'),'measured':p.get('measured'),'tuning_seconds':p.get('seconds'),'owner_lanes':[x['owner_lanes'] for x in record['segment_memory_by_rank']]}),flush=True)
print('EVIDENCE',root,flush=True)
