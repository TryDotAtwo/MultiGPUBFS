"""8 GPU S13: exact independent layer oracle and honest bounded outcome."""
import json,os,tempfile,time,math
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
n=13;gens=[]
for i in range(n-1):
 a=list(range(n));a[i],a[i+1]=a[i+1],a[i];gens.append(a)
g=GraphDefinition.permutation(gens,list(range(n)))
oracle=[1]
for j in range(1,n+1):
 y=[0]*(len(oracle)+j-1)
 for k,x in enumerate(oracle):
  for z in range(j):y[k+z]+=x
 oracle=y
assert sum(oracle)==math.factorial(n)
root=Path(tempfile.mkdtemp(prefix='seg24-eight-s13-',dir='/workspace'))
os.environ['MGBFS_SEGMENT_ROWS']='1048576'
os.environ['MGBFS_SEGMENT_OWNER_LANES']='8'
start=time.monotonic()
with persistent_graph_workers():
 r=run_graph(g,root/'result',devices=list(range(8)),memory_backend='segmented24',autotune=False,shards=8,_batch=8192,executable=os.environ['MGBFS_EXECUTABLE'],max_seconds=600)
assert r['layer_sizes']==oracle[:len(r['layer_sizes'])],(r['status'],r['reason'],r['layer_sizes'],oracle)
if r['status']=='COMPLETE':assert len(r['layer_sizes'])==len(oracle)
result=dict(status=r['status'],reason=r['reason'],layers=len(r['layer_sizes']),depth=len(r['layer_sizes'])-1,states=sum(r['layer_sizes']),bfs_seconds=r['bfs_seconds'],wall_seconds=time.monotonic()-start,segment_counts=[x['segment_count'] for x in r['segment_memory_by_rank']],rows_per_segment=[x['rows_per_bank'] for x in r['segment_memory_by_rank']],ranks=len(r['rank_worker_pids']),oracle='Mahonian inversion polynomial')
print(json.dumps(result),flush=True);print('EVIDENCE',root,flush=True)
