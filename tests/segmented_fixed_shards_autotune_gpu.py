"""Exercise fixed-shard tuning, matched exchange modes and graph correctness on GPUs."""
import json,os,tempfile,time
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
devices=list(range(int(os.environ.get('MGBFS_TEST_DEVICE_COUNT','2'))))
shards=int(os.environ.get('MGBFS_TEST_SHARDS','4'))
n=10;gens=[]
for i in range(n-1):
 g=list(range(n));g[i],g[i+1]=g[i+1],g[i];gens.append(g)
graph=GraphDefinition.permutation(gens,list(range(n)));oracle=[1]
for degree in range(1,n+1):
 nxt=[0]*(len(oracle)+degree-1)
 for j,x in enumerate(oracle):
  for k in range(degree):nxt[j+k]+=x
 oracle=nxt
root=Path(tempfile.mkdtemp(prefix='fixed-shard-tune-',dir='/workspace'))
os.environ['MGBFS_PROFILE_CACHE']=str(root/'cache');os.environ['MGBFS_SEGMENT_ROWS']='65536'
os.environ.pop('MGBFS_SEGMENT_EXACT_EXCHANGE',None)
with persistent_graph_workers():
 for backend in ('segmented24','permutation_bitmap'):
  t=time.monotonic()
  result=run_graph(graph,root/backend,devices=devices,shards=shards,autotune=True,
                   memory_backend=backend,executable=os.environ['MGBFS_EXECUTABLE'])
  profile=result['autotune']
  assert result['status']=='COMPLETE' and result['layer_sizes']==oracle,result
  assert profile['status']=='MEASURED_EQUAL_PREFIX_GPU_PROFILE',profile
  assert profile['shards']==shards and all(x['shards']==shards for x in profile['pilots']),profile
  assert len(profile['pilots'])>=2 and len({(x['batch'],x['segment_owner_lanes']) for x in profile['pilots']})>=2,profile
  measured=profile['pilots']+profile.get('excluded_prefix_candidates',[])
  assert any(a['segment_exact_exchange'] is False and b['segment_exact_exchange'] is True and (a['shards'],a['batch'],a['segment_owner_lanes'])==(b['shards'],b['batch'],b['segment_owner_lanes']) for a in measured for b in measured),profile
  assert all(x['exact_exchange']==profile['segment_exact_exchange'] for x in result['segment_memory_by_rank']),profile
  print(json.dumps({'backend':backend,'status':result['status'],'shards':profile['shards'],
   'pilots':len(profile['pilots']),'tuning_seconds':profile['seconds'],
   'bfs_seconds':result['bfs_seconds'],'wall_seconds':time.monotonic()-t}),flush=True)
print('FIXED_SHARD_TEST_PASS',root,flush=True)
