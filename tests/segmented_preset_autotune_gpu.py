"""Measured profile and reuse on two GPUs; S9 only, not a B200 capacity test."""
import json,os,tempfile,time
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
n=9;gens=[]
for i in range(n-1):
 x=list(range(n));x[i],x[i+1]=x[i+1],x[i];gens.append(x)
graph=GraphDefinition.permutation(gens,list(range(n)))
oracle=[1]
for degree in range(1,n+1):
 nxt=[0]*(len(oracle)+degree-1)
 for i,v in enumerate(oracle):
  for j in range(degree):nxt[i+j]+=v
 oracle=nxt
root=Path(tempfile.mkdtemp(prefix='seg-presets-autotune-',dir='/workspace'))
os.environ['MGBFS_PROFILE_CACHE']=str(root/'profile-cache')
native=os.environ['MGBFS_EXECUTABLE'];records=[];pids=None
with persistent_graph_workers():
 for backend in ('segmented24','permutation_bitmap'):
  for preset in ('balanced','max_capacity'):
   for repeat in range(2):
    started=time.monotonic()
    result=run_graph(graph,root/f'{backend}-{preset}-{repeat}',devices=[0,1],shards=4,executable=native,autotune=True,memory_backend=backend,memory_preset=preset)
    profile=result['autotune'];geometry=result['segment_memory_by_rank']
    assert result['status']=='COMPLETE' and result['layer_sizes']==oracle,(backend,preset,result)
    assert profile['status']=='MEASURED_EQUAL_PREFIX_GPU_PROFILE',profile
    assert profile['memory_preset']==preset and len(profile['pilots'])>=2,profile
    assert profile.get('cache_hit') is (repeat==1),profile
    if preset=='max_capacity':assert all(type(p.get('capacity_score_rows')) is int for p in profile['pilots']),profile
    if backend=='permutation_bitmap':assert all(g['banks']==(2 if preset=='max_capacity' else 3) for g in geometry),geometry
    if pids is None:pids=result['rank_worker_pids']
    assert result['rank_worker_pids']==pids,(result['rank_worker_pids'],pids)
    rec={'backend':backend,'preset':preset,'repeat':repeat,'status':profile['status'],'cache_hit':profile['cache_hit'],'pilots':len(profile['pilots']),'common_depth':profile['common_depth'],'score_rows':[p.get('capacity_score_rows') for p in profile['pilots']],'tuning_seconds':profile['seconds'],'bfs_seconds':result['bfs_seconds'],'wall_seconds':time.monotonic()-started,'pids':pids}
    records.append(rec);print(json.dumps(rec),flush=True)
(root/'evidence.json').write_text(json.dumps(records,indent=2))
print('PRESET_AUTOTUNE_TWO_GPU_PASS',root,flush=True)
