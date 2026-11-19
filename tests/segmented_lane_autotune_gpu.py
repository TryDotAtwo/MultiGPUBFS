"""Two-GPU measured lane selection and per-backend persistent profile reuse."""
import json,os,tempfile,time
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
n=11;gens=[]
for i in range(n-1):
    x=list(range(n));x[i],x[i+1]=x[i+1],x[i];gens.append(x)
graph=GraphDefinition.permutation(gens,list(range(n)))
oracle=[1]
for degree in range(1,n+1):
    nxt=[0]*(len(oracle)+degree-1)
    for i,v in enumerate(oracle):
        for j in range(degree):nxt[i+j]+=v
    oracle=nxt
root=Path(tempfile.mkdtemp(prefix='seg24-autotune-',dir='/workspace'))
os.environ['MGBFS_PROFILE_CACHE']=str(root/'profile-cache')
os.environ['MGBFS_SEGMENT_ROWS']='1048576'
native=os.environ['MGBFS_EXECUTABLE']
with persistent_graph_workers() as pool:
    first_pids=None
    for backend in ('segmented24','permutation_bitmap'):
        for run in range(2):
            start=time.monotonic()
            result=run_graph(graph,root/f'{backend}-{run}',devices=[0,1],capacity=None,executable=native,autotune=True,memory_backend=backend)
            profile=result['autotune']
            assert result['status']=='COMPLETE' and result['layer_sizes']==oracle,result
            assert profile['status']=='MEASURED_EQUAL_PREFIX_GPU_PROFILE',profile
            assert len(profile['pilots'])>=2 and profile['common_depth']>=4,profile
            requested=profile['segment_owner_lanes']
            actual=[x['owner_lanes'] for x in result['segment_memory_by_rank']]
            assert all(0<x<=requested for x in actual),(requested,actual)
            assert all('segment_owner_lanes' in p and len(p['segment_owner_lanes_by_rank'])==2 for p in profile['pilots'])
            assert profile.get('cache_hit') is (run==1),profile
            pids=result['rank_worker_pids']
            if first_pids is None:first_pids=pids
            assert pids==first_pids,(pids,first_pids)
            print(json.dumps(dict(backend=backend,run=run,status=result['status'],cache_hit=profile.get('cache_hit'),shards=profile['shards'],lanes_requested=requested,lanes_by_rank=actual,batch=profile['batch'],pilot_count=len(profile['pilots']),common_depth=profile['common_depth'],tuning_seconds=profile['seconds'],bfs_seconds=result['bfs_seconds'],wall_seconds=time.monotonic()-start)),flush=True)
print('EVIDENCE',root,flush=True)
