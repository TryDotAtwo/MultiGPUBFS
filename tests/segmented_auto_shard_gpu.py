"""Physical two-rank check of automatic shard geometry and exact BFS."""
import json,os,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

n=8
gens=[]
for i in range(n-1):
    g=list(range(n));g[i],g[i+1]=g[i+1],g[i];gens.append(g)
graph=GraphDefinition.permutation(gens,list(range(n)))
oracle=[1]
for degree in range(1,n+1):
    nxt=[0]*(len(oracle)+degree-1)
    for i,v in enumerate(oracle):
        for j in range(degree):nxt[i+j]+=v
    oracle=nxt
os.environ['MGBFS_SEGMENT_SHARD_TARGET_ROWS']='128'
os.environ['MGBFS_SEGMENT_ROWS']='1024'
root=Path(tempfile.mkdtemp(prefix='shard-sizing-',dir='/workspace'))
records=[]
with persistent_graph_workers() as pool:
    pids=None
    for backend in ('segmented24','permutation_bitmap'):
        for case,manual in [('auto',None),('manual',4)]:
            result=run_graph(graph,root/(backend+'-'+case),devices=[0,1],autotune=False,
                             memory_backend=backend,shards=manual)
            assert result['status']=='COMPLETE',result
            assert result['layer_sizes']==oracle,(backend,case,result['layer_sizes'],oracle)
            assert len(result['rank_worker_pids'])==2
            if pids is None:pids=result['rank_worker_pids']
            assert result['rank_worker_pids']==pids
            if manual is None:
                geometry=result['shard_geometry']
                assert geometry['admitted_shards']>=32,geometry
                assert result['plan']['shards']==geometry['admitted_shards']
            else:
                assert result['plan']['shards']==4
                assert 'shard_geometry' not in result
            records.append({'backend':backend,'case':case,'shards':result['plan']['shards'],
                            'bfs_seconds':result['bfs_seconds'],
                            'segment_counts':[p['segment_count'] for p in result['segment_memory_by_rank']]})
            print(json.dumps(records[-1]),flush=True)
(root/'evidence.json').write_text(json.dumps(records,indent=2))
print('EVIDENCE',root,flush=True)
