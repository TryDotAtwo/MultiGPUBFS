"""Matched physical 2-GPU Coxeter comparison of segmented hash and bitmap."""
import json,os,tempfile,time
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

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

os.environ['MGBFS_SEGMENT_ROWS']='1048576'
os.environ['MGBFS_SEGMENT_OWNER_LANES']='8'
os.environ.pop('MGBFS_SEGMENT_SHARD_TARGET_ROWS',None)
root=Path(tempfile.mkdtemp(prefix='segment-modes-s11-s12-',dir='/workspace'))
records=[]
with persistent_graph_workers():
    i=0
    for n,modes in ((11,('segmented24','permutation_bitmap','permutation_bitmap','segmented24')),
                    (12,('permutation_bitmap','segmented24','segmented24','permutation_bitmap'))):
        graph,oracle=graph_and_oracle(n)
        for backend in modes:
            t=time.monotonic()
            result=run_graph(graph,root/str(i),devices=[0,1],autotune=False,
                             memory_backend=backend,shards=8,_batch=8192,segment_rows=1<<32)
            assert result['status']=='COMPLETE',(n,backend,result)
            assert result['layer_sizes']==oracle,(n,backend,result['layer_sizes'],oracle)
            r=dict(n=n,backend=backend,shards=8,lanes=8,batch=8192,
                   forced_rows_per_bank=1048576,config_max=1<<32,
                   status=result['status'],bfs_seconds=result['bfs_seconds'],
                   wall_seconds=time.monotonic()-t,states=sum(oracle),
                   segment_memory_by_rank=result['segment_memory_by_rank'])
            records.append(r); print(json.dumps(r),flush=True)
            (root/'summary.json').write_text(json.dumps(records,indent=2))
            i+=1
print('EVIDENCE',root,flush=True)
