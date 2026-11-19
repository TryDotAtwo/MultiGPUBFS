"""Explicit 8-rank acceptance: exact small sets, substantial layers, persistent reuse."""
import json,os,tempfile,time
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

devices=list(range(8));native=os.environ['MGBFS_EXECUTABLE']
root=Path(tempfile.mkdtemp(prefix='seg24-eight-',dir='/workspace'))
os.environ['MGBFS_PROFILE_CACHE']=str(root/'profiles')
small=[
 ('s4',GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],list(range(4)))),
 ('repeated',GraphDefinition.permutation([[1,0,2,3,4],[0,2,1,3,4],[0,1,3,2,4],[0,1,2,4,3]],[0,0,1,2,3])),
 ('directed',GraphDefinition.permutation([[1,2,3,0]],list(range(4)))),
 ('wide',GraphDefinition.permutation([list(range(1,25))+[0]],[0]*24+[1])),
]
matrix=GraphDefinition.matrix(2,1,[([-1,0,0,1],0),([1,0,0,-1],0)],[-(1<<40),2])

def graph(n):
    gens=[]
    for i in range(n-1):
        a=list(range(n));a[i],a[i+1]=a[i+1],a[i];gens.append(a)
    return GraphDefinition.permutation(gens,list(range(n)))
def mahonian(n):
    x=[1]
    for i in range(1,n+1):
        y=[0]*(len(x)+i-1)
        for j,v in enumerate(x):
            for k in range(i):y[j+k]+=v
        x=y
    return x

env=dict(os.environ,MGBFS_SEGMENT_ROWS='3',MGBFS_SEGMENT_OWNER_LANES='8',MGBFS_GENERIC_GENERATOR='cuda',MGBFS_GENERIC_HISTORY='hash')
with persistent_graph_workers() as pool:
    pid=None
    for mode in ('segmented24','permutation_bitmap'):
        for name,g in small+([('matrix',matrix)] if mode=='segmented24' else []):
            out=root/(mode+'-'+name)
            report=run_graph(g,out,devices=devices,shards=8,autotune=False,memory_backend=mode,executable=native,max_seconds=120,_native_env=env)
            expected=g.exact_layers(1000)
            assert report['status']=='COMPLETE' and report['layer_sizes']==list(map(len,expected)),report
            parts=[json.loads((out/f'rank-{i}'/'report.json').read_text()) for i in devices]
            states=[json.loads((out/f'rank-{i}'/'states.json').read_text()) for i in devices]
            assert len(parts)==8 and all(x['layer_sizes']==report['layer_sizes'] for x in parts)
            assert sorted(sum((x['current'] for x in states),[]))==expected[-1]
            assert sorted(sum((x['previous_small'] for x in states),[]))==expected[-2]
            now=report['rank_worker_pids']
            if pid is None:pid=now
            assert now==pid and len(pid)==8,(now,pid)
            print('PASS eight',mode,name,'layers',len(expected),'states',sum(map(len,expected)),flush=True)
        out=root/(mode+'-capacity1')
        report=run_graph(graph(6),out,devices=devices,shards=8,autotune=False,memory_backend=mode,capacity=1,executable=native,max_seconds=120,_native_env=env)
        assert report['status']=='INCOMPLETE' and report['reason']=='RESOURCE_10',report
        parts=[json.loads((out/f'rank-{i}'/'report.json').read_text()) for i in devices]
        assert all(x['layer_sizes']==report['layer_sizes'] for x in parts),parts
        print('PASS eight',mode,'collective capacity stop at',len(report['layer_sizes'])-1,flush=True)
    env['MGBFS_SEGMENT_ROWS']='65536'
    g=graph(10);expected=mahonian(10)
    for mode in ('legacy','segmented24','permutation_bitmap'):
        out=root/(mode+'-s10')
        t=time.monotonic()
        report=run_graph(g,out,devices=devices,shards=8,autotune=False,memory_backend=mode,executable=native,max_seconds=600,_batch=8192,_native_env=env)
        assert report['status']=='COMPLETE' and report['layer_sizes']==expected,report
        assert len(report['rank_worker_pids'])==8 and report['rank_worker_pids']==pid
        if mode!='legacy':assert len(report['segment_memory_by_rank'])==8,report
        print(json.dumps(dict(mode=mode,states=sum(expected),layers=len(expected),bfs_seconds=report['bfs_seconds'],wall_seconds=time.monotonic()-t)),flush=True)
print('EVIDENCE',root,flush=True)
