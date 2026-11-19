"""One-GPU segmented exactness against independent layers."""
import json,os,subprocess,tempfile,time
from pathlib import Path
from multigpubfs import GraphDefinition

root=Path(tempfile.mkdtemp(prefix='seg24-one-gpu-',dir='/workspace'))
native=os.environ['MGBFS_EXECUTABLE']
env=dict(os.environ,MGBFS_EXPERIMENTAL_SEGMENTED24='1',MGBFS_EXPERIMENTAL_BITMAP='0',
         MGBFS_SEGMENT_ROWS='3',MGBFS_GENERIC_GENERATOR='cuda',MGBFS_GENERIC_HISTORY='hash')
def swap(n,i):
    a=list(range(n));a[i],a[i+1]=a[i+1],a[i];return a

def run(n,world,large_error=None):
    name='one-gpu-s6' if not large_error else large_error.lower()
    case=root/name;case.mkdir()
    g=GraphDefinition.permutation([swap(n,i) for i in range(n-1)] if n<7 else [swap(n,0)],list(range(n)))
    definition=case/'definition.json';definition.write_text(g.to_json())
    dev=','.join(map(str,range(world)))
    query=subprocess.run([native,'graph-info',str(definition),dev,'4','1024','4'],env=env,text=True,capture_output=True,timeout=60,check=True)
    plan=json.loads(query.stdout);plan['owner_cuts']=None
    config=case/'launch.json';config.write_text(json.dumps(plan))
    files=[];jobs=[]
    try:
        for rank in range(world):
            f=(case/f'rank-{rank}.log').open('w');files.append(f)
            jobs.append(subprocess.Popen([native,'graph-rank',str(definition),str(config),str(rank),str(case/'id'),str(case/f'rank-{rank}'),'0'],env=env,stdout=f,stderr=subprocess.STDOUT))
        deadline=time.monotonic()+60
        while any(p.poll() is None for p in jobs) and time.monotonic()<deadline:time.sleep(.05)
        assert all(p.poll() is not None for p in jobs),('workers hung',[p.poll() for p in jobs])
        for f in files:f.flush()
        logs='\n'.join((case/f'rank-{i}.log').read_text() for i in range(world))
        if large_error:
            assert all(p.returncode!=0 for p in jobs),(large_error,[p.returncode for p in jobs],logs)
            assert large_error in logs,(large_error,logs[-8000:])
            print('PASS expected rejection',large_error,'rank codes',[p.returncode for p in jobs],flush=True)
        else:
            assert all(p.returncode==0 for p in jobs),logs[-8000:]
            report=json.loads((case/'rank-0/report.json').read_text())
            states=json.loads((case/'rank-0/states.json').read_text())
            expected=g.exact_layers(1000)
            assert report['status']=='COMPLETE' and report['backend']=='EXPERIMENTAL_SEGMENTED24'
            assert report['layer_sizes']==list(map(len,expected))
            assert sorted(states['current'])==expected[-1]
            print('PASS exact segmented BFS 1 GPU',n,'states',sum(map(len,expected)),flush=True)
    finally:
        for p in jobs:
            if p.poll() is None:p.terminate()
        for p in jobs:
            if p.poll() is None:
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        for f in files:f.close()

run(6,1)
print('EVIDENCE',root)
