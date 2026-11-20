"""Distributed exact-transport mode must agree across all participating ranks."""
import json, os, subprocess, tempfile, time
from pathlib import Path
from multigpubfs import GraphDefinition
native=os.environ['MGBFS_EXECUTABLE']
root=Path(tempfile.mkdtemp(prefix='segment-exchange-consensus-',dir='/workspace'))
n=5
gens=[]
for i in range(n-1):
    g=list(range(n));g[i],g[i+1]=g[i+1],g[i];gens.append(g)
graph=GraphDefinition.permutation(gens,list(range(n)))
definition=root/'graph.json';definition.write_text(graph.to_json())
base=dict(os.environ,MGBFS_EXPERIMENTAL_SEGMENTED24='1',MGBFS_EXPERIMENTAL_BITMAP='1',MGBFS_SEGMENT_ROWS='64',MGBFS_SEGMENT_OWNER_LANES='4')
out=subprocess.run([native,'graph-info',str(definition),'0,1','4','10000','128'],env=base,text=True,capture_output=True,check=True,timeout=60)
plan=json.loads(out.stdout);plan['owner_cuts']=None
for name,flags in [('mismatch',('1','0')),('consistent',('1','1'))]:
    case=root/name;case.mkdir();config=case/'launch.json';config.write_text(json.dumps(plan));jobs=[];files=[]
    try:
        for rank,flag in enumerate(flags):
            env=dict(base,MGBFS_SEGMENT_EXACT_EXCHANGE=flag)
            log=(case/f'rank-{rank}.log').open('w');files.append(log)
            jobs.append(subprocess.Popen([native,'graph-rank',str(definition),str(config),str(rank),str(case/'id'),str(case/f'rank-{rank}'),'0'],env=env,stdout=log,stderr=subprocess.STDOUT))
        deadline=time.monotonic()+45
        while any(p.poll() is None for p in jobs) and time.monotonic()<deadline: time.sleep(.05)
        assert all(p.poll() is not None for p in jobs),f'{name}: ranks did not exit'
        for f in files:f.flush()
        logs='\n'.join((case/f'rank-{i}.log').read_text() for i in range(2))
        if name=='mismatch':
            assert all(p.returncode!=0 for p in jobs),(name,[p.returncode for p in jobs],logs)
            assert 'SEGMENTED_EXCHANGE_MODE_MISMATCH' in logs,logs
        else:
            assert all(p.returncode==0 for p in jobs),(name,[p.returncode for p in jobs],logs)
            parts=[json.loads((case/f'rank-{i}'/'report.json').read_text()) for i in range(2)]
            oracle=list(map(len,graph.exact_layers(1000)))
            assert all(p['status']=='COMPLETE' and p['layer_sizes']==oracle for p in parts),parts
        print('PASS',name,flush=True)
    finally:
        for p in jobs:
            if p.poll() is None:p.terminate()
        for p in jobs:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for f in files:f.close()
print('CONSENSUS_TEST_PASS',root,flush=True)
