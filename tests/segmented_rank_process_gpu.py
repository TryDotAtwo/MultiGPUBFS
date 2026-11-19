"""Bounded, two-process CUDA correctness gate for the experimental engine."""
import json,os,subprocess,time,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
root=Path(tempfile.mkdtemp(prefix='seg24-ranks-',dir='/workspace'))
native=Path(os.environ['MGBFS_EXECUTABLE']);env=dict(os.environ,MGBFS_EXPERIMENTAL_SEGMENTED24='1',MGBFS_SEGMENT_ROWS='3',MGBFS_GENERIC_GENERATOR='cuda',MGBFS_GENERIC_HISTORY='hash')
cases=[('s4',GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],list(range(4)))),('repeated',GraphDefinition.permutation([[1,0,2,3,4],[0,2,1,3,4],[0,1,3,2,4],[0,1,2,4,3]],[0,0,1,2,3])),('directed',GraphDefinition.permutation([[1,2,3,0]],list(range(4)))),
 ('s6',GraphDefinition.permutation([[1,0,2,3,4,5],[0,2,1,3,4,5],[0,1,3,2,4,5],[0,1,2,4,3,5],[0,1,2,3,5,4]],list(range(6)))),
 ('wide',GraphDefinition.permutation([list(range(1,25))+[0]],list(range(25)))),
 ('signed-matrix',GraphDefinition.matrix(2,1,[([-1,0,0,1],0),([1,0,0,-1],0)],[-(1<<40),2]))]
for name,g in cases:
 case=root/name;case.mkdir();definition=case/'definition.json';definition.write_text(g.to_json());expected=g.exact_layers(1000)
 probe=subprocess.run([str(native),'graph-info',str(definition),'0,1','2','1024','4'],env=env,text=True,capture_output=True,timeout=60)
 if probe.returncode:raise RuntimeError((name,'info',probe.stdout,probe.stderr))
 plan=json.loads(probe.stdout);config=case/'launch.json';config.write_text(json.dumps(plan));logs=[];jobs=[]
 try:
  for rank in range(2):
   f=(case/f'rank-{rank}.log').open('w');logs.append(f);jobs.append(subprocess.Popen([str(native),'graph-rank',str(definition),str(config),str(rank),str(case/'id'),str(case/f'rank-{rank}'),'0'],env=env,stdout=f,stderr=subprocess.STDOUT))
  start=time.monotonic()
  while any(p.poll() is None for p in jobs):
   if any(p.poll() not in (None,0) for p in jobs):raise RuntimeError((name,'rank failed',[(case/f'rank-{i}.log').read_text()[-8000:] for i in range(2)]))
   if time.monotonic()-start>90:raise RuntimeError((name,'timeout',[(case/f'rank-{i}.log').read_text()[-8000:] for i in range(2)]))
   time.sleep(.05)
  reports=[json.loads((case/f'rank-{i}'/'report.json').read_text()) for i in range(2)]
  states=[json.loads((case/f'rank-{i}'/'states.json').read_text()) for i in range(2)]
  assert all(x['status']=='COMPLETE' for x in reports),reports
  assert all(x['layer_sizes']==list(map(len,expected)) for x in reports),(reports,expected)
  assert sorted(states[0]['current']+states[1]['current'])==expected[-1],states
  assert sorted(states[0]['previous_small']+states[1]['previous_small'])==expected[-2],states
  print('PASS',name,'layers',len(expected),'states',sum(map(len,expected)),flush=True)
 finally:
  for p in jobs:
   if p.poll() is None:p.terminate()
  for p in jobs:
   if p.poll() is None:
    try:p.wait(timeout=8)
    except subprocess.TimeoutExpired:p.kill();p.wait()
  for f in logs:f.close()
print('EVIDENCE',root)
