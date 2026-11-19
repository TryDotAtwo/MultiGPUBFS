# Hard global per-rank limit under concurrent owner lanes.
import json,os,re,subprocess,tempfile,time
from pathlib import Path
from multigpubfs import GraphDefinition
root=Path(tempfile.mkdtemp(prefix='seg24-capacity-',dir='/workspace'));native=Path(os.environ['MGBFS_EXECUTABLE'])
env=dict(os.environ,MGBFS_EXPERIMENTAL_SEGMENTED24='1',MGBFS_SEGMENT_ROWS='3',MGBFS_SEGMENT_MAX_RANK_STATES='1',MGBFS_GENERIC_GENERATOR='cuda',MGBFS_GENERIC_HISTORY='hash')
g=GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],list(range(4)));expected=g.exact_layers(1000)
definition=root/'definition.json';definition.write_text(g.to_json())
probe=subprocess.run([str(native),'graph-info',str(definition),'0,1','4','1024','4'],env=env,text=True,capture_output=True,timeout=60,check=True)
plan=json.loads(probe.stdout);plan['owner_cuts']=[0,(1<<32)-1,1<<32]
config=root/'launch.json';config.write_text(json.dumps(plan))
for trial in range(5):
 case=root/str(trial);case.mkdir();jobs=[];logs=[]
 try:
  for rank in range(2):
   f=(case/f'{rank}.log').open('w');logs.append(f);jobs.append(subprocess.Popen([str(native),'graph-rank',str(definition),str(config),str(rank),str(case/'id'),str(case/f'rank-{rank}'),'0'],env=env,stdout=f,stderr=subprocess.STDOUT))
  deadline=time.monotonic()+60
  while any(p.poll() is None for p in jobs) and time.monotonic()<deadline:time.sleep(.05)
  assert all(p.poll()==0 for p in jobs),[p.poll() for p in jobs]
  for f in logs:f.flush()
  reports=[json.loads((case/f'rank-{i}'/'report.json').read_text()) for i in range(2)]
  states=[json.loads((case/f'rank-{i}'/'states.json').read_text()) for i in range(2)]
  assert all(r['status']=='INCOMPLETE' and r['reason']=='RESOURCE_10' for r in reports),reports
  depth=len(reports[0]['layer_sizes'])-1
  assert sorted(states[0]['current']+states[1]['current'])==expected[depth]
  found=[]
  for i in range(2):
   for rank,limit,count in re.findall(r'SEGMENTED_CAPACITY_OBSERVED rank=(\d+) limit=(\d+) count=(\d+)',(case/f'{i}.log').read_text()):
    assert int(limit)==1 and int(count)<=int(limit),(rank,limit,count)
    found.append((rank,limit,count))
  assert found,'limit stop did not expose a checked GPU-wide counter'
  print('PASS concurrent hard-cap',trial,found,flush=True)
 finally:
  for p in jobs:
   if p.poll() is None:p.terminate()
  for p in jobs:
   if p.poll() is None:
    try:p.wait(timeout=8)
    except subprocess.TimeoutExpired:p.kill();p.wait()
  for f in logs:f.close()
print('EVIDENCE',root)
