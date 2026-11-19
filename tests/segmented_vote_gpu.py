# Debug-only fault injection: fatal route must dominate a concurrent resource vote.
import json,os,subprocess,tempfile,time
from pathlib import Path
from multigpubfs import GraphDefinition
root=Path(tempfile.mkdtemp(prefix='seg24-vote-',dir='/workspace'))
native=Path(os.environ['MGBFS_EXECUTABLE'])
env=dict(os.environ,MGBFS_EXPERIMENTAL_SEGMENTED24='1',MGBFS_SEGMENT_ROWS='8',MGBFS_GENERIC_GENERATOR='cuda',MGBFS_GENERIC_HISTORY='hash')
gens=[]
for i in range(3):
 g=list(range(4));g[i],g[i+1]=g[i+1],g[i];gens.append(g)
graph=GraphDefinition.permutation(gens,list(range(4)))
definition=root/'definition.json';definition.write_text(graph.to_json())
probe=subprocess.run([str(native),'graph-info',str(definition),'0,1','4','1024','4'],env=env,text=True,capture_output=True,timeout=60,check=True)
config=root/'launch.json';config.write_text(probe.stdout)
for name,faults in [
 ('different-ranks',[{'MGBFS_SEGMENT_TEST_VOTE_SOURCE':'1'},{'MGBFS_SEGMENT_TEST_VOTE_OWNER_RESOURCE':'1'}]),
 ('same-rank',[{'MGBFS_SEGMENT_TEST_VOTE_SOURCE':'1','MGBFS_SEGMENT_TEST_VOTE_OWNER_RESOURCE':'1'},{}])]:
 case=root/name;case.mkdir();jobs=[];logs=[]
 try:
  for rank in range(2):
   f=(case/f'{rank}.log').open('w');logs.append(f)
   jobs.append(subprocess.Popen([str(native),'graph-rank',str(definition),str(config),str(rank),str(case/'id'),str(case/f'rank-{rank}'),'0'],env=dict(env,**faults[rank]),stdout=f,stderr=subprocess.STDOUT))
  deadline=time.monotonic()+60
  while any(p.poll() is None for p in jobs) and time.monotonic()<deadline:time.sleep(.05)
  assert all(p.poll() is not None for p in jobs),'rank hang'
  for f in logs:f.flush()
  text=[(case/f'{i}.log').read_text() for i in range(2)]
  assert all(p.returncode!=0 for p in jobs),(name,[p.returncode for p in jobs],text)
  assert all('SEGMENTED_ROUTE_66048' in x for x in text),(name,text)
  print('PASS',name,'both ranks report source fatal instead of resource',flush=True)
 finally:
  for p in jobs:
   if p.poll() is None:p.terminate()
  for p in jobs:
   if p.poll() is None:
    try:p.wait(timeout=8)
    except subprocess.TimeoutExpired:p.kill();p.wait()
  for f in logs:f.close()
print('EVIDENCE',root)
