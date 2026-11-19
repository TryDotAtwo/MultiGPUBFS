"""Two-process exact BFS oracle for bitmap rank ownership and spill."""
import json,os,subprocess,time,tempfile,re
from pathlib import Path
from multigpubfs import GraphDefinition
root=Path(tempfile.mkdtemp(prefix='bitmap-bfs-',dir='/workspace'))
native=Path(os.environ['MGBFS_EXECUTABLE'])
env=dict(os.environ,MGBFS_EXPERIMENTAL_SEGMENTED24='1',MGBFS_EXPERIMENTAL_BITMAP='1',
 MGBFS_SEGMENT_ROWS='3',MGBFS_SEGMENT_TEST_QUEUE_CAP='5',MGBFS_GENERIC_GENERATOR='cuda',MGBFS_GENERIC_HISTORY='hash')
swap=lambda n,i:list(range(i))+[i+1,i]+list(range(i+2,n))
wide_u=[0]*22+[1,2,3]
wide_i=[-(1<<40)]*22+[1<<39,2,3]
cases=[
 ('s4',GraphDefinition.permutation([swap(4,i) for i in range(3)],list(range(4)))),
 ('repeated',GraphDefinition.permutation([swap(5,i) for i in range(4)],[0,0,1,2,3])),
 ('directed',GraphDefinition.permutation([[1,2,3,0]],list(range(4)))),
 ('s6',GraphDefinition.permutation([swap(6,i) for i in range(5)],list(range(6)))),
 ('wide-u8',GraphDefinition.permutation([swap(25,i) for i in range(22,24)],wide_u)),
 ('wide-i64',GraphDefinition.permutation([swap(25,i) for i in range(22,24)],wide_i)),
]
def run(name,g,limit=None):
 case=root/name;case.mkdir();definition=case/'definition.json';definition.write_text(g.to_json());expected=g.exact_layers(1000)
 probe=subprocess.run([str(native),'graph-info',str(definition),'0,1','4','1024','4'],env=env,text=True,capture_output=True,timeout=60,check=True)
 plan=json.loads(probe.stdout)
 case_env=dict(env,MGBFS_SEGMENT_TEST_QUEUE_CAP=str(min(plan['plan']['queue_capacity'],max(3,g.generator_count))))
 if limit is not None:plan['profile_max_layers']=limit
 config=case/'launch.json';config.write_text(json.dumps(plan));logs=[];jobs=[]
 try:
  for rank in range(2):
   f=(case/f'rank-{rank}.log').open('w');logs.append(f);jobs.append(subprocess.Popen([str(native),'graph-rank',str(definition),str(config),str(rank),str(case/'id'),str(case/f'rank-{rank}'),'0'],env=case_env,stdout=f,stderr=subprocess.STDOUT))
  start=time.monotonic()
  while any(p.poll() is None for p in jobs):
   if any(p.poll() not in (None,0) for p in jobs):raise RuntimeError((name,'rank failed',[(case/f'rank-{i}.log').read_text()[-8000:] for i in range(2)]))
   if time.monotonic()-start>90:raise RuntimeError((name,'timeout',[(case/f'rank-{i}.log').read_text()[-8000:] for i in range(2)]))
   time.sleep(.03)
  reports=[json.loads((case/f'rank-{i}'/'report.json').read_text()) for i in range(2)]
  states=[json.loads((case/f'rank-{i}'/'states.json').read_text()) for i in range(2)]
  depth=len(reports[0]['layer_sizes'])-1
  assert all(x['backend']=='EXPERIMENTAL_PERMUTATION_BITMAP' for x in reports),reports
  assert all(x['layer_sizes']==list(map(len,expected[:depth+1])) for x in reports),(reports,expected)
  assert sorted(states[0]['current']+states[1]['current'])==expected[depth],states
  if depth:assert sorted(states[0]['previous_small']+states[1]['previous_small'])==expected[depth-1],states
  if limit is None:assert all(x['status']=='COMPLETE' and depth==len(expected)-1 for x in reports),reports
  else:assert depth==limit and all(x['status']=='INCOMPLETE' and x['reason']=='PROFILE_LAYER_LIMIT' for x in reports),reports
  if name=='s6' and limit is None and os.environ.get('MGBFS_SEGMENT_TEST_BOUNDS_TRACE')=='1':
   pattern=r'SEGMENT_BOUNDS rank=(\d+) depth=(\d+) local=(\d+) global=(\d+) upper=(\d+)'
   for f in logs:f.flush()
   traces=[re.findall(pattern,(case/f'rank-{i}.log').read_text()) for i in range(2)]
   assert traces[0] and len(traces[0])==len(traces[1]),traces
   reduced=False
   for a,b in zip(*traces):
    a=list(map(int,a));b=list(map(int,b))
    assert a[0]==0 and b[0]==1 and a[1]==b[1] and a[3:]==b[3:],(a,b)
    assert a[3]==max(a[2],b[2]) and 0<=a[3]<=a[4],(a,b)
    reduced |= a[3]<a[4]
   assert reduced,'actual queue bound never reduced padded transfer'
   print('PASS matched global bitmap queue bounds with reduced padding',len(traces[0]),flush=True)
  print('PASS',name,'depth',depth,'states',sum(map(len,expected)),'bitmap_bytes',[x['segment_memory']['bitmap_bytes'] for x in reports],flush=True)
 finally:
  for p in jobs:
   if p.poll() is None:p.terminate()
  for p in jobs:
   if p.poll() is None:
    try:p.wait(timeout=8)
    except subprocess.TimeoutExpired:p.kill();p.wait()
  for f in logs:f.close()
for name,g in cases:
 run(name,g)
 if name in ('s4','repeated','directed'):
  for depth in range(len(g.exact_layers(1000))):run(f'{name}-depth-{depth}',g,depth)
print('EVIDENCE',root)
