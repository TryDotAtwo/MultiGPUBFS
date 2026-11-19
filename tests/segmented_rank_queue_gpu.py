"""Unpacked u8/i64 owner skew, source retries and exact layer snapshots."""
import json,os,subprocess,time,tempfile,re
from pathlib import Path
from multigpubfs import GraphDefinition
root=Path(tempfile.mkdtemp(prefix='seg24-queue-',dir='/workspace'));native=Path(os.environ['MGBFS_EXECUTABLE'])
env=dict(os.environ,MGBFS_EXPERIMENTAL_SEGMENTED24='1',MGBFS_SEGMENT_ROWS='3',MGBFS_SEGMENT_TEST_QUEUE_CAP='3',MGBFS_GENERIC_GENERATOR='cuda',MGBFS_GENERIC_HISTORY='hash')
gens=[]
for i in range(3):
 gen=list(range(25));gen[i],gen[i+1]=gen[i+1],gen[i];gens.append(gen)
u8=GraphDefinition.permutation(gens,list(range(25)))
matrices=[]
for perm in gens:
 mx=[0]*(25*25)
 for i,j in enumerate(perm):mx[i*25+j]=1
 matrices.append((mx,0))
i64=GraphDefinition.matrix(25,1,matrices,[-(1<<40),(1<<39),-1234,6789]+list(range(4,25)))
def run(name,g,limit=None):
 case=root/name;case.mkdir();definition=case/'definition.json';definition.write_text(g.to_json());expected=g.exact_layers(1000)
 probe=subprocess.run([str(native),'graph-info',str(definition),'0,1','1','1024','4'],env=env,text=True,capture_output=True,timeout=60,check=True)
 plan=json.loads(probe.stdout);plan['owner_cuts']=[0,(1<<32)-1,1<<32]
 if limit is not None:plan['profile_max_layers']=limit
 config=case/'launch.json';config.write_text(json.dumps(plan));jobs=[];logs=[]
 try:
  for rank in range(2):
   f=(case/f'{rank}.log').open('w');logs.append(f);jobs.append(subprocess.Popen([str(native),'graph-rank',str(definition),str(config),str(rank),str(case/'id'),str(case/f'rank-{rank}'),'0'],env=env,stdout=f,stderr=subprocess.STDOUT))
  start=time.monotonic()
  while any(p.poll() is None for p in jobs):
   if any(p.poll() not in (None,0) for p in jobs):raise RuntimeError((name,'rank failed',[(case/f'{i}.log').read_text()[-5000:] for i in range(2)]))
   if time.monotonic()-start>90:raise RuntimeError((name,'timeout',[(case/f'{i}.log').read_text()[-5000:] for i in range(2)]))
   time.sleep(.05)
  reports=[json.loads((case/f'rank-{i}'/'report.json').read_text()) for i in range(2)]
  states=[json.loads((case/f'rank-{i}'/'states.json').read_text()) for i in range(2)]
  depth=len(reports[0]['layer_sizes'])-1
  assert all(x['layer_sizes']==list(map(len,expected[:depth+1])) for x in reports),reports
  assert sorted(states[0]['current']+states[1]['current'])==expected[depth]
  assert sorted(states[0]['previous_small']+states[1]['previous_small'])==(expected[depth-1] if depth else [])
  if limit is None:
   assert all(x['status']=='COMPLETE' and x['source_retries']>0 for x in reports),reports
  else:assert depth==limit and all(x['status']=='INCOMPLETE' and x['reason']=='PROFILE_LAYER_LIMIT' for x in reports)
  if limit is None and os.environ.get('SEG24_EXPECT_PREFETCH')=='1':
   assert all(x['segment_memory'].get('prefetched_chunks',0)>0 for x in reports),reports
  if limit is None and os.environ.get('MGBFS_SEGMENT_TEST_BOUNDS_TRACE')=='1':
   pattern=r'SEGMENT_BOUNDS rank=(\d+) depth=(\d+) local=(\d+) global=(\d+) upper=(\d+)'
   traces=[re.findall(pattern,(case/f'{i}.log').read_text()) for i in range(2)]
   assert traces[0] and len(traces[0])==len(traces[1]),traces
   reduced=False
   for a,b in zip(*traces):
    a=list(map(int,a));b=list(map(int,b))
    assert a[0]==0 and b[0]==1 and a[1]==b[1] and a[3:]==b[3:],(a,b)
    assert a[3]==max(a[2],b[2]) and 0<=a[3]<=a[4],(a,b)
    reduced |= a[3]<a[4]
   print('PASS matched global queue bounds',name,'rounds',len(traces[0]),flush=True)
  print('PASS',name,'depth',depth,'retries',reports[0]['source_retries'],flush=True)
 finally:
  for p in jobs:
   if p.poll() is None:p.terminate()
  for p in jobs:
   if p.poll() is None:
    try:p.wait(timeout=8)
    except subprocess.TimeoutExpired:p.kill();p.wait()
  for f in logs:f.close()
run('unpacked-u8',u8)
run('unpacked-i64',i64)
for depth in range(len(i64.exact_layers(1000))):run(f'unpacked-i64-depth-{depth}',i64,depth)
print('EVIDENCE',root)
