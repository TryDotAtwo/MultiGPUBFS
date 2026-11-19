"""Exact per-layer and asymmetric resource-stop checks for private rank mode."""
import json,os,subprocess,time,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
root=Path(tempfile.mkdtemp(prefix='seg24-boundaries-',dir='/workspace'));native=Path(os.environ['MGBFS_EXECUTABLE'])
env=dict(os.environ,MGBFS_EXPERIMENTAL_SEGMENTED24='1',MGBFS_SEGMENT_ROWS='3',MGBFS_GENERIC_GENERATOR='cuda',MGBFS_GENERIC_HISTORY='hash')
g=GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],list(range(4)));expected=g.exact_layers(1000)
def run(name,limit=None,rank1_capacity=None,rank1_real_oom_depth=None):
 case=root/name;case.mkdir();definition=case/'definition.json';definition.write_text(g.to_json())
 probe=subprocess.run([str(native),'graph-info',str(definition),'0,1','2','1024','4'],env=env,text=True,capture_output=True,timeout=60,check=True)
 plan=json.loads(probe.stdout);plan['owner_cuts']=None
 if limit is not None:plan['profile_max_layers']=limit
 config=case/'launch.json';config.write_text(json.dumps(plan));logs=[];jobs=[]
 try:
  for rank in range(2):
   e=env.copy()
   if rank==1 and rank1_capacity is not None:e['MGBFS_SEGMENT_TEST_LIMIT']=str(rank1_capacity)
   if rank==1 and rank1_real_oom_depth is not None:e['MGBFS_SEGMENT_TEST_REAL_OOM_AFTER_DEPTH']=str(rank1_real_oom_depth)
   f=(case/f'rank-{rank}.log').open('w');logs.append(f);jobs.append(subprocess.Popen([str(native),'graph-rank',str(definition),str(config),str(rank),str(case/'id'),str(case/f'rank-{rank}'),'0'],env=e,stdout=f,stderr=subprocess.STDOUT))
  start=time.monotonic()
  while any(p.poll() is None for p in jobs):
   if any(p.poll() not in (None,0) for p in jobs):raise RuntimeError((name,'rank failed',[(case/f'rank-{i}.log').read_text()[-4000:] for i in range(2)]))
   if time.monotonic()-start>60:raise RuntimeError((name,'timeout',[(case/f'rank-{i}.log').read_text()[-4000:] for i in range(2)]))
   time.sleep(.05)
  reports=[json.loads((case/f'rank-{i}'/'report.json').read_text()) for i in range(2)]
  states=[json.loads((case/f'rank-{i}'/'states.json').read_text()) for i in range(2)]
  assert reports[0]['status']==reports[1]['status']=='INCOMPLETE'
  if rank1_real_oom_depth is not None:assert 'SEGMENTED_TEST_REAL_OOM rank=1' in (case/'rank-1.log').read_text()
  assert reports[0]['reason']==reports[1]['reason']
  depth=len(reports[0]['layer_sizes'])-1
  assert all(x['layer_sizes']==list(map(len,expected[:depth+1])) for x in reports)
  assert sorted(states[0]['current']+states[1]['current'])==expected[depth]
  assert sorted(states[0]['previous_small']+states[1]['previous_small'])==(expected[depth-1] if depth else [])
  return depth,reports[0]['reason']
 finally:
  for p in jobs:
   if p.poll() is None:p.terminate()
  for p in jobs:
   if p.poll() is None:
    try:p.wait(timeout=8)
    except subprocess.TimeoutExpired:p.kill();p.wait()
  for f in logs:f.close()
for depth in range(len(expected)):
 actual,reason=run(f'depth-{depth}',limit=depth);assert (actual,reason)==(depth,'PROFILE_LAYER_LIMIT');print('PASS exact depth',depth,flush=True)
depth,reason=run('rank1-resource',rank1_capacity=0);assert reason=='RESOURCE_10';print('PASS asymmetric resource stop depth',depth,flush=True)
depth,reason=run('rank1-real-oom',rank1_real_oom_depth=2);assert reason=='RESOURCE_10' and depth>=2;print('PASS real asymmetric CUDA OOM depth',depth,flush=True)
print('EVIDENCE',root)
