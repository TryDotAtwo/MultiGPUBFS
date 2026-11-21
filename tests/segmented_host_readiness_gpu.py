"""Physical 2-GPU host-agreement and failure-cleanup regression.

Each trial owns a bootstrap directory and two child PIDs. There is no shared
process cleanup or timed termination of unrelated jobs.
"""
import json,os,subprocess,tempfile,time
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

root=Path(tempfile.mkdtemp(prefix='segmented-host-readiness-',dir='/workspace'));native=os.environ['MGBFS_EXECUTABLE']
def graph(n):
 gens=[]
 for i in range(n-1):
  g=list(range(n));g[i],g[i+1]=g[i+1],g[i];gens.append(g)
 return GraphDefinition.permutation(gens,list(range(n)))
graphs=[graph(4),graph(5)];configs=[];definitions=[]
with persistent_graph_workers():
 for i,g in enumerate(graphs):
  out=root/f'admit-{i}'
  report=run_graph(g,out,devices=[0,1],shards=2,autotune=False,_batch=8,memory_backend='segmented24',segment_strategy='atomic',segment_rows=3,executable=native,_profile_layers=0)
  assert report['status']=='INCOMPLETE'
  definitions.append(root/f'graph-{i}.json');definitions[-1].write_text(g.to_json());configs.append(out/'launch.json')

def trial(name,cap,graph_ids=(0,0),faults=('', ''),expected='',lost=False):
 folder=root/name;folder.mkdir();jobs=[];logs=[]
 try:
  for rank in range(2):
   env=dict(os.environ,MGBFS_EXPERIMENTAL_SEGMENTED24='1',MGBFS_SEGMENT_TEST_BATCH='1',
            MGBFS_SEGMENT_TEST_HOST_TIMEOUT_MS='5000',MGBFS_SEGMENT_STRATEGY='atomic',
            RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='2')
   env.pop('MGBFS_EXPERIMENTAL_BITMAP',None)
   if cap[rank] is None:env.pop('MGBFS_SEGMENT_TEST_QUEUE_CAP',None)
   else:env['MGBFS_SEGMENT_TEST_QUEUE_CAP']=cap[rank]
   if faults[rank]:env['MGBFS_SEGMENT_TEST_HOST_FAIL']=faults[rank]
   else:env.pop('MGBFS_SEGMENT_TEST_HOST_FAIL',None)
   log=(folder/f'rank-{rank}.log').open('w');logs.append(log)
   args=[native,'graph-rank',str(definitions[graph_ids[rank]]),str(configs[graph_ids[rank]]),str(rank),str(folder/'bootstrap'),str(folder/f'out-{rank}'),'0']
   jobs.append(subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT))
  deadline=time.monotonic()+30
  while any(p.poll() is None for p in jobs) and time.monotonic()<deadline:time.sleep(.05)
  assert all(p.poll() is not None for p in jobs),(name,'rank hung',[p.poll() for p in jobs])
  for log in logs:log.flush()
  texts=[(folder/f'rank-{r}.log').read_text() for r in range(2)]
  assert all(p.returncode!=0 for p in jobs),(name,[p.returncode for p in jobs],texts)
  if lost:
   assert expected in texts[0],(name,texts)
  else:assert all(expected in t for t in texts),(name,texts)
  for r in range(2):assert not (folder/f'out-{r}'/'report.json').exists()
  print('PASS HOST FAILURE',name,[p.returncode for p in jobs],expected,flush=True)
 finally:
  for p in jobs:
   if p.poll() is None:p.terminate()
  for p in jobs:
   if p.poll() is None:
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:p.kill();p.wait()
  for f in logs:f.close()
trial('invalid-one-rank',('0','4'),expected='SEGMENTED_TEST_QUEUE_CAP')
trial('different-valid-cap',('4','8'),expected='SEGMENTED_HOST_MISMATCH')
trial('different-graph',(None,None),(0,1),expected='SEGMENTED_HOST_MISMATCH')
trial('base-failure',(None,None),faults=('','base'),expected='SEGMENTED_TEST_HOST_FAIL_BASE')
trial('queue-failure',(None,None),faults=('queue',''),expected='SEGMENTED_TEST_HOST_FAIL_QUEUE')
trial('map-failure',(None,None),faults=('','map'),expected='SEGMENTED_TEST_HOST_FAIL_MAP')
trial('rank-lost',(None,None),faults=('','exit'),expected='SEGMENTED_HOST_TIMEOUT',lost=True)
with persistent_graph_workers():
 g=graphs[0];out=root/'clean-after-failures';r=run_graph(g,out,devices=[0,1],shards=2,autotune=False,_batch=8,memory_backend='segmented24',segment_strategy='atomic',segment_rows=3,executable=native,_profile_layers=999)
 assert r['status']=='COMPLETE' and r['layer_sizes']==[1,3,5,6,5,3,1],r
 print('PASS CLEAN SUBSEQUENT PERSISTENT JOB',flush=True)
print('HOST_READINESS_TWO_GPU_PASS',root,flush=True)
