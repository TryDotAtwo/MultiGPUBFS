"""Controlled 2-GPU bitmap comparison. S11 is not a B200 performance claim."""
import json,os,subprocess,tempfile,threading,time
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
n=11;gens=[]
for i in range(n-1):
 p=list(range(n));p[i],p[i+1]=p[i+1],p[i];gens.append(p)
g=GraphDefinition.permutation(gens,list(range(n)));oracle=[1]
for degree in range(1,n+1):
 q=[0]*(len(oracle)+degree-1)
 for i,a in enumerate(oracle):
  for j in range(degree):q[i+j]+=a
 oracle=q
root=Path(tempfile.mkdtemp(prefix='bitmap-2bank-s11-',dir='/workspace'))
native=os.environ['MGBFS_EXECUTABLE'];results=[];snapshots=[];pids=None
base=dict(os.environ,MGBFS_SEGMENT_ROWS=str(1<<18),MGBFS_SEGMENT_OWNER_LANES='4',MGBFS_SEGMENT_QUEUE_CAP='4096',MGBFS_SEGMENT_EXACT_EXCHANGE='1')
with persistent_graph_workers():
 for run,preset in enumerate(('balanced','max_capacity','max_capacity','balanced')):
  peaks=[0,0];stop=threading.Event()
  def sample():
   while not stop.is_set():
    q=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
    if q.returncode==0:
     try:
      for i,v in enumerate(q.stdout.splitlines()[:2]):peaks[i]=max(peaks[i],int(v.strip()))
     except ValueError:pass
    stop.wait(.2)
  t=threading.Thread(target=sample);t.start();start=time.monotonic()
  try:
   out=root/f'run-{run}'
   result=run_graph(g,out,devices=[0,1],shards=4,autotune=False,executable=native,memory_backend='permutation_bitmap',memory_preset=preset,_native_env=base)
  finally:stop.set();t.join()
  assert result['status']=='COMPLETE' and result['layer_sizes']==oracle,result
  geo=result['segment_memory_by_rank'];assert all(v['banks']==(2 if preset=='max_capacity' else 3) for v in geo)
  if pids is None:pids=result['rank_worker_pids']
  assert result['rank_worker_pids']==pids
  snap=json.loads((out/'states.json').read_text());snapshots.append((sorted(snap['current']),sorted(snap['previous_small'])))
  rec={'run':run,'preset':preset,'n':n,'states':sum(oracle),'layers':len(oracle),'bfs_seconds':result['bfs_seconds'],'wall_seconds':time.monotonic()-start,'nvidia_smi_peak_MiB_by_gpu':peaks,'storage_allocated_bytes_by_rank':[x['storage_allocated_bytes'] for x in geo],'banks':[x['banks'] for x in geo],'lanes':[x['owner_lanes'] for x in geo],'workers':pids}
  results.append(rec);print(json.dumps(rec),flush=True)
 assert all(x==snapshots[0] for x in snapshots)
(root/'evidence.json').write_text(json.dumps(results,indent=2))
print('BITMAP_S11_EQUAL_LAYERS_RETAINED_STATES_PASS',root,flush=True)
