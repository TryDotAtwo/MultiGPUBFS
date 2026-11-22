import importlib.metadata,json,math,os,subprocess,threading,time,traceback
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
import multigpubfs
assert importlib.metadata.version('multigpubfs')=='0.2.0.dev44'
assert str(multigpubfs.__file__).startswith('/workspace/mgbfs-dev44-public/')
root=Path('/workspace/mgbfs-advisory-evidence/dev44-public-s13-full-8gpu')
root.mkdir(parents=True,exist_ok=False)
os.environ.update(MGBFS_NATIVE_CACHE='/workspace/mgbfs-final-matrix/native-cache',
 MGBFS_NATIVE_OFFLINE='1',MGBFS_SEGMENT_ROWS='1048576',
 MGBFS_SEGMENT_EXACT_EXCHANGE='1',MGBFS_SEGMENT_BATCH_RECOVERY='0',
 MGBFS_SEGMENT_OWNER_LANES='4',MGBFS_SEGMENT_QUEUE_CAP='8192')
os.environ.pop('MGBFS_SEGMENT_MAX_RANK_STATES',None)
def apps():
 return subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
before=apps();(root/'gpu-before.txt').write_text(before)
assert not before.strip(),('FOREIGN_GPU_PROCESSES_AT_START',before)
(root/'hardware.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv,noheader'],text=True))
(root/'topology.txt').write_text(subprocess.check_output(['nvidia-smi','topo','-m'],text=True))
n=13;gens=[]
for i in range(n-1):
 a=list(range(n));a[i],a[i+1]=a[i+1],a[i];gens.append(a)
graph=GraphDefinition.permutation(gens,list(range(n)))
oracle=[1]
for j in range(1,n+1):
 z=[0]*(len(oracle)+j-1)
 for i,v in enumerate(oracle):
  for k in range(j):z[i+k]+=v
 oracle=z
assert len(oracle)==79 and sum(oracle)==math.factorial(n)
cases=[('hash-warp','segmented24','warp'),('bitmap-atomic','permutation_bitmap','atomic')]
def sampler(stop,rows):
 while not stop.is_set():
  try:
   p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=4)
   rows.append({'utc':time.time(),'vram_mib':{int(v.split(',')[0]):int(v.split(',')[1]) for v in p.stdout.splitlines()}})
  except Exception as e:rows.append({'utc':time.time(),'error':type(e).__name__})
  stop.wait(1.)
records=[]
try:
 with persistent_graph_workers() as pool:
  actual=[]
  for name,backend,strategy in cases:
   print('START smoke',name,flush=True)
   r=run_graph(graph,root/('smoke-'+name),devices=list(range(8)),autotune=False,shards=8,_batch=8192,
      memory_backend=backend,memory_preset='balanced',segment_rows=2**32,segment_strategy=strategy,_profile_layers=3)
   assert r['status']=='INCOMPLETE' and r['reason']=='PROFILE_LAYER_LIMIT' and r['layer_sizes']==oracle[:4]
   mem=r['segment_memory_by_rank'];q=[m['actual_queue_capacity'] for m in mem]
   assert len(set(q))==1 and all(m['owner_lanes']==4 for m in mem)
   actual.append(q[0]);print('SMOKE',name,'actual_queue',q[0],flush=True)
  assert actual[0]==actual[1],actual
  pids=[w.process.pid for w in pool.workers]
  for name,backend,strategy in cases:
   print('START',name,flush=True)
   (root/(name+'-apps-before.txt')).write_text(apps())
   stop=threading.Event();samples=[]
   thread=threading.Thread(target=sampler,args=(stop,samples),daemon=True);thread.start();start=time.monotonic()
   try:
    r=run_graph(graph,root/name,devices=list(range(8)),autotune=False,shards=8,_batch=8192,
       memory_backend=backend,memory_preset='balanced',segment_rows=2**32,segment_strategy=strategy)
   finally:wall=time.monotonic()-start;stop.set();thread.join(6)
   mem=r['segment_memory_by_rank']
   assert all(m['actual_queue_capacity']==actual[0] and m['owner_lanes']==4 for m in mem)
   assert r['status']=='COMPLETE' and r['layer_sizes']==oracle,name
   snapshot=json.loads((root/name/'states.json').read_text())
   reverse=tuple(range(n-1,-1,-1)); previous=[]
   for i in range(n-1):
    c=list(reverse);c[i],c[i+1]=c[i+1],c[i];previous.append(tuple(c))
   assert sorted(map(tuple,snapshot['current']))==[reverse]
   assert sorted(map(tuple,snapshot['previous_small']))==sorted(previous)
   assert [w.process.pid for w in pool.workers]==pids
   depth=len(r['layer_sizes'])-1
   sampled={i:max((s['vram_mib'].get(i,0) for s in samples if 'vram_mib' in s),default=0) for i in range(8)}
   rec={'name':name,'backend':backend,'strategy':strategy,'status':r['status'],'reason':r['reason'],
    'completed_depth':depth,'completed_states':sum(r['layer_sizes']),'layer_sizes':r['layer_sizes'],
    'all_completed_layer_sizes_match_mahonian':True,'full_layer_state_sets_checked':False,'last_two_exact_full_state_sets':True,
    'layer_seconds':r['layer_seconds'],'bfs_seconds':r['bfs_seconds'],'wall_seconds':wall,
    'window_depths':[34,44],'window_seconds':sum(r['layer_seconds'][33:44]),'window_new_states':sum(r['layer_sizes'][34:45]),'window_generated_edges':12*sum(r['layer_sizes'][33:44]),
    'rank_pids':pids,'actual_queue':actual[0],'requested_queue':8192,'owner_lanes':4,'shards':8,
    'segment_rows_configured':2**32,'batch_requested':8192,'segment_memory_by_rank':mem,
    'sampled_peak_mib_by_gpu':sampled,'sample_interval_s':1.,'sample_note':'periodic samples may miss brief peaks',
    'runtime_identity':r.get('runtime_identity'),'plan':r.get('plan')}
   records.append(rec);(root/'records.json').write_text(json.dumps(records,indent=2))
   (root/(name+'-samples.json')).write_text(json.dumps(samples))
   (root/(name+'-apps-after.txt')).write_text(apps())
   print('PASS',json.dumps({k:rec[k] for k in ('name','status','reason','completed_depth','bfs_seconds','wall_seconds','window_seconds','sampled_peak_mib_by_gpu')}),flush=True)
except BaseException:
 (root/'failure.txt').write_text(traceback.format_exc());raise
(root/'complete').write_text('ok\n')
print('S13_FULL_8GPU_FINISHED',flush=True)

