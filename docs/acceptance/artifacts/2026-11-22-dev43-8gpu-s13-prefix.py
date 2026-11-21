import importlib.metadata,json,math,os,subprocess,threading,time,traceback
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
assert importlib.metadata.version('multigpubfs')=='0.2.0.dev43'
root=Path('/workspace/capacity-eight/results/s13-prefix-dev43')
root.mkdir(parents=True,exist_ok=False)
os.environ.update(MGBFS_NATIVE_CACHE='/workspace/capacity-eight/native-cache',
 MGBFS_NATIVE_OFFLINE='1',MGBFS_SEGMENT_ROWS='1048576',
 MGBFS_SEGMENT_OWNER_LANES='4',MGBFS_SEGMENT_QUEUE_CAP='6144',
 MGBFS_SEGMENT_EXACT_EXCHANGE='1',MGBFS_SEGMENT_BATCH_RECOVERY='0')
busy=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,capture_output=True,check=True)
(root/'gpu-before.txt').write_text(busy.stdout)
assert not busy.stdout.strip(),'FOREIGN_GPU_PROCESSES_AT_START'
(root/'hardware.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv,noheader'],text=True))
(root/'topology.txt').write_text(subprocess.check_output(['nvidia-smi','topo','-m'],text=True))
n=13
gens=[]
for i in range(n-1):
 a=list(range(n));a[i],a[i+1]=a[i+1],a[i];gens.append(a)
graph=GraphDefinition.permutation(gens,list(range(n)))
oracle=[1]
for j in range(1,n+1):
 w=[0]*(len(oracle)+j-1)
 for k,x in enumerate(oracle):
  for z in range(j):w[k+z]+=x
 oracle=w
assert sum(oracle)==math.factorial(n)
def sampler(stop,rows):
 while not stop.is_set():
  try:
   p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=4)
   rows.append({'utc':time.time(),'vram_mib':{int(line.split(',')[0]):int(line.split(',')[1]) for line in p.stdout.splitlines()}})
  except Exception as e:rows.append({'utc':time.time(),'error':type(e).__name__})
  stop.wait(1.0)
records=[];ids=None
try:
 with persistent_graph_workers() as pool:
  for name,backend,strategy in [('bitmap13','permutation_bitmap','atomic'),('hash13','segmented24','warp')]:
   print('START',name,flush=True)
   stop=threading.Event();samples=[]
   monitor=threading.Thread(target=sampler,args=(stop,samples),daemon=True);monitor.start()
   start=time.monotonic()
   try:
    r=run_graph(graph,root/name,devices=list(range(8)),autotune=False,shards=8,
     _batch=8192,_profile_layers=38,memory_backend=backend,memory_preset='balanced',
     segment_rows=2**32,segment_strategy=strategy)
   finally: wall=time.monotonic()-start;stop.set();monitor.join(6)
   assert r['status']=='INCOMPLETE' and r['reason']=='PROFILE_LAYER_LIMIT',(name,r['status'],r['reason'])
   assert len(r['layer_sizes'])==39 and r['layer_sizes']==oracle[:39],name
   assert len(r['layer_seconds'])>=38
   now=[w.process.pid for w in pool.workers]
   if ids is None:ids=now
   assert now==ids and len(set(now))==8
   ranks=r['segment_memory_by_rank']
   observed={i:max((s['vram_mib'].get(i,0) for s in samples if 'vram_mib' in s),default=0) for i in range(8)}
   rec={'name':name,'backend':backend,'strategy':strategy,'graph':'Coxeter S13',
    'status':r['status'],'reason':r['reason'],'completed_depth':38,
    'states_in_prefix':sum(r['layer_sizes']),'layer_sizes_match_mahonian':True,
    'full_state_sets_checked':False,'bfs_seconds':r['bfs_seconds'],'wall_seconds':wall,
    'window_depths':[34,38],'window_seconds':sum(r['layer_seconds'][33:38]),
    'window_new_states':sum(r['layer_sizes'][34:39]),'rank_pids':now,
    'segment_memory_by_rank':ranks,'sampled_peak_mib_by_gpu':observed,
    'sample_interval_s':1.0,'sample_note':'NVML periodic samples may miss brief peaks',
    'runtime_identity':r.get('runtime_identity')}
   (root/(name+'-samples.json')).write_text(json.dumps(samples))
   records.append(rec);(root/'records.json').write_text(json.dumps(records,indent=2,default=str))
   print('PASS',json.dumps({k:rec[k] for k in ('name','completed_depth','bfs_seconds','wall_seconds','window_seconds','window_new_states','sampled_peak_mib_by_gpu')}),flush=True)
except BaseException:
 (root/'failure.txt').write_text(traceback.format_exc());raise
(root/'complete').write_text('ok\n')