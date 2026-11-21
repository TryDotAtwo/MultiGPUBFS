import importlib.metadata,json,math,os,subprocess,sys,threading,time,traceback
from pathlib import Path
from cayleypy import CayleyGraph,PermutationGroups
import multigpubfs
from multigpubfs import NativeOptions,persistent_graph_workers
version=sys.argv[1]
root=Path('/workspace/capacity-eight/results')/version
root.mkdir(parents=True,exist_ok=False)
assert importlib.metadata.version('multigpubfs')==version
os.environ.update(MGBFS_NATIVE_CACHE='/workspace/capacity-eight/native-cache',
                  MGBFS_NATIVE_OFFLINE='1',MGBFS_SEGMENT_ROWS='1048576',
                  MGBFS_SEGMENT_BATCH_RECOVERY='0',MGBFS_PROFILE_CACHE=str(root/'profiles'))
for key in ('MGBFS_SEGMENT_OWNER_LANES','MGBFS_SEGMENT_QUEUE_CAP','MGBFS_GENERIC_VRAM_RESERVE_BYTES','MGBFS_SEGMENT_EXACT_EXCHANGE'):
 os.environ.pop(key,None)
busy=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,capture_output=True,check=True)
(root/'gpu-before.txt').write_text(busy.stdout)
assert not busy.stdout.strip(), 'FOREIGN_GPU_PROCESSES_AT_START'
(root/'hardware.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv,noheader'],text=True))
(root/'topology.txt').write_text(subprocess.check_output(['nvidia-smi','topo','-m'],text=True))
def mahonian(n):
 v=[1]
 for k in range(2,n+1):
  w=[0]*(len(v)+k-1)
  for i,x in enumerate(v):
   for d in range(k):w[i+d]+=x
  v=w
 return v
def sampler(stop,rows):
 while not stop.is_set():
  try:
   p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=4)
   values={int(line.split(',')[0]):int(line.split(',')[1]) for line in p.stdout.splitlines()}
   rows.append({'utc':time.time(),'vram_mib':values})
  except Exception as e:rows.append({'utc':time.time(),'error':type(e).__name__})
  stop.wait(1.0)
records=[];ids=None
try:
 multigpubfs.enable_native(recovery=False)
 with persistent_graph_workers() as pool:
  for name,backend,strategy,n in [('hash12','segmented24','warp',12),('bitmap12','permutation_bitmap','atomic',12),('hash9','segmented24','warp',9)]:
   print('START',name,flush=True)
   opts=NativeOptions(devices=tuple(range(8)),autotune=True,memory_backend=backend,
      memory_preset='max_capacity',segment_rows=2**32,segment_strategy=strategy)
   graph=CayleyGraph(PermutationGroups.coxeter(n),device='cpu')
   stop=threading.Event();samples=[];monitor=threading.Thread(target=sampler,args=(stop,samples),daemon=True)
   monitor.start();started=time.monotonic()
   try:result=graph.bfs(native_options=opts,output=root/name)
   finally:wall=time.monotonic()-started;stop.set();monitor.join(6)
   report=result.report
   assert result.bfs_completed and result.layer_sizes==mahonian(n), (name,report.get('reason'))
   assert sum(result.layer_sizes)==math.factorial(n)
   now=[w.process.pid for w in pool.workers]
   assert len(now)==8 and len(set(now))==8
   if ids is None:ids=now
   assert now==ids,(name,now,ids)
   snapshot=json.loads((root/name/'states.json').read_text())
   reverse=tuple(range(n-1,-1,-1));previous=[]
   for i in range(n-1):
    row=list(reverse);row[i],row[i+1]=row[i+1],row[i];previous.append(tuple(row))
   assert sorted(map(tuple,snapshot['current']))==[reverse]
   assert sorted(map(tuple,snapshot['previous_small']))==sorted(previous)
   ranks=report['segment_memory_by_rank'];assert len(ranks)==8
   profile=report.get('autotune',{})
   observed={i:max((s['vram_mib'].get(i,0) for s in samples if 'vram_mib' in s),default=0) for i in range(8)}
   rec={'name':name,'backend':backend,'strategy':strategy,'n':n,'status':report['status'],
    'bfs_seconds':report['bfs_seconds'],'wall_seconds':wall,'layers':len(result.layer_sizes),
    'states':sum(result.layer_sizes),'all_layer_counts_match_mahonian':True,
    'last_two_exact_full_state_sets':True,'rank_pids':ids,'shards':report['plan']['shards'],
    'batch':report['plan']['batch'],'autotune':profile,'segment_memory_by_rank':ranks,
    'runtime_identity':report.get('runtime_identity'),
    'vram_sample_interval_s':1.0,'sampled_peak_mib_by_gpu':observed,
    'vram_sample_note':'nvidia-smi periodic samples, may miss transient peaks'}
   (root/(name+'-samples.json')).write_text(json.dumps(samples))
   records.append(rec);(root/'records.json').write_text(json.dumps(records,indent=2,default=str))
   print('PASS',json.dumps({k:rec[k] for k in ('name','bfs_seconds','wall_seconds','states','shards','batch','sampled_peak_mib_by_gpu')}),flush=True)
except BaseException:
 (root/'failure.txt').write_text(traceback.format_exc());raise
finally:
 multigpubfs.disable_native()
(root/'complete').write_text('ok\n')
