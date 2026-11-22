import hashlib,importlib.metadata,json,math,os,subprocess,threading,time,traceback
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
assert importlib.metadata.version('multigpubfs')=='0.2.0.dev44'
import multigpubfs
assert str(multigpubfs.__file__).startswith('/workspace/mgbfs-dev44-public/')
root=Path('/workspace/mgbfs-advisory-evidence/dev44-recovery-ab-s12-8gpu')
root.mkdir(parents=True,exist_ok=False)
os.environ.update(MGBFS_NATIVE_CACHE='/workspace/mgbfs-final-matrix/native-cache',
 MGBFS_NATIVE_OFFLINE='1',MGBFS_SEGMENT_ROWS='1048576',
 MGBFS_SEGMENT_EXACT_EXCHANGE='1',MGBFS_SEGMENT_BATCH_RECOVERY='0')
os.environ.pop('MGBFS_SEGMENT_MAX_RANK_STATES',None)
def gpu_apps():
 p=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
 return p.stdout
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
before=gpu_apps()
(root/'gpu-before.txt').write_text(before)
assert not before.strip(),('FOREIGN_GPU_PROCESSES_AT_START',before)
(root/'hardware.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv,noheader'],text=True))
(root/'topology.txt').write_text(subprocess.check_output(['nvidia-smi','topo','-m'],text=True))
meta={'wheel_sha256':'29b1ffdf6d31b194184e0d377e83e22b0f8b8c7b1355b36ff7b09eda5bdfd10b','source_commit':'5ed4b90e6ab3ff8ab16c78e36722e9c41fb5cd70', 'note':'sampled NVML is not an exact high-water; last two layers checked as full states, earlier layers by counts only'}
(root/'metadata.json').write_text(json.dumps(meta,indent=2))
n=12;gens=[]
for i in range(n-1):
 a=list(range(n));a[i],a[i+1]=a[i+1],a[i];gens.append(a)
graph=GraphDefinition.permutation(gens,list(range(n)))
oracle=[1]
for j in range(1,n+1):
 w=[0]*(len(oracle)+j-1)
 for k,x in enumerate(oracle):
  for z in range(j):w[k+z]+=x
 oracle=w
assert len(oracle)==67 and sum(oracle)==math.factorial(n)
def sampler(stop,rows):
 while not stop.is_set():
  try:
   p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=4)
   rows.append({'utc':time.time(),'vram_mib':{int(line.split(',')[0]):int(line.split(',')[1]) for line in p.stdout.splitlines()}})
  except Exception as e:rows.append({'utc':time.time(),'error':type(e).__name__})
  stop.wait(1.)
cases=[(i+1,r) for i,r in enumerate((0,1,1,0,0,1))]
records=[];pids=None;expected_q=None
try:
 with persistent_graph_workers() as pool:
  for rep,recovery in cases:
   q,lane=8192,4
   name=f'r{rep}-recovery{recovery}'
   os.environ['MGBFS_SEGMENT_BATCH_RECOVERY']=str(recovery)
   os.environ['MGBFS_SEGMENT_QUEUE_CAP']=str(q)
   os.environ['MGBFS_SEGMENT_OWNER_LANES']=str(lane)
   print('START',name,flush=True)
   (root/(name+'-apps-before.txt')).write_text(gpu_apps())
   stop=threading.Event();samples=[]
   monitor=threading.Thread(target=sampler,args=(stop,samples),daemon=True);monitor.start()
   start=time.monotonic()
   try:
    r=run_graph(graph,root/name,devices=list(range(8)),autotune=False,shards=8,
      _batch=8192,memory_backend='permutation_bitmap',memory_preset='balanced',
      segment_rows=2**32,segment_strategy='atomic')
   finally:wall=time.monotonic()-start;stop.set();monitor.join(6)
   assert r['status']=='COMPLETE',(name,r['status'],r.get('reason'))
   assert r['backend']=='EXPERIMENTAL_PERMUTATION_BITMAP'
   assert r['segment_strategy']=='atomic' and len(r['segment_memory_by_rank'])==8
   mem=r['segment_memory_by_rank']
   assert all(x['segment_strategy']=='atomic' and x['owner_lanes']==lane for x in mem),(name,mem)
   actual_q=mem[0]['actual_queue_capacity']
   if expected_q is None:expected_q=actual_q
   assert actual_q==expected_q and all(x['actual_queue_capacity']==actual_q and x['admitted_batch']==8192 and x['batch_recovery']==bool(recovery) for x in mem),(name,mem)
   assert r['layer_sizes']==oracle and len(r['layer_seconds'])>=66,name
   now=[w.process.pid for w in pool.workers]
   if pids is None:pids=now
   assert now==pids and len(set(now))==8
   snap=json.loads((root/name/'states.json').read_text())
   reverse=tuple(range(n-1,-1,-1));previous=[]
   for i in range(n-1):
    x=list(reverse);x[i],x[i+1]=x[i+1],x[i];previous.append(tuple(x))
   assert sorted(map(tuple,snap['current']))==[reverse],name
   assert sorted(map(tuple,snap['previous_small']))==sorted(previous),name
   observed={i:max((s['vram_mib'].get(i,0) for s in samples if 'vram_mib' in s),default=0) for i in range(8)}
   rec={'name':name,'requested_queue':q,'actual_queue':actual_q,'owner_lanes':lane,
    'strategy':'atomic','backend':'permutation_bitmap','graph':'Coxeter S12','batch_recovery':bool(recovery),
    'status':r['status'],'completed_depth':66,'states':sum(r['layer_sizes']),
    'all_layer_sizes_match_mahonian':True,'last_two_exact_full_state_sets':True,
    'bfs_seconds':r['bfs_seconds'],'wall_seconds':wall,
    'window_depths':[27,38],'window_seconds':sum(r['layer_seconds'][26:38]),
    'window_new_states':sum(r['layer_sizes'][27:39]),
    'window_generated_edges':11*sum(r['layer_sizes'][26:38]),
    'rank_pids':now,'layer_sizes':r['layer_sizes'],'layer_seconds':r['layer_seconds'],
    'segment_memory_by_rank':mem,'sampled_peak_mib_by_gpu':observed,
    'sample_interval_s':1.,'sample_note':'periodic samples may miss brief peaks',
    'runtime_identity':r.get('runtime_identity'),'plan':r.get('plan')}
   (root/(name+'-samples.json')).write_text(json.dumps(samples))
   (root/(name+'-apps-after.txt')).write_text(gpu_apps())
   records.append(rec);(root/'records.json').write_text(json.dumps(records,indent=2,default=str))
   print('PASS',json.dumps({k:rec[k] for k in ('name','actual_queue','owner_lanes','bfs_seconds','wall_seconds','window_seconds','sampled_peak_mib_by_gpu')}),flush=True)
except BaseException:
 (root/'failure.txt').write_text(traceback.format_exc());raise
(root/'complete').write_text('ok\n')
print('RECOVERY_AB_S12_8GPU_PASS',flush=True)

