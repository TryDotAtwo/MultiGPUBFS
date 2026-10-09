"""Bounded same-graph GPU prefix tuning; no CPU successor or dedup in production."""
import hashlib,json,os,subprocess,tempfile,time
from pathlib import Path

def _admit(graph,devices,capacity,shards,native,env,directory):
 path=Path(directory)/'definition.json';path.write_text(graph.to_json())
 command=[native,'graph-info',str(path),'auto' if devices is None else ','.join(map(str,devices)),str(shards)]
 if capacity is not None:command.append(str(capacity))
 p=subprocess.run(command,capture_output=True,text=True,env=env)
 if p.returncode:raise RuntimeError('PROFILE_ADMISSION_FAILED: '+p.stderr[-2000:])
 return json.loads(p.stdout)

def _system_info(command):
 try:return subprocess.run(command,capture_output=True,text=True,timeout=10)
 except (OSError,subprocess.TimeoutExpired):return subprocess.CompletedProcess(command,1,'unavailable','')

def choose_profile(graph,devices,capacity,max_seconds,native,env):
 started=time.monotonic()
 with tempfile.TemporaryDirectory(prefix='mgbfs-profile-') as temporary:
  baseline=_admit(graph,devices,capacity,1,native,env,temporary)
  selected=baseline['devices'];plan=baseline['plan']
  if plan['capacity']<=4096 or max_seconds<10:
   return {'status':'SMALL_OR_SHORT_WORKLOAD_CONSERVATIVE_PROFILE','shards':1,'batch_fraction':1.0,'measured':False,'seconds':time.monotonic()-started}
  # Native content, graph and driver/topology identity prevent stale reuse.
  hardware=_system_info(['nvidia-smi','--query-gpu=uuid,driver_version,name','--format=csv,noheader'])
  topology=_system_info(['nvidia-smi','topo','-m'])
  identity={'schema':2,'graph':graph.digest(),'native_sha256':hashlib.sha256(Path(native).read_bytes()).hexdigest(),'devices':selected,'hardware':hardware.stdout,'topology':topology.stdout,'capacity_override':capacity,'environment':{k:env.get(k) for k in ('CUDA_VISIBLE_DEVICES','NCCL_P2P_DISABLE','NCCL_SHM_DISABLE','NCCL_SOCKET_IFNAME')}}
  key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
  root=Path(os.environ.get('MGBFS_PROFILE_CACHE',str(Path.home()/'.cache/multigpubfs/profiles')));cache=root/(key+'.json')
  if hardware.returncode==0 and topology.returncode==0 and cache.is_file():
   try:saved=json.loads(cache.read_text())
   except (OSError,ValueError):saved={}
   if saved.get('identity')==identity and saved.get('status')=='MEASURED_EQUAL_PREFIX_GPU_PROFILE' and saved.get('shards') in (1,4) and saved.get('batch_fraction') in (1.0,.25):
    saved=dict(saved);saved['cache_hit']=True;saved['seconds']=time.monotonic()-started;return saved
  variants=[(1,1.0),(4,1.0),(4,.25)]
  admitted=[]
  for shards,fraction in variants:
   try:admission=_admit(graph,selected,capacity,shards,native,env,temporary)
   except RuntimeError as error:
    if any(code in str(error) for code in ('REQUESTED_CAPACITY_EXCEEDS_ADMISSION','GENERIC_DISTRIBUTED_NO_CAPACITY','GENERIC_DISTRIBUTED_HEADROOM')):continue
    raise
   admitted.append((shards,fraction,admission['plan']))
  if len(admitted)<2:return {'status':'NO_ADMITTED_ALTERNATIVE_CONSERVATIVE_PROFILE','shards':1,'batch_fraction':1.0,'measured':False,'seconds':time.monotonic()-started}
  common_capacity=min(p['capacity'] for _,_,p in admitted)
  pilots=[];limit=min(3,max(1,max_seconds//20))
  from .launch import run_graph
  for index,(shards,fraction,_) in enumerate(admitted):
   admission=_admit(graph,selected,common_capacity,shards,native,env,temporary)
   batch=max(1,int(admission['plan']['batch']*fraction))
   report=run_graph(graph,Path(temporary)/('pilot-'+str(index)),devices=selected,capacity=common_capacity,max_seconds=limit,executable=native,shards=shards,autotune=False,_batch=batch,_profile_layers=36,_native_env=env)
   pilots.append({'shards':shards,'batch_fraction':fraction,'batch':batch,'layer_sizes':report['layer_sizes'],'layer_seconds':report['layer_seconds'],'status':report['status'],'reason':report['reason']})
  depth=min(len(p['layer_seconds']) for p in pilots)
  sizes=pilots[0]['layer_sizes'][:depth+1]
  if any(p['layer_sizes'][:depth+1]!=sizes for p in pilots):raise RuntimeError('AUTOTUNE_PREFIX_CORRECTNESS_MISMATCH')
  # Exclude first collective warmup. Tiny prefixes cannot justify a speed claim.
  if depth<4 or sum(sizes[2:])<32768:
   return {'status':'INSUFFICIENT_PREFIX_CONSERVATIVE_PROFILE','shards':1,'batch_fraction':1.0,'measured':False,'pilots':pilots,'common_depth':depth,'seconds':time.monotonic()-started}
  scores=[sum(p['layer_seconds'][1:depth]) for p in pilots]
  winner=min(range(len(scores)),key=lambda i:scores[i])
  # A single bounded sample cannot justify a marginal switch. Keep baseline
  # unless the alternative beats it by at least five percent.
  if scores[winner]>=scores[0]*.95:winner=0
  chosen=pilots[winner]
  result={'status':'MEASURED_EQUAL_PREFIX_GPU_PROFILE','shards':chosen['shards'],'batch_fraction':chosen['batch_fraction'],'measured':True,'minimum_switch_improvement':.05,'common_depth':depth,'common_layer_sizes':sizes,'scores_seconds':scores,'pilots':pilots,'identity':identity,'cache_hit':False,'seconds':time.monotonic()-started,'scope':'bounded prefix among three admitted profiles; no claim of global optimum or large-rank acceptance'}
  if hardware.returncode==0 and topology.returncode==0:
   root.mkdir(parents=True,exist_ok=True)
   fd,path=tempfile.mkstemp(prefix=key+'-',suffix='.tmp',dir=root)
   with os.fdopen(fd,'w') as f:json.dump(result,f)
   os.replace(path,cache)
  return result
