"""Single-host native launch. CPU control never generates or deduplicates states."""
import hashlib,json,os,shutil,subprocess,tempfile,time,signal
from pathlib import Path
from .graph_definition import GraphDefinition,from_cayleypy
from .native_distribution import native_runtime

def _receipt(output,digest):
 report=json.loads((output/'report.json').read_text(encoding='utf-8'))
 if report['graph_digest']!=digest:raise RuntimeError('GRAPH_RECEIPT_IDENTITY_MISMATCH')
 if report['status'] not in ('COMPLETE','INCOMPLETE'):raise RuntimeError('GRAPH_RECEIPT_STATUS')
 if hashlib.sha256((output/'states.json').read_bytes()).hexdigest()!=report['states_sha256']:raise RuntimeError('GRAPH_RECEIPT_CHECKSUM')
 return report

def run_graph(graph,output,*,device=None,devices=None,capacity=None,max_seconds=3600,executable=None,shards=None,autotune=True,transport='auto',_batch=None,_profile_layers=None,_native_env=None):
 """Launch on all visible GPUs by default, or an explicit device/device list.

 Memory is admitted from actual free VRAM. With shards omitted, bounded
 GPU prefix measurements select a cached profile for substantial workloads.
 With WORLD_SIZE/RANK/LOCAL_RANK and MASTER_ADDR/MASTER_PORT it uses
 the external-rank network launcher; one process owns one local GPU.
 """
 if not isinstance(graph,GraphDefinition):graph=from_cayleypy(graph)
 if device is not None and devices is not None:raise ValueError('DEVICE_SELECTION_CONFLICT')
 if device is not None:
  if type(device) is not int or device<0:raise ValueError('INVALID_DEVICE')
  devices=[device]
 if devices is not None and (not isinstance(devices,(list,tuple)) or not devices or any(type(v) is not int or v<0 for v in devices) or len(set(devices))!=len(devices)):raise ValueError('INVALID_DEVICES')
 if type(max_seconds) is not int or max_seconds<1:raise ValueError('INVALID_MAX_SECONDS')
 if shards is not None and (type(shards) is not int or not 1<=shards<=4096):raise ValueError('INVALID_SHARDS')
 if type(autotune) is not bool:raise ValueError('INVALID_AUTOTUNE')
 if transport not in ('auto','full','parent'):raise ValueError('INVALID_TRANSPORT')
 if capacity is not None and (type(capacity) is not int or not 1<=capacity<=1<<28):raise ValueError('INVALID_CAPACITY')
 native,native_env=native_runtime(executable)
 if _native_env is not None:native_env=dict(_native_env)
 if transport!='auto':native_env=dict(native_env,MGBFS_GENERIC_TRANSPORT=transport)
 if os.environ.get('WORLD_SIZE','1')!='1':
  from .distributed_launch import run_external
  return run_external(graph,output,devices=devices,capacity=capacity,max_seconds=max_seconds,native=native,native_env=native_env,shards=shards,autotune=autotune)
 output=Path(output).absolute()
 if output.exists():raise FileExistsError(output)
 profile=None
 if shards is None:
  if autotune:
   from .autotune import choose_profile
   profile=choose_profile(graph,devices,capacity,max_seconds,native,native_env)
   shards=profile['shards']
   native_env=dict(native_env,MGBFS_GENERIC_TRANSPORT=profile.get('transport','full'))
   max_seconds=max(1,max_seconds-int(profile['seconds']+.999))
  else:shards=1
 output=Path(output).absolute()
 if output.exists():raise FileExistsError(output)
 output.parent.mkdir(parents=True,exist_ok=True);digest=graph.digest()
 with tempfile.TemporaryDirectory(prefix='mgbfs-definition-',dir=output.parent) as temporary:
  definition=Path(temporary)/'graph.json';definition.write_text(graph.to_json(),encoding='utf-8')
  selection='auto' if devices is None else ','.join(map(str,devices));command=[str(native),'graph-info',str(definition),selection,str(shards)]
  if capacity is not None:command.append(str(capacity))
  if profile and profile.get('batch_fraction',1.0)!=1.0:
   initial=subprocess.run(command,capture_output=True,text=True,env=native_env)
   if initial.returncode:raise RuntimeError('NATIVE_GRAPH_ADMISSION_FAILED: '+initial.stderr[-4000:])
   _batch=max(1,int(json.loads(initial.stdout)['plan']['batch']*profile['batch_fraction']))
  if _batch is not None:
   if capacity is None:
    initial=subprocess.run(command,capture_output=True,text=True,env=native_env)
    if initial.returncode:raise RuntimeError('NATIVE_GRAPH_ADMISSION_FAILED: '+initial.stderr[-4000:])
    command.append('auto')
   command.append(str(_batch))
  probe=subprocess.run(command,capture_output=True,text=True,env=native_env)
  if probe.returncode:raise RuntimeError('NATIVE_GRAPH_ADMISSION_FAILED: '+probe.stderr[-4000:])
  admission=json.loads(probe.stdout);devices=admission['devices']
  if _profile_layers is not None:admission['profile_max_layers']=_profile_layers
  if admission['graph_digest']!=digest:raise RuntimeError('GRAPH_ADMISSION_IDENTITY')
  if len(devices)==1 and shards==1 and _profile_layers is None and admission['plan'].get('history_layers',1)==1 and not admission['plan'].get('parent_transport',False):
   command=[str(native),'graph',str(definition),str(output),'--device',str(devices[0]),'--seconds',str(max_seconds)]
   if capacity is not None:command+=['--capacity',str(capacity)]
   process=subprocess.run(command,capture_output=True,text=True,env=native_env)
   if process.returncode:raise RuntimeError('NATIVE_GRAPH_FAILED: '+process.stderr[-4000:])
   result=_receipt(output,digest)
   if profile:
    result['autotune']=profile;(output/'report.json').write_text(json.dumps(result,indent=2))
   return result
  output.mkdir();configuration=output/'launch.json';configuration.write_text(json.dumps(admission));bootstrap=Path(temporary)/'nccl-id';jobs=[];logs=[];started=time.monotonic()
  try:
   for rank in range(len(devices)):
    log=(output/f'rank-{rank}.log').open('w');logs.append(log)
    env=dict(native_env,RANK=str(rank),WORLD_SIZE=str(len(devices)),LOCAL_RANK=str(devices[rank]))
    jobs.append(subprocess.Popen([str(native),'graph-rank',str(definition),str(configuration),str(rank),str(bootstrap),str(output/f'rank-{rank}'),str(max_seconds)],env=env,stdout=log,stderr=subprocess.STDOUT))
   while any(p.poll() is None for p in jobs):
    failed=[(rank,p.returncode) for rank,p in enumerate(jobs) if p.poll() not in (None,0)]
    if failed:raise RuntimeError('NATIVE_DISTRIBUTED_RANK_FAILED '+repr(failed))
    if time.monotonic()-started>max_seconds+180:raise RuntimeError('DISTRIBUTED_COMPLETION_TIMEOUT: worker state remains in rank logs; no restart')
    time.sleep(.02)
   if any(p.returncode for p in jobs):raise RuntimeError('NATIVE_DISTRIBUTED_RANK_FAILED')
  finally:
   for p in jobs:
    if p.poll() is None:p.terminate()
   for p in jobs:
    if p.poll() is None:
     try:p.wait(timeout=10)
     except subprocess.TimeoutExpired:p.kill();p.wait()
   for log in logs:log.close()
  parts=[_receipt(output/f'rank-{rank}',digest) for rank in range(len(devices))]
  for rank,p in enumerate(parts):
   if p['rank']!=rank or p['world']!=len(devices) or p['device']!=devices[rank] or p['plan']!=admission.get('rank_plans',[admission['plan']]*len(devices))[rank]:raise RuntimeError('DISTRIBUTED_RANK_GEOMETRY_MISMATCH')
   if any(p[key]!=parts[0][key] for key in ('status','reason','layer_sizes')):raise RuntimeError('DISTRIBUTED_LAYER_CONSENSUS_MISMATCH')
  snapshots=[json.loads((output/f'rank-{rank}'/'states.json').read_text()) for rank in range(len(devices))]
  current=[v for p in snapshots for v in p['current']];previous=[v for p in snapshots for v in p['previous_small']]
  if len(current)>1000 or len(previous)>=1000:raise RuntimeError('DISTRIBUTED_RETENTION_BOUND')
  states={'schema':2,'state_encoding':'signed_int64_vectors','current':current,'previous_small':previous,'current_sample_limit':1000};raw=json.dumps(states).encode();(output/'states.json').write_bytes(raw)
  report=dict(parts[0]);report.update(rank_plans=[p['plan'] for p in parts],owner_cuts=admission.get('owner_cuts'));report.pop('rank');report.pop('device');report.update(devices=devices,states_sha256=hashlib.sha256(raw).hexdigest(),rank_receipts=[f'rank-{r}/report.json' for r in range(len(devices))],bfs_seconds=max(p['bfs_seconds'] for p in parts),setup_seconds=max(p['setup_seconds'] for p in parts),launch_wall_seconds=time.monotonic()-started,scope='single host general exact retained-history path; bounded profile evidence in autotune receipt when enabled; larger hardware not verified')
  if profile:report['autotune']=profile;report['profile_status']=profile['status']
  (output/'report.json.tmp').write_text(json.dumps(report,indent=2));(output/'report.json.tmp').replace(output/'report.json')
  return _receipt(output,digest)
