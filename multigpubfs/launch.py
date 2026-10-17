"""Single-host native launch. CPU control never generates or deduplicates states."""
import hashlib,json,os,shutil,subprocess,tempfile,time,signal
from pathlib import Path
from .process_control import query as native_query, run as native_run
from .graph_definition import GraphDefinition,from_cayleypy
from .native_distribution import native_runtime

def _receipt(output,digest):
 report=json.loads((output/'report.json').read_text(encoding='utf-8'))
 if report['graph_digest']!=digest:raise RuntimeError('GRAPH_RECEIPT_IDENTITY_MISMATCH')
 if report['status'] not in ('COMPLETE','INCOMPLETE'):raise RuntimeError('GRAPH_RECEIPT_STATUS')
 if hashlib.sha256((output/'states.json').read_bytes()).hexdigest()!=report['states_sha256']:raise RuntimeError('GRAPH_RECEIPT_CHECKSUM')
 return report


def _job_phase(job,output,rank):
 if hasattr(job,'phase'):return job.phase()
 path=Path(output)/('rank-'+str(rank)+'.phase.json')
 try:
  if path.stat().st_size>262144:return None
  value=json.loads(path.read_text())
 except (OSError,ValueError):return None
 if not isinstance(value,dict) or value.get('schema')!=1 or value.get('pid')!=job.pid:return None
 if not isinstance(value.get('entered'),dict) or not isinstance(value.get('phase'),str):return None
 return value

def _phase_failures(phases):
 failures=[]
 for rank,phase in enumerate(phases):
  error=phase.get('native_error') if isinstance(phase,dict) else None
  if isinstance(error,dict) and type(error.get('code')) is int and error['code']!=0:failures.append((rank,1))
 return failures

def run_graph(graph,output,*,device=None,devices=None,capacity=None,max_seconds=3600,executable=None,shards=None,autotune=True,backend='auto',peer_transport='auto',transport='auto',candidate_order='auto',_batch=None,_profile_layers=None,_native_env=None):
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
 if backend not in ('auto','generic','shard_ab_hash','shard_ab_sort_merge'):raise ValueError('INVALID_BACKEND')
 if peer_transport not in ('auto','host','lsa'):raise ValueError('INVALID_PEER_TRANSPORT')
 if peer_transport=='lsa' and not backend.startswith('shard_ab_'):raise ValueError('LSA_REQUIRES_SPECIALIZED_BACKEND')
 if transport not in ('auto','full','parent'):raise ValueError('INVALID_TRANSPORT')
 if candidate_order not in ('auto','none','radix'):raise ValueError('INVALID_CANDIDATE_ORDER')
 if capacity is not None and (type(capacity) is not int or not 1<=capacity<=0x7ffffffe):raise ValueError('INVALID_CAPACITY')
 from .host_memory import admit
 host_memory=admit(graph,output=output) if os.environ.get('WORLD_SIZE','1')=='1' else None
 native,native_env=native_runtime(executable)
 geometry_bound=None
 if capacity is None and _profile_layers is None and not backend.startswith('shard_ab_'):
  from .workload_geometry import certified_frontier_bound
  geometry_bound=certified_frontier_bound(graph)
  if geometry_bound is not None:
   from .autotune import _admit
   with tempfile.TemporaryDirectory(prefix='mgbfs-bound-') as bound_dir:
    bound_admission=_admit(graph,devices,None,shards or 1,native,native_env,bound_dir)
   capacity=min(geometry_bound['maximum_frontier'],bound_admission['plan']['capacity'])

 if _native_env is not None:native_env=dict(_native_env)
 native_env=dict(native_env,MGBFS_PEER_TRANSPORT=peer_transport)
 def finish(report):
  if geometry_bound is not None:report['workload_geometry_bound']=geometry_bound
  from .size_profiles import save_online_profiles
  try:save_online_profiles(report)
  except OSError:report['profile_cache_write_status']='UNAVAILABLE'
  from .native_distribution import runtime_identity
  report['runtime_identity']=runtime_identity(native,native_env)
  if host_memory is not None:report['host_memory']=[host_memory]
  path=Path(output)/'report.json';temporary=path.with_name('report.json.tmp');temporary.write_text(json.dumps(report,indent=2));temporary.replace(path)
  return report
 if transport!='auto':native_env=dict(native_env,MGBFS_GENERIC_TRANSPORT=transport)
 if candidate_order!='auto':native_env=dict(native_env,MGBFS_GENERIC_SORT=candidate_order)
 if os.environ.get('WORLD_SIZE','1')!='1':
  if backend not in ('auto','generic') or peer_transport=='lsa':raise ValueError('SPECIALIZED_EXTERNAL_RANK_UNSUPPORTED')
  from .distributed_launch import run_external
  return finish(run_external(graph,output,devices=devices,capacity=capacity,max_seconds=max_seconds,native=native,native_env=native_env,shards=shards,autotune=autotune))
 output=Path(output).absolute()
 if output.exists():raise FileExistsError(output)
 profile=None
 if backend.startswith('shard_ab_'):
  if shards is not None or transport!='auto' or candidate_order!='auto':raise ValueError('SPECIALIZED_ADAPTIVE_GEOMETRY_OR_GENERIC_OPTION_CONFLICT')
  return finish(_run_specialized_admitted(graph,output,native,native_env,devices,capacity,max_seconds,backend,_batch,_profile_layers,None))
 if shards is None:
  if autotune:
   from .autotune import choose_size_profile
   # Reserve most of the caller's deadline for BFS, not cold tuning.
   tuning_budget=min(60,max(1,max_seconds//3))
   native_env=dict(native_env,MGBFS_TUNING_EVIDENCE_ROOT=str(Path(output).parent/(Path(output).name+'-tuning-evidence')))
   profile=choose_size_profile(graph,devices,capacity,tuning_budget,native,native_env,allow_specialized=backend=='auto')
   shards=profile['shards']
   if profile.get('backend','generic')!='generic':
    remaining=max(1,max_seconds-int(profile['seconds']+.999))
    return finish(_run_specialized_admitted(graph,output,native,native_env,devices,capacity,remaining,profile['backend'],_batch,_profile_layers,profile))
   native_env=dict(native_env,MGBFS_GENERIC_TRANSPORT=profile.get('transport','full'),MGBFS_GENERIC_SORT=profile.get('candidate_order','none'),MGBFS_GENERIC_GENERATOR=profile.get('generator_backend',native_env.get('MGBFS_GENERIC_GENERATOR','cuda')))
   if profile.get('history_algorithm')=='SORTED_RUNS':native_env=dict(native_env,MGBFS_GENERIC_HISTORY='sorted',MGBFS_GENERIC_OWNER_LANES=str(profile['owner_lanes']))
   elif profile.get('history_algorithm')=='HASH':native_env=dict(native_env,MGBFS_GENERIC_HISTORY='hash')
   if (profile.get('measured') or profile.get('status')=='REUSED_COMPATIBLE_GPU_PROFILE') and profile.get('backend','generic')=='generic':_batch=profile['batch']
   max_seconds=max(1,max_seconds-int(profile['seconds']+.999))
  else:shards=1
 output=Path(output).absolute()
 if output.exists():raise FileExistsError(output)
 output.parent.mkdir(parents=True,exist_ok=True);digest=graph.digest()
 with tempfile.TemporaryDirectory(prefix='mgbfs-definition-',dir=output.parent) as temporary:
  definition=Path(temporary)/'graph.json';definition.write_text(graph.to_json(),encoding='utf-8')
  selection='auto' if devices is None else ','.join(map(str,devices));command=[str(native),'graph-info',str(definition),selection,str(shards)]
  # A bounded pilot may reuse the exclusive startup's inventory. Real runs
  # always query live VRAM again; no shared inventory on production admission.
  if _profile_layers is not None and native_env.get('_MGBFS_PROFILE_INVENTORY'):
   cached_inventory=Path(temporary)/'inventory.json';cached_inventory.write_text(native_env['_MGBFS_PROFILE_INVENTORY'])
   command=[str(native),'graph-plan',str(definition),str(cached_inventory),str(shards)]
  if capacity is not None:command.append(str(capacity))
  if _batch is None and profile and profile.get('batch_fraction',1.0)!=1.0:
   initial=native_query(command,capture_output=True,text=True,env=native_env)
   if initial.returncode:raise RuntimeError('NATIVE_GRAPH_ADMISSION_FAILED: '+initial.stderr[-4000:])
   _batch=max(1,int(json.loads(initial.stdout)['plan']['batch']*profile['batch_fraction']))
  if _batch is not None:
   if capacity is None:command.append('auto')
   command.append(str(_batch))
  from .generic_session import take_launch_admission
  accepted=take_launch_admission(graph,native,native_env,devices,capacity,shards,_batch) if profile is not None and _profile_layers is None else None
  probe=subprocess.CompletedProcess(command,0,json.dumps(accepted),'') if accepted is not None else native_query(command,capture_output=True,text=True,env=native_env)
  if profile is not None:profile['launch_admission_reused']=accepted is not None
  if probe.returncode:raise RuntimeError('NATIVE_GRAPH_ADMISSION_FAILED: '+probe.stderr[-4000:])
  admission=json.loads(probe.stdout);devices=admission['devices']
  if admission.get('owned_resource_admission'):native_env=dict(native_env,MGBFS_REQUIRE_OWNED_RAW='1')
  if profile and profile.get('size_profiles'):admission.update(size_profiles=profile['size_profiles'],online_size_tuning=profile.get('online_size_tuning',False))
  if _profile_layers is not None:admission['profile_max_layers']=_profile_layers
  if admission['graph_digest']!=digest:raise RuntimeError('GRAPH_ADMISSION_IDENTITY')
  if admission['plan'].get('generator_backend','cuda')!='gemm' and admission['plan'].get('history_algorithm','HASH')=='HASH' and len(devices)==1 and shards==1 and _profile_layers is None and not (profile and profile.get('size_profiles')) and admission['plan'].get('history_layers',1)==1 and not admission['plan'].get('parent_transport',False) and not admission['plan'].get('sort_candidates',False):
   command=[str(native),'graph',str(definition),str(output),'--device',str(devices[0]),'--seconds',str(max_seconds)]
   if capacity is not None:command+=['--capacity',str(capacity)]
   process=native_run(command,capture_output=True,text=True,env=native_env,timeout=max_seconds+180)
   if process.returncode:raise RuntimeError('NATIVE_GRAPH_FAILED: '+process.stderr[-4000:])
   result=_receipt(output,digest)
   if profile:
    result['autotune']=profile;(output/'report.json').write_text(json.dumps(result,indent=2))
   return finish(result)
  output.mkdir();configuration=output/'launch.json';configuration.write_text(json.dumps(admission));bootstrap=Path(temporary)/'nccl-id';jobs=[];logs=[];failure=None;started=time.monotonic()
  try:
   from .generic_session import launch_group_selection
   selection=launch_group_selection(str(native),devices,native_env)
   for rank in range(len(devices)):
    log=(output/f'rank-{rank}.log').open('w');logs.append(log)
    env=dict(native_env,RANK=str(rank),WORLD_SIZE=str(len(devices)),LOCAL_RANK=str(devices[rank]),MGBFS_JOB_PHASE_PATH=str(output/f'rank-{rank}.phase.json'))
    from .generic_session import launch as launch_rank
    jobs.append(launch_rank([str(native),'graph-rank',str(definition),str(configuration),str(rank),str(bootstrap),str(output/f'rank-{rank}'),str(max_seconds)],env,log,devices,_selection=selection))
   next_phase_check=0.
   while any(p.poll() is None for p in jobs):
    failed=[(rank,p.returncode) for rank,p in enumerate(jobs) if p.poll() not in (None,0)]
    if failed:raise RuntimeError('NATIVE_DISTRIBUTED_RANK_FAILED '+repr(failed))
    now=time.monotonic()
    if now>=next_phase_check:
     phases=[_job_phase(p,output,rank) for rank,p in enumerate(jobs)]
     failed=_phase_failures(phases)
     if failed:raise RuntimeError('NATIVE_DISTRIBUTED_RANK_FAILED '+repr(failed))
     timed_out=any(
      (phase['phase']=='preparing' and time.time()-phase['entered']['preparing']>600) or
      (phase['phase']=='bfs' and time.time()-phase['entered']['bfs']>max_seconds+60) or
      (phase['phase']=='retiring' and time.time()-phase['entered']['retiring']>600)
      for phase in phases if phase is not None)
     if timed_out or (not all(phases) and now-started>max_seconds+1200):raise RuntimeError('DISTRIBUTED_COMPLETION_TIMEOUT: worker state remains in rank logs; no restart')
     next_phase_check=now+.05
    time.sleep(.002)
   if any(p.returncode for p in jobs):raise RuntimeError('NATIVE_DISTRIBUTED_RANK_FAILED')
   failed=_phase_failures([_job_phase(p,output,rank) for rank,p in enumerate(jobs)])
   if failed:raise RuntimeError('NATIVE_DISTRIBUTED_RANK_FAILED '+repr(failed))

