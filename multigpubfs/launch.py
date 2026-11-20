"""Single-host native launch. CPU control never generates or deduplicates states."""
import hashlib,json,os,shutil,subprocess,tempfile,time,signal
from pathlib import Path
from .process_control import query as native_query, run as native_run
from .graph_definition import GraphDefinition,from_cayleypy
from .native_distribution import native_runtime

def _normalize_memory_backend(name):
 # Keep the previous spelling as an alias for existing clients and receipts;
 # it is no longer the width of the local index.
 if name in ('segmented','segmented32','segmented24'):
  return 'segmented24'
 if name in ('legacy','permutation_bitmap'):
  return name
 raise ValueError('INVALID_MEMORY_BACKEND')

def _normalize_segment_strategy(name):
 if name not in ('atomic','warp','batch'):raise ValueError('INVALID_SEGMENT_STRATEGY')
 return name

def _apply_memory_preset(env,backend,preset):
 if preset not in ('balanced','max_capacity'):raise ValueError('INVALID_MEMORY_PRESET')
 selected=dict(env,MGBFS_MEMORY_PRESET=preset)
 if preset=='max_capacity':
  if backend not in ('segmented24','permutation_bitmap'):raise ValueError('MEMORY_PRESET_UNSUPPORTED_FOR_LEGACY')
  selected.setdefault('MGBFS_SEGMENT_OWNER_LANES','1')
  selected.setdefault('MGBFS_SEGMENT_QUEUE_CAP','2048')
  if backend=='segmented24':selected.setdefault('MGBFS_SEGMENT_COMPACT_POS','1')
  else:selected.setdefault('MGBFS_BITMAP_BANKS','2')
 else:
  # Private settings supplied by the user remain explicit; defaults do not
  # contaminate a later balanced job in a persistent Python process.
  pass
 return selected

# Cache only successful capability queries. The key follows the executable
# identity and loader search path, so replacing a binary or library path
# cannot silently reuse a capability learned from a different installation.
_memory_backend_capabilities={}
def _require_memory_backend(native,env,backend):
 key_name={'segmented24':'segmented24_v1','permutation_bitmap':'permutation_bitmap_v1'}[backend]
 resolved=shutil.which(native) or native
 st=os.stat(resolved)
 # A wheel may update its CUDA library without changing the executable or
 # loader path. Never reuse capabilities from different dependency bytes.
 # In persistent mode the resolver checks file stamps and caches the digest.
 from .autotune import _dependency_identity
 dependency=_dependency_identity(str(resolved),env)
 dependency_key=None if dependency is None else (
  dependency.get('cuda_library_sha256'),
  tuple(tuple(item) for item in dependency.get('resolved_dependencies',())))
 loader=tuple(env.get(k,'') for k in ('LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT'))
 test_options=tuple(env.get(k,'') for k in ('MGBFS_SEGMENT_TEST_BATCH','MGBFS_SEGMENT_TEST_BATCH_ABI_MISSING'))
 identity=(str(Path(resolved).resolve()),st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,loader,test_options,dependency_key,key_name)
 if dependency is not None and identity in _memory_backend_capabilities:return _memory_backend_capabilities[identity]
 result=native_query([str(native),'key-info'],env=env,capture_output=True,text=True)
 try: info=json.loads(result.stdout)
 except (TypeError,ValueError):info={}
 if result.returncode or not isinstance(info,dict) or info.get('schema')!=1 or info.get(key_name) is not True or info.get('segmented32_v2') is not True or info.get('segmented_memory_presets_v1') is not True:
  raise RuntimeError('NATIVE_MEMORY_BACKEND_UNAVAILABLE: '+backend+' requires a matching native executable and CUDA library built from the segmented backend source')
 if dependency is not None:
  if len(_memory_backend_capabilities)>=64:_memory_backend_capabilities.clear()
  _memory_backend_capabilities[identity]=info
 return info

def _check_memory_backend_receipt(report,backend):
 expected={'segmented24':'EXPERIMENTAL_SEGMENTED24','permutation_bitmap':'EXPERIMENTAL_PERMUTATION_BITMAP'}.get(backend)
 if expected is not None and report.get('backend')!=expected:
  raise RuntimeError('NATIVE_MEMORY_BACKEND_RECEIPT_MISMATCH: expected '+expected+', got '+str(report.get('backend')))

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
  from .metadata_io import read_json
  value=read_json(path)
 except ValueError:return None
 if not isinstance(value,dict) or value.get('schema')!=1 or value.get('pid')!=job.pid:return None
 if not isinstance(value.get('entered'),dict) or not isinstance(value.get('phase'),str):return None
 return value

def _phase_failures(phases):
 failures=[]
 for rank,phase in enumerate(phases):
  error=phase.get('native_error') if isinstance(phase,dict) else None
  if isinstance(error,dict) and type(error.get('code')) is int and error['code']!=0:failures.append((rank,1))
 return failures

from .work_limits import normalize, wire, remaining as remaining_work, phase_expired

def run_graph(graph,output,*,device=None,devices=None,capacity=None,max_seconds=None,executable=None,shards=None,autotune=True,backend='auto',memory_backend='legacy',segment_rows=None,memory_preset='balanced',segment_strategy=None,peer_transport='auto',transport='auto',candidate_order='auto',_batch=None,_profile_layers=None,_native_env=None):
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
 max_seconds=normalize(max_seconds)
 if shards is not None and (type(shards) is not int or not 1<=shards<=4096):raise ValueError('INVALID_SHARDS')
 if type(autotune) is not bool:raise ValueError('INVALID_AUTOTUNE')
 if segment_rows is not None and (type(segment_rows) is not int or not 1<=segment_rows<=1<<32):raise ValueError('INVALID_SEGMENT_ROWS')
 if backend not in ('auto','generic','shard_ab_hash','shard_ab_sort_merge'):raise ValueError('INVALID_BACKEND')
 memory_backend=_normalize_memory_backend(memory_backend)
 strategy_source=_native_env if _native_env is not None else os.environ
 if segment_strategy is None:segment_strategy=strategy_source.get('MGBFS_SEGMENT_STRATEGY','atomic')
 segment_strategy=_normalize_segment_strategy(segment_strategy)
 if memory_backend=='legacy' and segment_strategy!='atomic':raise ValueError('SEGMENT_STRATEGY_REQUIRES_SEGMENTED_BACKEND')
 if memory_preset not in ('balanced','max_capacity'):raise ValueError('INVALID_MEMORY_PRESET')
 if memory_backend=='permutation_bitmap':
  if graph.action['kind']!='permutation':raise ValueError('BITMAP_REQUIRES_PERMUTATION_ACTION')
  if not 1<=len(graph.start)<=64:raise ValueError('BITMAP_UNSUPPORTED_DEGREE')
 if peer_transport not in ('auto','host','lsa'):raise ValueError('INVALID_PEER_TRANSPORT')
 if peer_transport=='lsa' and not backend.startswith('shard_ab_'):raise ValueError('LSA_REQUIRES_SPECIALIZED_BACKEND')
 if transport not in ('auto','full','parent'):raise ValueError('INVALID_TRANSPORT')
 if candidate_order not in ('auto','none','radix'):raise ValueError('INVALID_CANDIDATE_ORDER')
 if capacity is not None and (type(capacity) is not int or not 1<=capacity<=(0xffffffffffffffff if memory_backend in ('segmented24','permutation_bitmap') else 0x7ffffffe)):raise ValueError('INVALID_CAPACITY')
 from .host_memory import admit
 host_memory=admit(graph,output=output) if os.environ.get('WORLD_SIZE','1')=='1' else None
 native,native_env=native_runtime(executable)
 if _native_env is not None:native_env=dict(_native_env)
 native_env=_apply_memory_preset(native_env,memory_backend,memory_preset)
 segmented=(memory_backend in ('segmented24','permutation_bitmap'))
 if not segmented:
  # An inherited private experiment flag must not override an explicit API choice.
  native_env.pop('MGBFS_EXPERIMENTAL_SEGMENTED24',None);native_env.pop('MGBFS_EXPERIMENTAL_BITMAP',None);native_env.pop('MGBFS_SEGMENT_MAX_RANK_STATES',None)
 if segmented:
  if segment_rows is not None:native_env['MGBFS_SEGMENT_MAX_ROWS']=str(segment_rows)
  else:native_env.setdefault('MGBFS_SEGMENT_MAX_ROWS',str(1<<32))
  if backend not in ('auto','generic') or peer_transport=='lsa' or transport not in ('auto','full') or candidate_order not in ('auto','none'):raise ValueError('SEGMENTED_BACKEND_OPTION_CONFLICT')
  if os.environ.get('WORLD_SIZE','1')!='1':raise ValueError('SEGMENTED_EXTERNAL_RANK_NOT_VERIFIED')
  capabilities=_require_memory_backend(native,native_env,memory_backend)
  if segment_strategy!='atomic':
   supported=capabilities.get('segmented_strategies_v1')
   if not isinstance(supported,dict) or supported.get(segment_strategy) is not True:
    raise RuntimeError('NATIVE_SEGMENT_STRATEGY_UNAVAILABLE: '+segment_strategy+' requires a matching verified native executable and CUDA library')
  native_env['MGBFS_SEGMENT_STRATEGY']=segment_strategy
  exchange_capable=capabilities.get('segmented_exact_exchange_v1') is True
  if native_env.get('MGBFS_SEGMENT_EXACT_EXCHANGE')=='1' and not exchange_capable:raise RuntimeError('NATIVE_SEGMENTED_EXCHANGE_UNAVAILABLE: update the native runtime before requesting exact exchange')
  native_env['_MGBFS_SEGMENT_EXCHANGE_CAPABLE']='1' if exchange_capable else '0'
  native_env=dict(native_env,MGBFS_EXPERIMENTAL_SEGMENTED24='1',MGBFS_GENERIC_HISTORY='hash',MGBFS_GENERIC_GENERATOR='cuda',MGBFS_GENERIC_TRANSPORT='full',MGBFS_GENERIC_SORT='none')
  if memory_backend=='permutation_bitmap':native_env['MGBFS_EXPERIMENTAL_BITMAP']='1'
  else:native_env.pop('MGBFS_EXPERIMENTAL_BITMAP',None)
  if capacity is not None:
   prior=int(native_env.get('MGBFS_SEGMENT_MAX_RANK_STATES',str(capacity)))
   native_env['MGBFS_SEGMENT_MAX_RANK_STATES']=str(min(prior,capacity))
   # Legacy graph-info capacity is only used to size bounded transport; it
   # must not become an implicit rank-wide segment capacity.
   capacity=None
 communication=None
 if _profile_layers is None and os.environ.get('WORLD_SIZE','1')=='1' and not backend.startswith('shard_ab_'):
  if Path(output).exists():raise FileExistsError(output)
  selected=devices
  if selected is None:
   count=native_query([native,'graph-count'],env=native_env,capture_output=True,text=True)
   if count.returncode:raise RuntimeError('NATIVE_GPU_DISCOVERY_FAILED: '+count.stderr[-4000:])
   selected=list(range(int(count.stdout.strip())))
  from .nccl_preflight import prepare
  # The transport smoke graph must not inherit an explicit production-state
  # cap: a tiny cap is a valid request for the real BFS, not an NCCL failure.
  preflight_env=dict(native_env);segment_cap=preflight_env.pop('MGBFS_SEGMENT_MAX_RANK_STATES',None)
  native_env,communication=prepare(native,selected,preflight_env,output,graph)
  if segment_cap is not None:native_env['MGBFS_SEGMENT_MAX_RANK_STATES']=segment_cap
 geometry_bound=None;shard_geometry=None
 if capacity is None and _profile_layers is None and not segmented and not backend.startswith('shard_ab_'):
  from .workload_geometry import certified_frontier_bound
  geometry_bound=certified_frontier_bound(graph)
  if geometry_bound is not None:
   from .autotune import _admit
   with tempfile.TemporaryDirectory(prefix='mgbfs-bound-') as bound_dir:
    bound_admission=_admit(graph,devices,None,shards or 1,native,native_env,bound_dir)
   capacity=min(geometry_bound['maximum_frontier'],bound_admission['plan']['capacity'])

 native_env=dict(native_env,MGBFS_PEER_TRANSPORT=peer_transport)
 def finish(report):
  _check_memory_backend_receipt(report,memory_backend)
  if communication is not None:report['communication_preflight']=communication
  if geometry_bound is not None:report['workload_geometry_bound']=geometry_bound
  if shard_geometry is not None:report['shard_geometry']=shard_geometry
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
 fixed_segmented_tuning=segmented and autotune and shards is not None and _batch is None and _profile_layers is None
 if shards is None or fixed_segmented_tuning:
  if autotune:
   from .autotune import choose_size_profile,choose_profile
   # Reserve most of the caller's deadline for BFS, not cold tuning.
   tuning_budget=60 if max_seconds is None else min(60,max(1,max_seconds//3))
   native_env=dict(native_env,MGBFS_TUNING_EVIDENCE_ROOT=str(Path(output).parent/(Path(output).name+'-tuning-evidence')))
   profile=(choose_profile(graph,devices,capacity,tuning_budget,native,native_env,allow_specialized=False,fixed_shards=shards if fixed_segmented_tuning else None) if segmented else choose_size_profile(graph,devices,capacity,tuning_budget,native,native_env,allow_specialized=backend=='auto'))
   shards=profile['shards']
   if segmented:shard_geometry=profile.get('shard_geometry')
   if profile.get('backend','generic')!='generic':
    remaining=remaining_work(max_seconds,profile['seconds'])
    return finish(_run_specialized_admitted(graph,output,native,native_env,devices,capacity,remaining,profile['backend'],_batch,_profile_layers,profile))
   from .autotune import _profile_environment
   native_env=_profile_environment(native_env,profile)
   if (profile.get('measured') or profile.get('status')=='REUSED_COMPATIBLE_GPU_PROFILE') and profile.get('backend','generic')=='generic':_batch=profile['batch']
   max_seconds=remaining_work(max_seconds,profile['seconds'])
  elif segmented:
   from .autotune import _segmented_shard_plan
   with tempfile.TemporaryDirectory(prefix='mgbfs-segment-shards-') as shard_dir:
    shard_geometry,_=_segmented_shard_plan(graph,devices,capacity,native,native_env,shard_dir)
   shards=shard_geometry['admitted_shards']
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
  if segmented:
   # Scale a known maximum frontier per bank to the admitted GPU and shard
   # count. Other graph families are sized natively from their orbit bound.
   from .workload_geometry import coxeter_frontier_bound
   frontier=coxeter_frontier_bound(graph) if admission['plan'].get('history_layers',3)==3 else None
   if frontier:
    divisor=4*len(devices)*shards
    expected=(frontier['maximum_frontier']*5+divisor-1)//divisor
    native_env['MGBFS_SEGMENT_EXPECTED_ROWS']=str(max(32,expected))
  if memory_backend=='permutation_bitmap':
   # graph-info derives hash cut points, while bitmap uses exact rank modulo
   # world. Never send irrelevant hash cuts into the bitmap runtime.
   admission['owner_cuts']=None
  if admission.get('owned_resource_admission'):native_env=dict(native_env,MGBFS_REQUIRE_OWNED_RAW='1')
  if profile and profile.get('size_profiles'):admission.update(size_profiles=profile['size_profiles'],online_size_tuning=profile.get('online_size_tuning',False))
  if _profile_layers is not None:admission['profile_max_layers']=_profile_layers
  if admission['graph_digest']!=digest:raise RuntimeError('GRAPH_ADMISSION_IDENTITY')
  if not segmented and max_seconds is not None and admission['plan'].get('generator_backend','cuda')!='gemm' and admission['plan'].get('history_algorithm','HASH')=='HASH' and len(devices)==1 and shards==1 and _profile_layers is None and not (profile and profile.get('size_profiles')) and admission['plan'].get('history_layers',1)==1 and not admission['plan'].get('parent_transport',False) and not admission['plan'].get('sort_candidates',False):
   command=[str(native),'graph',str(definition),str(output),'--device',str(devices[0]),'--seconds',str(wire(max_seconds))]
   if capacity is not None:command+=['--capacity',str(capacity)]
   process=native_run(command,capture_output=True,text=True,env=native_env,timeout=None if max_seconds is None else max_seconds+180)
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
    jobs.append(launch_rank([str(native),'graph-rank',str(definition),str(configuration),str(rank),str(bootstrap),str(output/f'rank-{rank}'),str(wire(max_seconds))],env,log,devices,_selection=selection))
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
      phase_expired(phase,time.time(),max_seconds)
      for phase in phases if phase is not None)
     if timed_out or (not all(phases) and now-started>(max_seconds or 0)+1200):raise RuntimeError('DISTRIBUTED_COMPLETION_TIMEOUT: worker state remains in rank logs; no restart')
     next_phase_check=now+.05
    time.sleep(.002)
   if any(p.returncode for p in jobs):raise RuntimeError('NATIVE_DISTRIBUTED_RANK_FAILED')
   failed=_phase_failures([_job_phase(p,output,rank) for rank,p in enumerate(jobs)])
   if failed:raise RuntimeError('NATIVE_DISTRIBUTED_RANK_FAILED '+repr(failed))

  except BaseException as error:
   failure=error
   from .failure_evidence import preserve_native_failure
   preserve_native_failure(jobs,output,error,when='before_cleanup')
   if isinstance(error,RuntimeError):error.args=(str(error)+'; evidence='+str(output/'failure-evidence'),)
   raise
  finally:
   for p in jobs:
    if p.poll() is None:p.terminate()
   for p in jobs:
    if p.poll() is None:
     try:p.wait(timeout=10)
     except subprocess.TimeoutExpired:p.kill();p.wait()
   if failure is not None:
    from .failure_evidence import preserve_native_failure
    preserve_native_failure(jobs,output,failure,when='after_cleanup')
   for log in logs:log.close()
  parts=[_receipt(output/f'rank-{rank}',digest) for rank in range(len(devices))]
  for rank,p in enumerate(parts):
   if p['rank']!=rank or p['world']!=len(devices) or p['device']!=devices[rank] or p['plan']!=admission.get('rank_plans',[admission['plan']]*len(devices))[rank]:raise RuntimeError('DISTRIBUTED_RANK_GEOMETRY_MISMATCH')
   if any(p[key]!=parts[0][key] for key in ('status','reason','layer_sizes')):raise RuntimeError('DISTRIBUTED_LAYER_CONSENSUS_MISMATCH')
  snapshots=[json.loads((output/f'rank-{rank}'/'states.json').read_text()) for rank in range(len(devices))]
  current=[v for p in snapshots for v in p['current']];previous=[v for p in snapshots for v in p['previous_small']]
  if len(current)>1000 or len(previous)>=1000:raise RuntimeError('DISTRIBUTED_RETENTION_BOUND')
  states={'schema':2,'state_encoding':'signed_int64_vectors','current':current,'previous_small':previous,'current_sample_limit':1000};raw=json.dumps(states).encode();(output/'states.json').write_bytes(raw)
  report=dict(parts[0]);report.update(rank_plans=[p['plan'] for p in parts],owner_cuts=admission.get('owner_cuts'));report.pop('rank');report.pop('device')
  if segmented:
   report['segment_memory_by_rank']=[dict(p['segment_memory'],rank=i,device=devices[i]) for i,p in enumerate(parts)]
   report['segment_memory_scope']='rank 0 only; see segment_memory_by_rank for every selected GPU'
  report.update(devices=devices,states_sha256=hashlib.sha256(raw).hexdigest(),rank_receipts=[f'rank-{r}/report.json' for r in range(len(devices))],bfs_seconds=max(p['bfs_seconds'] for p in parts),setup_seconds=max(p['setup_seconds'] for p in parts),launch_wall_seconds=time.monotonic()-started,rank_worker_pids=[p.pid for p in jobs],scope='single host general exact retained-history path; bounded profile evidence in autotune receipt when enabled; larger hardware not verified')
  if profile:report['autotune']=profile;report['profile_status']=profile['status']
  (output/'report.json.tmp').write_text(json.dumps(report,indent=2));(output/'report.json.tmp').replace(output/'report.json')
  return finish(_receipt(output,digest))


def _run_specialized_admitted(graph,output,native,env,devices,capacity,max_seconds,backend,batch,profile_layers,profile):
 from .specialized import admit_specialized,match_lrx,run_specialized,exact_specialized_supported
 if not exact_specialized_supported(graph):raise RuntimeError('SPECIALIZED_LOSSLESS_KEY_DOMAIN_UNSUPPORTED: no proved injective key for this graph')
 if match_lrx(graph) is None:raise ValueError('SPECIALIZED_GRAPH_UNSUPPORTED')
 output=Path(output).absolute();output.parent.mkdir(parents=True,exist_ok=True)
 if output.exists():raise FileExistsError(output)
 started=time.monotonic()
 with tempfile.TemporaryDirectory(prefix='mgbfs-specialized-admit-',dir=output.parent) as temporary:
  definition=Path(temporary)/'definition.json';definition.write_text(graph.to_json());selection='auto' if devices is None else ','.join(map(str,devices))
  probe=native_query([native,'graph-info',str(definition),selection,'1'],capture_output=True,text=True,env=env)
  if probe.returncode:raise RuntimeError('NATIVE_GRAPH_ADMISSION_FAILED: '+probe.stderr[-4000:])
  inventory=json.loads(probe.stdout)
  if inventory['graph_digest']!=graph.digest():raise RuntimeError('GRAPH_ADMISSION_IDENTITY')
  selected=inventory['devices'];mode='HASH' if backend=='shard_ab_hash' else 'SORT_MERGE'
  from .transport import select_peer_transport
  request=('lsa' if profile['specialized_transport']=='NCCL_LSA' else 'host') if profile and profile.get('specialized_transport') else env.get('MGBFS_PEER_TRANSPORT','auto')
  decision=select_peer_transport(native,env,selected,Path(temporary)/'transport',request=request,seconds=60 if max_seconds is None else max_seconds)
  env=dict(env,MGBFS_SPECIALIZED_TRANSPORT=decision['selected'])
  if decision['selected']=='NCCL_LSA':env['NCCL_CUMEM_ENABLE']='1'
  plan=admit_specialized(graph,native,env,selected,Path(temporary)/'queries',capacity=capacity,mode=mode,seconds=60 if max_seconds is None else max(1,max_seconds-(time.monotonic()-started)))
  actual_batch=plan['batch'] if batch is None else batch
  if actual_batch!=plan['batch']:raise ValueError('SPECIALIZED_BATCH_REQUIRES_MATCHED_ADMISSION')
  report=run_specialized(graph,output,native=native,env=env,devices=selected,capacity=plan['capacity'],batch=actual_batch,max_seconds=remaining_work(max_seconds,time.monotonic()-started),mode=mode,profile_layers=profile_layers)
  report['plan'].update(plan);report['startup_admission_seconds']=time.monotonic()-started-report['launch_wall_seconds']
  report['transport_selection']=decision
  if profile:report['autotune']=profile
  (output/'report.json').write_text(json.dumps(report,indent=2));return _receipt(output,graph.digest())
