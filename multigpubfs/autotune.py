"""Bounded same-graph GPU prefix tuning; no CPU successor or dedup in production."""
import hashlib,json,math,os,subprocess,tempfile,time
from pathlib import Path
from .artifact_paths import artifact_path
from .process_control import ProfileBudgetExpired, profile_budget, profile_timeout, query as native_query
from .profile_cache import read_mapping as _read_profile_cache, write_mapping as _write_profile_cache

def _admit(graph,devices,capacity,shards,native,env,directory,*,batch=None):
 path=Path(directory)/'definition.json';path.write_text(graph.to_json())
 inventory=env.get('_MGBFS_PROFILE_INVENTORY')
 if inventory:
  inventory_path=Path(directory)/'inventory.json';inventory_path.write_text(inventory)
  command=[native,'graph-plan',str(path),str(inventory_path),str(shards),'auto' if capacity is None else str(capacity)]
 else:
  command=[native,'graph-info',str(path),'auto' if devices is None else ','.join(map(str,devices)),str(shards)]
  if capacity is not None:command.append(str(capacity))
 if batch is not None:
  if capacity is None and command[1]=='graph-info':command.append('auto')
  command.append(str(batch))
 p=native_query(command,capture_output=True,text=True,env=env)
 if p.returncode:raise RuntimeError('PROFILE_ADMISSION_FAILED: '+p.stderr[-2000:])
 return json.loads(p.stdout)

def _system_info(command):
 # Only immutable hardware metadata is cached. Free VRAM and rank health
 # remain live native admission queries on every production launch.
 from .generic_session import _active
 immutable=tuple(command) in (
  ('nvidia-smi','--query-gpu=uuid,driver_version,name','--format=csv,noheader'),
  ('nvidia-smi','topo','-m'),
  ('nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'))
 cache=None
 if _active is not None and immutable:
  cache=getattr(_active,'hardware_metadata',None)
  if cache is None:cache={};_active.hardware_metadata=cache
  key=(tuple(command),tuple(os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER')), _active.generation)
  if key in cache:return subprocess.CompletedProcess(command,0,cache[key],'')
 try:result=subprocess.run(command,capture_output=True,text=True,timeout=profile_timeout(10))
 except (OSError,subprocess.TimeoutExpired):return subprocess.CompletedProcess(command,1,'unavailable','')
 if cache is not None and result.returncode==0:
  if len(cache)>=32:cache.clear()
  cache[key]=result.stdout
 return result

def _content_digest(*paths):
 from .generic_session import _active
 cache=None
 if _active is not None:
  cache=getattr(_active,'selector_digests',None)
  if cache is None:cache={};_active.selector_digests=cache
  key=tuple((str(Path(p).resolve()),_stat_identity(p)) for p in paths)
  if key in cache:return cache[key]
 digest=hashlib.sha256()
 for path in paths:digest.update(Path(path).read_bytes())
 result=digest.hexdigest()
 if cache is not None:
  if len(cache)>=32:cache.clear()
  cache[key]=result
 return result

def _gemm_hardware_available():
 caps=_system_info(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'])
 try:return caps.returncode==0 and bool(caps.stdout.strip()) and all(float(v.strip())>=8.0 for v in caps.stdout.splitlines())
 except ValueError:return False

def _dependency_identity_uncached(native,env):
 from .native_distribution import _file_digest
 try:resolved=subprocess.run(['ldd',native],capture_output=True,text=True,env=env,timeout=profile_timeout(10))
 except (OSError,subprocess.TimeoutExpired):return None
 dependencies=[]
 for line in resolved.stdout.splitlines():
  fields=line.split()
  if len(fields)>=3 and fields[1]=='=>' and Path(fields[2]).is_file():
   dep=Path(fields[2]).resolve();stat=dep.stat();dependencies.append([fields[0],str(dep),stat.st_size,stat.st_mtime_ns])
 for line in resolved.stdout.splitlines():
  fields=line.split()
  if len(fields)>=3 and fields[0]=='libmgbfs_cuda.so' and fields[1]=='=>':
   path=Path(fields[2])
   if path.is_file():return {'cuda_library_sha256':_file_digest(path),'resolved_dependencies':dependencies}
 return None

_dependency_cache={}
def _stat_identity(path):
 try:
  s=Path(path).stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 except OSError:return None

def _dependency_identity(native,env):
 from .generic_session import _active
 from .cluster import current_cluster
 if _active is None and current_cluster() is None:return _dependency_identity_uncached(native,env)
 search=tuple(env.get(k,'') for k in ('LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT'))
 directories=tuple((x,_stat_identity(x)) for x in search[0].split(':') if x)
 key=(str(native),_stat_identity(native),search,directories)
 cached=_dependency_cache.get(key)
 if cached is not None:
  value,stamps=cached
  if all(_stat_identity(path)==stamp for path,stamp in stamps):return value
 value=_dependency_identity_uncached(native,env)
 if value is not None:
  stamps=tuple((x[1],_stat_identity(x[1])) for x in value['resolved_dependencies'])
  _dependency_cache.clear();_dependency_cache[key]=(value,stamps)
 return value

def _transport_variants(graph,env,hardware_shards=None):
 forced=env.get('MGBFS_GENERIC_TRANSPORT')
 if forced not in (None,'full','parent'):raise ValueError('INVALID_TRANSPORT')
 order=env.get('MGBFS_GENERIC_SORT')
 if order not in (None,'none','radix'):raise ValueError('INVALID_CANDIDATE_ORDER')
 mode=forced or 'full';order_default=order or 'none';shard_options=hardware_shards or [1,4,16];central=shard_options[len(shard_options)//2];variants=[(sh,1.0,mode,order_default) for sh in shard_options]+[(central,.25,mode,order_default)]
 packed=graph.action['kind']=='permutation' and graph.state_elements<=24 and all(0<=v<=255 for v in graph.start)
 if forced is None and not packed:variants += [(central,1.0,'parent',order_default),(central,.25,'parent',order_default)]
 if order is None:variants.append((central,.25,'parent' if forced is None and not packed else mode,'radix'))
 history=env.get('MGBFS_GENERIC_HISTORY')
 if history not in (None,'hash','sorted'):raise ValueError('INVALID_HISTORY_ALGORITHM')
 if history=='sorted':
  lane=int(env.get('MGBFS_GENERIC_OWNER_LANES','1'))
  if not 1<=lane<=8:raise ValueError('INVALID_OWNER_LANES')
  variants=list(dict.fromkeys((max(sh,lane),f,tr,'none') for sh,f,tr,_ in variants))
 return variants

def _configuration_digest(env):
 # Values are hashed, never copied into a public receipt (tokens may coexist).
 ignored={'MGBFS_PROFILE_CACHE','MGBFS_CONTROL_TOKEN','MGBFS_RUN_ID','MGBFS_TUNING_EVIDENCE_ROOT','MGBFS_ARTIFACT_ROOT','MGBFS_JOB_PHASE_PATH','MGBFS_JOB_SEQUENCE','MGBFS_JOB_CANCEL_PATH'}
 values={k:v for k,v in env.items() if (k.startswith(('NCCL_','MGBFS_','_MGBFS_PRESET_IMPLICIT_')) or k in ('CUDA_VISIBLE_DEVICES','CUDA_MODULE_LOADING')) and k not in ignored}
 for address in ('MGBFS_SEGMENT_PACKET_CONTROL_ADDR','MGBFS_SEGMENT_PACKET_BIND_ADDR'):
  if address in values:values[address]='configured'
 return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()

def _segmented_deep_prefix_candidates(pilots):
 # A stalled candidate must not force all comparisons onto tiny early layers.
 # Always check its completed prefix before excluding it.
 if not pilots:return [],[]
 reference=max(pilots,key=lambda p:len(p['layer_seconds']))['layer_sizes']
 for p in pilots:
  d=len(p['layer_seconds'])
  if p['layer_sizes'][:d+1]!=reference[:d+1]:raise RuntimeError('AUTOTUNE_PREFIX_CORRECTNESS_MISMATCH')
 max_depth=max(len(p['layer_seconds']) for p in pilots)
 threshold=(3*max_depth+3)//4
 deep=[p for p in pilots if len(p['layer_seconds'])>=threshold]
 if len(deep)<2:return pilots,[]
 excluded=[dict(p,completed_depth=len(p['layer_seconds']),minimum_comparable_depth=threshold,reason='SHORT_COMMON_PREFIX_RELATIVE_TO_BEST_COMPLETED_DEPTH') for p in pilots if len(p['layer_seconds'])<threshold]
 return deep,excluded

def _segmented_shard_plan(graph,devices,capacity,native,env,directory,baseline=None,fixed_shards=None,*,minimum_shards=1):
 from .shard_policy import choose_segmented_shards
 if baseline is None:baseline=_admit(graph,devices,capacity,fixed_shards or 1,native,env,directory)
 try:target_rows=int(env.get('MGBFS_SEGMENT_SHARD_TARGET_ROWS',str(1<<32)))
 except (TypeError,ValueError):raise ValueError('INVALID_SHARD_TARGET_ROWS') from None
 try:rank_limit=int(env['MGBFS_SEGMENT_MAX_RANK_STATES']) if 'MGBFS_SEGMENT_MAX_RANK_STATES' in env else None
 except (TypeError,ValueError):raise ValueError('INVALID_SEGMENT_RANK_LIMIT') from None
 try:segment_rows=int(env.get('MGBFS_SEGMENT_ROWS',env.get('MGBFS_SEGMENT_MAX_ROWS',str(1<<32))))
 except (TypeError,ValueError):raise ValueError('INVALID_SEGMENT_ROWS') from None
 geometry=choose_segmented_shards(graph,baseline,bitmap=env.get('MGBFS_EXPERIMENTAL_BITMAP')=='1',target_rows=min(target_rows,segment_rows),rank_limit=rank_limit,env=env,minimum_shards=minimum_shards)
 if fixed_shards is not None:
  geometry.update(admitted_shards=fixed_shards,shards_fixed_by_user=True,admission_rejections=[])
  return geometry,baseline
 shards=geometry['target_shards'];rejected=[]
 while True:
  try:
   admitted=baseline if shards==1 else _admit(graph,devices,capacity,shards,native,env,directory)
   geometry.update(admitted_shards=shards,admission_rejections=rejected)
   return geometry,admitted
  except RuntimeError as error:
   if shards<=1 or not any(code in str(error) for code in ('REQUESTED_CAPACITY_EXCEEDS_ADMISSION','GENERIC_DISTRIBUTED_NO_CAPACITY','GENERIC_DISTRIBUTED_HEADROOM','GENERIC_VRAM_CAPACITY')):
    raise
   rejected.append({'shards':shards,'reason':str(error)})
   shards=max(1,shards//2)

def _segmented_capacity_priorities(pilots):
 # Old internal callers without detailed receipts remain usable only when all
 # candidates lack them. Production segmented pilots include per-rank data.
 detailed=['capacity_limit_per_rank_by_rank' in p for p in pilots]
 if any(detailed) and not all(detailed):raise RuntimeError('SEGMENTED_CAPACITY_EVIDENCE_MISSING')
 if any(type(p.get('capacity_score_rows')) is not int or p['capacity_score_rows']<0 for p in pilots):
  raise RuntimeError('SEGMENTED_CAPACITY_EVIDENCE_MISSING')
 if not any(detailed):
  return [(p['capacity_score_rows'],p.get('capacity_equivalent_rows',0)) for p in pilots]
 if any(type(p.get('capacity_limit_scope')) is not str for p in pilots):raise RuntimeError('SEGMENTED_CAPACITY_SCOPE_MISMATCH')
 scopes={p.get('capacity_limit_scope') for p in pilots}
 if len(scopes)!=1 or not scopes.issubset({'one next layer per rank','all visited states per rank'}):
  raise RuntimeError('SEGMENTED_CAPACITY_SCOPE_MISMATCH')
 common_limits=None;priorities=[]
 for p in pilots:
  rows=p.get('capacity_score_rows_by_rank');equiv=p.get('capacity_equivalent_rows_by_rank');limits=p.get('capacity_limit_per_rank_by_rank')
  if not isinstance(rows,list) or not rows or not isinstance(equiv,list) or not isinstance(limits,list) or not len(rows)==len(equiv)==len(limits):
   raise RuntimeError('SEGMENTED_CAPACITY_EVIDENCE_MISSING')
  if any(type(x) is not int or x<0 for x in rows+equiv) or any(x is not None and (type(x) is not int or x<=0) for x in limits):
   raise RuntimeError('SEGMENTED_CAPACITY_EVIDENCE_INVALID')
  if common_limits is None:common_limits=tuple(limits)
  elif tuple(limits)!=common_limits:raise RuntimeError('SEGMENTED_CAPACITY_LIMIT_MISMATCH')
  bounded=lambda xs:min(v if cap is None else min(v,cap) for v,cap in zip(xs,limits))
  priorities.append((bounded(rows),bounded(equiv)))
 return priorities

def _pick_segmented_profile(pilots,scores,preset):
 if preset=='balanced':return min(range(len(scores)),key=lambda i:scores[i])
 if preset!='max_capacity':raise ValueError('INVALID_MEMORY_PRESET')
 priorities=_segmented_capacity_priorities(pilots)
 # The segment budget is a planning estimate, not a physical pressure proof.
 # Limit each rank in the same logical units before using time as a tie-break.
 return min(range(len(scores)),key=lambda i:(-priorities[i][0],-priorities[i][1],scores[i]))

def _balanced_segment_environment(env):
 balanced=dict(env,MGBFS_MEMORY_PRESET='balanced')
 for marker,fields in (
  ('_MGBFS_PRESET_IMPLICIT_LANES',('MGBFS_SEGMENT_OWNER_LANES',)),
  ('_MGBFS_PRESET_IMPLICIT_QUEUE',('MGBFS_SEGMENT_QUEUE_CAP',)),
  ('_MGBFS_PRESET_IMPLICIT_PACKET',('MGBFS_SEGMENT_PACKET_CONTROL_ADDR',)),
  ('_MGBFS_PRESET_IMPLICIT_LAYOUT',('MGBFS_SEGMENT_COMPACT_POS','MGBFS_BITMAP_BANKS')),
 ):
  if balanced.pop(marker,None)=='1':
   for field in fields:balanced.pop(field,None)
 return balanced

def _bounded_segmented_profile(pilots,limit):
 """Compare each capacity candidate with actual balanced runs on one fixed work window."""
 baseline=[i for i,p in enumerate(pilots) if p['segment_runtime_preset']=='balanced']
 if not baseline or len(baseline)==len(pilots):return None
 depth=min(len(p['layer_seconds']) for p in pilots)
 def full_time(p):
  elapsed=p.get('bfs_seconds')
  if p['status']=='COMPLETE' and type(elapsed) in (int,float) and 0<elapsed<float('inf'):return elapsed
  return None
 full_times=[full_time(p) for p in pilots]
 full_window=all(t is not None and len(p['layer_seconds'])==depth for p,t in zip(pilots,full_times))
 scores=full_times if full_window else [sum(p['layer_seconds'][1:depth]) for p in pilots]
 if any(not math.isfinite(t) or t<=0 for t in scores):
  raise RuntimeError('SEGMENTED_TUNER_INVALID_PREFIX_TIME')
 prefix_base=min(baseline,key=scores.__getitem__)
 full_bases={}
 for i in baseline:
  elapsed=full_times[i]
  if elapsed is None:continue
  completed_depth=len(pilots[i]['layer_seconds'])
  previous=full_bases.get(completed_depth)
  if previous is None or elapsed<full_times[previous]:
   full_bases[completed_depth]=i
 timings=[]
 for p,t,score in zip(pilots,full_times,scores):
  completed_depth=len(p['layer_seconds'])
  full_base=full_bases.get(completed_depth) if t is not None else None
  if full_base is not None:
   timings.append((full_times[full_base],t,True,full_base))
  else:
   timings.append((scores[prefix_base],score,False,prefix_base))
 eligible=[i for i,(b,t,_,_) in enumerate(timings) if t<=limit*b]
 priorities=_segmented_capacity_priorities(pilots)
 winner=min(eligible,key=lambda i:(-priorities[i][0],-priorities[i][1],timings[i][1]/timings[i][0]))
 baseline_time,selected_time,full,base=timings[winner]
 base_depth=len(pilots[base]['layer_seconds'])
 evidence={
  'status':'MEASURED_FULL_BFS_BOUND' if full else 'MEASURED_MATCHED_PREFIX_BOUND',
  'scope':('measured balanced anchor on the same backend, devices and entire completed BFS'
           if full else 'measured balanced anchor on the same backend, devices and completed graph layers; first layer excluded'),
  'timing_scope':'FULL_BFS' if full else 'COMPLETED_LAYER_PREFIX',
  'limit':limit,'depth_start':1 if full else 2,'depth_end':base_depth if full else depth,
  'baseline_candidate':pilots[base]['candidate'],
  'selected_candidate':pilots[winner]['candidate'],
  'baseline_geometry':{key:pilots[base].get(key) for key in (
   'shards','batch','segment_owner_lanes','segment_queue_cap','segment_exact_exchange')},
  'baseline_seconds':baseline_time,'selected_seconds':selected_time,
  'observed_ratio':selected_time/baseline_time,
  'rejected_over_limit':[p['candidate'] for i,p in enumerate(pilots) if i not in eligible],
 }
 return pilots,scores,winner,depth,evidence

def _validate_segmented_resource_report(report,output,digest,devices,env,state_elements):
 # run_graph already checked rank identities and state checksums; recheck them
 # while the bounded pilot directory is still alive, before calling it a
 # normal resource stop. This is not a substitute for an independent oracle.
 from .launch import _receipt
 if report.get('status')!='INCOMPLETE' or report.get('reason')!='RESOURCE_10':
  raise RuntimeError('SEGMENTED_TUNING_RESOURCE_UNSUPPORTED: '+str(report.get('reason')))
 output=Path(output);aggregate=_receipt(output,digest)
 fields=('graph_digest','status','reason','layer_sizes','layer_seconds','states_sha256','devices','rank_receipts','rank_plans')
 if any(aggregate.get(k)!=report.get(k) for k in fields):raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_MISMATCH')
 sizes=report.get('layer_sizes');seconds=report.get('layer_seconds')
 if not isinstance(sizes,list) or not isinstance(seconds,list) or len(sizes)!=len(seconds)+1 or not sizes or sizes[0]!=1 or any(type(v) is not int or v<1 for v in sizes):
  raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_INVALID_PREFIX')
 if any(type(v) not in (int,float) or not __import__('math').isfinite(v) or v<0 for v in seconds):
  raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_INVALID_TIMING')
 if report.get('devices')!=list(devices) or report.get('rank_receipts')!=[f'rank-{r}/report.json' for r in range(len(devices))]:
  raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_RANKS')
 rank_plans=report.get('rank_plans');rank_memory=report.get('segment_memory_by_rank')
 if not isinstance(rank_plans,list) or len(rank_plans)!=len(devices) or not isinstance(rank_memory,list) or len(rank_memory)!=len(devices):raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_RANKS')
 try:expected_lanes=int(env['MGBFS_SEGMENT_OWNER_LANES'])
 except (KeyError,TypeError,ValueError):raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_GEOMETRY') from None
 expected_exchange=env.get('MGBFS_SEGMENT_EXACT_EXCHANGE','0')=='1'
 expected_strategy=env.get('MGBFS_SEGMENT_STRATEGY','atomic')
 expected_cap=int(env['MGBFS_SEGMENT_MAX_RANK_STATES']) if 'MGBFS_SEGMENT_MAX_RANK_STATES' in env else None
 def snapshot(directory):
  value=json.loads((directory/'states.json').read_text())
  if not isinstance(value,dict) or value.get('schema')!=2 or value.get('state_encoding')!='signed_int64_vectors' or not isinstance(value.get('current'),list) or not isinstance(value.get('previous_small'),list):
   raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_SNAPSHOT')
  for row in value['current']+value['previous_small']:
   if not isinstance(row,list) or len(row)!=state_elements or any(type(x) is not int or not -(1<<63)<=x<(1<<63) for x in row):
    raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_SNAPSHOT')
  return value
 aggregate_snapshot=snapshot(output)
 checksums=[];rank_snapshots=[]
 for rank,device in enumerate(devices):
  directory=output/f'rank-{rank}'
  part=_receipt(directory,digest)
  if part.get('rank')!=rank or part.get('world')!=len(devices) or part.get('device')!=device or part.get('plan')!=rank_plans[rank]:
   raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_RANKS')
  if any(part.get(k)!=report.get(k) for k in ('status','reason','layer_sizes')):
   raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_MISMATCH')
  mem=part.get('segment_memory')
  if not isinstance(mem,dict) or dict(mem,rank=rank,device=device)!=rank_memory[rank]:
   raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_GEOMETRY')
  actual_lanes=mem.get('owner_lanes')
  if type(actual_lanes) is not int or not 1<=actual_lanes<=expected_lanes or mem.get('segment_strategy')!=expected_strategy:
   raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_GEOMETRY')
  actual_exchange=mem.get('exact_exchange')
  if env.get('_MGBFS_SEGMENT_EXCHANGE_CAPABLE')=='1':
   if type(actual_exchange) is not bool or actual_exchange!=expected_exchange:raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_GEOMETRY')
  elif expected_exchange or actual_exchange not in (None,False):raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_GEOMETRY')
  if expected_cap is not None and (mem.get('capacity_limit_per_rank')!=expected_cap or mem.get('capacity_limit_is_hard') is not True):
   raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_GEOMETRY')
  if mem.get('capacity_limit_scope') not in ('one next layer per rank','all visited states per rank'):
   raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_GEOMETRY')
  rank_snapshots.append(snapshot(directory));checksums.append(part['states_sha256'])
 for key in ('current','previous_small'):
  combined=[row for part in rank_snapshots for row in part[key]]
  if combined!=aggregate_snapshot[key]:raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_SNAPSHOT_RANK_MISMATCH')
 current=aggregate_snapshot['current'];previous=aggregate_snapshot['previous_small']
 if len(current)!=min(sizes[-1],1000):raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_SNAPSHOT_SIZE')
 expected_previous=sizes[-2] if len(sizes)>1 and sizes[-2]<1000 else 0
 if len(previous)!=expected_previous:raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_SNAPSHOT_SIZE')
 if len({tuple(row) for row in current})!=len(current) or len({tuple(row) for row in previous})!=len(previous):
  raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_SNAPSHOT_DUPLICATE')
 if {tuple(row) for row in current}.intersection(tuple(row) for row in previous):
  raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_SNAPSHOT_CROSS_LAYER_DUPLICATE')
 full_layers={}
 for depth,rows in ((len(sizes)-1,current),(len(sizes)-2,previous)):
  if depth>=0 and len(rows)==sizes[depth]:
   canonical=json.dumps(sorted(rows),separators=(',',':')).encode()
   full_layers[str(depth)]=hashlib.sha256(canonical).hexdigest()
 return {'validated_ranks':len(devices),'graph_digest':digest,'states_sha256':report['states_sha256'],
         'rank_states_sha256':checksums,'fully_retained_layer_sha256':full_layers,
         'snapshot_current_count':len(current),'snapshot_current_layer_size':sizes[-1],
         'validation_scope':'rank consensus, layer counts and bounded snapshot integrity; fully retained layers can be compared across probes; not an independent full-prefix state proof'}

def choose_profile(graph,devices,capacity,max_seconds,native,env,*,allow_specialized=True,fixed_shards=None,capacity_slowdown_limit=None,pilot_limit=None,pilot_wall_seconds=None,startup_deadline=None,preserve_implicit_bitmap_banks=False):
 if pilot_wall_seconds is not None and (type(pilot_wall_seconds) is not int or pilot_wall_seconds<2):raise ValueError('INVALID_PROFILE_PILOT_WALL_SECONDS')
 # A real balanced anchor drops the implicit layout and allocates a third bank.
 skip_balanced_anchor=(preserve_implicit_bitmap_banks and capacity_slowdown_limit is not None
  and env.get('MGBFS_EXPERIMENTAL_SEGMENTED24')=='1' and env.get('MGBFS_EXPERIMENTAL_BITMAP')=='1'
  and env.get('MGBFS_MEMORY_PRESET')=='max_capacity'
  and env.get('_MGBFS_PRESET_IMPLICIT_LAYOUT')=='1' and env.get('MGBFS_BITMAP_BANKS')=='2')
 started=time.monotonic() if pilot_wall_seconds is not None or startup_deadline is not None else None
 deadline=startup_deadline
 if pilot_wall_seconds is not None:
  local=started+min(max_seconds,pilot_wall_seconds)
  deadline=local if deadline is None else min(deadline,local)
 with profile_budget(deadline):
  try:
   return _choose_profile(graph,devices,capacity,max_seconds,native,env,
    allow_specialized=allow_specialized,fixed_shards=fixed_shards,capacity_slowdown_limit=capacity_slowdown_limit,
    pilot_limit=pilot_limit,pilot_wall_seconds=pilot_wall_seconds,budget_deadline=deadline,started=started,
    skip_balanced_anchor=skip_balanced_anchor)
  except ProfileBudgetExpired:
   result={'status':'TUNING_BUDGET_EXHAUSTED_CONSERVATIVE_PROFILE','shards':fixed_shards,
    'measured':False,'seconds':time.monotonic()-started}
   if skip_balanced_anchor:
    result.update(segment_runtime_preset='max_capacity',capacity_slowdown={
     'status':'UNVERIFIED','reason':'MATCHED_PREFIX_UNAVAILABLE','limit':capacity_slowdown_limit})
   return result


def _choose_profile(graph,devices,capacity,max_seconds,native,env,*,allow_specialized,fixed_shards,capacity_slowdown_limit,pilot_limit,pilot_wall_seconds,budget_deadline,started,skip_balanced_anchor):
 # Omitted peer policy and public default auto are the same configuration.
 env=dict(env);env.setdefault('MGBFS_PEER_TRANSPORT','auto')
 if capacity_slowdown_limit is not None and (type(capacity_slowdown_limit) not in (int,float) or not 1<=capacity_slowdown_limit<float('inf')):raise ValueError('INVALID_CAPACITY_SLOWDOWN_LIMIT')
 if pilot_limit is not None and (type(pilot_limit) is not int or pilot_limit<1):raise ValueError('INVALID_PROFILE_PILOT_LIMIT')
 from .specialized import exact_specialized_supported
 allow_specialized=allow_specialized and exact_specialized_supported(graph)
 from .generation import gemm_supported
 if fixed_shards is not None and (env.get('MGBFS_EXPERIMENTAL_SEGMENTED24')!='1' or type(fixed_shards) is not int or not 1<=fixed_shards<=4096):raise ValueError('INVALID_FIXED_SEGMENTED_SHARDS')
 if env.get('MGBFS_EXPERIMENTAL_SEGMENTED24')=='1' and env.get('MGBFS_SEGMENT_EXACT_EXCHANGE','0') not in ('0','1'):raise ValueError('INVALID_SEGMENTED_EXCHANGE_MODE')
 if env.get('MGBFS_EXPERIMENTAL_SEGMENTED24')=='1' and env.get('MGBFS_SEGMENT_EXACT_EXCHANGE')=='1' and env.get('_MGBFS_SEGMENT_EXCHANGE_CAPABLE')!='1':raise RuntimeError('NATIVE_SEGMENTED_EXCHANGE_UNAVAILABLE')
 forced_generator=env.get('MGBFS_GENERIC_GENERATOR','cuda')
 if forced_generator not in ('cuda','gemm'):raise ValueError('INVALID_GENERIC_GENERATOR')
 if forced_generator=='gemm' and not gemm_supported(graph):raise ValueError('GEMM_GRAPH_NOT_EXACTLY_SUPPORTED')
 if started is None:started=time.monotonic()
 profile_timeout(max_seconds)
 variants=_transport_variants(graph,env);default_transport=variants[0][2];default_order=variants[0][3]
 with _evidence_directory(env,'mgbfs-profile-') as temporary:
  baseline_shards=fixed_shards if fixed_shards is not None else variants[0][0]
  baseline=_admit(graph,devices,capacity,baseline_shards,native,env,temporary)
  selected=baseline['devices'];plan=baseline['plan']
  # Independent shard owners are scheduled on a bounded number of GPU lanes.
  # Both dimensions must be measured: increasing shards alone does not add lanes.
  segmented=env.get('MGBFS_EXPERIMENTAL_SEGMENTED24')=='1'
  bitmap=segmented and env.get('MGBFS_EXPERIMENTAL_BITMAP')=='1'
  capacity_preset=segmented and env.get('MGBFS_MEMORY_PRESET')=='max_capacity'
  bounded_capacity=capacity_preset and capacity_slowdown_limit is not None
  def fallback(result):
   if bounded_capacity:
    alternatives=[p for p in result.get('pilots',[]) if p['segment_runtime_preset']=='max_capacity']
    result.update(segment_runtime_preset='max_capacity',capacity_slowdown={
     'status':'UNVERIFIED','reason':'MATCHED_PREFIX_UNAVAILABLE','limit':capacity_slowdown_limit})
    if alternatives:
     priorities=_segmented_capacity_priorities(alternatives)
     candidate=alternatives[max(range(len(alternatives)),key=priorities.__getitem__)]
     result.update({k:candidate[k] for k in ('shards','batch','batch_fraction','transport','candidate_order',
      'segment_owner_lanes','segment_queue_cap','segment_exact_exchange')})
     result.update(apply_profile_batch=True,capacity_planning_candidate=candidate['candidate'])
   return result
  implicit_lanes=capacity_preset and env.get('_MGBFS_PRESET_IMPLICIT_LANES')=='1'
  implicit_queue=capacity_preset and env.get('_MGBFS_PRESET_IMPLICIT_QUEUE')=='1'
  queue_base=None
  if capacity_preset:
   try:queue_base=int(env['MGBFS_SEGMENT_QUEUE_CAP'])
   except (KeyError,TypeError,ValueError):raise ValueError('INVALID_SEGMENTED_QUEUE_CAP') from None
   if not 1<=queue_base<=0xffffffff:raise ValueError('INVALID_SEGMENTED_QUEUE_CAP')
  shard_geometry=None
  if segmented:
   shard_geometry,baseline=_segmented_shard_plan(graph,devices,capacity,native,env,temporary,baseline,fixed_shards=fixed_shards)
   baseline_shards=shard_geometry['admitted_shards'];selected=baseline['devices'];plan=baseline['plan']
  from .hardware_profiles import startup_policy
  hardware_policy=startup_policy(baseline.get('inventory',[]),plan)
  shard_choices=(list(dict.fromkeys([1,4,8]+hardware_policy['shards'])) if segmented else hardware_policy['shards'])
  variants=[(*v,None) for v in _transport_variants(graph,env,shard_choices)]
  if segmented:
   forced=None if implicit_lanes else env.get('MGBFS_SEGMENT_OWNER_LANES')
   if forced is not None:
    try:forced=int(forced)
    except ValueError:raise ValueError('INVALID_SEGMENTED_OWNER_LANES') from None
    if not 1<=forced<=64:raise ValueError('INVALID_SEGMENTED_OWNER_LANES')
   # Start with inexpensive, empirically useful geometries, then vary batch
   # and shard count. The timer limits the number actually executed.
   trial=[(4,.0625,4),(8,.0625,4),(8,.0625,8),(16,.0625,8),
          (16,.0625,16),(4,1.0,4),(8,1.0,8),(1,.25,1)]
   trial.extend((sh,.0625,min(sh,8)) for sh in shard_choices)
   floor=baseline_shards
   if floor>1:
    # Probe the memory-sized geometry first, then more shards/lanes. Never
    # silently benchmark a geometry below the admitted segment target.
    trial=[(floor,.0625,min(floor,8)),(floor,.0625,min(floor,16)),
           (min(4096,floor*2),.0625,min(floor*2,16)),
           (floor,1.0,min(floor,8))]+[v for v in trial if v[0]>=floor]
   ceiling=shard_geometry['minimum_shard_segment_budget']
   if fixed_shards is not None:
    # A user-selected shard count is a constraint, not a reason to skip
    # empirical lane and batch selection or silently alter that count.
    if forced is not None and forced>fixed_shards:raise ValueError('SEGMENTED_FIXED_SHARDS_LANES_CONFLICT')
    primary_lane=forced if forced is not None else min(fixed_shards,8)
    lanes=([forced] if forced is not None else list(dict.fromkeys((primary_lane,1,min(fixed_shards,4)))))
    trial=[(fixed_shards,.0625,lane) for lane in lanes]+[(fixed_shards,f,primary_lane) for f in (.25,1.0)]
   else:
    trial=[v for v in trial if v[0]<=ceiling]
    if forced is not None:
     if forced>ceiling:raise RuntimeError('SEGMENTED_LANES_EXCEED_SHARD_STORAGE_BUDGET')
     trial=[(max(sh,forced),f,forced) for sh,f,_ in trial]
   if capacity_preset and implicit_lanes and fixed_shards is None:
    # Preserve the memory-conservative geometry as a measured candidate.
    trial.insert(0,(floor,.0625,1))
    trial.insert(1,(floor,.0625,min(floor,4)))
   variants=list(dict.fromkeys((sh,f,default_transport,default_order,lane) for sh,f,lane in trial))
  # The same sequential startup owns these selected devices; pilot processes
  # release their allocations before the next admission. Production queries live VRAM again.
  if baseline.get('inventory'):env['_MGBFS_PROFILE_INVENTORY']=json.dumps(baseline['inventory'])
  if forced_generator=='gemm' and plan.get('generator_backend')!='gemm':raise RuntimeError('NATIVE_GEMM_CAPABILITY_REQUIRED')
  from .hardware_profiles import small_permutation_workload
  if plan['capacity']<=4096 or max_seconds<10 or small_permutation_workload(graph,baseline.get('inventory',[])):
   return fallback({'shard_geometry':shard_geometry,'status':'SMALL_OR_SHORT_WORKLOAD_CONSERVATIVE_PROFILE','shards':baseline_shards,'batch_fraction':1.0,'transport':default_transport,'candidate_order':default_order,'measured':False,'seconds':time.monotonic()-started})
  # Native content, graph and driver/topology identity prevent stale reuse.
  hardware=_system_info(['nvidia-smi','--query-gpu=uuid,driver_version,name','--format=csv,noheader'])
  topology=_system_info(['nvidia-smi','topo','-m'])
  dependency=_dependency_identity(native,env)
  capacity_inventory=None
  if capacity_preset:
   inventory=baseline.get('inventory',[])
   by_device={i.get('device'):i for i in inventory if isinstance(i,dict)}
   if len(by_device)==len(selected) and all(d in by_device and type(by_device[d].get('free_bytes')) is int and by_device[d]['free_bytes']>=0 for d in selected):
    # Native inventory's effective free_bytes already includes reclaimable
    # session and device-pool storage. Their partition can change at a backend
    # switch without changing admission. Key on the budget itself, while
    # production still rechecks live admission for every launch.
    memory_fields=('device','free_bytes','total_bytes','additional_reserve_bytes')
    capacity_inventory=[{k:by_device[d].get(k) for k in memory_fields} for d in selected]
  capacity_cache_safe=not capacity_preset or capacity_inventory is not None
  identity={'schema':20,'capacity_inventory':capacity_inventory,'selector_sha256':_content_digest(__file__,Path(__file__).with_name('profile_families.py'),Path(__file__).with_name('profile_cache.py'),Path(__file__).with_name('size_profiles.py'),Path(__file__).with_name('shard_policy.py'),Path(__file__).with_name('workload_geometry.py')),'configuration_digest':_configuration_digest(env),'baseline_geometry':{'plan':{k:plan.get(k) for k in ('capacity','batch','queue_capacity','state_bytes','history_layers','history_algorithm','owner_lanes','sorted_owner_bytes')},'rank_capacities':[p['capacity'] for p in baseline.get('rank_plans',[])]},'hardware_policy':hardware_policy,'allow_specialized':allow_specialized,'native_dependencies':dependency,'graph':graph.digest(),'native_sha256':_content_digest(native),'devices':selected,'hardware':hardware.stdout,'topology':topology.stdout,'capacity_override':capacity,'fixed_shards':fixed_shards,'environment':{k:env.get(k) for k in ('CUDA_VISIBLE_DEVICES','NCCL_P2P_DISABLE','NCCL_SHM_DISABLE','NCCL_SOCKET_IFNAME','NCCL_NET_GDR_LEVEL','NCCL_IB_DISABLE','NCCL_NET','MGBFS_GENERIC_VRAM_RESERVE_BYTES','MGBFS_GENERIC_TRANSPORT','MGBFS_GENERIC_SORT','MGBFS_PEER_TRANSPORT')}}
  if any(x is not None for x in (capacity_slowdown_limit,pilot_limit,pilot_wall_seconds)):
   identity['selector_policy']={'capacity_slowdown_limit':capacity_slowdown_limit,'pilot_limit':pilot_limit,'pilot_wall_seconds':pilot_wall_seconds}
  key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
  root=Path(os.environ.get('MGBFS_PROFILE_CACHE',str(Path.home()/'.cache/multigpubfs/profiles')));cache=root/(key+'.json')
  profile_cache_read_status={'status':'DISABLED','stage':'read','errno':None}
  if not skip_balanced_anchor and capacity_cache_safe and dependency is not None and hardware.returncode==0 and topology.returncode==0:
   saved,profile_cache_read_status=_read_profile_cache(cache)
   saved=saved or {}
   if bounded_capacity:
    stored=saved.get('capacity_slowdown')
    if (not isinstance(stored,dict) or stored.get('status') not in ('MEASURED_MATCHED_PREFIX_BOUND','MEASURED_FULL_BFS_BOUND')
        or stored.get('limit')!=capacity_slowdown_limit
        or type(stored.get('observed_ratio')) not in (int,float)
        or not 0<stored['observed_ratio']<=capacity_slowdown_limit
        or saved.get('segment_runtime_preset') not in ('balanced','max_capacity')):
     saved={}
   from .specialized import match_lrx
   specialized_eligible=allow_specialized and len(selected) in (1,2,4,8) and not env.get('MGBFS_GENERIC_TRANSPORT') and not env.get('MGBFS_GENERIC_SORT') and not env.get('MGBFS_GENERIC_HISTORY') and match_lrx(graph) is not None
   if (not segmented or (type(saved.get('segment_exact_exchange')) is bool and (env.get('MGBFS_SEGMENT_EXACT_EXCHANGE') is None or saved['segment_exact_exchange']==(env['MGBFS_SEGMENT_EXACT_EXCHANGE']=='1')))) and (not capacity_preset or (type(saved.get('segment_queue_cap')) is int and saved['segment_queue_cap']>0 and (implicit_queue or saved['segment_queue_cap']==queue_base))) and saved.get('generator_backend','cuda') in (('cuda','gemm') if gemm_supported(graph) else ('cuda',)) and saved.get('identity')==identity and saved.get('status')=='MEASURED_EQUAL_PREFIX_GPU_PROFILE' and type(saved.get('shards')) is int and 1<=saved['shards']<=4096 and saved.get('batch_fraction') in ((1.0,.25,.0625) if segmented else (1.0,.25)) and ((saved.get('backend','generic')=='generic' and (saved.get('shards'),saved.get('batch_fraction'),saved.get('transport'),saved.get('candidate_order'),saved.get('segment_owner_lanes') if segmented else None) in variants and saved.get('history_algorithm','HASH')=='HASH' or saved.get('backend','generic')=='generic' and saved.get('history_algorithm')=='SORTED_RUNS' and saved.get('owner_lanes') in (1,2,4,8) and saved.get('shards')==max(4,saved['owner_lanes']) and saved.get('transport')==default_transport and saved.get('candidate_order')=='none') or (specialized_eligible and saved.get('backend') in ('shard_ab_hash','shard_ab_sort_merge') and type(saved.get('specialized_capacity')) is int and saved['specialized_capacity']>0)):
    saved=dict(saved);saved['cache_hit']=True;saved['profile_cache_read_status']=profile_cache_read_status;saved['seconds']=time.monotonic()-started;saved['_admission_inventory']=baseline.get('inventory',[]);return saved
  from .profile_families import identity as family_identity,load as family_load,transferred
  shared_identity=family_identity(graph,plan,identity)
  if not segmented and dependency is not None and hardware.returncode==0 and topology.returncode==0:
   borrowed=family_load(root,shared_identity)
   if borrowed is not None:
    candidate_env=dict(env,MGBFS_GENERIC_TRANSPORT=borrowed.get('transport','full'),MGBFS_GENERIC_SORT=borrowed.get('candidate_order','none'),MGBFS_GENERIC_GENERATOR=borrowed.get('generator_backend','cuda'),MGBFS_GENERIC_HISTORY='sorted' if borrowed.get('history_algorithm')=='SORTED_RUNS' else 'hash',MGBFS_GENERIC_OWNER_LANES=str(borrowed.get('owner_lanes',1)))
    try:
     if borrowed['backend']=='generic':fresh=_admit(graph,selected,capacity,borrowed['shards'],native,candidate_env,temporary)['plan']
     else:
      from .specialized import admit_specialized
      specialized_env=dict(env,MGBFS_SPECIALIZED_TRANSPORT=borrowed['specialized_transport'])
      if borrowed['specialized_transport']=='NCCL_LSA':specialized_env['NCCL_CUMEM_ENABLE']='1'
      fresh=admit_specialized(graph,native,specialized_env,selected,Path(temporary)/'shared-specialized',capacity=capacity,mode=borrowed['specialized_mode'],seconds=min(60,max_seconds))
     return dict(transferred(borrowed,identity,fresh,time.monotonic()-started),_admission_inventory=baseline.get('inventory',[]))
    except (RuntimeError,ValueError):pass
  admitted=[]
  for shards,fraction,transport,order,lane in variants:
   if budget_deadline is not None and time.monotonic()>=budget_deadline:break
   if segmented and pilot_limit is not None and len(admitted)>=max(2,pilot_limit):break
   candidate_env=dict(env,MGBFS_GENERIC_TRANSPORT=transport,MGBFS_GENERIC_SORT=order)
   if lane is not None:candidate_env['MGBFS_SEGMENT_OWNER_LANES']=str(lane)
   try:admission=_admit(graph,selected,capacity,shards,native,candidate_env,temporary)
   except RuntimeError as error:
    if any(code in str(error) for code in ('REQUESTED_CAPACITY_EXCEEDS_ADMISSION','REQUESTED_CAPACITY_EXCEEDS_SORTED_ADMISSION','GENERIC_DISTRIBUTED_NO_CAPACITY','GENERIC_DISTRIBUTED_HEADROOM')):continue
    raise
   admitted.append((shards,fraction,transport,order,lane,admission['plan']))
  history=env.get('MGBFS_GENERIC_HISTORY');sorted_admitted=[]
  if not segmented and history in (None,'sorted'):
   forced_lanes=env.get('MGBFS_GENERIC_OWNER_LANES')
   lanes=[int(forced_lanes)] if forced_lanes is not None else [1,2,4,8]
   for lane in lanes:
    if not 1<=lane<=8:raise ValueError('INVALID_OWNER_LANES')
    shards=max(4,lane);transport=default_transport
    candidate_env=dict(env,MGBFS_GENERIC_HISTORY='sorted',MGBFS_GENERIC_OWNER_LANES=str(lane),MGBFS_GENERIC_TRANSPORT=transport,MGBFS_GENERIC_SORT='none')
    try:admission=_admit(graph,selected,capacity,shards,native,candidate_env,temporary)
    except RuntimeError as error:
     if any(code in str(error) for code in ('REQUESTED_CAPACITY_EXCEEDS_ADMISSION','REQUESTED_CAPACITY_EXCEEDS_SORTED_ADMISSION','GENERIC_DISTRIBUTED_NO_CAPACITY','GENERIC_DISTRIBUTED_HEADROOM','SORTED_OWNER_NO_ADMISSION')):continue
     raise
    sorted_admitted.append((lane,shards,transport,admission['plan']))
  if len(admitted)+len(sorted_admitted)<(1 if bounded_capacity else 2):return fallback({'shard_geometry':shard_geometry,'status':'NO_ADMITTED_ALTERNATIVE_CONSERVATIVE_PROFILE','shards':baseline_shards,'batch_fraction':1.0,'transport':default_transport,'candidate_order':default_order,'measured':False,'seconds':time.monotonic()-started})
  # Segments are allocated lazily. Capping them to the legacy probe budget
  # prematurely ended every pilot on the same small layer and selected a
  # geometry that was slow on the larger layers of the actual graph.
  capacity_limits=[p['capacity'] for _,_,_,_,_,p in admitted]+[p['capacity'] for _,_,_,p in sorted_admitted]
  if not segmented:capacity_limits.append(hardware_policy['probe_capacity_per_rank'])
  common_capacity=min(capacity_limits)
  pilots=[];limit=5;candidate_failures=[];resource_rejections=[];timed_out_trials=[]
  from .launch import run_graph
  def measure(label,candidate_env,run):
   # Only the failed candidate is rejected. Successful candidates still have
   # to establish a real matched-prefix comparison on the selected GPU group.
   try:
    with _evidence_directory(candidate_env,'mgbfs-candidate-'+label+'-') as trial:
     target=Path(trial)/label
     report=run(target)
     if segmented and str(report.get('reason','')).startswith('RESOURCE_'):
      report=dict(report,_validated_resource_receipt=_validate_segmented_resource_report(report,target,graph.digest(),selected,candidate_env,graph.state_elements))
     return report
   except ProfileBudgetExpired:
    timed_out_trials.append(label)
    return None
   except RuntimeError as error:
    if pilot_wall_seconds is not None and str(error).split(';',1)[0]=='AUTOTUNE_PILOT_WALL_DEADLINE' and 'TEARDOWN_INCOMPLETE' not in str(error):
     timed_out_trials.append(label)
     return None
    failure=_native_probe_failure(error)
    if failure is None:raise
    failure=dict(failure,candidate=label,source_graph_digest=graph.digest())
    candidate_failures.append(failure)
    from .generic_session import _active
    if _active is not None:
     _active.close_workers();_active.launch_admission=None
    # Queries from the destroyed group cannot admit another candidate.
    env.pop('_MGBFS_PROFILE_INVENTORY',None)
    import warnings
    warnings.warn('GPU autotuning rejected candidate '+label+' after CUDA 999; remaining candidates will be measured on a fresh rank group; evidence='+failure['evidence'],RuntimeWarning,stacklevel=2)
    return None
  def no_measured_profile():
   payload={'schema':1,'status':'AUTOTUNE_NO_MEASURED_PROFILE_AFTER_FAILURE','successful_candidates':len(pilots),'candidate_failures':candidate_failures,'scope':'no unmeasured fallback and no failed-candidate retry'}
   (Path(temporary)/'candidate-failures.json').write_text(json.dumps(payload,indent=2))
   error=RuntimeError('AUTOTUNE_NO_MEASURED_PROFILE_AFTER_FAILURE: successful_candidates='+str(len(pilots))+'; failed_candidates='+str(len(candidate_failures))+'; evidence='+', '.join(f['evidence'] for f in candidate_failures))
   error.candidate_failures=candidate_failures
   raise error
  # Compare exchange modes on the same geometry before other choices consume
  # the startup budget. Explicit user settings remain hard constraints.
  forced_exchange=env.get('MGBFS_SEGMENT_EXACT_EXCHANGE') if segmented else None
  if capacity_preset and admitted:
   # Compare the conservative preset and a hardware-admitted queue on the
   # same shard geometry early, before other choices spend the time budget.
   # Explicit user settings are never relaxed. Native admission remains live.
   base_lane=1 if implicit_lanes else int(env['MGBFS_SEGMENT_OWNER_LANES'])
   base=next((e for e in admitted if e[0]==baseline_shards and e[4]==base_lane),admitted[0])
   fast=next((e for e in admitted if e[0]==base[0] and e[4]==min(base[0],4)),base)
   def wider(entry):
    q=entry[-1].get('queue_capacity')
    return max(queue_base,q) if implicit_queue and type(q) is int and q>0 else queue_base
   # Exercise the wider geometry first so later comparisons include its
   # persistent CUDA/NCCL high-water allocations on the worker group.
   candidates=[(fast,wider(fast)),(base,queue_base),(base,wider(base)),(fast,queue_base)]
   candidates.extend((entry,queue_base) for entry in admitted)
   unique=[];seen=set()
   for entry,q in candidates:
    candidate_key=(*entry[:5],q)
    if candidate_key not in seen:seen.add(candidate_key);unique.append((entry,q))
   runs=[(entry,forced_exchange or '0',q) for entry,q in unique]
   if env.get('_MGBFS_SEGMENT_EXCHANGE_CAPABLE')=='1' and forced_exchange is None and len(selected)>1:
    # Compare exact exchange after warming the wider geometry as well.
    runs.insert(0,(fast,'1',wider(fast)))
    runs.insert(1,(base,'1',queue_base))
    distinct=[];seen=set()
    for entry,mode,q in runs:
     candidate_key=(*entry[:5],mode,q)
     if candidate_key not in seen:seen.add(candidate_key);distinct.append((entry,mode,q))
    runs=distinct
  else:
   runs=[(entry,forced_exchange or '0',None) for entry in admitted]
   if segmented and env.get('_MGBFS_SEGMENT_EXCHANGE_CAPABLE')=='1' and forced_exchange is None and len(selected)>1 and runs:
    runs.insert(1,(runs[0][0],'1',None))
  runtime_preset=env.get('MGBFS_MEMORY_PRESET','balanced')
  scheduled=[(*r,runtime_preset) for r in runs]
  if bounded_capacity and not skip_balanced_anchor:
   balanced_env=_balanced_segment_environment(env)
   first_admitted=next((e for e in admitted if e[0]==baseline_shards and e[4]==min(baseline_shards,4)),admitted[0])
   balanced_queue=int(balanced_env.get('MGBFS_SEGMENT_QUEUE_CAP',first_admitted[-1]['queue_capacity']))
   anchor_exchange=forced_exchange or ('1' if env.get('_MGBFS_SEGMENT_EXCHANGE_CAPABLE')=='1' and len(selected)>1 else '0')
   scheduled.insert(0,(first_admitted,anchor_exchange,balanced_queue,'balanced'))
   if pilot_limit is None or pilot_limit>3:
    if forced_exchange is None and anchor_exchange=='1':
     scheduled.insert(3,(first_admitted,'0',balanced_queue,'balanced'))
  for index,((shards,fraction,transport,order,lane,_),exchange,queue_cap,runtime_preset) in enumerate(scheduled):
   if pilot_limit is not None and index>=pilot_limit:break
   now=time.monotonic()
   remaining=max_seconds-(now-started) if budget_deadline is None else budget_deadline-now
   if remaining<(limit+1 if pilot_wall_seconds is None else 2):break
   candidate_base=balanced_env if bounded_capacity and not skip_balanced_anchor and runtime_preset=='balanced' else env
   candidate_env=dict(candidate_base,MGBFS_GENERIC_TRANSPORT=transport,MGBFS_GENERIC_SORT=order)
   if lane is not None:candidate_env['MGBFS_SEGMENT_OWNER_LANES']=str(lane)
   if segmented:candidate_env['MGBFS_SEGMENT_EXACT_EXCHANGE']=exchange
   if queue_cap is not None:candidate_env['MGBFS_SEGMENT_QUEUE_CAP']=str(queue_cap)
   # Segmented pilots allocate segments lazily and are not bounded by the
   # legacy common-capacity proxy. Re-admit the actual user request against
   # current free memory; keep legacy pilots on their common fixed workload.
   pilot_deadline=None
   if budget_deadline is not None:
    pilot_deadline=min(budget_deadline,time.monotonic()+pilot_wall_seconds)
    candidate_env['MGBFS_NATIVE_QUERY_SECONDS']=str(min(pilot_wall_seconds,float(env.get('MGBFS_NATIVE_QUERY_SECONDS','60'))))
   try:admission=_admit(graph,selected,capacity if segmented else common_capacity,shards,native,candidate_env,temporary)
   except ProfileBudgetExpired:
    timed_out_trials.append('pilot-'+str(index))
    break
   except RuntimeError as error:
    if pilot_deadline is None or not str(error).startswith('NATIVE_STARTUP_QUERY_TIMEOUT: standalone query child terminated'):raise
    timed_out_trials.append('pilot-'+str(index));continue
   if pilot_deadline is not None and time.monotonic()>=pilot_deadline:
    timed_out_trials.append('pilot-'+str(index));continue
   batch=max(1,int(admission['plan']['batch']*fraction))
   label='pilot-'+str(index)
   # For segmented modes the user cap, if any, is already in
   # candidate_env. Passing the legacy probe capacity here would invent an
   # extra global atomic limit in uncapped production and skew the timings.
   pilot_deadline_option={} if pilot_deadline is None else {'_profile_deadline':pilot_deadline}
   bfs_limit=limit if pilot_deadline is None else max(1,math.ceil(pilot_deadline-time.monotonic())-1)
   report=measure(label,candidate_env,lambda target:run_graph(graph,target,devices=selected,capacity=None if segmented else common_capacity,max_seconds=bfs_limit,executable=native,shards=shards,autotune=False,memory_backend='permutation_bitmap' if bitmap else 'segmented24' if segmented else 'legacy',memory_preset=runtime_preset,_batch=batch,_profile_layers=36,_native_env=candidate_env,**pilot_deadline_option))
   if report is None:continue
   if pilot_deadline is not None and time.monotonic()>=pilot_deadline:
    timed_out_trials.append(label);continue
   if segmented and report.get('backend')!=('EXPERIMENTAL_PERMUTATION_BITMAP' if bitmap else 'EXPERIMENTAL_SEGMENTED24'):raise RuntimeError('SEGMENTED_TUNER_WRONG_BACKEND')
   if segmented and str(report.get('reason','')).startswith('RESOURCE_'):
    evidence=report.get('_validated_resource_receipt')
    if not isinstance(evidence,dict):raise RuntimeError('SEGMENTED_RESOURCE_RECEIPT_MISSING')
    resource_rejections.append(dict(evidence,candidate=label,shards=shards,reason=report['reason'],
     completed_layers=len(report['layer_sizes']),layer_sizes=report['layer_sizes'],layer_seconds=report['layer_seconds']))
    continue
   pilot={'candidate':label,'executed_backend':report.get('backend'),'rank_worker_pids':report.get('rank_worker_pids',[]),'backend':'generic','history_algorithm':admission['plan'].get('history_algorithm','HASH'),'owner_lanes':admission['plan'].get('owner_lanes',0),'candidate_order':order,'transport':transport,'shards':shards,'batch_fraction':fraction,'batch':batch,'state_bytes':report.get('state_bytes',report.get('plan',{}).get('state_bytes')),'layer_sizes':report['layer_sizes'],'layer_seconds':report['layer_seconds'],'status':report['status'],'reason':report['reason']}
   if bounded_capacity:pilot['bfs_seconds']=report.get('bfs_seconds')
   if segmented:
    receipts=report.get('segment_memory_by_rank',[])
    actual=[x.get('owner_lanes') for x in receipts]
    if len(actual)!=len(selected) or any(type(n) is not int or not 1<=n<=lane for n in actual):raise RuntimeError('SEGMENTED_TUNER_LANE_RECEIPT_MISSING')
    actual_exchange=[x.get('exact_exchange') for x in receipts]
    if env.get('_MGBFS_SEGMENT_EXCHANGE_CAPABLE')=='1':
     if any(type(x) is not bool or x!=(exchange=='1') for x in actual_exchange):raise RuntimeError('SEGMENTED_TUNER_EXCHANGE_RECEIPT_MISMATCH')
    elif exchange!='0' or any(x not in (None,False) for x in actual_exchange):raise RuntimeError('SEGMENTED_TUNER_EXCHANGE_RECEIPT_MISMATCH')
    if capacity_preset:
     actual_queue=[x.get('actual_queue_capacity') for x in receipts]
     if len(actual_queue)!=len(selected) or any(type(q) is not int or not 1<=q<=queue_cap for q in actual_queue) or len(set(actual_queue))!=1:
      raise RuntimeError('SEGMENTED_TUNER_QUEUE_RECEIPT_MISMATCH')
     pilot.update(segment_queue_cap=queue_cap,segment_queue_capacity_by_rank=actual_queue)
     capacities=[];equivalents=[];budgets=[];limits=[];scopes=[];allocated=[];peak_live=[]
     for r in receipts:
      budget=r.get('storage_budget_bytes');unit=r.get('bytes_per_segment');rows=r.get('rows_per_bank')
      if not all(type(x) is int and x>0 for x in (budget,unit,rows)):raise RuntimeError('SEGMENTED_CAPACITY_EVIDENCE_MISSING')
      rank_limit=r.get('capacity_limit_per_rank');scope=r.get('capacity_limit_scope')
      if rank_limit is not None and (type(rank_limit) is not int or rank_limit<=0 or r.get('capacity_limit_is_hard') is not True):raise RuntimeError('SEGMENTED_CAPACITY_LIMIT_INVALID')
      if scope not in ('one next layer per rank','all visited states per rank'):raise RuntimeError('SEGMENTED_CAPACITY_SCOPE_MISMATCH')
      capacities.append((budget//unit)*rows)
      equivalents.append((budget*rows)//unit)
      budgets.append(budget);limits.append(rank_limit);scopes.append(scope)
      segment_rows=r.get('segment_rows_per_bank_by_shard')
      if isinstance(segment_rows,list) and all(isinstance(bucket,list) and all(type(v) is int and v>=0 for v in bucket) for bucket in segment_rows):
       allocated.append(sum(sum(bucket) for bucket in segment_rows))
      else:allocated.append(None)
      peak_live.append(r.get('peak_live_rows'))
     if len(set(scopes))!=1:raise RuntimeError('SEGMENTED_CAPACITY_SCOPE_MISMATCH')
     pilot.update(capacity_score_rows=min(capacities),capacity_score_rows_by_rank=capacities,
      capacity_equivalent_rows=min(equivalents),capacity_equivalent_rows_by_rank=equivalents,
      capacity_storage_budget_bytes_by_rank=budgets,capacity_limit_per_rank_by_rank=limits,
      capacity_limit_scope=scopes[0],capacity_allocated_rows_per_bank_by_rank=allocated,
      capacity_peak_live_rows_by_rank=peak_live)
    pilot.update(segment_owner_lanes=lane,segment_owner_lanes_by_rank=actual,segment_exact_exchange=exchange=='1')
    if bounded_capacity:pilot['segment_runtime_preset']=runtime_preset
   pilots.append(pilot)
  # Compare generator on the same graph, history, capacity and transport.
  # GEMM has its own explicit workspace admission and a bounded launch batch.
  if pilots and env.get('MGBFS_EXPERIMENTAL_SEGMENTED24')!='1' and gemm_supported(graph) and _gemm_hardware_available() and forced_generator=='cuda' and max_seconds-(time.monotonic()-started)>=limit+1 :
   base=pilots[0];candidate_env=dict(env,MGBFS_GENERIC_GENERATOR='gemm',MGBFS_GENERIC_TRANSPORT=base['transport'],MGBFS_GENERIC_SORT=base['candidate_order'],MGBFS_GENERIC_HISTORY='sorted' if base.get('history_algorithm')=='SORTED_RUNS' else 'hash',MGBFS_GENERIC_OWNER_LANES=str(base.get('owner_lanes',1)))
   try:
    admission=_admit(graph,selected,common_capacity,base['shards'],native,candidate_env,temporary)
    if admission['plan'].get('generator_backend')!='gemm':raise RuntimeError('NATIVE_GEMM_CAPABILITY_REQUIRED')
    batch=min(base['batch'],admission['plan']['batch'])
    report=measure('generator-gemm',candidate_env,lambda target:run_graph(graph,target,devices=selected,capacity=common_capacity,max_seconds=min(limit,max(1,int(max_seconds-(time.monotonic()-started)))),executable=native,shards=base['shards'],autotune=False,_batch=batch,_profile_layers=36,_native_env=candidate_env))
    if report is not None:pilots.append(dict(base,candidate='generator-gemm',rank_worker_pids=report.get('rank_worker_pids',[]),generator_backend='gemm',batch=batch,layer_sizes=report['layer_sizes'],layer_seconds=report['layer_seconds'],status=report['status'],reason=report['reason']))
   except RuntimeError as error:
    if not any(code in str(error) for code in ('GEMM_BATCH_EXCEEDS_ADMISSION','REQUESTED_CAPACITY_EXCEEDS_ADMISSION','GENERIC_CUDA_801','GENERIC_CUDA_209','NATIVE_GEMM_CAPABILITY_REQUIRED')):raise

  # Same canonical state capacity: history/lane alternatives must pass actual
  # native workspace admission before they can become timing candidates.
  if sorted_admitted:
   for lane,shards,transport,_ in sorted_admitted:
    remaining=max_seconds-(time.monotonic()-started)
    if remaining<3:break
    shards=max(4,lane);transport=default_transport
    candidate_env=dict(env,MGBFS_GENERIC_HISTORY='sorted',MGBFS_GENERIC_OWNER_LANES=str(lane),MGBFS_GENERIC_TRANSPORT=transport,MGBFS_GENERIC_SORT='none')
    try:admission=_admit(graph,selected,common_capacity,shards,native,candidate_env,temporary)
    except RuntimeError as error:
     if any(code in str(error) for code in ('REQUESTED_CAPACITY_EXCEEDS_ADMISSION','REQUESTED_CAPACITY_EXCEEDS_SORTED_ADMISSION','GENERIC_DISTRIBUTED_NO_CAPACITY','GENERIC_DISTRIBUTED_HEADROOM','SORTED_OWNER_NO_ADMISSION')):continue
     raise
    batch=max(1,min(admission['plan']['batch'],pilots[0]['batch'] if pilots else admission['plan']['batch']))
    label='sorted-lanes-'+str(lane)
    report=measure(label,candidate_env,lambda target:run_graph(graph,target,devices=selected,capacity=common_capacity,max_seconds=min(limit,max(1,int(remaining))),executable=native,shards=shards,autotune=False,_batch=batch,_profile_layers=36,_native_env=candidate_env))
    if report is None:continue
    pilots.append({'candidate':label,'rank_worker_pids':report.get('rank_worker_pids',[]),'backend':'generic','history_algorithm':'SORTED_RUNS','owner_lanes':lane,'candidate_order':'none','transport':transport,'shards':shards,'batch_fraction':1.0,'batch':batch,'layer_sizes':report['layer_sizes'],'layer_seconds':report['layer_seconds'],'status':report['status'],'reason':report['reason']})
  # Specialized candidates preserve only a proved identical LRX action/root.
  # Unsupported definitions/topologies remain on the general exact backend.
  if allow_specialized and len(selected) in (1,2,4,8) and not env.get('MGBFS_GENERIC_TRANSPORT') and not env.get('MGBFS_GENERIC_SORT') and not env.get('MGBFS_GENERIC_HISTORY') and max_seconds-(time.monotonic()-started)>=8 :
   from .specialized import match_lrx,admit_specialized,run_specialized
   matched=match_lrx(graph)
   if matched is not None and matched['order']>=32768:
    from .transport import select_peer_transport
    transport_decision=select_peer_transport(native,env,selected,Path(temporary)/'peer-capability',request=env.get('MGBFS_PEER_TRANSPORT','auto'),seconds=max_seconds-(time.monotonic()-started))
    peer_options=['HOST_SIZED_NCCL']
    if transport_decision['selected']=='NCCL_LSA':peer_options.append('NCCL_LSA')
    for mode,backend,peer in [(mode,backend,peer) for peer in peer_options for mode,backend in [('HASH','shard_ab_hash'),('SORT_MERGE','shard_ab_sort_merge')]]:
     remaining=max_seconds-(time.monotonic()-started)
     if remaining<15:break
     candidate_env=dict(env,MGBFS_SPECIALIZED_TRANSPORT=peer)
     if peer=='NCCL_LSA':candidate_env['NCCL_CUMEM_ENABLE']='1'
     label=backend+'-'+peer
     plan=admit_specialized(graph,native,candidate_env,selected,Path(temporary)/('admit-'+label),capacity=capacity,mode=mode,seconds=min(60,remaining))
     remaining=max_seconds-(time.monotonic()-started)
     if remaining<2:break
     report=run_specialized(graph,Path(temporary)/label,native=native,env=candidate_env,devices=selected,capacity=plan['capacity'],batch=plan['batch'],max_seconds=min(15,max(2,int(remaining))),mode=mode,profile_layers=36)
     if len(report['layer_seconds'])>=4 and sum(report['layer_sizes'][2:])>=32768:
      pilots.append({'backend':backend,'candidate_order':'none' if mode=='HASH' else 'radix','transport':'specialized_key_first_host','shards':1,'batch_fraction':1.0,'batch':plan['batch'],'specialized_capacity':plan['capacity'],'specialized_mode':mode,'specialized_transport':peer,'transport_selection':transport_decision,'state_bytes':1,'layer_sizes':report['layer_sizes'],'layer_seconds':report['layer_seconds'],'status':report['status'],'reason':report['reason'],'specialized_admission':plan})
  # Resource-stopped and too-short probes must not disappear before the
  # completed prefixes have been checked against every other candidate.
  full_layer_digests={}
  for probe in resource_rejections:
   for depth,digest in probe['fully_retained_layer_sha256'].items():
    if depth in full_layer_digests and full_layer_digests[depth]!=digest:
     raise RuntimeError('AUTOTUNE_PREFIX_CORRECTNESS_MISMATCH: fully retained layer '+depth)
    full_layer_digests[depth]=digest
  prefixes=pilots+resource_rejections
  if prefixes:
   reference=max(prefixes,key=lambda p:len(p['layer_seconds']))['layer_sizes']
   for probe in prefixes:
    depth=len(probe['layer_seconds'])
    if len(probe['layer_sizes'])!=depth+1 or probe['layer_sizes'][:depth+1]!=reference[:depth+1]:
     raise RuntimeError('AUTOTUNE_PREFIX_CORRECTNESS_MISMATCH')
  if not pilots:
   if candidate_failures:no_measured_profile()
   if resource_rejections:
    return fallback({'shard_geometry':shard_geometry,'status':'VALIDATED_RESOURCE_LIMIT_UNMEASURED_PROFILE',
     'shards':baseline_shards,'batch_fraction':1.0,'transport':default_transport,'candidate_order':default_order,
     'measured':False,'resource_rejections':resource_rejections,'seconds':time.monotonic()-started,
     '_admission_inventory':baseline.get('inventory',[]),
     'scope':'All completed pilots reached a validated resource stop; production uses a freshly admitted conservative geometry and may return an exact incomplete prefix.'})
   return fallback({'shard_geometry':shard_geometry,'status':'TUNING_BUDGET_EXHAUSTED_CONSERVATIVE_PROFILE','shards':baseline_shards,'batch_fraction':1.0,'transport':default_transport,'candidate_order':default_order,'measured':False,'seconds':time.monotonic()-started,'_admission_inventory':baseline.get('inventory',[]),'timed_out_trials':timed_out_trials})
  all_pilots=pilots
  reference=max(all_pilots,key=lambda p:len(p['layer_seconds']))['layer_sizes']
  excluded_prefix_candidates=[];pilots=[]
  for pilot in all_pilots:
   completed_depth=len(pilot['layer_seconds']);completed_sizes=pilot['layer_sizes'][:completed_depth+1]
   # Validate even a too-short probe before excluding it from timing. A wrong
   # layer vector must never disappear as a harmless sampling limitation.
   if completed_sizes!=reference[:completed_depth+1]:raise RuntimeError('AUTOTUNE_PREFIX_CORRECTNESS_MISMATCH')
   completed_work=sum(completed_sizes[2:])
   if completed_depth<4 or completed_work<32768:
    excluded_prefix_candidates.append(dict(pilot,completed_depth=completed_depth,completed_work_after_warmup=completed_work,minimum_completed_depth=4,minimum_post_warmup_states=32768,native_reason=pilot['reason'],reason='INSUFFICIENT_COMPLETED_PREFIX'))
   else:pilots.append(pilot)
  if capacity_preset and not bounded_capacity and all_pilots:
   priorities=_segmented_capacity_priorities(all_pilots)
   priority_by_id={id(p):v for p,v in zip(all_pilots,priorities)}
  if capacity_preset and not bounded_capacity and pilots and excluded_prefix_candidates:
   # The excluded dictionaries are copies; evaluate them under the same cap.
   excluded_priorities=_segmented_capacity_priorities(excluded_prefix_candidates)
   if max(excluded_priorities)>max(priority_by_id[id(p)] for p in pilots):
    # An uncomparable high-capacity candidate cannot silently lose to a
    # faster, lower-capacity one. Keep the scoped conservative preset.
    if candidate_failures:no_measured_profile()
    return {'shard_geometry':shard_geometry,'status':'CAPACITY_COMPARISON_INSUFFICIENT_PREFIX','shards':baseline_shards,'batch_fraction':1.0,'transport':default_transport,'candidate_order':default_order,'measured':False,'pilots':all_pilots,'excluded_prefix_candidates':excluded_prefix_candidates,'seconds':time.monotonic()-started,'_admission_inventory':baseline.get('inventory',[]),'scope':'A higher-capacity probe did not complete a comparable prefix; preset geometry retained.'}
  bounded_selection=_bounded_segmented_profile(pilots,capacity_slowdown_limit) if bounded_capacity and len(pilots)>=2 else None
  if bounded_capacity and (bounded_selection is None or len(bounded_selection[0])<2):
   if candidate_failures:no_measured_profile()
   return fallback({'shard_geometry':shard_geometry,'status':'CAPACITY_COMPARISON_INSUFFICIENT_PREFIX',
     'shards':baseline_shards,'batch_fraction':1.0,'transport':default_transport,
     'candidate_order':default_order,'measured':False,'pilots':all_pilots,
     'excluded_prefix_candidates':excluded_prefix_candidates,'timed_out_trials':timed_out_trials,
     'seconds':time.monotonic()-started,'_admission_inventory':baseline.get('inventory',[])})
  if len(pilots)<2:
   if candidate_failures:no_measured_profile()
   return {'shard_geometry':shard_geometry,'status':'INSUFFICIENT_PREFIX_CONSERVATIVE_PROFILE','shards':baseline_shards,'batch_fraction':1.0,'transport':default_transport,'candidate_order':default_order,'measured':False,'pilots':all_pilots,'excluded_prefix_candidates':excluded_prefix_candidates,'comparable_candidates':len(pilots),'common_depth':min(len(p['layer_seconds']) for p in all_pilots),'seconds':time.monotonic()-started,'_admission_inventory':baseline.get('inventory',[])}
  if segmented and not capacity_preset:
   pilots,shallow=_segmented_deep_prefix_candidates(pilots)
   excluded_prefix_candidates.extend(shallow)
  # In max_capacity, a slow short prefix must not discard a candidate with
  # higher estimated capacity before applying the capacity-first selection.
  # Keep usable INCOMPLETE samples. Only the common completed substantial
  # prefix of the eligible candidates is compared; warmup is excluded below.
  if bounded_selection is not None:
   pilots,scores,winner,depth,bound_evidence=bounded_selection
  else:
   depth=min(len(p['layer_seconds']) for p in pilots)
  sizes=pilots[0]['layer_sizes'][:depth+1]
  if any(p['layer_sizes'][:depth+1]!=sizes for p in pilots):raise RuntimeError('AUTOTUNE_PREFIX_CORRECTNESS_MISMATCH')
  if bounded_selection is None:
   scores=[sum(p['layer_seconds'][1:depth]) for p in pilots]
   if segmented:
    winner=_pick_segmented_profile(pilots,scores,env.get('MGBFS_MEMORY_PRESET','balanced'))
   else:
    winner=min(range(len(scores)),key=lambda i:scores[i])
    if scores[winner]>=scores[0]*.95:winner=0
  chosen=pilots[winner]
  chosen_generator=chosen.get('generator_backend',forced_generator)
  result={'generator_backend':chosen_generator,'backend':chosen['backend'],'status':'MEASURED_EQUAL_PREFIX_GPU_PROFILE','shards':chosen['shards'],'batch_fraction':chosen['batch_fraction'],'transport':chosen['transport'],'candidate_order':chosen['candidate_order'],'measured':True,'minimum_switch_improvement':.05,'common_depth':depth,'common_layer_sizes':sizes,'scores_seconds':scores,'pilots':pilots,'identity':identity,'cache_hit':False,'seconds':time.monotonic()-started,'hardware_policy':hardware_policy,'pilot_capacity_per_rank':common_capacity,'scope':'hardware-derived prefix among admitted shard/batch and full-child/parent-origin and unsorted/radix-index transport profiles and exact bounded matrix CUDA/GEMM generators and eligible SHARD_AB hash/sorted-history owners; no claim of global optimum or large-rank acceptance'}
  if segmented:result.update(memory_preset=env.get('MGBFS_MEMORY_PRESET','balanced'),capacity_metric_scope='minimum per-rank planned full-segment rows bounded by the hard cap, then proportional residual estimate; not a physical capacity guarantee; allocation, shard skew and fragmentation can differ',shard_geometry=shard_geometry,resource_rejections=resource_rejections,segment_owner_lanes=chosen['segment_owner_lanes'],segment_exact_exchange=chosen['segment_exact_exchange'],winner_index=winner,scope=('permutation_bitmap' if bitmap else 'segmented24')+' GPU prefix timing among admitted shard, owner-lane, queue, batch and exchange choices; no claim of a global optimum or large-rank acceptance')
  if capacity_preset:result['segment_queue_cap']=chosen['segment_queue_cap']
  if bounded_selection is not None:
   result.update(segment_runtime_preset=chosen['segment_runtime_preset'],capacity_slowdown=bound_evidence)
  if timed_out_trials:result['timed_out_trials']=timed_out_trials
  result['excluded_prefix_candidates']=excluded_prefix_candidates
  result['_admission_inventory']=[] if candidate_failures else baseline.get('inventory',[])
  if candidate_failures:result.update(candidate_failures=candidate_failures,failed_candidate_policy='REJECT_AND_MEASURE_REMAINING',winner_candidate=chosen.get('candidate'),failed_probes_retried=False)
  result.update(history_algorithm=chosen.get('history_algorithm','SORTED_RUNS' if env.get('MGBFS_GENERIC_HISTORY')=='sorted' else 'HASH'),owner_lanes=chosen.get('owner_lanes',int(env.get('MGBFS_GENERIC_OWNER_LANES','0'))))
  if chosen['backend']=='generic':result['batch']=chosen['batch']
  if chosen['backend']!='generic':result.update(specialized_capacity=chosen['specialized_capacity'],specialized_mode=chosen['specialized_mode'],specialized_transport=chosen['specialized_transport'],specialized_admission=chosen['specialized_admission'],batch=chosen['batch'])
  result['profile_cache_read_status']=profile_cache_read_status
  if capacity_cache_safe and dependency is not None and hardware.returncode==0 and topology.returncode==0:
   result['profile_cache_write_status']=_write_profile_cache(cache,result,prefix=key+'-')
   if not segmented:
    from .profile_families import store as family_store
    result['family_cache_write_status']=family_store(root,shared_identity,result)
  return result


def _session_profile_key(pool,graph,devices,capacity,max_seconds,native,env,allow_specialized):
 dependency=_dependency_identity(native,env)
 if dependency is None:return None
 files=(native,__file__,Path(__file__).with_name('profile_families.py'),Path(__file__).with_name('profile_cache.py'),Path(__file__).with_name('size_profiles.py'))
 return (pool.generation,graph.digest(),None if devices is None else tuple(devices),capacity,allow_specialized,_configuration_digest(env),tuple(env.get(k) for k in ('LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT','CUDA_DEVICE_ORDER','CUDA_DISABLE_PTX_JIT')),tuple((str(p),_stat_identity(p)) for p in files),tuple((d[1],_stat_identity(d[1])) for d in dependency['resolved_dependencies']))

def _profile_environment(env,result):
 if result.get('segment_runtime_preset')=='balanced':
  env=_balanced_segment_environment(env)
 selected=dict(env,MGBFS_GENERIC_TRANSPORT=result.get('transport','full'),MGBFS_GENERIC_SORT=result.get('candidate_order','none'),MGBFS_GENERIC_GENERATOR=result.get('generator_backend','cuda'))
 if result.get('history_algorithm')=='SORTED_RUNS':selected.update(MGBFS_GENERIC_HISTORY='sorted',MGBFS_GENERIC_OWNER_LANES=str(result['owner_lanes']))
 elif result.get('history_algorithm')=='HASH':selected['MGBFS_GENERIC_HISTORY']='hash'
 if result.get('segment_owner_lanes') is not None:selected['MGBFS_SEGMENT_OWNER_LANES']=str(result['segment_owner_lanes'])
 if result.get('segment_queue_cap') is not None:selected['MGBFS_SEGMENT_QUEUE_CAP']=str(result['segment_queue_cap'])
 if result.get('segment_exact_exchange') is not None:selected['MGBFS_SEGMENT_EXACT_EXCHANGE']='1' if result['segment_exact_exchange'] else '0'
 return selected

def _profile_geometry(admission):
 return {k:admission.get(k) for k in ('devices','plan','rank_plans','size_profile_capability','owner_cuts')}

def _lrx_resource_alias_key(key,graph):
 from .specialized import match_lrx
 match=match_lrx(graph)
 if key is None or match is None:return None
 preferred=1 if all(0<=v<=255 for v in graph.start) else 8
 return (key[:1]+key[2:],json.dumps(graph.action,sort_keys=True),preferred)

def _rebase_owned_profile(saved,graph):
 import copy
 result=copy.deepcopy(saved);source=result.get('source_graph_digest',result.get('identity',{}).get('graph'))
 result.update(status='REUSED_COMPATIBLE_GPU_PROFILE',measured=False,cache_hit=True,source_graph_digest=source,source_profile_identity=copy.deepcopy(saved.get('identity')),online_size_tuning=True,scope='same LRX action and physically owned exact allocation geometry; source measured size choices reused, target optimum not claimed')
 if result.get('identity'):result['identity']['graph']=graph.digest()
 for profile in result.get('size_profiles',[]):
  if profile.get('status')=='LIVE_CHUNK_EMPIRICAL_PROFILE':profile['status']='REUSED_SOURCE_LIVE_BUCKET'
  else:profile['status']='REUSED_SOURCE_SIZE_PROFILE'
  profile['measured']=False
 return result


def _native_probe_failure(error):
 """Only a captured CUDA 999 from an optional native rank job is recoverable."""
 if not str(error).startswith('NATIVE_DISTRIBUTED_RANK_FAILED'):return None
 root=getattr(error,'tuning_evidence_root',None)
 if not root:return None
 for evidence in Path(root).glob('**/failure-evidence'):
  if not (evidence/'summary.json').is_file():continue
  for phase in evidence.glob('rank-*/phase.json'):
   try:
    if phase.stat().st_size>262144:continue
    fault=json.loads(phase.read_text()).get('native_error')
   except (OSError,ValueError):continue
   if isinstance(fault,dict) and fault.get('code')==999:
    return {'error':str(error),'evidence':str(evidence.resolve()),'native_error':fault,'probe_retried':False}
 return None

def choose_size_profile(graph,devices,capacity,max_seconds,native,env,*,allow_specialized=True):
 # Session reuse skips selector work, never live memory admission. A changed
 # geometry, runtime, environment or rank generation takes the full path.
 import copy
 from .generic_session import _active
 started=time.monotonic();env=dict(env);env.setdefault('MGBFS_PEER_TRANSPORT','auto')
 pool=_active;key=None if pool is None else _session_profile_key(pool,graph,devices,capacity,max_seconds,native,env,allow_specialized)
 cache=None if pool is None else getattr(pool,'size_profile_results',None)
 alias_key=_lrx_resource_alias_key(key,graph)
 aliases={} if pool is None else getattr(pool,'size_profile_aliases',{})
 if alias_key in aliases and (cache is None or key not in cache):
  saved,geometry=aliases[alias_key]
  from .specialized import match_lrx
  bound=match_lrx(graph)['order'];explicit=graph.to_dict()['expected_max_unique_states']
  if explicit is not None:bound=min(bound,explicit)
  if all(p['capacity']<=bound for p in geometry.get('rank_plans') or [geometry['plan']]):
   from .generic_session import owned_resource_admission,remember_launch_admission
   selected=_profile_environment(env,saved);fresh=owned_resource_admission(graph,native,selected,geometry)
   if fresh is not None:
    result=_rebase_owned_profile(saved,graph);result.update(session_profile_hit=True,owned_resource_hit=True,seconds=time.monotonic()-started)
    remember_launch_admission(graph,native,selected,devices,capacity,result['shards'],result['batch'],fresh)
    if cache is None:cache={};pool.size_profile_results=cache
    cache[key]=(copy.deepcopy(result),geometry)
    return result
 if key is not None and cache is not None and key in cache:
  saved,geometry=cache[key];selected=_profile_environment(env,saved)
  from .generic_session import owned_resource_admission
  fresh=owned_resource_admission(graph,native,selected,geometry)
  if fresh is None:
   with _evidence_directory(env,'mgbfs-session-profile-') as temporary:
    fresh=_admit(graph,devices,capacity,saved['shards'],native,selected,temporary,batch=saved['batch'])
  if key==_session_profile_key(pool,graph,devices,capacity,max_seconds,native,env,allow_specialized) and _profile_geometry(fresh)==geometry:
   from .generic_session import remember_launch_admission
   remember_launch_admission(graph,native,selected,devices,capacity,saved['shards'],saved['batch'],fresh)
   result=copy.deepcopy(saved);result.update(cache_hit=True,session_profile_hit=True,owned_resource_hit=fresh.get('owned_resource_admission',False),seconds=time.monotonic()-started,_admission_inventory=fresh.get('inventory',[]));return result
  del cache[key]
 result=_choose_size_profile_uncached(graph,devices,capacity,max_seconds,native,env,allow_specialized=allow_specialized)
 if pool is not None and result.get('backend','generic')=='generic' and result.get('size_profiles') and (result.get('measured') or result.get('cache_hit')):
  selected=_profile_environment(env,result)
  with _evidence_directory(env,'mgbfs-session-profile-store-') as temporary:
   admitted=_admit(graph,devices,capacity,result['shards'],native,selected,temporary,batch=result['batch'])
  key=_session_profile_key(pool,graph,devices,capacity,max_seconds,native,env,allow_specialized)
  if key is not None:
   if cache is None:cache={};pool.size_profile_results=cache
   if len(cache)>=256:cache.clear()
   saved=copy.deepcopy(result);saved.pop('_admission_inventory',None);cache[key]=(saved,_profile_geometry(admitted))
   alias_key=_lrx_resource_alias_key(key,graph)
   if alias_key is not None:
    aliases=getattr(pool,'size_profile_aliases',{})
    if len(aliases)>=256:aliases.clear()
    aliases[alias_key]=(copy.deepcopy(saved),_profile_geometry(admitted));pool.size_profile_aliases=aliases
   from .generic_session import remember_launch_admission
   remember_launch_admission(graph,native,selected,devices,capacity,result['shards'],result['batch'],admitted)
 result['seconds']=time.monotonic()-started
 return result

def _choose_size_profile_uncached(graph,devices,capacity,max_seconds,native,env,*,allow_specialized=True):
 """Bounded real-prefix measurements; unknown large tiers remain conservative."""
 from .size_profiles import select_size_profiles,valid_profile_entries
 from .generation import gemm_supported
 started=time.monotonic();env=dict(env)
 # Avoid replaying an entire prefix for bounded modest permutation orbits.
 # Large unknown families retain measured selection; matrix GEMM is unchanged.
 from .workload_geometry import certified_frontier_bound
 from .hardware_profiles import small_permutation_workload
 from .specialized import match_lrx
 forced=any(env.get(k) for k in ('MGBFS_GENERIC_HISTORY','MGBFS_GENERIC_TRANSPORT','MGBFS_GENERIC_SORT','MGBFS_GENERIC_GENERATOR'))
 matrix_fast=graph.action['kind']=='matrix' and certified_frontier_bound(graph) is not None and (not gemm_supported(graph) or not _gemm_hardware_available())
 if (graph.action['kind']=='permutation' and match_lrx(graph) is None or matrix_fast) and not forced:
  with _evidence_directory(env,'mgbfs-fast-profile-') as directory:
   baseline=_admit(graph,devices,capacity,1,native,env,directory)
  inventory=baseline.get('inventory',[])
  modest=[dict(i,sm_count=max(1,int(i.get('sm_count',1)))*128) for i in inventory]
  if certified_frontier_bound(graph) is not None or small_permutation_workload(graph,modest):
   plan=baseline['plan'];batch=plan['batch']
   return {'backend':'generic','status':'FAST_START_LIVE_SIZE_TUNING','shards':1,'batch':batch,'batch_fraction':1.,'transport':'full','candidate_order':'none','history_algorithm':'HASH','generator_backend':'cuda','measured':False,'online_size_tuning':True,'size_profiles':[{'minimum_frontier':0,'batch':batch,'generator_backend':'cuda','measured':False,'status':'CONSERVATIVE_LIVE_TUNING_SEED'}],'seconds':time.monotonic()-started,'scope':'cold prefix replay skipped; actual disjoint GPU chunks tune batch online; no global optimum claim'}
 global_profile=choose_profile(graph,devices,capacity,max_seconds,native,env,allow_specialized=allow_specialized)
 if global_profile.get('backend','generic')!='generic' or max_seconds<10 or global_profile.get('status') in ('SMALL_OR_SHORT_WORKLOAD_CONSERVATIVE_PROFILE','TUNING_BUDGET_EXHAUSTED_CONSERVATIVE_PROFILE') or time.monotonic()-started>=max_seconds:return global_profile
 shards=global_profile['shards']
 if global_profile.get('_admission_inventory'):env['_MGBFS_PROFILE_INVENTORY']=json.dumps(global_profile['_admission_inventory'])
 env.update(MGBFS_GENERIC_TRANSPORT=global_profile.get('transport','full'),MGBFS_GENERIC_SORT=global_profile.get('candidate_order','none'),MGBFS_GENERIC_HISTORY='sorted' if global_profile.get('history_algorithm')=='SORTED_RUNS' else 'hash',MGBFS_GENERIC_OWNER_LANES=str(global_profile.get('owner_lanes',1)),MGBFS_GENERIC_GENERATOR=global_profile.get('generator_backend','cuda'))
 with _evidence_directory(env,'mgbfs-size-profile-') as temporary:
  first=_admit(graph,devices,capacity,shards,native,env,temporary);selected=first['devices']
  if first.get('size_profile_capability')!=1:return dict(global_profile,size_tuning_status='NATIVE_SIZE_PROFILE_CAPABILITY_UNAVAILABLE')
  default_generator=env.get('MGBFS_GENERIC_GENERATOR','cuda')
  eligible=default_generator=='gemm' or (gemm_supported(graph) and _gemm_hardware_available())
  reserve_env=dict(env,MGBFS_GENERIC_GENERATOR='gemm' if eligible else 'cuda')
  size_generator_rejections=[]
  try:
   reserve=first if reserve_env['MGBFS_GENERIC_GENERATOR']==default_generator else _admit(graph,selected,capacity,shards,native,reserve_env,temporary)
   common=min(first['plan']['capacity'],reserve['plan']['capacity'])
   reserve=reserve if capacity==common else _admit(graph,selected,common,shards,native,reserve_env,temporary)
  except RuntimeError as error:
   # A separate optional generator may not fit the measured owner's geometry.
   # Keep the already admitted measured choice; never hide its own failure.
   resource_codes=('REQUESTED_CAPACITY_EXCEEDS_SORTED_ADMISSION','REQUESTED_CAPACITY_EXCEEDS_ADMISSION','GEMM_BATCH_EXCEEDS_ADMISSION')
   if reserve_env['MGBFS_GENERIC_GENERATOR']==default_generator or not any(code in str(error) for code in resource_codes):raise
   size_generator_rejections.append({'generator':reserve_env['MGBFS_GENERIC_GENERATOR'],'stage':'size-profile-workspace-admission','reason':str(error),'selected_generator':default_generator,'selected_capacity':first['plan']['capacity'],'shards':shards,'history_algorithm':first['plan'].get('history_algorithm','HASH')})
   eligible=False;reserve_env=dict(env);reserve=first;common=first['plan']['capacity']
  admitted_generators=['cuda','gemm'] if eligible else [default_generator]
  large_batch=reserve['plan']['batch'];small_batch=min(4096,large_batch)
  dependency=_dependency_identity(native,env)
  hardware=_system_info(['nvidia-smi','--query-gpu=uuid,driver_version,name','--format=csv,noheader'])
  topology=_system_info(['nvidia-smi','topo','-m'])
  identity={'schema':1,'graph':graph.digest(),'devices':selected,'capacity':common,'batch':large_batch,'plan':reserve['plan'],'config':_configuration_digest(env),'native':_content_digest(native),'selector':_content_digest(__file__,Path(__file__).with_name('profile_cache.py'),Path(__file__).with_name('size_profiles.py')),'dependency':dependency,'hardware':hardware.stdout,'topology':topology.stdout,'admitted_generators':admitted_generators}
  key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
  root=Path(os.environ.get('MGBFS_PROFILE_CACHE',str(Path.home()/'.cache/multigpubfs/profiles')));cache=root/('size-'+key+'.json')
  profile_cache_read_status={'status':'DISABLED','stage':'read','errno':None}
  if dependency and hardware.returncode==0 and topology.returncode==0:
   result,profile_cache_read_status=_read_profile_cache(cache)
   result=result or {}
   profiles=result.get('size_profiles',[])
   if result.get('identity')==identity and result.get('status')=='MEASURED_FRONTIER_SIZE_PROFILES' and valid_profile_entries(profiles,large_batch,admitted_generators):
    result=dict(result,cache_hit=True,profile_cache_read_status=profile_cache_read_status,global_profile=global_profile,size_generator_rejections=size_generator_rejections,admitted_generators=admitted_generators,seconds=time.monotonic()-started);return result
  if global_profile.get('status')=='REUSED_COMPATIBLE_GPU_PROFILE':
   source=global_profile.get('source_pilots',[])
   compatible=[p for p in source if p.get('backend')=='generic' and p.get('history_algorithm','HASH')==global_profile.get('history_algorithm','HASH') and p.get('shards')==shards and p.get('transport')==global_profile.get('transport') and p.get('candidate_order')==global_profile.get('candidate_order')]
   threshold=len(selected)*max(1024,min(int(i['sm_count']) for i in first['inventory'])*512)
   compatible=[dict(p,generator_backend=p.get('generator_backend','cuda')) for p in compatible if p.get('generator_backend','cuda') in admitted_generators]
   tiers=select_size_profiles(compatible,len(selected),threshold=threshold) if compatible else []
   profiles=[]
   for tier in tiers:
    donor_batch=next((p.get('batch',1) for p in compatible if p.get('generator_backend','cuda')==tier['generator_backend']),1)
    batch=max(1,min(large_batch,int(large_batch*tier['batch']/max(1,donor_batch))))
    profiles.append(dict(minimum_frontier=tier['minimum_frontier'],batch=batch,generator_backend=tier['generator_backend'],measured=False,status='REUSED_SOURCE_SIZE_PROFILE',source_maximum_measured_frontier=tier['maximum_measured_frontier']))
   if not profiles:profiles=[dict(minimum_frontier=0,batch=max(1,int(large_batch*global_profile['batch_fraction'])),generator_backend=default_generator,measured=False,status='REUSED_SOURCE_CONSERVATIVE_PROFILE')]
   return dict(global_profile,batch=max(1,int(large_batch*global_profile['batch_fraction'])),size_profiles=profiles,online_size_tuning=True,seconds=time.monotonic()-started,global_profile=global_profile,pilot_budget_seconds=0.,target_pilots_executed=0,size_generator_rejections=size_generator_rejections,admitted_generators=admitted_generators)
  base={'backend':'generic','shards':shards,'batch':large_batch,'batch_fraction':1.,'transport':env.get('MGBFS_GENERIC_TRANSPORT','full'),'candidate_order':env.get('MGBFS_GENERIC_SORT','none'),'history_algorithm':reserve['plan'].get('history_algorithm','HASH'),'owner_lanes':reserve['plan'].get('owner_lanes',0),'generator_backend':reserve['plan'].get('generator_backend','cuda'),'size_generator_rejections':size_generator_rejections,'admitted_generators':admitted_generators,'size_profiles':[],'seconds':time.monotonic()-started,'measured':False}
  if max_seconds<10 or common<=4096:return dict(base,status='SMALL_OR_SHORT_WORKLOAD_CONSERVATIVE_PROFILE')
  # Reuse completed matching startup pilots, never rerun the prefix for tiers.
  pilots=[]
  chosen={'shards':shards,'transport':env.get('MGBFS_GENERIC_TRANSPORT','full'),'candidate_order':env.get('MGBFS_GENERIC_SORT','none'),'history_algorithm':reserve['plan'].get('history_algorithm','HASH'),'owner_lanes':reserve['plan'].get('owner_lanes',0)}
  for p in global_profile.get('pilots',[]):
   if all(p.get(k,0 if k=='owner_lanes' else 'HASH' if k=='history_algorithm' else None)==v for k,v in chosen.items()) and p.get('batch',large_batch)<=large_batch and p.get('generator_backend','cuda') in admitted_generators:
    pilots.append(dict(p,batch=p.get('batch',large_batch),generator_backend=p.get('generator_backend','cuda')))
  pilots.sort(key=lambda p:(p['generator_backend']!=global_profile.get('generator_backend','cuda'),p['batch']!=global_profile.get('batch')))
  if not pilots:pilots=[dict(batch=large_batch,generator_backend=default_generator,layer_sizes=[1],layer_seconds=[])]
  budget=0.
  profiles=select_size_profiles(pilots,len(selected),threshold=len(selected)*global_profile.get('hardware_policy',{}).get('size_threshold_per_rank',16384))
  result=dict(base,online_size_tuning=True,global_profile=global_profile,status='MEASURED_FRONTIER_SIZE_PROFILES',size_profiles=profiles,pilots=pilots,identity=identity,cache_hit=False,measured=global_profile.get('measured',False) or any(p['measured'] for p in profiles),selection_measured=global_profile.get('measured',False),size_profiles_measured=any(p['measured'] for p in profiles),seconds=time.monotonic()-started,pilot_budget_seconds=budget,scope='same admitted shard/history/transport geometry; layer-boundary batch and exact generator selection; unmeasured large frontiers are conservative, no extrapolated optimum')
  if global_profile.get('candidate_failures'):result.update(candidate_failures=global_profile['candidate_failures'],failed_candidate_policy=global_profile['failed_candidate_policy'],failed_probes_retried=False)
  result['profile_cache_read_status']=profile_cache_read_status
  if dependency and hardware.returncode==0 and topology.returncode==0:
   result['profile_cache_write_status']=_write_profile_cache(cache,result,prefix='size-'+key+'-')
  return result


@__import__('contextlib').contextmanager
def _evidence_directory(env,prefix):
 import shutil,traceback
 temporary=tempfile.TemporaryDirectory(prefix=prefix)
 try:
  yield temporary.name
 except BaseException as error:
  root=artifact_path('tuning-evidence',env=env,override='MGBFS_TUNING_EVIDENCE_ROOT')
  try:
   root.mkdir(parents=True,exist_ok=True)
   destination=Path(tempfile.mkdtemp(prefix=prefix,dir=root))
  except OSError as save_error:
   if any(env.get(key,os.environ.get(key)) for key in ('MGBFS_TUNING_EVIDENCE_ROOT','MGBFS_ARTIFACT_ROOT')):
    raise error from save_error
   try:destination=Path(tempfile.mkdtemp(prefix=prefix))
   except OSError as fallback_error:raise error from fallback_error
  try:shutil.copytree(temporary.name,destination,dirs_exist_ok=True)
  except OSError as save_error:raise error from save_error
  original_error=str(error)
  old=str(Path(temporary.name));new=str(destination.resolve())
  error.args=tuple(arg.replace(old,new) if isinstance(arg,str) else arg for arg in error.args)
  error.tuning_evidence_root=new
  # Rebase metadata, preserving verbatim worker logs and their checksums.
  def rebase(value):
   if isinstance(value,str):return value.replace(old,new)
   if isinstance(value,list):return [rebase(item) for item in value]
   if isinstance(value,dict):return {key:rebase(item) for key,item in value.items()}
   return value
  for summary in destination.glob('**/failure-evidence/summary.json'):
   try:summary.write_text(json.dumps(rebase(json.loads(summary.read_text())),indent=2))
   except (OSError,ValueError):pass
  (destination/'failure-context.json').write_text(json.dumps(dict(schema=1,error_type=type(error).__name__,error=str(error),original_error=original_error,evidence=new,trace=traceback.format_exc())))
  if hasattr(error,'add_note'):error.add_note('Tuning evidence: '+new)
  raise
 finally:
  temporary.cleanup()
