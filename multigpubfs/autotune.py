"""Bounded same-graph GPU prefix tuning; no CPU successor or dedup in production."""
import hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
from .process_control import query as native_query

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
 try:result=subprocess.run(command,capture_output=True,text=True,timeout=10)
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
 try:resolved=subprocess.run(['ldd',native],capture_output=True,text=True,env=env,timeout=10)
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
   if path.is_file():return {'cuda_library_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'resolved_dependencies':dependencies}
 return None

_dependency_cache={}
def _stat_identity(path):
 try:
  s=Path(path).stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 except OSError:return None

def _dependency_identity(native,env):
 from .generic_session import _active
 if _active is None:return _dependency_identity_uncached(native,env)
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
 ignored={'MGBFS_PROFILE_CACHE','MGBFS_CONTROL_TOKEN','MGBFS_RUN_ID','MGBFS_TUNING_EVIDENCE_ROOT','MGBFS_JOB_PHASE_PATH','MGBFS_JOB_SEQUENCE','MGBFS_JOB_CANCEL_PATH'}
 values={k:v for k,v in env.items() if (k.startswith(('NCCL_','MGBFS_')) or k in ('CUDA_VISIBLE_DEVICES','CUDA_MODULE_LOADING')) and k not in ignored}
 return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()

def choose_profile(graph,devices,capacity,max_seconds,native,env,*,allow_specialized=True):
 # Omitted peer policy and public default auto are the same configuration.
 env=dict(env);env.setdefault('MGBFS_PEER_TRANSPORT','auto')
 from .specialized import exact_specialized_supported
 allow_specialized=allow_specialized and exact_specialized_supported(graph)
 from .generation import gemm_supported
 forced_generator=env.get('MGBFS_GENERIC_GENERATOR','cuda')
 if forced_generator not in ('cuda','gemm'):raise ValueError('INVALID_GENERIC_GENERATOR')
 if forced_generator=='gemm' and not gemm_supported(graph):raise ValueError('GEMM_GRAPH_NOT_EXACTLY_SUPPORTED')
 started=time.monotonic();variants=_transport_variants(graph,env);default_transport=variants[0][2];default_order=variants[0][3]
 with _evidence_directory(env,'mgbfs-profile-') as temporary:
  baseline_shards=variants[0][0]
  baseline=_admit(graph,devices,capacity,baseline_shards,native,env,temporary)
  selected=baseline['devices'];plan=baseline['plan']
  from .hardware_profiles import startup_policy
  hardware_policy=startup_policy(baseline.get('inventory',[]),plan)
  variants=_transport_variants(graph,env,hardware_policy['shards'])
  # The same sequential startup owns these selected devices; pilot processes
  # release their allocations before the next admission. Production queries live VRAM again.
  if baseline.get('inventory'):env['_MGBFS_PROFILE_INVENTORY']=json.dumps(baseline['inventory'])
  if forced_generator=='gemm' and plan.get('generator_backend')!='gemm':raise RuntimeError('NATIVE_GEMM_CAPABILITY_REQUIRED')
  from .hardware_profiles import small_permutation_workload
  if plan['capacity']<=4096 or max_seconds<10 or small_permutation_workload(graph,baseline.get('inventory',[])):
   return {'status':'SMALL_OR_SHORT_WORKLOAD_CONSERVATIVE_PROFILE','shards':baseline_shards,'batch_fraction':1.0,'transport':default_transport,'candidate_order':default_order,'measured':False,'seconds':time.monotonic()-started}
  # Native content, graph and driver/topology identity prevent stale reuse.
  hardware=_system_info(['nvidia-smi','--query-gpu=uuid,driver_version,name','--format=csv,noheader'])
  topology=_system_info(['nvidia-smi','topo','-m'])
  dependency=_dependency_identity(native,env)
  identity={'schema':17,'selector_sha256':_content_digest(__file__,Path(__file__).with_name('profile_families.py'),Path(__file__).with_name('size_profiles.py')),'configuration_digest':_configuration_digest(env),'baseline_geometry':{'plan':{k:plan.get(k) for k in ('capacity','batch','state_bytes','history_layers','history_algorithm','owner_lanes','sorted_owner_bytes')},'rank_capacities':[p['capacity'] for p in baseline.get('rank_plans',[])]},'hardware_policy':hardware_policy,'allow_specialized':allow_specialized,'native_dependencies':dependency,'graph':graph.digest(),'native_sha256':_content_digest(native),'devices':selected,'hardware':hardware.stdout,'topology':topology.stdout,'capacity_override':capacity,'environment':{k:env.get(k) for k in ('CUDA_VISIBLE_DEVICES','NCCL_P2P_DISABLE','NCCL_SHM_DISABLE','NCCL_SOCKET_IFNAME','MGBFS_GENERIC_TRANSPORT','MGBFS_GENERIC_SORT','MGBFS_PEER_TRANSPORT')}}
  key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
  root=Path(os.environ.get('MGBFS_PROFILE_CACHE',str(Path.home()/'.cache/multigpubfs/profiles')));cache=root/(key+'.json')
  if dependency is not None and hardware.returncode==0 and topology.returncode==0 and cache.is_file():
   try:saved=json.loads(cache.read_text())
   except (OSError,ValueError):saved={}
   from .specialized import match_lrx
   specialized_eligible=allow_specialized and len(selected) in (1,2,4,8) and not env.get('MGBFS_GENERIC_TRANSPORT') and not env.get('MGBFS_GENERIC_SORT') and not env.get('MGBFS_GENERIC_HISTORY') and match_lrx(graph) is not None
   if saved.get('generator_backend','cuda') in (('cuda','gemm') if gemm_supported(graph) else ('cuda',)) and saved.get('identity')==identity and saved.get('status')=='MEASURED_EQUAL_PREFIX_GPU_PROFILE' and type(saved.get('shards')) is int and 1<=saved['shards']<=4096 and saved.get('batch_fraction') in (1.0,.25) and ((saved.get('backend','generic')=='generic' and (saved.get('shards'),saved.get('batch_fraction'),saved.get('transport'),saved.get('candidate_order')) in variants and saved.get('history_algorithm','HASH')=='HASH' or saved.get('backend','generic')=='generic' and saved.get('history_algorithm')=='SORTED_RUNS' and saved.get('owner_lanes') in (1,2,4,8) and saved.get('shards')==max(4,saved['owner_lanes']) and saved.get('transport')==default_transport and saved.get('candidate_order')=='none') or (specialized_eligible and saved.get('backend') in ('shard_ab_hash','shard_ab_sort_merge') and type(saved.get('specialized_capacity')) is int and saved['specialized_capacity']>0)):
    saved=dict(saved);saved['cache_hit']=True;saved['seconds']=time.monotonic()-started;saved['_admission_inventory']=baseline.get('inventory',[]);return saved
  from .profile_families import identity as family_identity,load as family_load,transferred
  shared_identity=family_identity(graph,plan,identity)
  if dependency is not None and hardware.returncode==0 and topology.returncode==0:
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
  for shards,fraction,transport,order in variants:
   candidate_env=dict(env,MGBFS_GENERIC_TRANSPORT=transport,MGBFS_GENERIC_SORT=order)
   try:admission=_admit(graph,selected,capacity,shards,native,candidate_env,temporary)
   except RuntimeError as error:
    if any(code in str(error) for code in ('REQUESTED_CAPACITY_EXCEEDS_ADMISSION','REQUESTED_CAPACITY_EXCEEDS_SORTED_ADMISSION','GENERIC_DISTRIBUTED_NO_CAPACITY','GENERIC_DISTRIBUTED_HEADROOM')):continue
    raise
   admitted.append((shards,fraction,transport,order,admission['plan']))
  history=env.get('MGBFS_GENERIC_HISTORY');sorted_admitted=[]
  if history in (None,'sorted'):
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
  if len(admitted)+len(sorted_admitted)<2:return {'status':'NO_ADMITTED_ALTERNATIVE_CONSERVATIVE_PROFILE','shards':baseline_shards,'batch_fraction':1.0,'transport':default_transport,'candidate_order':default_order,'measured':False,'seconds':time.monotonic()-started}
  common_capacity=min([p['capacity'] for _,_,_,_,p in admitted]+[p['capacity'] for _,_,_,p in sorted_admitted]+[hardware_policy['probe_capacity_per_rank']])
  pilots=[];limit=5;candidate_failures=[]
  from .launch import run_graph
  def measure(label,candidate_env,run):
   # Only the failed candidate is rejected. Successful candidates still have
   # to establish a real matched-prefix comparison on the selected GPU group.
   try:
    with _evidence_directory(candidate_env,'mgbfs-candidate-'+label+'-') as trial:
     return run(Path(trial)/label)
   except RuntimeError as error:
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
  for index,(shards,fraction,transport,order,_) in enumerate(admitted):
   if max_seconds-(time.monotonic()-started)<limit+1:break
   candidate_env=dict(env,MGBFS_GENERIC_TRANSPORT=transport,MGBFS_GENERIC_SORT=order)
   admission=_admit(graph,selected,common_capacity,shards,native,candidate_env,temporary)
   batch=max(1,int(admission['plan']['batch']*fraction))
   label='pilot-'+str(index)
   report=measure(label,candidate_env,lambda target:run_graph(graph,target,devices=selected,capacity=common_capacity,max_seconds=limit,executable=native,shards=shards,autotune=False,_batch=batch,_profile_layers=36,_native_env=candidate_env))
   if report is None:continue
   pilots.append({'candidate':label,'rank_worker_pids':report.get('rank_worker_pids',[]),'backend':'generic','history_algorithm':admission['plan'].get('history_algorithm','HASH'),'owner_lanes':admission['plan'].get('owner_lanes',0),'candidate_order':order,'transport':transport,'shards':shards,'batch_fraction':fraction,'batch':batch,'state_bytes':report.get('state_bytes',report.get('plan',{}).get('state_bytes')),'layer_sizes':report['layer_sizes'],'layer_seconds':report['layer_seconds'],'status':report['status'],'reason':report['reason']})
  # Compare generator on the same graph, history, capacity and transport.
  # GEMM has its own explicit workspace admission and a bounded launch batch.
  if pilots and gemm_supported(graph) and _gemm_hardware_available() and forced_generator=='cuda' and max_seconds-(time.monotonic()-started)>=limit+1 :
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
  if candidate_failures and len(pilots)<2:no_measured_profile()
  if not pilots:return {'status':'TUNING_BUDGET_EXHAUSTED_CONSERVATIVE_PROFILE','shards':baseline_shards,'batch_fraction':1.0,'transport':default_transport,'candidate_order':default_order,'measured':False,'seconds':time.monotonic()-started,'_admission_inventory':baseline.get('inventory',[])}
  depth=min(len(p['layer_seconds']) for p in pilots)
  sizes=pilots[0]['layer_sizes'][:depth+1]
  if any(p['layer_sizes'][:depth+1]!=sizes for p in pilots):raise RuntimeError('AUTOTUNE_PREFIX_CORRECTNESS_MISMATCH')
  # Exclude first collective warmup. Tiny prefixes cannot justify a speed claim.
  if depth<4 or sum(sizes[2:])<32768:
   if candidate_failures:no_measured_profile()
   return {'status':'INSUFFICIENT_PREFIX_CONSERVATIVE_PROFILE','shards':baseline_shards,'batch_fraction':1.0,'transport':default_transport,'candidate_order':default_order,'measured':False,'pilots':pilots,'common_depth':depth,'seconds':time.monotonic()-started,'_admission_inventory':baseline.get('inventory',[])}
  scores=[sum(p['layer_seconds'][1:depth]) for p in pilots]
  winner=min(range(len(scores)),key=lambda i:scores[i])
  # A single bounded sample cannot justify a marginal switch. Keep baseline
  # unless the alternative beats it by at least five percent.
  if scores[winner]>=scores[0]*.95:winner=0
  chosen=pilots[winner]
  chosen_generator=chosen.get('generator_backend',forced_generator)
  result={'generator_backend':chosen_generator,'backend':chosen['backend'],'status':'MEASURED_EQUAL_PREFIX_GPU_PROFILE','shards':chosen['shards'],'batch_fraction':chosen['batch_fraction'],'transport':chosen['transport'],'candidate_order':chosen['candidate_order'],'measured':True,'minimum_switch_improvement':.05,'common_depth':depth,'common_layer_sizes':sizes,'scores_seconds':scores,'pilots':pilots,'identity':identity,'cache_hit':False,'seconds':time.monotonic()-started,'hardware_policy':hardware_policy,'pilot_capacity_per_rank':common_capacity,'scope':'hardware-derived prefix among admitted shard/batch and full-child/parent-origin and unsorted/radix-index transport profiles and exact bounded matrix CUDA/GEMM generators and eligible SHARD_AB hash/sorted-history owners; no claim of global optimum or large-rank acceptance'}
  result['_admission_inventory']=[] if candidate_failures else baseline.get('inventory',[])
  if candidate_failures:result.update(candidate_failures=candidate_failures,failed_candidate_policy='REJECT_AND_MEASURE_REMAINING',winner_candidate=chosen.get('candidate'),failed_probes_retried=False)
  result.update(history_algorithm=chosen.get('history_algorithm','SORTED_RUNS' if env.get('MGBFS_GENERIC_HISTORY')=='sorted' else 'HASH'),owner_lanes=chosen.get('owner_lanes',int(env.get('MGBFS_GENERIC_OWNER_LANES','0'))))
  if chosen['backend']=='generic':result['batch']=chosen['batch']
  if chosen['backend']!='generic':result.update(specialized_capacity=chosen['specialized_capacity'],specialized_mode=chosen['specialized_mode'],specialized_transport=chosen['specialized_transport'],specialized_admission=chosen['specialized_admission'],batch=chosen['batch'])
  if dependency is not None and hardware.returncode==0 and topology.returncode==0:
   root.mkdir(parents=True,exist_ok=True)
   fd,path=tempfile.mkstemp(prefix=key+'-',suffix='.tmp',dir=root)
   with os.fdopen(fd,'w') as f:json.dump(result,f)
   os.replace(path,cache)
   from .profile_families import store as family_store
   family_store(root,shared_identity,result)
  return result


def _session_profile_key(pool,graph,devices,capacity,max_seconds,native,env,allow_specialized):
 dependency=_dependency_identity(native,env)
 if dependency is None:return None
 files=(native,__file__,Path(__file__).with_name('profile_families.py'),Path(__file__).with_name('size_profiles.py'))
 return (pool.generation,graph.digest(),None if devices is None else tuple(devices),capacity,allow_specialized,_configuration_digest(env),tuple(env.get(k) for k in ('LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT','CUDA_DEVICE_ORDER','CUDA_DISABLE_PTX_JIT')),tuple((str(p),_stat_identity(p)) for p in files),tuple((d[1],_stat_identity(d[1])) for d in dependency['resolved_dependencies']))

def _profile_environment(env,result):
 selected=dict(env,MGBFS_GENERIC_TRANSPORT=result.get('transport','full'),MGBFS_GENERIC_SORT=result.get('candidate_order','none'),MGBFS_GENERIC_GENERATOR=result.get('generator_backend','cuda'))
 if result.get('history_algorithm')=='SORTED_RUNS':selected.update(MGBFS_GENERIC_HISTORY='sorted',MGBFS_GENERIC_OWNER_LANES=str(result['owner_lanes']))
 elif result.get('history_algorithm')=='HASH':selected['MGBFS_GENERIC_HISTORY']='hash'
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
  reserve=first if reserve_env['MGBFS_GENERIC_GENERATOR']==env['MGBFS_GENERIC_GENERATOR'] else _admit(graph,selected,capacity,shards,native,reserve_env,temporary)
  common=min(first['plan']['capacity'],reserve['plan']['capacity'])
  reserve=reserve if capacity==common else _admit(graph,selected,common,shards,native,reserve_env,temporary)
  large_batch=reserve['plan']['batch'];small_batch=min(4096,large_batch)
  dependency=_dependency_identity(native,env)
  hardware=_system_info(['nvidia-smi','--query-gpu=uuid,driver_version,name','--format=csv,noheader'])
  topology=_system_info(['nvidia-smi','topo','-m'])
  identity={'schema':1,'graph':graph.digest(),'devices':selected,'capacity':common,'batch':large_batch,'plan':reserve['plan'],'config':_configuration_digest(env),'native':_content_digest(native),'selector':_content_digest(__file__,Path(__file__).with_name('size_profiles.py')),'dependency':dependency,'hardware':hardware.stdout,'topology':topology.stdout}
  key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
  root=Path(os.environ.get('MGBFS_PROFILE_CACHE',str(Path.home()/'.cache/multigpubfs/profiles')));cache=root/('size-'+key+'.json')
  if dependency and hardware.returncode==0 and topology.returncode==0 and cache.is_file():
   try:result=json.loads(cache.read_text())
   except (OSError,ValueError):result={}
   if not isinstance(result,dict):result={}
   profiles=result.get('size_profiles',[])
   if result.get('identity')==identity and result.get('status')=='MEASURED_FRONTIER_SIZE_PROFILES' and valid_profile_entries(profiles,large_batch,('cuda','gemm') if eligible else ('cuda',)):
    result=dict(result,cache_hit=True,global_profile=global_profile,seconds=time.monotonic()-started);return result
  if global_profile.get('status')=='REUSED_COMPATIBLE_GPU_PROFILE':
   source=global_profile.get('source_pilots',[])
   compatible=[p for p in source if p.get('backend')=='generic' and p.get('history_algorithm','HASH')==global_profile.get('history_algorithm','HASH') and p.get('shards')==shards and p.get('transport')==global_profile.get('transport') and p.get('candidate_order')==global_profile.get('candidate_order')]
   threshold=len(selected)*max(1024,min(int(i['sm_count']) for i in first['inventory'])*512)
   compatible=[dict(p,generator_backend=p.get('generator_backend','cuda')) for p in compatible]
   tiers=select_size_profiles(compatible,len(selected),threshold=threshold) if compatible else []
   profiles=[]
   for tier in tiers:
    donor_batch=next((p.get('batch',1) for p in compatible if p.get('generator_backend','cuda')==tier['generator_backend']),1)
    batch=max(1,min(large_batch,int(large_batch*tier['batch']/max(1,donor_batch))))
    profiles.append(dict(minimum_frontier=tier['minimum_frontier'],batch=batch,generator_backend=tier['generator_backend'],measured=False,status='REUSED_SOURCE_SIZE_PROFILE',source_maximum_measured_frontier=tier['maximum_measured_frontier']))
   if not profiles:profiles=[dict(minimum_frontier=0,batch=max(1,int(large_batch*global_profile['batch_fraction'])),generator_backend=default_generator,measured=False,status='REUSED_SOURCE_CONSERVATIVE_PROFILE')]
   return dict(global_profile,batch=max(1,int(large_batch*global_profile['batch_fraction'])),size_profiles=profiles,online_size_tuning=True,seconds=time.monotonic()-started,global_profile=global_profile,pilot_budget_seconds=0.,target_pilots_executed=0)
  base={'backend':'generic','shards':shards,'batch':large_batch,'batch_fraction':1.,'transport':env.get('MGBFS_GENERIC_TRANSPORT','full'),'candidate_order':env.get('MGBFS_GENERIC_SORT','none'),'history_algorithm':reserve['plan'].get('history_algorithm','HASH'),'owner_lanes':reserve['plan'].get('owner_lanes',0),'generator_backend':reserve['plan'].get('generator_backend','cuda'),'size_profiles':[],'seconds':time.monotonic()-started,'measured':False}
  if max_seconds<10 or common<=4096:return dict(base,status='SMALL_OR_SHORT_WORKLOAD_CONSERVATIVE_PROFILE')
  # Reuse completed matching startup pilots, never rerun the prefix for tiers.
  pilots=[]
  chosen={'shards':shards,'transport':env.get('MGBFS_GENERIC_TRANSPORT','full'),'candidate_order':env.get('MGBFS_GENERIC_SORT','none'),'history_algorithm':reserve['plan'].get('history_algorithm','HASH'),'owner_lanes':reserve['plan'].get('owner_lanes',0)}
  for p in global_profile.get('pilots',[]):
   if all(p.get(k,0 if k=='owner_lanes' else 'HASH' if k=='history_algorithm' else None)==v for k,v in chosen.items()) and p.get('batch',large_batch)<=large_batch:
    pilots.append(dict(p,batch=p.get('batch',large_batch),generator_backend=p.get('generator_backend','cuda')))
  pilots.sort(key=lambda p:(p['generator_backend']!=global_profile.get('generator_backend','cuda'),p['batch']!=global_profile.get('batch')))
  if not pilots:pilots=[dict(batch=large_batch,generator_backend=default_generator,layer_sizes=[1],layer_seconds=[])]
  budget=0.
  profiles=select_size_profiles(pilots,len(selected),threshold=len(selected)*global_profile.get('hardware_policy',{}).get('size_threshold_per_rank',16384))
  result=dict(base,online_size_tuning=True,global_profile=global_profile,status='MEASURED_FRONTIER_SIZE_PROFILES',size_profiles=profiles,pilots=pilots,identity=identity,cache_hit=False,measured=global_profile.get('measured',False) or any(p['measured'] for p in profiles),selection_measured=global_profile.get('measured',False),size_profiles_measured=any(p['measured'] for p in profiles),seconds=time.monotonic()-started,pilot_budget_seconds=budget,scope='same admitted shard/history/transport geometry; layer-boundary batch and exact generator selection; unmeasured large frontiers are conservative, no extrapolated optimum')
  if global_profile.get('candidate_failures'):result.update(candidate_failures=global_profile['candidate_failures'],failed_candidate_policy=global_profile['failed_candidate_policy'],failed_probes_retried=False)
  if dependency and hardware.returncode==0 and topology.returncode==0:
   root.mkdir(parents=True,exist_ok=True);fd,path=tempfile.mkstemp(prefix='size-'+key+'-',suffix='.tmp',dir=root)
   with os.fdopen(fd,'w') as f:json.dump(result,f)
   os.replace(path,cache)
  return result


@__import__('contextlib').contextmanager
def _evidence_directory(env,prefix):
 import shutil,traceback
 temporary=tempfile.TemporaryDirectory(prefix=prefix)
 try:
  yield temporary.name
 except BaseException as error:
  root=Path(env.get('MGBFS_TUNING_EVIDENCE_ROOT',str(Path.cwd()/'mgbfs-tuning-failures')))
  root.mkdir(parents=True,exist_ok=True)
  destination=Path(tempfile.mkdtemp(prefix=prefix,dir=root))
  shutil.copytree(temporary.name,destination,dirs_exist_ok=True)
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
