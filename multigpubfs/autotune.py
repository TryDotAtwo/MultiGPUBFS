"""Bounded same-graph GPU prefix tuning; no CPU successor or dedup in production."""
import hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
from .process_control import query as native_query

def _admit(graph,devices,capacity,shards,native,env,directory):
 path=Path(directory)/'definition.json';path.write_text(graph.to_json())
 command=[native,'graph-info',str(path),'auto' if devices is None else ','.join(map(str,devices)),str(shards)]
 if capacity is not None:command.append(str(capacity))
 p=native_query(command,capture_output=True,text=True,env=env)
 if p.returncode:raise RuntimeError('PROFILE_ADMISSION_FAILED: '+p.stderr[-2000:])
 return json.loads(p.stdout)

def _system_info(command):
 try:return subprocess.run(command,capture_output=True,text=True,timeout=10)
 except (OSError,subprocess.TimeoutExpired):return subprocess.CompletedProcess(command,1,'unavailable','')

def _gemm_hardware_available():
 caps=_system_info(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'])
 try:return caps.returncode==0 and bool(caps.stdout.strip()) and all(float(v.strip())>=8.0 for v in caps.stdout.splitlines())
 except ValueError:return False

def _dependency_identity(native,env):
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

def _transport_variants(graph,env):
 forced=env.get('MGBFS_GENERIC_TRANSPORT')
 if forced not in (None,'full','parent'):raise ValueError('INVALID_TRANSPORT')
 order=env.get('MGBFS_GENERIC_SORT')
 if order not in (None,'none','radix'):raise ValueError('INVALID_CANDIDATE_ORDER')
 mode=forced or 'full';order_default=order or 'none';variants=[(1,1.0,mode,order_default),(4,1.0,mode,order_default),(16,1.0,mode,order_default),(4,.25,mode,order_default)]
 packed=graph.action['kind']=='permutation' and graph.state_elements<=24 and all(0<=v<=255 for v in graph.start)
 if forced is None and not packed:variants += [(4,1.0,'parent',order_default),(4,.25,'parent',order_default)]
 if order is None:variants.append((4,.25,'parent' if forced is None and not packed else mode,'radix'))
 history=env.get('MGBFS_GENERIC_HISTORY')
 if history not in (None,'hash','sorted'):raise ValueError('INVALID_HISTORY_ALGORITHM')
 if history=='sorted':
  lane=int(env.get('MGBFS_GENERIC_OWNER_LANES','1'))
  if not 1<=lane<=8:raise ValueError('INVALID_OWNER_LANES')
  variants=list(dict.fromkeys((max(sh,lane),f,tr,'none') for sh,f,tr,_ in variants))
 return variants

def _configuration_digest(env):
 # Values are hashed, never copied into a public receipt (tokens may coexist).
 ignored={'MGBFS_PROFILE_CACHE','MGBFS_CONTROL_TOKEN','MGBFS_RUN_ID'}
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
 with tempfile.TemporaryDirectory(prefix='mgbfs-profile-') as temporary:
  baseline_shards=variants[0][0]
  baseline=_admit(graph,devices,capacity,baseline_shards,native,env,temporary)
  selected=baseline['devices'];plan=baseline['plan']
  if forced_generator=='gemm' and plan.get('generator_backend')!='gemm':raise RuntimeError('NATIVE_GEMM_CAPABILITY_REQUIRED')
  if plan['capacity']<=4096 or max_seconds<10:
   return {'status':'SMALL_OR_SHORT_WORKLOAD_CONSERVATIVE_PROFILE','shards':baseline_shards,'batch_fraction':1.0,'transport':default_transport,'candidate_order':default_order,'measured':False,'seconds':time.monotonic()-started}
  # Native content, graph and driver/topology identity prevent stale reuse.
  hardware=_system_info(['nvidia-smi','--query-gpu=uuid,driver_version,name','--format=csv,noheader'])
  topology=_system_info(['nvidia-smi','topo','-m'])
  dependency=_dependency_identity(native,env)
  identity={'schema':15,'selector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'configuration_digest':_configuration_digest(env),'baseline_geometry':{'plan':{k:plan.get(k) for k in ('capacity','batch','state_bytes','history_layers','history_algorithm','owner_lanes','sorted_owner_bytes')},'rank_capacities':[p['capacity'] for p in baseline.get('rank_plans',[])]},'allow_specialized':allow_specialized,'native_dependencies':dependency,'graph':graph.digest(),'native_sha256':hashlib.sha256(Path(native).read_bytes()).hexdigest(),'devices':selected,'hardware':hardware.stdout,'topology':topology.stdout,'capacity_override':capacity,'environment':{k:env.get(k) for k in ('CUDA_VISIBLE_DEVICES','NCCL_P2P_DISABLE','NCCL_SHM_DISABLE','NCCL_SOCKET_IFNAME','MGBFS_GENERIC_TRANSPORT','MGBFS_GENERIC_SORT','MGBFS_PEER_TRANSPORT')}}
  key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
  root=Path(os.environ.get('MGBFS_PROFILE_CACHE',str(Path.home()/'.cache/multigpubfs/profiles')));cache=root/(key+'.json')
  if dependency is not None and hardware.returncode==0 and topology.returncode==0 and cache.is_file():
   try:saved=json.loads(cache.read_text())
   except (OSError,ValueError):saved={}
   from .specialized import match_lrx
   specialized_eligible=allow_specialized and len(selected) in (1,2,4,8) and not env.get('MGBFS_GENERIC_TRANSPORT') and not env.get('MGBFS_GENERIC_SORT') and not env.get('MGBFS_GENERIC_HISTORY') and match_lrx(graph) is not None
   if saved.get('generator_backend','cuda') in (('cuda','gemm') if gemm_supported(graph) else ('cuda',)) and saved.get('identity')==identity and saved.get('status')=='MEASURED_EQUAL_PREFIX_GPU_PROFILE' and saved.get('shards') in (1,4,8,16) and saved.get('batch_fraction') in (1.0,.25) and ((saved.get('backend','generic')=='generic' and (saved.get('shards'),saved.get('batch_fraction'),saved.get('transport'),saved.get('candidate_order')) in variants and saved.get('history_algorithm','HASH')=='HASH' or saved.get('backend','generic')=='generic' and saved.get('history_algorithm')=='SORTED_RUNS' and saved.get('owner_lanes') in (1,2,4,8) and saved.get('shards')==max(4,saved['owner_lanes']) and saved.get('transport')==default_transport and saved.get('candidate_order')=='none') or (specialized_eligible and saved.get('backend') in ('shard_ab_hash','shard_ab_sort_merge') and type(saved.get('specialized_capacity')) is int and saved['specialized_capacity']>0)):
    saved=dict(saved);saved['cache_hit']=True;saved['seconds']=time.monotonic()-started;return saved
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
  common_capacity=min([p['capacity'] for _,_,_,_,p in admitted]+[p['capacity'] for _,_,_,p in sorted_admitted])
  pilots=[];limit=1
  startup_budget=min(8.,max_seconds*.1)
  from .launch import run_graph
  for index,(shards,fraction,transport,order,_) in enumerate(admitted):
   if index>=2 and time.monotonic()-started>=startup_budget:break
   candidate_env=dict(env,MGBFS_GENERIC_TRANSPORT=transport,MGBFS_GENERIC_SORT=order)
   admission=_admit(graph,selected,common_capacity,shards,native,candidate_env,temporary)
   batch=max(1,int(admission['plan']['batch']*fraction))
   report=run_graph(graph,Path(temporary)/('pilot-'+str(index)),devices=selected,capacity=common_capacity,max_seconds=limit,executable=native,shards=shards,autotune=False,_batch=batch,_profile_layers=36,_native_env=candidate_env)
   pilots.append({'backend':'generic','history_algorithm':admission['plan'].get('history_algorithm','HASH'),'owner_lanes':admission['plan'].get('owner_lanes',0),'candidate_order':order,'transport':transport,'shards':shards,'batch_fraction':fraction,'batch':batch,'state_bytes':report.get('state_bytes',report.get('plan',{}).get('state_bytes')),'layer_sizes':report['layer_sizes'],'layer_seconds':report['layer_seconds'],'status':report['status'],'reason':report['reason']})
  # Same canonical state capacity: history/lane alternatives must pass actual
  # native workspace admission before they can become timing candidates.
  if sorted_admitted:
   for lane,shards,transport,_ in sorted_admitted:
    remaining=max_seconds-(time.monotonic()-started)
    if remaining<3 or time.monotonic()-started>=startup_budget:break
    shards=max(4,lane);transport=default_transport
    candidate_env=dict(env,MGBFS_GENERIC_HISTORY='sorted',MGBFS_GENERIC_OWNER_LANES=str(lane),MGBFS_GENERIC_TRANSPORT=transport,MGBFS_GENERIC_SORT='none')
    try:admission=_admit(graph,selected,common_capacity,shards,native,candidate_env,temporary)
    except RuntimeError as error:
     if any(code in str(error) for code in ('REQUESTED_CAPACITY_EXCEEDS_ADMISSION','REQUESTED_CAPACITY_EXCEEDS_SORTED_ADMISSION','GENERIC_DISTRIBUTED_NO_CAPACITY','GENERIC_DISTRIBUTED_HEADROOM','SORTED_OWNER_NO_ADMISSION')):continue
     raise
    batch=max(1,min(admission['plan']['batch'],pilots[0]['batch']))
    report=run_graph(graph,Path(temporary)/('sorted-lanes-'+str(lane)),devices=selected,capacity=common_capacity,max_seconds=min(limit,max(1,int(remaining))),executable=native,shards=shards,autotune=False,_batch=batch,_profile_layers=36,_native_env=candidate_env)
    pilots.append({'backend':'generic','history_algorithm':'SORTED_RUNS','owner_lanes':lane,'candidate_order':'none','transport':transport,'shards':shards,'batch_fraction':1.0,'batch':batch,'layer_sizes':report['layer_sizes'],'layer_seconds':report['layer_seconds'],'status':report['status'],'reason':report['reason']})
  # Compare generator on the same graph, history, capacity and transport.
  # GEMM has its own explicit workspace admission and a bounded launch batch.
  if gemm_supported(graph) and _gemm_hardware_available() and forced_generator=='cuda' and max_seconds-(time.monotonic()-started)>=3 and time.monotonic()-started<startup_budget:
   base=pilots[0];candidate_env=dict(env,MGBFS_GENERIC_GENERATOR='gemm',MGBFS_GENERIC_TRANSPORT=base['transport'],MGBFS_GENERIC_SORT=base['candidate_order'],MGBFS_GENERIC_HISTORY='sorted' if base.get('history_algorithm')=='SORTED_RUNS' else 'hash',MGBFS_GENERIC_OWNER_LANES=str(base.get('owner_lanes',1)))
   try:
    admission=_admit(graph,selected,common_capacity,base['shards'],native,candidate_env,temporary)
    if admission['plan'].get('generator_backend')!='gemm':raise RuntimeError('NATIVE_GEMM_CAPABILITY_REQUIRED')
    batch=min(base['batch'],admission['plan']['batch'])
    report=run_graph(graph,Path(temporary)/'generator-gemm',devices=selected,capacity=common_capacity,max_seconds=min(limit,max(1,int(max_seconds-(time.monotonic()-started)))),executable=native,shards=base['shards'],autotune=False,_batch=batch,_profile_layers=36,_native_env=candidate_env)
    pilots.append(dict(base,generator_backend='gemm',batch=batch,layer_sizes=report['layer_sizes'],layer_seconds=report['layer_seconds'],status=report['status'],reason=report['reason']))
   except RuntimeError as error:
    if not any(code in str(error) for code in ('GEMM_BATCH_EXCEEDS_ADMISSION','REQUESTED_CAPACITY_EXCEEDS_ADMISSION','GENERIC_CUDA_801','GENERIC_CUDA_209','NATIVE_GEMM_CAPABILITY_REQUIRED')):raise

  # Specialized candidates preserve only a proved identical LRX action/root.
  # Unsupported definitions/topologies remain on the general exact backend.
  if allow_specialized and len(selected) in (1,2,4,8) and not env.get('MGBFS_GENERIC_TRANSPORT') and not env.get('MGBFS_GENERIC_SORT') and not env.get('MGBFS_GENERIC_HISTORY') and max_seconds-(time.monotonic()-started)>=8 and time.monotonic()-started<startup_budget:
   from .specialized import match_lrx,admit_specialized,run_specialized
   matched=match_lrx(graph)
   if matched is not None and matched['order']>=32768:
    from .transport import select_peer_transport
    transport_decision=select_peer_transport(native,env,selected,Path(temporary)/'peer-capability',request=env.get('MGBFS_PEER_TRANSPORT','auto'),seconds=max_seconds-(time.monotonic()-started))
    peer_options=['HOST_SIZED_NCCL']
    if transport_decision['selected']=='NCCL_LSA':peer_options.append('NCCL_LSA')
    for mode,backend,peer in [(mode,backend,peer) for peer in peer_options for mode,backend in [('HASH','shard_ab_hash'),('SORT_MERGE','shard_ab_sort_merge')]]:
     remaining=max_seconds-(time.monotonic()-started)
     if remaining<5:break
     candidate_env=dict(env,MGBFS_SPECIALIZED_TRANSPORT=peer)
     if peer=='NCCL_LSA':candidate_env['NCCL_CUMEM_ENABLE']='1'
     label=backend+'-'+peer
     plan=admit_specialized(graph,native,candidate_env,selected,Path(temporary)/('admit-'+label),capacity=capacity,mode=mode,seconds=min(60,remaining))
     remaining=max_seconds-(time.monotonic()-started)
     if remaining<2:break
     report=run_specialized(graph,Path(temporary)/label,native=native,env=candidate_env,devices=selected,capacity=plan['capacity'],batch=plan['batch'],max_seconds=min(5,max(2,int(remaining))),mode=mode,profile_layers=36)
     if len(report['layer_seconds'])>=4 and sum(report['layer_sizes'][2:])>=32768:
      pilots.append({'backend':backend,'candidate_order':'none' if mode=='HASH' else 'radix','transport':'specialized_key_first_host','shards':1,'batch_fraction':1.0,'batch':plan['batch'],'specialized_capacity':plan['capacity'],'specialized_mode':mode,'specialized_transport':peer,'transport_selection':transport_decision,'state_bytes':1,'layer_sizes':report['layer_sizes'],'layer_seconds':report['layer_seconds'],'status':report['status'],'reason':report['reason'],'specialized_admission':plan})
  depth=min(len(p['layer_seconds']) for p in pilots)
  sizes=pilots[0]['layer_sizes'][:depth+1]
  if any(p['layer_sizes'][:depth+1]!=sizes for p in pilots):raise RuntimeError('AUTOTUNE_PREFIX_CORRECTNESS_MISMATCH')
  # Exclude first collective warmup. Tiny prefixes cannot justify a speed claim.
  if depth<4 or sum(sizes[2:])<32768:
   return {'status':'INSUFFICIENT_PREFIX_CONSERVATIVE_PROFILE','shards':baseline_shards,'batch_fraction':1.0,'transport':default_transport,'candidate_order':default_order,'measured':False,'pilots':pilots,'common_depth':depth,'seconds':time.monotonic()-started}
  scores=[sum(p['layer_seconds'][1:depth]) for p in pilots]
  winner=min(range(len(scores)),key=lambda i:scores[i])
  # A single bounded sample cannot justify a marginal switch. Keep baseline
  # unless the alternative beats it by at least five percent.
  if scores[winner]>=scores[0]*.95:winner=0
  chosen=pilots[winner]
  chosen_generator=chosen.get('generator_backend',forced_generator)
  result={'generator_backend':chosen_generator,'backend':chosen['backend'],'status':'MEASURED_EQUAL_PREFIX_GPU_PROFILE','shards':chosen['shards'],'batch_fraction':chosen['batch_fraction'],'transport':chosen['transport'],'candidate_order':chosen['candidate_order'],'measured':True,'minimum_switch_improvement':.05,'common_depth':depth,'common_layer_sizes':sizes,'scores_seconds':scores,'pilots':pilots,'identity':identity,'cache_hit':False,'seconds':time.monotonic()-started,'scope':'bounded prefix among admitted shard/batch and full-child/parent-origin and unsorted/radix-index transport profiles and exact bounded matrix CUDA/GEMM generators and eligible SHARD_AB hash/sorted-history owners; no claim of global optimum or large-rank acceptance'}
  result.update(history_algorithm=chosen.get('history_algorithm','SORTED_RUNS' if env.get('MGBFS_GENERIC_HISTORY')=='sorted' else 'HASH'),owner_lanes=chosen.get('owner_lanes',int(env.get('MGBFS_GENERIC_OWNER_LANES','0'))))
  if chosen['backend']=='generic':result['batch']=chosen['batch']
  if chosen['backend']!='generic':result.update(specialized_capacity=chosen['specialized_capacity'],specialized_mode=chosen['specialized_mode'],specialized_transport=chosen['specialized_transport'],specialized_admission=chosen['specialized_admission'],batch=chosen['batch'])
  if dependency is not None and hardware.returncode==0 and topology.returncode==0:
   root.mkdir(parents=True,exist_ok=True)
   fd,path=tempfile.mkstemp(prefix=key+'-',suffix='.tmp',dir=root)
   with os.fdopen(fd,'w') as f:json.dump(result,f)
   os.replace(path,cache)
  return result


def choose_size_profile(graph,devices,capacity,max_seconds,native,env,*,allow_specialized=True):
 """Bounded real-prefix measurements; unknown large tiers remain conservative."""
 from .size_profiles import select_size_profiles
 from .generation import gemm_supported
 started=time.monotonic();env=dict(env)
 global_profile=choose_profile(graph,devices,capacity,max_seconds,native,env,allow_specialized=allow_specialized)
 if global_profile.get('backend','generic')!='generic' or max_seconds<10 or global_profile.get('status')=='SMALL_OR_SHORT_WORKLOAD_CONSERVATIVE_PROFILE':return global_profile
 shards=global_profile['shards']
 env.update(MGBFS_GENERIC_TRANSPORT=global_profile.get('transport','full'),MGBFS_GENERIC_SORT=global_profile.get('candidate_order','none'),MGBFS_GENERIC_HISTORY='sorted' if global_profile.get('history_algorithm')=='SORTED_RUNS' else 'hash',MGBFS_GENERIC_OWNER_LANES=str(global_profile.get('owner_lanes',1)),MGBFS_GENERIC_GENERATOR=global_profile.get('generator_backend','cuda'))
 with tempfile.TemporaryDirectory(prefix='mgbfs-size-profile-') as temporary:
  first=_admit(graph,devices,capacity,shards,native,env,temporary);selected=first['devices']
  if first.get('size_profile_capability')!=1:return dict(global_profile,size_tuning_status='NATIVE_SIZE_PROFILE_CAPABILITY_UNAVAILABLE')
  default_generator=env.get('MGBFS_GENERIC_GENERATOR','cuda')
  eligible=default_generator=='gemm' or (gemm_supported(graph) and _gemm_hardware_available())
  reserve_env=dict(env,MGBFS_GENERIC_GENERATOR='gemm' if eligible else 'cuda')
  reserve=_admit(graph,selected,capacity,shards,native,reserve_env,temporary)
  common=min(first['plan']['capacity'],reserve['plan']['capacity'])
  reserve=_admit(graph,selected,common,shards,native,reserve_env,temporary)
  large_batch=reserve['plan']['batch'];small_batch=min(4096,large_batch)
  dependency=_dependency_identity(native,env)
  hardware=_system_info(['nvidia-smi','--query-gpu=uuid,driver_version,name','--format=csv,noheader'])
  topology=_system_info(['nvidia-smi','topo','-m'])
  identity={'schema':1,'graph':graph.digest(),'devices':selected,'capacity':common,'batch':large_batch,'plan':reserve['plan'],'config':_configuration_digest(env),'native':hashlib.sha256(Path(native).read_bytes()).hexdigest(),'selector':hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('size_profiles.py').read_bytes()).hexdigest(),'dependency':dependency,'hardware':hardware.stdout,'topology':topology.stdout}
  key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
  root=Path(os.environ.get('MGBFS_PROFILE_CACHE',str(Path.home()/'.cache/multigpubfs/profiles')));cache=root/('size-'+key+'.json')
  if dependency and hardware.returncode==0 and topology.returncode==0 and cache.is_file():
   try:result=json.loads(cache.read_text())
   except (OSError,ValueError):result={}
   if not isinstance(result,dict):result={}
   profiles=result.get('size_profiles',[])
   if result.get('identity')==identity and result.get('status')=='MEASURED_FRONTIER_SIZE_PROFILES' and isinstance(profiles,list) and profiles and all(isinstance(p,dict) and type(p.get('minimum_frontier')) is int and p['minimum_frontier']>=0 and type(p.get('batch')) is int and 0<p['batch']<=large_batch and p.get('generator_backend') in (('cuda','gemm') if eligible else ('cuda',)) for p in profiles) and profiles[0]['minimum_frontier']==0 and all(a['minimum_frontier']<b['minimum_frontier'] for a,b in zip(profiles,profiles[1:])):
    result=dict(result,cache_hit=True,global_profile=global_profile,seconds=time.monotonic()-started);return result
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
  profiles=select_size_profiles(pilots,len(selected))
  result=dict(base,online_size_tuning=True,global_profile=global_profile,status='MEASURED_FRONTIER_SIZE_PROFILES',size_profiles=profiles,pilots=pilots,identity=identity,cache_hit=False,measured=any(p['measured'] for p in profiles),seconds=time.monotonic()-started,pilot_budget_seconds=budget,scope='same admitted shard/history/transport geometry; layer-boundary batch and exact generator selection; unmeasured large frontiers are conservative, no extrapolated optimum')
  if dependency and hardware.returncode==0 and topology.returncode==0:
   root.mkdir(parents=True,exist_ok=True);fd,path=tempfile.mkstemp(prefix='size-'+key+'-',suffix='.tmp',dir=root)
   with os.fdopen(fd,'w') as f:json.dump(result,f)
   os.replace(path,cache)
  return result
