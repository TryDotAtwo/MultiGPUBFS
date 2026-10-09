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
 return variants

def _configuration_digest(env):
 # Values are hashed, never copied into a public receipt (tokens may coexist).
 ignored={'MGBFS_PROFILE_CACHE','MGBFS_CONTROL_TOKEN','MGBFS_RUN_ID'}
 values={k:v for k,v in env.items() if (k.startswith(('NCCL_','MGBFS_')) or k in ('CUDA_VISIBLE_DEVICES','CUDA_MODULE_LOADING')) and k not in ignored}
 return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()

def choose_profile(graph,devices,capacity,max_seconds,native,env,*,allow_specialized=True):
 from .specialized import exact_specialized_supported
 allow_specialized=allow_specialized and exact_specialized_supported()
 started=time.monotonic();variants=_transport_variants(graph,env);default_transport=variants[0][2];default_order=variants[0][3]
 with tempfile.TemporaryDirectory(prefix='mgbfs-profile-') as temporary:
  baseline=_admit(graph,devices,capacity,1,native,env,temporary)
  selected=baseline['devices'];plan=baseline['plan']
  if plan['capacity']<=4096 or max_seconds<10:
   return {'status':'SMALL_OR_SHORT_WORKLOAD_CONSERVATIVE_PROFILE','shards':1,'batch_fraction':1.0,'transport':default_transport,'candidate_order':default_order,'measured':False,'seconds':time.monotonic()-started}
  # Native content, graph and driver/topology identity prevent stale reuse.
  hardware=_system_info(['nvidia-smi','--query-gpu=uuid,driver_version,name','--format=csv,noheader'])
  topology=_system_info(['nvidia-smi','topo','-m'])
  dependency=_dependency_identity(native,env)
  identity={'schema':11,'configuration_digest':_configuration_digest(env),'baseline_geometry':{'plan':{k:plan.get(k) for k in ('capacity','batch','state_bytes','history_layers')},'rank_capacities':[p['capacity'] for p in baseline.get('rank_plans',[])]},'allow_specialized':allow_specialized,'native_dependencies':dependency,'graph':graph.digest(),'native_sha256':hashlib.sha256(Path(native).read_bytes()).hexdigest(),'devices':selected,'hardware':hardware.stdout,'topology':topology.stdout,'capacity_override':capacity,'environment':{k:env.get(k) for k in ('CUDA_VISIBLE_DEVICES','NCCL_P2P_DISABLE','NCCL_SHM_DISABLE','NCCL_SOCKET_IFNAME','MGBFS_GENERIC_TRANSPORT','MGBFS_GENERIC_SORT','MGBFS_PEER_TRANSPORT')}}
  key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
  root=Path(os.environ.get('MGBFS_PROFILE_CACHE',str(Path.home()/'.cache/multigpubfs/profiles')));cache=root/(key+'.json')
  if dependency is not None and hardware.returncode==0 and topology.returncode==0 and cache.is_file():
   try:saved=json.loads(cache.read_text())
   except (OSError,ValueError):saved={}
   from .specialized import match_lrx
   specialized_eligible=allow_specialized and len(selected) in (1,2,4,8) and not env.get('MGBFS_GENERIC_TRANSPORT') and not env.get('MGBFS_GENERIC_SORT') and match_lrx(graph) is not None
   if saved.get('identity')==identity and saved.get('status')=='MEASURED_EQUAL_PREFIX_GPU_PROFILE' and saved.get('shards') in (1,4,16) and saved.get('batch_fraction') in (1.0,.25) and ((saved.get('backend','generic')=='generic' and (saved.get('shards'),saved.get('batch_fraction'),saved.get('transport'),saved.get('candidate_order')) in variants) or (specialized_eligible and saved.get('backend') in ('shard_ab_hash','shard_ab_sort_merge') and type(saved.get('specialized_capacity')) is int and saved['specialized_capacity']>0)):
    saved=dict(saved);saved['cache_hit']=True;saved['seconds']=time.monotonic()-started;return saved
  admitted=[]
  for shards,fraction,transport,order in variants:
   candidate_env=dict(env,MGBFS_GENERIC_TRANSPORT=transport,MGBFS_GENERIC_SORT=order)
   try:admission=_admit(graph,selected,capacity,shards,native,candidate_env,temporary)
   except RuntimeError as error:
    if any(code in str(error) for code in ('REQUESTED_CAPACITY_EXCEEDS_ADMISSION','GENERIC_DISTRIBUTED_NO_CAPACITY','GENERIC_DISTRIBUTED_HEADROOM')):continue
    raise
   admitted.append((shards,fraction,transport,order,admission['plan']))
  if len(admitted)<2:return {'status':'NO_ADMITTED_ALTERNATIVE_CONSERVATIVE_PROFILE','shards':1,'batch_fraction':1.0,'transport':default_transport,'candidate_order':default_order,'measured':False,'seconds':time.monotonic()-started}
  common_capacity=min(p['capacity'] for _,_,_,_,p in admitted)
  pilots=[];limit=min(3,max(1,max_seconds//20))
  from .launch import run_graph
  for index,(shards,fraction,transport,order,_) in enumerate(admitted):
   candidate_env=dict(env,MGBFS_GENERIC_TRANSPORT=transport,MGBFS_GENERIC_SORT=order)
   admission=_admit(graph,selected,common_capacity,shards,native,candidate_env,temporary)
   batch=max(1,int(admission['plan']['batch']*fraction))
   report=run_graph(graph,Path(temporary)/('pilot-'+str(index)),devices=selected,capacity=common_capacity,max_seconds=limit,executable=native,shards=shards,autotune=False,_batch=batch,_profile_layers=36,_native_env=candidate_env)
   pilots.append({'backend':'generic','candidate_order':order,'transport':transport,'shards':shards,'batch_fraction':fraction,'batch':batch,'state_bytes':report.get('state_bytes',report.get('plan',{}).get('state_bytes')),'layer_sizes':report['layer_sizes'],'layer_seconds':report['layer_seconds'],'status':report['status'],'reason':report['reason']})
  # Specialized candidates preserve only a proved identical LRX action/root.
  # Unsupported definitions/topologies remain on the general exact backend.
  if allow_specialized and len(selected) in (1,2,4,8) and not env.get('MGBFS_GENERIC_TRANSPORT') and not env.get('MGBFS_GENERIC_SORT') and max_seconds-(time.monotonic()-started)>=8:
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
   return {'status':'INSUFFICIENT_PREFIX_CONSERVATIVE_PROFILE','shards':1,'batch_fraction':1.0,'transport':default_transport,'candidate_order':default_order,'measured':False,'pilots':pilots,'common_depth':depth,'seconds':time.monotonic()-started}
  scores=[sum(p['layer_seconds'][1:depth]) for p in pilots]
  winner=min(range(len(scores)),key=lambda i:scores[i])
  # A single bounded sample cannot justify a marginal switch. Keep baseline
  # unless the alternative beats it by at least five percent.
  if scores[winner]>=scores[0]*.95:winner=0
  chosen=pilots[winner]
  result={'backend':chosen['backend'],'status':'MEASURED_EQUAL_PREFIX_GPU_PROFILE','shards':chosen['shards'],'batch_fraction':chosen['batch_fraction'],'transport':chosen['transport'],'candidate_order':chosen['candidate_order'],'measured':True,'minimum_switch_improvement':.05,'common_depth':depth,'common_layer_sizes':sizes,'scores_seconds':scores,'pilots':pilots,'identity':identity,'cache_hit':False,'seconds':time.monotonic()-started,'scope':'bounded prefix among admitted shard/batch and full-child/parent-origin and unsorted/radix-index transport profiles and eligible SHARD_AB hash/sorted-history owners; no claim of global optimum or large-rank acceptance'}
  if chosen['backend']!='generic':result.update(specialized_capacity=chosen['specialized_capacity'],specialized_mode=chosen['specialized_mode'],specialized_transport=chosen['specialized_transport'],specialized_admission=chosen['specialized_admission'],batch=chosen['batch'])
  if dependency is not None and hardware.returncode==0 and topology.returncode==0:
   root.mkdir(parents=True,exist_ok=True)
   fd,path=tempfile.mkstemp(prefix=key+'-',suffix='.tmp',dir=root)
   with os.fdopen(fd,'w') as f:json.dump(result,f)
   os.replace(path,cache)
  return result
