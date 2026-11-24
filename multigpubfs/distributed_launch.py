"""External rank launch with network bootstrap and bounded collective tuning."""
import os,json,time,tempfile,subprocess,hashlib,statistics,base64,socket
from pathlib import Path
from .process_control import query as native_query
from .network_control import ControlStore
from .autotune import _dependency_identity,_transport_variants

def _external_variants(graph,env,allow_gemm=False,hardware_shards=None):
 history=env.get('MGBFS_GENERIC_HISTORY')
 base=[v+(history or 'hash',int(env.get('MGBFS_GENERIC_OWNER_LANES','1')) if history=='sorted' else 0) for v in _transport_variants(graph,env,hardware_shards)]
 if history is None and env.get('MGBFS_GENERIC_SORT') in (None,'none'):
  packed=graph.action['kind']=='permutation' and graph.state_elements<=24 and all(0<=v<=255 for v in graph.start)
  transport=env.get('MGBFS_GENERIC_TRANSPORT') or ('full' if packed else 'parent')
  base += [(max(4,lane),1.0,transport,'none','sorted',lane) for lane in (1,2,4,8)]
 from .generation import gemm_supported
 forced=env.get('MGBFS_GENERIC_GENERATOR','cuda')
 if forced not in ('cuda','gemm'):raise ValueError('INVALID_GENERIC_GENERATOR')
 if forced=='gemm' and not gemm_supported(graph):raise ValueError('GEMM_GRAPH_NOT_EXACTLY_SUPPORTED')
 base=[v+(forced,) for v in base]
 if allow_gemm and forced=='cuda' and gemm_supported(graph):base.append(base[0][:-1]+('gemm',))
 return base

def _phase_environment(env,transport=None,order=None,history=None,lanes=None,generator=None):
 result=dict(env)
 for key,value in [('MGBFS_GENERIC_TRANSPORT',transport),('MGBFS_GENERIC_SORT',order),('MGBFS_GENERIC_HISTORY',history),('MGBFS_GENERIC_OWNER_LANES',lanes),('MGBFS_GENERIC_GENERATOR',generator)]:
  if value is not None:result[key]=str(value)
 return result

def _match_resident_reserve(env, inventory):
 values=[item.get('additional_reserve_bytes') for item in inventory]
 if not values or any(type(value) is not int or value<0 or value!=values[0] for value in values):
  raise RuntimeError('EXTERNAL_RESIDENT_RESERVE_CONSENSUS')
 return dict(env,MGBFS_GENERIC_VRAM_RESERVE_BYTES=str(values[0]))

def _merge_local_sorted_plans(parts,digest,requested):
 if not parts or len(parts)>128:raise RuntimeError('EXTERNAL_LOCAL_PLAN_WORLD')
 plans=[v['plan'] for v in parts];first=plans[0]
 fields=('world','shards','elements','state_bytes','history_layers','history_algorithm','owner_lanes','batch','queue_capacity','generator_bytes','parent_transport','sort_candidates','generator_backend')
 for item in parts:
  if item['graph_digest']!=digest or item['plan']['world']!=len(parts) or item['plan']['history_algorithm']!='SORTED_RUNS':raise RuntimeError('EXTERNAL_LOCAL_PLAN_IDENTITY')
  if any(item['plan'].get(k,'cuda' if k=='generator_backend' else None)!=first.get(k,'cuda' if k=='generator_backend' else None) for k in fields):raise RuntimeError('EXTERNAL_LOCAL_PLAN_TRANSPORT_GEOMETRY')
  if requested is not None and item['plan']['capacity']!=requested:raise RuntimeError('REQUESTED_CAPACITY_EXCEEDS_SORTED_ADMISSION')
 cuts=None
 if requested is None:
  total=sum(p['capacity'] for p in plans)
  if total<=0:raise RuntimeError('EXTERNAL_LOCAL_PLAN_CAPACITY')
  cuts=[0];cumulative=0
  for p in plans:
   cumulative+=p['capacity'];cuts.append((cumulative*(1<<32)+total-1)//total)
 return {'size_profile_capability':1 if all(p.get('size_profile_capability')==1 for p in parts) else 0,'graph_digest':digest,'devices':[v['device'] for v in parts],'inventory':[{k:v[k] for k in ('device','free_bytes','total_bytes','sm_count','compute_major','compute_minor','l2_bytes') if k in v} for v in parts],'plan':first,'rank_plans':plans,'owner_cuts':cuts}

from .work_limits import wire, remaining as remaining_work, phase_expired

def _native_artifact_hashes(native,env):
 from .native_distribution import _file_digest
 dependency=_dependency_identity(native,env)
 return {'native':_file_digest(native),'cuda':dependency['cuda_library_sha256'] if dependency else None}

def run_external(graph,output,*,devices,capacity,max_seconds,native,native_env,shards,autotune,batch=None,path_targets=None,path_bidirectional=False,path_window=None,path_max_depth=None):
 from .launch import (_PACKET_ADDR,_PACKET_RING,_PACKET_RX,_PATH_COMPACT,_PATH_PACKET_ENV,
                      _PATH_SEGMENTED_NEAREST,_PATH_SEGMENTED_WINDOW,_compact_path_env,_rank_launch_config,
                      _receipt,_with_packet_job_token)
 from .cluster import current_cluster
 cluster=current_cluster()
 segmented=native_env.get('MGBFS_SEGMENT_EXTERNAL_LINK')=='1'
 resident_admission=cluster is not None and (segmented or path_targets is not None)
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ.get('LOCAL_RANK','0'))
 if not 1<world<=128 or not 0<=rank<world:raise ValueError('EXTERNAL_RANK_RANGE')
 device=local if devices is None else devices[0]
 if devices is not None and len(devices)!=1:raise ValueError('EXTERNAL_RANK_REQUIRES_ONE_LOCAL_DEVICE')
 output=Path(output).absolute()
 if output.exists():raise FileExistsError(output)
 output.parent.mkdir(parents=True,exist_ok=True)
 digest=graph.digest();host=os.environ['MASTER_ADDR'];port=int(os.environ.get('MGBFS_CONTROL_PORT',str(int(os.environ['MASTER_PORT'])+1)))
 session=os.environ.get('MGBFS_RUN_ID','mgbfs-'+os.environ['MASTER_PORT']+'-'+str(world));store=ControlStore(host,port,rank,session,600 if max_seconds is None else max_seconds+180,os.environ.get('MGBFS_CONTROL_TOKEN',''))
 started=time.monotonic();profile=None
 try:
  segmented_nearest=path_targets is not None and native_env.get(_PATH_SEGMENTED_NEAREST)=='1'
  segmented_window=path_window is not None and native_env.get(_PATH_SEGMENTED_WINDOW)=='1'
  path_compact=path_targets is not None and (segmented_nearest or segmented_window or native_env.get(_PATH_COMPACT)=='1')
  if path_compact:
   native_env=_compact_path_env(native,native_env,world,path_bidirectional,external=True)
  from .host_memory import admit as admit_host
  host_memory=admit_host(graph,world,True,output)
  store.put(f'host-memory/{rank}',host_memory)
  with tempfile.TemporaryDirectory(prefix=f'mgbfs-rank-{rank}-',dir=output.parent) as directory:
   temporary=Path(directory);definition=temporary/'graph.json';definition.write_text(graph.to_json())
   identity={'graph':digest,'world':world,'capacity':capacity,'seconds':max_seconds,'shards':shards,'autotune':autotune,'transport':native_env.get('MGBFS_GENERIC_TRANSPORT'),'candidate_order':native_env.get('MGBFS_GENERIC_SORT'),'history_algorithm':native_env.get('MGBFS_GENERIC_HISTORY','hash'),'owner_lanes':native_env.get('MGBFS_GENERIC_OWNER_LANES','1'),'generator_backend':native_env.get('MGBFS_GENERIC_GENERATOR','cuda'),'gemm_variant':native_env.get('MGBFS_GEMM_VARIANT','auto'),**_native_artifact_hashes(native,native_env)}
   identity['batch']=batch
   if path_targets is not None:
    identity['path_bidirectional']=path_bidirectional
    identity['path_window']=path_window
    identity['path_maxmemory']=native_env.get('MGBFS_GENERIC_PATH_MAXMEMORY','0')
    identity['path_fused_route']=native_env.get('MGBFS_GENERIC_PATH_FUSED_ROUTE','1')!='0'
    identity['path_exact_exchange']=native_env.get('MGBFS_GENERIC_PATH_EXACT_EXCHANGE')=='1'
    identity['path_targets_sha256']=hashlib.sha256(json.dumps(path_targets,separators=(',',':')).encode()).hexdigest()
    if path_compact:
     # Listen address can differ on rank 0 behind port forwarding.
     identity['path_compact']={key:native_env[key] for key in (_PACKET_ADDR,_PACKET_RING,_PACKET_RX)}
    if segmented_nearest:
     identity['path_segmented_nearest']={'max_depth':path_max_depth}
    if segmented_window:
     identity['path_segmented_window']=True
   if segmented:
    identity['segmented']={k:v for k,v in native_env.items() if (k.startswith(('MGBFS_SEGMENT_', 'MGBFS_BITMAP_')) or k in ('MGBFS_MEMORY_PRESET','MGBFS_EXPERIMENTAL_BITMAP')) and k!='MGBFS_SEGMENT_PACKET_BIND_ADDR'}
   store.put(f'identity/{rank}',identity)
   if rank==0:
    peers=[store.get(f'identity/{i}') for i in range(world)]
    if any(v!=identity for v in peers):raise RuntimeError('EXTERNAL_GRAPH_NATIVE_OR_PARAMETERS_MISMATCH')
    if identity['cuda'] is None:raise RuntimeError('EXTERNAL_NATIVE_LIBRARY_IDENTITY_UNAVAILABLE')
    store.put('identity-ready',True)
   store.get('identity-ready')
   def inventories(label):
    if resident_admission:
     inventory_env=native_env
     if segmented_nearest or segmented_window:
      inventory_env=dict(inventory_env)
      inventory_env.pop(_PATH_SEGMENTED_NEAREST,None)
      inventory_env.pop(_PATH_SEGMENTED_WINDOW,None)
     inventory=cluster.inventory(native,inventory_env,device)
     store.put(label+f'/inventory/{rank}',inventory)
     if rank==0:return [store.get(label+f'/inventory/{i}') for i in range(world)]
     return None
    inventory_env=dict(native_env,MGBFS_GENERIC_HISTORY='hash')
    if path_compact:
     # This legacy one-GPU probe contributes hardware only. Rank 0 admits
     # the actual compact plan from the inventory of the entire world.
     for key in (_PATH_COMPACT,_PATH_SEGMENTED_NEAREST,_PATH_SEGMENTED_WINDOW,*_PATH_PACKET_ENV):inventory_env.pop(key,None)
    probe=native_query([str(native),'graph-info',str(definition),str(device),'1'],env=inventory_env,capture_output=True,text=True)
    if probe.returncode:raise RuntimeError('EXTERNAL_LOCAL_ADMISSION_FAILED '+probe.stderr[-2000:])
    v=json.loads(probe.stdout);store.put(label+f'/inventory/{rank}',v['inventory'][0])
    from .autotune import _gemm_hardware_available
    store.put(label+f'/gemm/{rank}',_gemm_hardware_available() and 'generator_backend' in v['plan'])
    store.put(label+f'/size-profile/{rank}',v.get('size_profile_capability')==1)
    if rank==0:return [store.get(label+f'/inventory/{i}') for i in range(world)]
   def admit(label,inventory,count,requested,batch=None,transport=None,order=None,history=None,lanes=None,generator=None):
    admission_env=_phase_environment(native_env,transport,order,history,lanes,generator)
    if admission_env.get('MGBFS_GENERIC_HISTORY')=='sorted':
     target=batch or max(1,(1<<20)//max(1,len(graph.action['generators'])))
     def local_probe(stage,target):
      cmd=[str(native),'graph-local-plan',str(definition),str(device),str(world),str(count),str(requested) if requested is not None else 'auto',str(target)]
      p=native_query(cmd,env=admission_env,capture_output=True,text=True)
      item={'value':json.loads(p.stdout)} if p.returncode==0 else {'error':'EXTERNAL_LOCAL_ADMISSION_FAILED '+p.stderr[-2000:]}
      store.put(label+f'/{stage}/{rank}',item)
      if rank==0:
       parts=[store.get(label+f'/{stage}/{i}') for i in range(world)];errors=[v['error'] for v in parts if 'error' in v]
       store.put(label+'/'+stage+'-result',{'error':errors[0]} if errors else {'parts':[v['value'] for v in parts]})
      result=store.get(label+'/'+stage+'-result')
      if 'error' in result:raise RuntimeError(result['error'])
      return result['parts']
     initial=local_probe('initial',target);common=min(v['plan']['batch'] for v in initial)
     if batch is not None and common!=batch:raise RuntimeError('REQUESTED_BATCH_EXCEEDS_SORTED_ADMISSION')
     parts=local_probe('matched',common);return _merge_local_sorted_plans(parts,digest,requested)
    if rank==0:
     if resident_admission:admission_env=_match_resident_reserve(admission_env,inventory)
     path=temporary/(label.replace('/','-')+'-inventory.json');path.write_text(json.dumps(inventory));cmd=[str(native),'graph-plan',str(definition),str(path),str(count),str(requested) if requested is not None else 'auto']
     if batch is not None:cmd.append(str(batch))
     p=native_query(cmd,env=admission_env,capture_output=True,text=True)
     store.put(label+'/admitted',{'value':json.loads(p.stdout)} if p.returncode==0 else {'error':'EXTERNAL_GLOBAL_ADMISSION_FAILED '+p.stderr[-2000:]})
    result=store.get(label+'/admitted')
    if 'error' in result:raise RuntimeError(result['error'])
    return result['value']
   def phase(label,count,requested,batch_fraction,seconds,depth=None,transport=None,order=None,history=None,lanes=None,generator=None):
    phase_env=_phase_environment(native_env,transport,order,history,lanes,generator)
    inventory=inventories(label)
    config=admit(label,inventory,count,requested,batch if label=='production' else None,transport=transport,order=order,history=history,lanes=lanes,generator=generator)
    if batch_fraction!=1.0:config=admit(label+'-batch',inventory,count,requested,max(1,int(config['plan']['batch']*batch_fraction)),transport=transport,order=order,history=history,lanes=lanes,generator=generator)
    if rank==0:
     config=_rank_launch_config(
      config,digest,bitmap=segmented and native_env.get('MGBFS_EXPERIMENTAL_BITMAP')=='1',depth=depth,
      path_targets=path_targets,path_bidirectional=path_bidirectional,path_window=path_window,
      path_max_depth=path_max_depth)
     config['max_seconds']=seconds
     if depth is None and profile and config.get('size_profile_capability')==1:
      config['online_size_tuning']=True
      config['size_profiles']=[{'minimum_frontier':0,'batch':config['plan']['batch'],'generator_backend':profile.get('generator_backend','cuda')}]
     store.put(label+'/config',_with_packet_job_token(config,phase_env))
    config=store.get(label+'/config');seconds=config['max_seconds'];folder=temporary/label if depth is not None else output;folder.mkdir();configuration=folder/'launch.json';configuration.write_text(json.dumps(config));bootstrap=folder/'nccl-id'
    if rank!=0 and not segmented:bootstrap.write_bytes(base64.b64decode(store.get(label+'/nccl-id')))
    if segmented:
     from .segmented_network import AgreementRelay
     relay=AgreementRelay(store,label,bootstrap,rank,world)
    log=(folder/f'rank-{rank}.log').open('w');job=None
    try:
     command=[str(native),'graph-rank',str(definition),str(configuration),str(rank),str(bootstrap),str(folder/f'rank-{rank}'),str(wire(seconds))]
     job_env=dict(phase_env,RANK=str(rank),WORLD_SIZE=str(world),LOCAL_RANK=str(device),MGBFS_JOB_PHASE_PATH=str(folder/f'rank-{rank}.phase.json'))
     if resident_admission:job_env['MGBFS_GENERIC_VRAM_RESERVE_BYTES']=str(config['additional_reserve_bytes'])
     job=cluster.launch_rank(command,job_env,log) if cluster is not None else subprocess.Popen(command,env=job_env,stdout=log,stderr=subprocess.STDOUT)
     if rank==0 and not segmented:
      bootstrap_deadline=time.monotonic()+600
      while not bootstrap.exists():
       if job.poll() is not None:raise RuntimeError('EXTERNAL_BOOTSTRAP_WORKER_EXIT')
       if time.monotonic()>=bootstrap_deadline:raise RuntimeError('EXTERNAL_BOOTSTRAP_TIMEOUT_NO_RESTART')
       store.check_error();time.sleep(.02)
      store.put(label+'/nccl-id',base64.b64encode(bootstrap.read_bytes()).decode())
     deadline=None if seconds is None else time.monotonic()+seconds+180
     observed_start=time.monotonic()
     bootstrap_done=not segmented
     next_error_poll=0.
     while job.poll() is None:
      from .launch import _job_phase
      observed=_job_phase(job,folder,rank)
      now=time.monotonic()
      if now>=next_error_poll:
       store.check_error();next_error_poll=now+.2
      host_phase=(observed or {}).get('phase','')
      if segmented and host_phase.startswith('segmented-host-'):
       relay.step(host_phase.removeprefix('segmented-host-'))
      if not bootstrap_done:
       if rank==0 and bootstrap.is_file():
        store.put(label+'/nccl-id',base64.b64encode(bootstrap.read_bytes()).decode());bootstrap_done=True
       if rank!=0:
        shared_id=store.try_get(label+'/nccl-id')
        if shared_id is not None:
         temporary_id=bootstrap.with_suffix('.relay-tmp');temporary_id.write_bytes(base64.b64decode(shared_id,validate=True));temporary_id.replace(bootstrap);bootstrap_done=True
       if time.monotonic()-observed_start>600:raise RuntimeError('EXTERNAL_BOOTSTRAP_TIMEOUT_NO_RESTART')
      if (deadline is not None and time.monotonic()>deadline) or (observed is not None and phase_expired(observed,time.time(),seconds)) or (observed is None and time.monotonic()-observed_start>600):raise RuntimeError('EXTERNAL_WORKER_OBSERVATION_TIMEOUT_NO_RESTART')
      time.sleep(.02 if segmented and (not bootstrap_done or host_phase.startswith('segmented-host-')) else .05)
     if job.returncode:raise RuntimeError('EXTERNAL_NATIVE_RANK_FAILED '+str(rank))
    finally:
     if job is not None and job.poll() is None:
      job.terminate()
      try:job.wait(timeout=10)
      except subprocess.TimeoutExpired:job.kill();job.wait()
     log.close()
    if path_targets is not None:
     local=json.loads((folder/f'rank-{rank}'/'path.json').read_text(encoding='utf-8'))
     from .generic_session import path_workspace
     workspace=path_workspace(job)
     if workspace is not None:workspace['hostname']=socket.gethostname()
     store.put(label+f'/receipt/{rank}',{'path':local,'workspace':workspace})
     if rank==0:
      replies=[store.get(label+f'/receipt/{i}') for i in range(world)]
      workspaces={'ranks':[reply['workspace'] for reply in replies]}
      if path_window is not None:
       from .path_search import _agree_window_receipts
       result=_agree_window_receipts([reply['path'] for reply in replies])
      else:
       if any(reply['path']!=local for reply in replies):raise RuntimeError('NATIVE_DISTRIBUTED_PATH_CONSENSUS_MISMATCH')
       result=local
      store.put(label+'/result',dict(result,native_workspace=workspaces))
     result=store.get(label+'/result')
     store.put(label+f'/consumed/{rank}',True)
     if rank==0:
      for i in range(world):store.get(label+f'/consumed/{i}')
      store.release_prefix(label+'/')
     return result
    report=_receipt(folder/f'rank-{rank}',digest);states=json.loads((folder/f'rank-{rank}/states.json').read_text());store.put(label+f'/receipt/{rank}',{'report':report,'states':states,'pid':job.pid,'hostname':socket.gethostname()})
    if rank==0:
     parts=[store.get(label+f'/receipt/{i}') for i in range(world)];reports=[v['report'] for v in parts]
     from .traversal_receipt import merge_rank_traversal
     result,states=merge_rank_traversal(reports,[part['states'] for part in parts],config,error_prefix='EXTERNAL',
         segmented=segmented,bitmap=native_env.get('MGBFS_EXPERIMENTAL_BITMAP')=='1',require_admitted_cuts=True)
     result.update(host_memory=[store.get(f'host-memory/{i}') for i in range(world)],
                   scope='network external-rank exact path; actual multi-host hardware acceptance separately required')
     result['rank_worker_pids']=[part['pid'] for part in parts]
     result['rank_hostnames']=[part['hostname'] for part in parts]
     store.put(label+'/result',{'report':result,'states':states})
    result=store.get(label+'/result');(folder/'states.json').write_bytes(json.dumps(result['states']).encode());(folder/'report.json').write_text(json.dumps(result['report'],indent=2))
    store.put(label+f'/consumed/{rank}',True)
    if rank==0:
     for i in range(world):store.get(label+f'/consumed/{i}')
     store.release_prefix(label+'/')
    return result['report']
   if segmented and shards is None:
    inventory=inventories('segmented-geometry')
    baseline=admit('segmented-baseline',inventory,1,capacity)
    if rank==0:
     from .shard_policy import choose_segmented_shards
     bitmap=native_env.get('MGBFS_EXPERIMENTAL_BITMAP')=='1'
     geometry=choose_segmented_shards(graph,baseline,bitmap=bitmap,target_rows=int(native_env['MGBFS_SEGMENT_MAX_ROWS']),env=native_env)
     store.put('segmented-shards',geometry['target_shards'])
    shards=store.get('segmented-shards')
    while shards>1:
     try:admit(f'segmented-admit-{shards}',inventory,shards,capacity)
     except RuntimeError as error:
      if not any(code in str(error) for code in ('GENERIC_DISTRIBUTED_HEADROOM','GENERIC_DISTRIBUTED_NO_CAPACITY','GENERIC_VRAM_CAPACITY')):raise
      shards=max(1,shards//2)
     else:break
   if not segmented and shards is None and autotune and (max_seconds is None or max_seconds>=10):
    inventory=inventories('profile-admission')
    from .hardware_profiles import startup_policy
    baseline=admit('hardware-baseline',inventory,1,capacity)
    hardware_policy=startup_policy(baseline['inventory'],baseline['plan'])
    variants=[];plans=[]
    for i,(c,f,transport,order,history,lanes,generator) in enumerate(_external_variants(graph,native_env,all(store.get('profile-admission'+f'/gemm/{i}') for i in range(world)),hardware_policy['shards'])):
     try:plan=admit('profile-'+str(i),inventory,c,capacity,transport=transport,order=order,history=history,lanes=lanes,generator=generator)['plan']
     except RuntimeError as error:
      if any(code in str(error) for code in ('REQUESTED_CAPACITY_EXCEEDS_ADMISSION','REQUESTED_CAPACITY_EXCEEDS_SORTED_ADMISSION','GENERIC_DISTRIBUTED_NO_CAPACITY','GENERIC_DISTRIBUTED_HEADROOM')):continue
      raise
     variants.append((c,f,transport,order,history,lanes,generator));plans.append(plan)
    common=min([p['capacity'] for p in plans]+[hardware_policy['probe_capacity_per_rank']]) if plans else 0
    if rank==0:store.put('profile-plan',{'capacity':common,'run':common>4096 and len(variants)>1,'variants':variants})
    proposal=store.get('profile-plan')
    if proposal['run']:
     pilots=[]
     for i,(count,fraction,transport,order,history,lanes,generator) in enumerate(proposal['variants']):
      if rank==0:store.put('profile-go-'+str(i),i==0 or (60 if max_seconds is None else min(60,max_seconds))-(time.monotonic()-started)>=3)
      if not store.get('profile-go-'+str(i)):break
      v=phase('pilot-'+str(i),count,proposal['capacity'],fraction,1,128,transport=transport,order=order,history=history,lanes=lanes,generator=generator);pilots.append({'generator_backend':v['plan'].get('generator_backend','cuda'),'history_algorithm':v['plan']['history_algorithm'],'owner_lanes':v['plan'].get('owner_lanes',0),'candidate_order':order,'transport':transport,'shards':count,'batch_fraction':fraction,'layer_sizes':v['layer_sizes'],'layer_seconds':v['layer_seconds'],'status':v['status']})
     depth=min(len(p['layer_seconds']) for p in pilots);sizes=pilots[0]['layer_sizes'][:depth+1]
     if any(p['layer_sizes'][:depth+1]!=sizes for p in pilots):raise RuntimeError('EXTERNAL_PROFILE_PREFIX_MISMATCH')
     scores=[sum(p['layer_seconds'][1:depth]) for p in pilots];winner=min(range(len(pilots)),key=lambda i:scores[i]) if depth>=4 and sum(sizes[2:])>=32768 else 0
     if scores[winner]>=scores[0]*.95:winner=0
     profile={'status':'MEASURED_EQUAL_PREFIX_EXTERNAL_PROFILE' if depth>=4 and sum(sizes[2:])>=32768 else 'INSUFFICIENT_PREFIX_CONSERVATIVE_PROFILE','generator_backend':pilots[winner].get('generator_backend','cuda'),'shards':pilots[winner]['shards'],'batch_fraction':pilots[winner]['batch_fraction'],'transport':pilots[winner]['transport'],'candidate_order':pilots[winner]['candidate_order'],'history_algorithm':pilots[winner]['history_algorithm'],'owner_lanes':pilots[winner]['owner_lanes'],'pilots':pilots,'common_depth':depth,'scores_seconds':scores,'hardware_policy':hardware_policy,'pilot_capacity_per_rank':common,'scope':'bounded hardware-derived collective prefix, no global optimum claim'}
   if profile:
    from .generation import gemm_supported
    profile['reserve_generator_backend']='gemm' if gemm_supported(graph) and all(store.get('profile-admission'+f'/gemm/{i}') and store.get('profile-admission'+f'/size-profile/{i}') for i in range(world)) else profile.get('generator_backend','cuda')
    profile['online_size_tuning']=True
   shards=shards if shards is not None else (profile['shards'] if profile else 1)
   remaining=remaining_work(max_seconds,time.monotonic()-started);result=phase('production',shards,capacity,profile['batch_fraction'] if profile else 1.0,remaining,transport=profile['transport'] if profile else native_env.get('MGBFS_GENERIC_TRANSPORT'),order=profile['candidate_order'] if profile else native_env.get('MGBFS_GENERIC_SORT'),history=('sorted' if profile['history_algorithm']=='SORTED_RUNS' else 'hash') if profile else native_env.get('MGBFS_GENERIC_HISTORY'),lanes=profile['owner_lanes'] if profile and profile['history_algorithm']=='SORTED_RUNS' else None,generator=profile.get('reserve_generator_backend',profile.get('generator_backend','cuda')) if profile else native_env.get('MGBFS_GENERIC_GENERATOR'))
   if profile:profile['online_size_tuning']=result.get('size_profile_count',0)>0;result['autotune']=profile;(output/'report.json').write_text(json.dumps(result,indent=2))
   store.put(f'finished/{rank}',True)
   if rank==0:
    for i in range(world):store.get(f'finished/{i}')
   return result if path_targets is not None else _receipt(output,digest)
 except BaseException as error:
  try:
   store.report_error(error,world)
  except Exception:pass
  raise
 finally:store.close()
