"""External rank launch with network bootstrap and bounded collective tuning."""
import os,json,time,tempfile,subprocess,hashlib,statistics,base64
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

def run_external(graph,output,*,devices,capacity,max_seconds,native,native_env,shards,autotune):
 from .launch import _receipt
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ.get('LOCAL_RANK','0'))
 if not 1<world<=128 or not 0<=rank<world:raise ValueError('EXTERNAL_RANK_RANGE')
 device=local if devices is None else devices[0]
 if devices is not None and len(devices)!=1:raise ValueError('EXTERNAL_RANK_REQUIRES_ONE_LOCAL_DEVICE')
 output=Path(output).absolute()
 if output.exists():raise FileExistsError(output)
 output.parent.mkdir(parents=True,exist_ok=True)
 digest=graph.digest();host=os.environ['MASTER_ADDR'];port=int(os.environ.get('MGBFS_CONTROL_PORT',str(int(os.environ['MASTER_PORT'])+1)))
 session=os.environ.get('MGBFS_RUN_ID','mgbfs-'+os.environ['MASTER_PORT']+'-'+str(world));store=ControlStore(host,port,rank,session,max_seconds+180,os.environ.get('MGBFS_CONTROL_TOKEN',''))
 started=time.monotonic();profile=None
 try:
  from .host_memory import admit as admit_host
  host_memory=admit_host(graph,world,True,output)
  store.put(f'host-memory/{rank}',host_memory)
  with tempfile.TemporaryDirectory(prefix=f'mgbfs-rank-{rank}-',dir=output.parent) as directory:
   temporary=Path(directory);definition=temporary/'graph.json';definition.write_text(graph.to_json())
   dependency=_dependency_identity(native,native_env)
   identity={'graph':digest,'world':world,'capacity':capacity,'seconds':max_seconds,'shards':shards,'autotune':autotune,'transport':native_env.get('MGBFS_GENERIC_TRANSPORT'),'candidate_order':native_env.get('MGBFS_GENERIC_SORT'),'history_algorithm':native_env.get('MGBFS_GENERIC_HISTORY','hash'),'owner_lanes':native_env.get('MGBFS_GENERIC_OWNER_LANES','1'),'generator_backend':native_env.get('MGBFS_GENERIC_GENERATOR','cuda'),'gemm_variant':native_env.get('MGBFS_GEMM_VARIANT','auto'),'native':hashlib.sha256(Path(native).read_bytes()).hexdigest(),'cuda':dependency['cuda_library_sha256'] if dependency else None}
   store.put(f'identity/{rank}',identity)
   if rank==0:
    peers=[store.get(f'identity/{i}') for i in range(world)]
    if any(v!=identity for v in peers):raise RuntimeError('EXTERNAL_GRAPH_NATIVE_OR_PARAMETERS_MISMATCH')
    if identity['cuda'] is None:raise RuntimeError('EXTERNAL_NATIVE_LIBRARY_IDENTITY_UNAVAILABLE')
    store.put('identity-ready',True)
   store.get('identity-ready')
   def inventories(label):
    probe=native_query([str(native),'graph-info',str(definition),str(device),'1'],env=dict(native_env,MGBFS_GENERIC_HISTORY='hash'),capture_output=True,text=True)
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
    config=admit(label,inventory,count,requested,transport=transport,order=order,history=history,lanes=lanes,generator=generator)
    if batch_fraction!=1.0:config=admit(label+'-batch',inventory,count,requested,max(1,int(config['plan']['batch']*batch_fraction)),transport=transport,order=order,history=history,lanes=lanes,generator=generator)
    if rank==0:
     config['max_seconds']=seconds
     if depth is None and profile and config.get('size_profile_capability')==1:
      config['online_size_tuning']=True
      config['size_profiles']=[{'minimum_frontier':0,'batch':config['plan']['batch'],'generator_backend':profile.get('generator_backend','cuda')}]
     if depth is not None:config['profile_max_layers']=depth
     store.put(label+'/config',config)
    config=store.get(label+'/config');seconds=config['max_seconds'];folder=temporary/label if depth is not None else output;folder.mkdir();configuration=folder/'launch.json';configuration.write_text(json.dumps(config));bootstrap=folder/'nccl-id'
    if rank!=0:bootstrap.write_bytes(base64.b64decode(store.get(label+'/nccl-id')))
    log=(folder/f'rank-{rank}.log').open('w');job=None
    try:
     job=subprocess.Popen([str(native),'graph-rank',str(definition),str(configuration),str(rank),str(bootstrap),str(folder/f'rank-{rank}'),str(seconds)],env=dict(phase_env,RANK=str(rank),WORLD_SIZE=str(world),LOCAL_RANK=str(device)),stdout=log,stderr=subprocess.STDOUT)
     if rank==0:
      while not bootstrap.exists():
       if job.poll() is not None:raise RuntimeError('EXTERNAL_BOOTSTRAP_WORKER_EXIT')
       store.check_error();time.sleep(.02)
      store.put(label+'/nccl-id',base64.b64encode(bootstrap.read_bytes()).decode())
     deadline=time.monotonic()+seconds+180
     while job.poll() is None:
      store.check_error()
      if time.monotonic()>deadline:raise RuntimeError('EXTERNAL_WORKER_OBSERVATION_TIMEOUT_NO_RESTART')
      time.sleep(.05)
     if job.returncode:raise RuntimeError('EXTERNAL_NATIVE_RANK_FAILED '+str(rank))
    finally:
     if job is not None and job.poll() is None:
      job.terminate()
      try:job.wait(timeout=10)
      except subprocess.TimeoutExpired:job.kill();job.wait()
     log.close()
    report=_receipt(folder/f'rank-{rank}',digest);states=json.loads((folder/f'rank-{rank}/states.json').read_text());store.put(label+f'/receipt/{rank}',{'report':report,'states':states})
    if rank==0:
     parts=[store.get(label+f'/receipt/{i}') for i in range(world)];reports=[v['report'] for v in parts]
     for i,v in enumerate(reports):
      if v['rank']!=i or v['world']!=world or v['device']!=config['devices'][i] or v['plan']!=config['rank_plans'][i]:raise RuntimeError('EXTERNAL_RANK_GEOMETRY_MISMATCH')
      if any(v[k]!=reports[0][k] for k in ('status','reason','layer_sizes')):raise RuntimeError('EXTERNAL_LAYER_CONSENSUS_MISMATCH')
     current=[row for v in parts for row in v['states']['current']];previous=[row for v in parts for row in v['states']['previous_small']]
     if len(current)>1000 or len(previous)>=1000:raise RuntimeError('EXTERNAL_RETENTION_BOUND')
     states={'schema':2,'state_encoding':'signed_int64_vectors','current':current,'previous_small':previous};raw=json.dumps(states).encode();result=dict(reports[0]);result.pop('rank');result.pop('device');result.update(layer_seconds=[max(v['layer_seconds'][i] for v in reports) for i in range(len(reports[0]['layer_seconds']))],devices=config['devices'],rank_plans=config['rank_plans'],owner_cuts=config['owner_cuts'],bfs_seconds=max(v['bfs_seconds'] for v in reports),setup_seconds=max(v['setup_seconds'] for v in reports),states_sha256=hashlib.sha256(raw).hexdigest(),host_memory=[store.get(f'host-memory/{i}') for i in range(world)],scope='network external-rank exact path; actual multi-host hardware acceptance separately required')
     store.put(label+'/result',{'report':result,'states':states})
    result=store.get(label+'/result');(folder/'states.json').write_bytes(json.dumps(result['states']).encode());(folder/'report.json').write_text(json.dumps(result['report'],indent=2))
    store.put(label+f'/consumed/{rank}',True)
    if rank==0:
     for i in range(world):store.get(label+f'/consumed/{i}')
     store.release_prefix(label+'/')
    return result['report']
   if shards is None and autotune and max_seconds>=10:
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
      if rank==0:store.put('profile-go-'+str(i),max_seconds-(time.monotonic()-started)>=3)
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
   remaining=max(1,max_seconds-int(time.monotonic()-started+.999));result=phase('production',shards,capacity,profile['batch_fraction'] if profile else 1.0,remaining,transport=profile['transport'] if profile else native_env.get('MGBFS_GENERIC_TRANSPORT'),order=profile['candidate_order'] if profile else native_env.get('MGBFS_GENERIC_SORT'),history=('sorted' if profile['history_algorithm']=='SORTED_RUNS' else 'hash') if profile else native_env.get('MGBFS_GENERIC_HISTORY'),lanes=profile['owner_lanes'] if profile and profile['history_algorithm']=='SORTED_RUNS' else None,generator=profile.get('reserve_generator_backend',profile.get('generator_backend','cuda')) if profile else native_env.get('MGBFS_GENERIC_GENERATOR'))
   if profile:profile['online_size_tuning']=result.get('size_profile_count',0)>0;result['autotune']=profile;(output/'report.json').write_text(json.dumps(result,indent=2))
   store.put(f'finished/{rank}',True)
   if rank==0:
    for i in range(world):store.get(f'finished/{i}')
   return _receipt(output,digest)
 except BaseException as error:
  try:
   store.put('error',str(error));store.put(f'error-ack/{rank}',True)
   if rank==0:
    for i in range(world):store.get(f'error-ack/{i}',timeout=5)
  except Exception:pass
  raise
 finally:store.close()
