"""External rank launch with network bootstrap and bounded collective tuning."""
import os,json,time,tempfile,subprocess,hashlib,statistics,base64
from pathlib import Path
from .network_control import ControlStore
from .autotune import _dependency_identity

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
  with tempfile.TemporaryDirectory(prefix=f'mgbfs-rank-{rank}-',dir=output.parent) as directory:
   temporary=Path(directory);definition=temporary/'graph.json';definition.write_text(graph.to_json())
   dependency=_dependency_identity(native,native_env)
   identity={'graph':digest,'world':world,'capacity':capacity,'seconds':max_seconds,'shards':shards,'autotune':autotune,'native':hashlib.sha256(Path(native).read_bytes()).hexdigest(),'cuda':dependency['cuda_library_sha256'] if dependency else None}
   store.put(f'identity/{rank}',identity)
   if rank==0:
    peers=[store.get(f'identity/{i}') for i in range(world)]
    if any(v!=identity for v in peers):raise RuntimeError('EXTERNAL_GRAPH_NATIVE_OR_PARAMETERS_MISMATCH')
    if identity['cuda'] is None:raise RuntimeError('EXTERNAL_NATIVE_LIBRARY_IDENTITY_UNAVAILABLE')
    store.put('identity-ready',True)
   store.get('identity-ready')
   def inventories(label):
    probe=subprocess.run([str(native),'graph-info',str(definition),str(device),'1'],env=native_env,capture_output=True,text=True)
    if probe.returncode:raise RuntimeError('EXTERNAL_LOCAL_ADMISSION_FAILED '+probe.stderr[-2000:])
    v=json.loads(probe.stdout);store.put(label+f'/inventory/{rank}',v['inventory'][0])
    if rank==0:return [store.get(label+f'/inventory/{i}') for i in range(world)]
   def admit(label,inventory,count,requested,batch=None):
    path=temporary/(label.replace('/','-')+'-inventory.json');path.write_text(json.dumps(inventory));cmd=[str(native),'graph-plan',str(definition),str(path),str(count),str(requested) if requested is not None else 'auto']
    if batch is not None:cmd.append(str(batch))
    p=subprocess.run(cmd,env=native_env,capture_output=True,text=True)
    if p.returncode:raise RuntimeError('EXTERNAL_GLOBAL_ADMISSION_FAILED '+p.stderr[-2000:])
    return json.loads(p.stdout)
   def phase(label,count,requested,batch_fraction,seconds,depth=None):
    inventory=inventories(label)
    if rank==0:
     config=admit(label,inventory,count,requested)
     if batch_fraction!=1.0:config=admit(label+'-batch',inventory,count,requested,max(1,int(config['plan']['batch']*batch_fraction)))
     config['max_seconds']=seconds
     if depth is not None:config['profile_max_layers']=depth
     store.put(label+'/config',config)
    config=store.get(label+'/config');seconds=config['max_seconds'];folder=temporary/label if depth is not None else output;folder.mkdir();configuration=folder/'launch.json';configuration.write_text(json.dumps(config));bootstrap=folder/'nccl-id'
    if rank!=0:bootstrap.write_bytes(base64.b64decode(store.get(label+'/nccl-id')))
    log=(folder/f'rank-{rank}.log').open('w');job=None
    try:
     job=subprocess.Popen([str(native),'graph-rank',str(definition),str(configuration),str(rank),str(bootstrap),str(folder/f'rank-{rank}'),str(seconds)],env=dict(native_env,RANK=str(rank),WORLD_SIZE=str(world),LOCAL_RANK=str(device)),stdout=log,stderr=subprocess.STDOUT)
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
     states={'schema':2,'state_encoding':'signed_int64_vectors','current':current,'previous_small':previous};raw=json.dumps(states).encode();result=dict(reports[0]);result.pop('rank');result.pop('device');result.update(layer_seconds=[max(v['layer_seconds'][i] for v in reports) for i in range(len(reports[0]['layer_seconds']))],devices=config['devices'],rank_plans=config['rank_plans'],owner_cuts=config['owner_cuts'],bfs_seconds=max(v['bfs_seconds'] for v in reports),setup_seconds=max(v['setup_seconds'] for v in reports),states_sha256=hashlib.sha256(raw).hexdigest(),scope='network external-rank exact path; actual multi-host hardware acceptance separately required')
     store.put(label+'/result',{'report':result,'states':states})
    result=store.get(label+'/result');(folder/'states.json').write_bytes(json.dumps(result['states']).encode());(folder/'report.json').write_text(json.dumps(result['report'],indent=2));return result['report']
   if shards is None and autotune and max_seconds>=10:
    inventory=inventories('profile-admission')
    if rank==0:
     variants=[];plans=[]
     for i,(c,f) in enumerate(((1,1.0),(4,1.0),(16,1.0),(4,.25))):
      try:plan=admit('profile-'+str(i),inventory,c,capacity)['plan']
      except RuntimeError as error:
       if any(code in str(error) for code in ('REQUESTED_CAPACITY_EXCEEDS_ADMISSION','GENERIC_DISTRIBUTED_NO_CAPACITY','GENERIC_DISTRIBUTED_HEADROOM')):continue
       raise
      variants.append((c,f));plans.append(plan)
     common=min(p['capacity'] for p in plans) if plans else 0;store.put('profile-plan',{'capacity':common,'run':common>4096 and len(variants)>1,'variants':variants})
    proposal=store.get('profile-plan')
    if proposal['run']:
     pilots=[]
     for i,(count,fraction) in enumerate(proposal['variants']):
      v=phase('pilot-'+str(i),count,proposal['capacity'],fraction,min(3,max(1,max_seconds//20)),36);pilots.append({'shards':count,'batch_fraction':fraction,'layer_sizes':v['layer_sizes'],'layer_seconds':v['layer_seconds'],'status':v['status']})
     depth=min(len(p['layer_seconds']) for p in pilots);sizes=pilots[0]['layer_sizes'][:depth+1]
     if any(p['layer_sizes'][:depth+1]!=sizes for p in pilots):raise RuntimeError('EXTERNAL_PROFILE_PREFIX_MISMATCH')
     scores=[sum(p['layer_seconds'][1:depth]) for p in pilots];winner=min(range(len(pilots)),key=lambda i:scores[i]) if depth>=4 and sum(sizes[2:])>=32768 else 0
     if scores[winner]>=scores[0]*.95:winner=0
     profile={'status':'MEASURED_EQUAL_PREFIX_EXTERNAL_PROFILE' if depth>=4 and sum(sizes[2:])>=32768 else 'INSUFFICIENT_PREFIX_CONSERVATIVE_PROFILE','shards':pilots[winner]['shards'],'batch_fraction':pilots[winner]['batch_fraction'],'pilots':pilots,'common_depth':depth,'scores_seconds':scores,'scope':'bounded collective prefix, no global optimum claim'}
   shards=shards if shards is not None else (profile['shards'] if profile else 1)
   remaining=max(1,max_seconds-int(time.monotonic()-started+.999));result=phase('production',shards,capacity,profile['batch_fraction'] if profile else 1.0,remaining)
   if profile:result['autotune']=profile;(output/'report.json').write_text(json.dumps(result,indent=2))
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
