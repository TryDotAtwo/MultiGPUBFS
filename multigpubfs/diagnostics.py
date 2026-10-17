"""Isolated, bounded single-host native diagnostics. No BFS restart or GPU reset."""
import argparse,ctypes as C,datetime,hashlib,importlib.metadata,json,math,os,platform,shutil,signal,subprocess,sys,tempfile,threading,time,traceback,zipfile
from pathlib import Path

def _write(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w',encoding='utf-8') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def _phase(value):print('PHASE '+value,flush=True)

def _tool(name):
 found=shutil.which(name)
 if found:return found
 for prefix in ('/usr/bin','/bin','/usr/sbin','/sbin','/usr/local/nvidia/bin'):
  path=Path(prefix)/name
  if path.is_file() and os.access(path,os.X_OK):return str(path)
 return name

def _paths():
 from .native_distribution import native_runtime
 native,env=native_runtime()
 try:query=subprocess.run([_tool('ldd'),native],env=env,capture_output=True,text=True,timeout=10)
 except OSError as error:raise RuntimeError('DEPENDENCY_INSPECTION_UNAVAILABLE: cannot execute ldd: '+str(error)) from error
 except subprocess.TimeoutExpired as error:raise RuntimeError('DEPENDENCY_INSPECTION_TIMEOUT: ldd did not finish; this is not a GPU calculation timeout') from error
 if query.returncode:raise RuntimeError('DEPENDENCY_INSPECTION_FAILED: '+(query.stdout+query.stderr)[-6000:])
 libs={}
 for line in query.stdout.splitlines():
  if '=>' not in line:continue
  name,location=line.split('=>',1);location=location.strip().rsplit(' (',1)[0]
  if Path(location).is_file():libs[name.strip()]=str(Path(location).resolve())
 for name in ('libmgbfs_cuda.so','libcudart.so.12','libnccl.so.2'):
  if name not in libs:raise RuntimeError('DEPENDENCY_RESOLUTION_FAILED: '+name+'; loader output: '+(query.stdout+query.stderr)[-6000:])
 return native,env,libs


def _cuda(device=None,engine=False):
 _phase('resolve and load CUDA runtime')
 native,env,paths=_paths();cuda=C.CDLL(paths['libcudart.so.12'],mode=C.RTLD_GLOBAL)
 cuda.cudaGetErrorString.argtypes=[C.c_int];cuda.cudaGetErrorString.restype=C.c_char_p
 cuda.cudaSetDevice.argtypes=[C.c_int];cuda.cudaFree.argtypes=[C.c_void_p]
 cuda.cudaMalloc.argtypes=[C.POINTER(C.c_void_p),C.c_size_t]
 cuda.cudaMemcpy.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int]
 def ck(code):
  if code:raise RuntimeError('CUDA status '+str(code)+': '+cuda.cudaGetErrorString(code).decode(errors='replace'))
 if device is not None:
  _phase('create CUDA context on device '+str(device));ck(cuda.cudaSetDevice(device));ck(cuda.cudaFree(None))
 lib=None
 if engine:
  C.CDLL(paths['libnccl.so.2'],mode=C.RTLD_GLOBAL);lib=C.CDLL(paths['libmgbfs_cuda.so'],mode=C.RTLD_GLOBAL)
 return cuda,lib,ck

def _kernel(device):
 cuda,lib,ck=_cuda(device,True);alloc=[]
 def upload(values,ty):
  host=(ty*len(values))(*values);ptr=C.c_void_p();ck(cuda.cudaMalloc(C.byref(ptr),C.sizeof(host)));alloc.append(ptr);ck(cuda.cudaMemcpy(ptr,host,C.sizeof(host),1));return ptr
 try:
  _phase('allocate kernel input/output');parents=upload([0,1],C.c_int64);perms=upload([1,0],C.c_uint32);out=upload([0,0],C.c_int64);error=upload([0],C.c_uint32)
  fn=lib.mgbfs_generic_generate_i64;fn.argtypes=[C.c_uint32]*5+[C.c_void_p,C.c_uint32,C.c_uint32]+[C.c_void_p]*4+[C.c_uint32,C.c_void_p,C.c_uint32,C.c_void_p,C.c_void_p];fn.restype=C.c_int
  _phase('launch real native successor CUDA kernel');ck(fn(0,2,2,1,1,parents,1,1,perms,None,None,None,1,out,1,error,None));ck(cuda.cudaDeviceSynchronize())
  host=(C.c_int64*2)();err=C.c_uint32();ck(cuda.cudaMemcpy(host,out,16,2));ck(cuda.cudaMemcpy(C.byref(err),error,4,2))
  if list(host)!=[1,0] or err.value:raise RuntimeError('Native CUDA successor mismatch')
  return {'device':device,'kernel':'mgbfs_generic_generate_i64','exact_output_verified':True}
 finally:
  for ptr in alloc:cuda.cudaFree(ptr)

def _save_pool(pool,root):
 for f in pool.root.rglob('*'):
  if f.is_file() and (f.name=='worker.log' or f.name.startswith(('phase-','response-'))):
   dest=root/'persistent-logs'/f.relative_to(pool.root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)

def _rank(stage,root,devices,rank):
 cuda,lib,ck=_cuda(devices[rank],True);comm=C.c_void_p();error=C.create_string_buffer(4096);uid=C.create_string_buffer((root/'nccl-id.bin').read_bytes(),128)
 lib.mgbfs_nccl_create.argtypes=[C.c_uint32,C.c_uint32,C.c_uint32,C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p,C.c_size_t];lib.mgbfs_nccl_create.restype=C.c_int
 lib.mgbfs_nccl_destroy.argtypes=[C.c_void_p];lib.mgbfs_nccl_destroy.restype=None
 _phase('rank '+str(rank)+' NCCL communicator initialization')
 code=lib.mgbfs_nccl_create(rank,len(devices),devices[rank],uid,C.byref(comm),error,len(error))
 if code:raise RuntimeError('NCCL initialization status '+str(code)+': '+error.value.decode(errors='replace'))
 alloc=[]
 def barrier(suffix):
  (root/('rank-'+str(rank)+suffix)).touch()
  _phase('rank '+str(rank)+' wait for all ranks '+suffix)
  while not all((root/('rank-'+str(r)+suffix)).exists() for r in range(len(devices))):
   if any((root/('rank-'+str(r)+'-error.json')).exists() for r in range(len(devices))):raise RuntimeError('Another NCCL rank failed; inspect rank logs')
   time.sleep(.02)
 try:
  barrier('-ready')
  if stage=='nccl_exchange':
   send=C.c_void_p();receive=C.c_void_p()
   for ptr in (send,receive):ck(cuda.cudaMalloc(C.byref(ptr),4));alloc.append(ptr)
   value=C.c_uint32(rank+1);ck(cuda.cudaMemcpy(send,C.byref(value),4,1))
   fn=lib.mgbfs_nccl_all_reduce_max_u32;fn.argtypes=[C.c_void_p]*4;fn.restype=C.c_int
   _phase('rank '+str(rank)+' NCCL all-reduce')
   code=fn(comm,send,receive,None)
   if code:raise RuntimeError('NCCL collective returned status '+str(code))
   ck(cuda.cudaDeviceSynchronize());value=C.c_uint32();ck(cuda.cudaMemcpy(C.byref(value),receive,4,2))
   if value.value!=len(devices):raise RuntimeError('NCCL all-reduce value mismatch')
  barrier('-done');_phase('rank '+str(rank)+' destroy NCCL communicator')
 finally:
  for ptr in alloc:cuda.cudaFree(ptr)
  lib.mgbfs_nccl_destroy(comm)
 return {'rank':rank,'device':devices[rank],'world':len(devices),'exchange_verified':stage=='nccl_exchange'}

def _component(stage,root,devices,device,seconds):
 os.environ['MGBFS_DIAGNOSTIC_TRACE_FILE']=str(root/'native-startup.log')
 if stage=='assets':
  from .native_distribution import native_runtime
  from .native_assets import status
  _phase('detect visible CUDA architecture and prepare verified precompiled components');native,env=native_runtime();data=status(Path(native).parent.parent);_write(root/'native-assets.json',data);return data
 if stage=='thread':
  _phase('start and join Python control thread')
  event=threading.Event();worker=threading.Thread(target=event.set);worker.start();worker.join(timeout=5)
  if worker.is_alive() or not event.is_set():raise RuntimeError('CONTROL_THREAD_JOIN_FAILED')
  return {'thread_start_join_verified':True}
 if stage in ('meminfo','hardware'):
  cuda,lib,ck=_cuda(device,stage=='hardware')
  if stage=='meminfo':
   _phase('cudaMemGetInfo independent native-runtime call')
   free=C.c_size_t();total=C.c_size_t();ck(cuda.cudaMemGetInfo(C.byref(free),C.byref(total)))
   return {'device':device,'free_bytes':free.value,'total_bytes':total.value}
  _phase('mgbfs_generic_hardware_info / cudaGetDeviceProperties independent wrapper')
  values=(C.c_uint64*8)();fn=lib.mgbfs_generic_hardware_info;fn.argtypes=[C.c_int,C.POINTER(C.c_uint64)];fn.restype=C.c_int
  ck(fn(device,values));return {'device':device,'hardware_values':list(values)}
 if stage in ('environment','nvml','torch','contrasts','loader-trace'):
  from .system_diagnostics import survey,nvml,torch_probe,contrasts,loader_trace
  return {'environment':lambda:survey(root),'nvml':nvml,'torch':torch_probe,'contrasts':lambda:contrasts(root,seconds),'loader-trace':lambda:loader_trace(root,seconds)}[stage]()
 if stage=='driver':
  from .native_distribution import native_runtime
  try:native,env=native_runtime();context='native_runtime loader environment'
  except Exception as error:env=dict(os.environ);context='inherited environment; native runtime unavailable: '+str(error)
  _phase('direct NVIDIA driver/runtime API probe; no nvidia-smi needed')
  subprocess.run([sys.executable,'-m','multigpubfs.driver_diagnostics','--output',str(root/'driver.json')],env=env,check=True)
  data=json.loads((root/'driver.json').read_text());data['loader_context']=context;return data
 if stage=='system':
  _phase('query nvidia-smi')
  r=subprocess.run([_tool('nvidia-smi'),'--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True,capture_output=True,check=True,timeout=min(10,seconds*.8));print(r.stdout,flush=True);return {'nvidia_smi':r.stdout.strip()}
 if stage=='filesystem':
  roots=[root,Path(os.environ.get('MGBFS_GENERIC_SESSION_LOG_ROOT') or tempfile.gettempdir())]
  for directory in roots:
   _phase('atomic file exchange in '+str(directory))
   with tempfile.TemporaryDirectory(prefix='mgbfs-doctor-',dir=directory) as tmp:
    f=Path(tmp)/'probe.json';_write(f,{'sequence':1});assert json.loads(f.read_text())=={'sequence':1}
  return {'atomic_write_read_verified':True,'directories':list(map(str,roots))}
 if stage=='runtime':
  from .native_distribution import native_runtime
  _phase('verify native bundle checksums and executable');native,env=native_runtime()
  if not os.access(native,os.X_OK):raise RuntimeError('Native executable has no execute permission')
  _phase('native binary loader/key-info');r=subprocess.run([native,'key-info'],env=env,capture_output=True,text=True,check=True)
  info=json.loads(r.stdout);_write(root/'runtime.json',{'status':'RUNNING','native':native,'key_info':info,'binary_executed':True});_phase('resolve native dependencies');_,_,libs=_paths()
  return {'native':native,'libraries':libs,'key_info':info}
 if stage=='count':
  from .native_distribution import native_runtime
  native,env=native_runtime();_phase('native graph-count / CUDA device discovery');r=subprocess.run([native,'graph-count'],env=env,capture_output=True,text=True,check=True);count=int(r.stdout.strip())
  if count<1:raise RuntimeError('No CUDA devices visible to the native engine')
  return {'visible_devices':count}
 if stage=='context':
  _cuda(device);return {'device':device,'cuda_context_verified':True}
 if stage=='memory':
  cuda,_,ck=_cuda(device);ptr=C.c_void_p();_phase('allocate/copy/read back 4096 bytes');ck(cuda.cudaMalloc(C.byref(ptr),4096))
  try:
   source=(C.c_uint32*1024)(*range(1024));out=(C.c_uint32*1024)();ck(cuda.cudaMemcpy(ptr,source,4096,1));ck(cuda.cudaMemcpy(out,ptr,4096,2));assert list(out)==list(source)
  finally:cuda.cudaFree(ptr)
  return {'device':device,'allocated_bytes':4096,'readback_verified':True}
 if stage=='kernel':return _kernel(device)
 if stage=='persistent':
  from .generic_session import Pool
  from .native_distribution import native_runtime
  native,env=native_runtime();log_root=root/'persistent-runtime';log_root.mkdir();previous=os.environ.get('MGBFS_GENERIC_SESSION_LOG_ROOT');os.environ['MGBFS_GENERIC_SESSION_LOG_ROOT']=str(log_root)
  try:pool=Pool()
  finally:
   if previous is None:os.environ.pop('MGBFS_GENERIC_SESSION_LOG_ROOT',None)
   else:os.environ['MGBFS_GENERIC_SESSION_LOG_ROOT']=previous
  try:
   _phase('persistent worker first file request');first=pool.inventory(native,[device],env,max(.1,seconds*.5));pid=pool.workers[0].process.pid
   _phase('persistent worker second file request');second=pool.inventory(native,[device],env,max(.1,seconds*.2));assert pid==pool.workers[0].process.pid and second[0]['device']==device
   return {'device':device,'worker_pid':pid,'requests':2,'same_worker_verified':True}
  finally:
   _save_pool(pool,root);_phase('close diagnostic persistent worker');pool.close()
 if stage in ('nccl_init','nccl_exchange'):
  _,lib,ck=_cuda(engine=True);uid=C.create_string_buffer(128);lib.mgbfs_nccl_unique_id.argtypes=[C.c_void_p];lib.mgbfs_nccl_unique_id.restype=C.c_int
  if lib.mgbfs_nccl_unique_id(uid):raise RuntimeError('NCCL unique ID creation failed')
  (root/'nccl-id.bin').write_bytes(uid.raw);jobs=[];logs=[]
  try:
   for rank in range(len(devices)):
    log=(root/('rank-'+str(rank)+'.log')).open('w');logs.append(log)
    command=[sys.executable,'-m','multigpubfs.diagnostics','--_rank',str(rank),'--_child',stage,'--output',str(root),'--devices',','.join(map(str,devices))]
    env=dict(os.environ);env.setdefault('NCCL_DEBUG','INFO');jobs.append(subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL));_phase('NCCL rank '+str(rank)+' started PID '+str(jobs[-1].pid))
   while any(p.poll() is None for p in jobs):
    if any(p.poll() not in (None,0) for p in jobs):raise RuntimeError('NCCL rank process failed; inspect rank logs')
    time.sleep(.02)
   if any(p.returncode for p in jobs):raise RuntimeError('NCCL rank process failed')
   return {'devices':devices,'ranks':[json.loads((root/('rank-'+str(r)+'.json')).read_text()) for r in range(len(devices))]}
  finally:
   for job in jobs:
    if job.poll() is None:job.terminate()
   for log in logs:log.close()
 if stage=='bfs':
  from .graph_definition import GraphDefinition
  from .launch import run_graph
  g=GraphDefinition.permutation([[1,0,2],[0,2,1]],[0,1,2]);_phase('native BFS, known S3 layers')
  report=run_graph(g,root/'bfs',devices=devices,capacity=64,shards=1,autotune=False,max_seconds=max(1,min(10,int(seconds*.4))))
  if report['status']!='COMPLETE' or report['layer_sizes']!=[1,2,2,1]:raise RuntimeError('Tiny BFS did not match exact S3 oracle: '+repr(report))
  return {'devices':devices,'layer_sizes':report['layer_sizes'],'bfs_seconds':report['bfs_seconds'],'exact_oracle_verified':True}
 raise ValueError('Unknown diagnostic component '+stage)

class Runner:
 def __init__(self,root,timeout):
  self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False);self.timeout=timeout
  try:version=importlib.metadata.version('multigpubfs')
  except importlib.metadata.PackageNotFoundError:version='source-checkout'
  self.report={'schema':1,'status':'RUNNING','scope':'single-host diagnostic; no multi-node acceptance or automatic BFS restart','python':sys.version,'platform':platform.platform(),'package':version,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage_timeout_seconds':timeout,'environment':{k:os.environ[k] for k in ('MGBFS_NATIVE_QUERY_SECONDS','CUDA_MODULE_LOADING','CUDA_DISABLE_PTX_JIT','NCCL_P2P_DISABLE','NCCL_SHM_DISABLE','NCCL_IB_DISABLE','NCCL_SOCKET_IFNAME') if k in os.environ},'steps':[]}
  try:
   from .system_diagnostics import limits
   self.report['parent_resource_limits']=limits()
  except (ImportError,OSError):self.report['parent_resource_limits']={'status':'UNAVAILABLE'}
  self.save()
 def save(self):_write(self.root/'report.json',self.report)
 def skip(self,name,reason):
  self.report['steps'].append({'name':name,'status':'SKIPPED','reason':reason});self.save();print('SKIP '+name+': '+reason,flush=True)
 def run(self,name,command,*,optional=False,environment=None):
  if getattr(self,'deep_cleanup_blocked',False):
   self.skip(name,'OWNED_DIAGNOSTIC_PROCESS_COULD_NOT_BE_REAPED; no further probes started');return self.report['steps'][-1]
  entry={'name':name,'status':'RUNNING','command':command,'timeout_seconds':self.timeout,'log':name+'.log'};self.report['steps'].append(entry);self.save();print('START '+name,flush=True)
  started=time.monotonic();process=None;thread=None
  try:
   with (self.root/(name+'.log')).open('w',encoding='utf-8') as log:
    env=dict(os.environ if environment is None else environment)
    if getattr(self,'native_assets_failed',False):env['MGBFS_NATIVE_OFFLINE']='1'
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT','TORCHELASTIC_RUN_ID'):env.pop(key,None)
    process=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,text=True,encoding='utf-8',errors='replace',start_new_session=os.name=='posix')
    entry['pid']=process.pid;self.save()
    def pump():
     for line in process.stdout:
      log.write(line);log.flush()
      if line.startswith('PHASE '):print('  '+name+': '+line.strip()[6:],flush=True)
    thread=threading.Thread(target=pump,daemon=True);thread.start()
    try:process.wait(timeout=self.timeout)
    except subprocess.TimeoutExpired:
     entry['status']='TIMEOUT';entry['error']='Stage deadline exceeded; only this diagnostic process group is terminated'
     if getattr(self,'deep_capture',False):
      from .deep_diagnostics import process_snapshot
      folder=self.root/name;folder.mkdir(exist_ok=True);process_snapshot(process.pid,folder/'process-snapshot.json')
    except BaseException:self.stop_probe(process,self.root/name);raise
    cleanup=self.stop_probe(process,self.root/name);thread.join(timeout=2)
    if not cleanup['complete']:
     entry['status']='TIMEOUT' if entry['status']=='TIMEOUT' else 'FAIL'
     entry['error']=entry.get('error','')+'; OWNED_DIAGNOSTIC_PROCESS_COULD_NOT_BE_REAPED: '+str(cleanup['surviving_group_pids'])
    if not thread.is_alive() and process.stdout:process.stdout.close()
    if entry['status']=='RUNNING':
     entry['exit_code']=process.returncode;entry['status']='PASS' if process.returncode==0 else 'FAIL'
     if process.returncode:entry['error']='Component process exited with code '+str(process.returncode)
  except (OSError,ValueError) as error:entry.update(status='FAIL',error=str(error),errno=getattr(error,'errno',None),exception_type=type(error).__name__)
  entry['seconds']=time.monotonic()-started
  if optional and entry['status'] in ('FAIL','TIMEOUT') and not getattr(self,'deep_cleanup_blocked',False):entry['status']='WARNING'
  entry['log_tail']=(self.root/(name+'.log')).read_text(encoding='utf-8',errors='replace')[-6000:] if (self.root/(name+'.log')).exists() else ''
  phases=[line[6:] for line in entry['log_tail'].splitlines() if line.startswith('PHASE ')]
  if phases:entry['last_phase']=phases[-1]
  for file in ('driver.json','environment.json','runtime.json'):
   snapshot=self.root/name/file
   if snapshot.is_file():
    try:entry.setdefault('partial_evidence',{})[file]=json.loads(snapshot.read_text())
    except (OSError,ValueError):pass
  result=self.root/name/'result.json'
  if result.is_file():
   try:entry['result']=json.loads(result.read_text())
   except (OSError,ValueError) as error:entry.update(status='FAIL',error='Invalid component result: '+str(error))
  if entry.get('result',{}).get('error'):entry['error']=entry['result']['error']
  data=entry.get('result',{}).get('data',{})
  if entry['status']=='PASS' and data.get('status')=='UNAVAILABLE':entry.update(status='SKIPPED',reason=data.get('reason',data.get('error','Auxiliary component unavailable')))
  self.save();print(entry['status']+' '+name+' ('+format(entry['seconds'],'.2f')+' s)',flush=True);return entry
 def stop_probe(self,process,folder=None):
  from .deep_diagnostics import bounded_stop
  result=bounded_stop(process)
  self.report.setdefault('diagnostic_cleanup',[]).append(result)
  if folder is not None:
   folder.mkdir(parents=True,exist_ok=True)
   _write(folder/'cleanup.json',result)
  if not result['complete']:
   self.deep_cleanup_blocked=True
   if folder is not None:
    from .deep_diagnostics import process_snapshot
    process_snapshot(process.pid,folder/'post-cleanup-snapshot.json',process_group=process.pid)
  self.save();return result
 @staticmethod
 def stop(process):
  if os.name=='posix':
   try:os.killpg(process.pid,signal.SIGTERM)
   except ProcessLookupError:pass
  elif process.poll() is None:process.terminate()
  try:process.wait(timeout=1)
  except subprocess.TimeoutExpired:pass
  if os.name=='posix':
   try:os.killpg(process.pid,signal.SIGKILL)
   except ProcessLookupError:pass
  elif process.poll() is None:process.kill()
  process.wait()
 def component(self,name,stage,*,devices=None,device=None,optional=False):
  folder=self.root/name;folder.mkdir()
  command=[sys.executable,'-m','multigpubfs.diagnostics','--_child',stage,'--output',str(folder),'--stage-timeout',str(self.timeout)]
  if devices is not None:command+=['--devices',','.join(map(str,devices))]
  if device is not None:command+=['--_device',str(device)]
  return self.run(name,command,optional=optional)
 def finish(self):
  if getattr(self,'deep_cleanup_blocked',False) and self.report.get('working_configuration',{}).get('status')=='READY':
   self.report['working_configuration']={'status':'BLOCKED','reason':'A later diagnostic child could not be reaped; no ready launch command'}
   for name in ('working-config.json','server-command.txt','architecture.json','native-assets.json'):(self.root/name).unlink(missing_ok=True)
  failures=[x['name'] for x in self.report['steps'] if x['status'] in ('FAIL','TIMEOUT')]
  from .diagnostic_analysis import analyze,render
  self.report.update(status='FAIL' if failures else 'PASS',failed_components=failures,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());self.report['analysis']=analyze(self.report)
  if not failures and any(x.get('reason')=='GLOBAL_DIAGNOSTIC_BUDGET_EXHAUSTED' for x in self.report['steps']):self.report['status']='INCOMPLETE'
  self.save();_write(self.root/'analysis.json',self.report['analysis']);(self.root/'analysis.txt').write_text(render(self.report['analysis']),encoding='utf-8')
  lines=['MultiGPUBFS diagnostic: '+self.report['status'],'Report: '+str(self.root/'report.json')]
  if self.report.get('production_bfs_acceptance'):lines.append('Production BFS acceptance: '+self.report['production_bfs_acceptance']['status']+' ('+self.report['production_bfs_acceptance']['build']+')')
  working=self.report.get('working_configuration',{})
  if working:
   lines.append('Working server configuration: '+working['status'])
   if working.get('command'):
    lines.append('VERIFIED GPU SET '+str(working['selected']['devices'])+(' (single-GPU fallback)' if working.get('single_gpu_fallback') else ''))
    lines.append('READY COMMAND '+working['command']);lines.append('CALCULATION EXAMPLE '+working['calculation_example'])
   elif working.get('reason'):lines.append('NO READY COMMAND '+working['reason'])
  for step in self.report['steps']:lines.append(step['status']+' '+step['name']+' '+step.get('reason',step.get('error','')))
  lines.append(render(self.report['analysis']));(self.root/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
  files=[f for f in self.root.rglob('*') if f.is_file() and (f.suffix=='.log' or f.name in ('report.json','report.txt','result.json','driver.json','environment.json','runtime.json','analysis.json','analysis.txt','probe.json','process-snapshot.json','deep-analysis.json','deep-summary.txt','variant.json','cleanup.json','post-cleanup-snapshot.json','working-config.json','server-command.txt','architecture.json','native-assets.json') or f.name.startswith(('rank-','response-','phase-')) and f.suffix=='.json' or 'failure-evidence' in f.relative_to(self.root).parts and f.name in ('phase.json','response.json','summary.json'))]
  manifest=[]
  for f in sorted(files):manifest.append({'path':str(f.relative_to(self.root)),'size':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
  _write(self.root/'manifest.json',{'schema':1,'files':manifest,'scope':'logs/metadata only; BFS state payloads and job environment files excluded'})
  with zipfile.ZipFile(self.root/'diagnostics.zip','w',zipfile.ZIP_DEFLATED) as archive:
   for f in files+[self.root/'manifest.json']:archive.write(f,str(f.relative_to(self.root)))
  print('\n'.join(lines),flush=True);print('BUNDLE '+str(self.root/'diagnostics.zip'),flush=True);return self.report

def prepare_delivery(runner,deadline):
 """Downloads have their own phase/deadline, before native query timeouts."""
 from .native_distribution import native_root
 package=Path(__file__).parent
 if os.environ.get('MGBFS_EXECUTABLE') or not (package/'native_assets.json').is_file() or (package/'_native/bin/mgbfs').is_file():return None
 previous=runner.timeout;runner.timeout=max(.1,min(600,deadline-time.monotonic()-8))
 try:step=runner.component('native-assets','assets')
 finally:runner.timeout=previous
 runner.report['native_delivery']=step.get('result',{}).get('data',{'status':step['status'],'error':step.get('error','Native component preparation did not complete')})
 if step['status']!='PASS':runner.native_assets_failed=True
 runner.save();return step

def recover_visible_devices(driver_step,requested=None):
 """Use agreed real CUDA driver/runtime evidence, never NVML or scheduler counts."""
 if driver_step.get('status')!='PASS':return None
 data=driver_step.get('result',{}).get('data',{})
 if data.get('classification')!='DRIVER_RUNTIME_READY':return None
 driver=data.get('driver',{});runtime=data.get('runtime',{});count=driver.get('visible_devices')
 if type(count) is not int or not 0<count<=128 or runtime.get('visible_devices')!=count:return None
 rows=driver.get('devices',[])
 if len(rows)!=count or {r.get('logical_id') for r in rows}!=set(range(count)):return None
 selected=list(range(count)) if requested is None else list(requested)
 if not selected or len(set(selected))!=len(selected) or any(type(d) is not int or not 0<=d<count for d in selected):return None
 return selected

def record_architecture(runner,driver_step):
 from .architecture_support import coverage
 if driver_step.get('status')!='PASS':return
 manifest=Path(__file__).parent/'_native/manifest.json'
 if not manifest.is_file():
  from .native_distribution import native_root
  delivery=runner.report.get('native_delivery',{})
  delivered=delivery.get('root') if delivery.get('status')=='READY' else None
  manifest=(Path(delivered) if delivered else native_root(prepare=False))/'manifest.json'
 if not manifest.is_file():return
 data=driver_step.get('result',{}).get('data',{}).get('driver',{})
 value=coverage(json.loads(manifest.read_text()),data.get('devices',[]))
 runner.report['architecture_support']=value;_write(runner.root/'architecture.json',value);runner.save()
 for device in value['devices']:print('GPU architecture: logical '+str(device['logical_id'])+' SM'+str(device['sm'])+' '+device['code'],flush=True)

def run_diagnostics(output=None,devices=None,timeout=60,*,deep=False,total_timeout=1800):
 if deep:
  from .deep_diagnostics import run_deep
  return run_deep(output,devices,timeout,total_timeout)
 if not math.isfinite(timeout) or timeout<=0:raise ValueError('Stage timeout must be positive and finite')
 if devices is not None and (not devices or len(devices)!=len(set(devices)) or any(type(d) is not int or d<0 for d in devices)):raise ValueError('Devices must be distinct nonnegative integer IDs')
 search_deadline=time.monotonic()+total_timeout
 root=Path(output or ('mgbfs-diagnostics-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+str(os.getpid()))).absolute();runner=Runner(root,timeout)
 try:
  prepare_delivery(runner,search_deadline)
  runner.component('control-thread','thread');runner.component('system-info','system',optional=True);runner.component('filesystem','filesystem');runner.component('environment-evidence','environment',optional=True);runner.component('nvml-evidence','nvml',optional=True);runner.component('torch-evidence','torch',optional=True);driver=runner.component('driver-api','driver')
  runtime=runner.component('native-runtime','runtime')
  if runtime['status']!='PASS':runner.component('native-loader-trace','loader-trace',optional=True);runner.skip('gpu-components','Native runtime loader/integrity failed');return runner.finish()
  record_architecture(runner,driver)
  runner.deep_capture=True
  count=runner.component('cuda-discovery','count')
  runner.component('loader-contrasts','contrasts',optional=True);runner.component('native-loader-trace','loader-trace',optional=True)
  if count['status']!='PASS':
   selected=recover_visible_devices(driver,devices)
   if selected is not None:
    runner.report['devices']=selected
    runner.report['native_discovery_recovery']={'status':'INDEPENDENT_CUDA_EVIDENCE','devices':selected,'scope':'Original native count failure is retained. Only an exact native BFS and executed launcher can produce READY.'}
    from .server_recipe import find_working
    runner.report['working_configuration']=find_working(runner,selected,search_deadline,full_gpu_only=True);runner.save()
   else:runner.skip('gpu-components','CUDA discovery failed and no agreed independent CUDA count is available')
   return runner.finish()
  selected=devices or list(range(count['result']['data']['visible_devices']))
  if any(d>=count['result']['data']['visible_devices'] for d in selected):
   runner.report['steps'].append({'name':'device-selection','status':'FAIL','error':'Selected GPU ID is not visible'});runner.save();return runner.finish()
  runner.report['devices']=selected;runner.save()
  from .deep_diagnostics import run_case
  def joint(name,stage):return run_case(runner,{'name':name,'config':{'kind':'component','stage':stage,'devices':selected,'seconds':timeout},'seconds':timeout},search_deadline)
  context=joint('all-gpus-context','context')
  if context['status']=='PASS':
   joint('all-gpus-meminfo','meminfo');joint('all-gpus-hardware','hardware')
   memory=joint('all-gpus-memory','memory')
   kernel=joint('all-gpus-kernel','kernel') if memory['status']=='PASS' else None
   if kernel is None:runner.skip('all-gpus-kernel','Memory allocation/readback failed on the selected GPU set')
   joint('all-gpus-persistent','persistent')
   if kernel is not None and kernel['status']=='PASS':
    if len(selected)>1:
     init=joint('nccl-init','nccl_init')
     if init['status']=='PASS':
      exchange=joint('nccl-exchange','nccl_exchange')
      if exchange['status']=='PASS':joint('multi-gpu-bfs','bfs')
      else:runner.skip('multi-gpu-bfs','NCCL exchange failed')
     else:runner.skip('nccl-exchange','NCCL initialization failed');runner.skip('multi-gpu-bfs','NCCL initialization failed')
    else:joint('all-gpus-bfs','bfs')
   else:runner.skip('joint-bfs','Kernel did not pass on the complete selected GPU set')
  else:runner.skip('all-gpus-remaining','CUDA context failed on the selected GPU set; no subset fallback')
  from .server_recipe import find_working
  runner.report['working_configuration']=find_working(runner,selected,search_deadline,full_gpu_only=True);runner.save()
  return runner.finish()
 except KeyboardInterrupt:
  runner.report['steps'].append({'name':'interrupted','status':'FAIL','error':'Interrupted by caller'});return runner.finish()
 except Exception as error:
  runner.report['steps'].append({'name':'diagnostic-controller','status':'FAIL','error':str(error),'traceback':traceback.format_exc()});return runner.finish()

def main(argv=None):
 parser=argparse.ArgumentParser(description='Check native loader, filesystem, every visible GPU, kernel, persistent workers, NCCL and tiny exact BFS; write report/log bundle. No GPU reset or automatic retry.')
 parser.add_argument('--output',type=Path,help='New diagnostic directory (must not already exist)');parser.add_argument('--devices',help='Logical local IDs, e.g. 0,1; default all visible GPUs');parser.add_argument('--stage-timeout',type=float,default=60)
 parser.add_argument('--deep',action='store_true',help='Isolated comprehensive synthetic diagnostic suite; no production retries or GPU resets');parser.add_argument('--total-timeout',type=float,default=1800,help='Deep suite total execution budget, seconds (default 30 minutes)')
 parser.add_argument('--_child',help=argparse.SUPPRESS);parser.add_argument('--_device',type=int,help=argparse.SUPPRESS);parser.add_argument('--_rank',type=int,help=argparse.SUPPRESS)
 args=parser.parse_args(argv)
 try:
  devices=None if args.devices is None else [int(v) for v in args.devices.split(',')]
  if args._child:
   root=args.output
   try:
    data=_rank(args._child,root,devices,args._rank) if args._rank is not None else _component(args._child,root,devices,args._device,args.stage_timeout)
    result={'status':'PASS','data':data};code=0
    if args._child=='driver' and data['status']!='PASS':result.update(status='FAIL',error=data['summary']);code=1
    if args._child in ('nvml','torch','contrasts','loader-trace') and data.get('status')=='FAIL':result.update(status='FAIL',error=data.get('error','Auxiliary probe failed'));code=1
   except Exception as error:
    result={'status':'FAIL','error':str(error),'exception_type':type(error).__name__,'traceback':traceback.format_exc()}
    for key in ('stdout','stderr'):
     value=getattr(error,key,None)
     if value:
      value=value.decode(errors='replace') if isinstance(value,bytes) else str(value)
      result[key]=value[-12000:];print(key.upper()+': '+result[key],flush=True)
    print(result['traceback'],flush=True);code=1
   name='result.json' if args._rank is None else 'rank-'+str(args._rank)+('.json' if code==0 else '-error.json')
   _write(root/name,result);return code
  report=run_diagnostics(args.output,devices,args.stage_timeout,deep=args.deep,total_timeout=args.total_timeout);return 0 if report['status']=='PASS' else 1
 except (OSError,ValueError) as error:print('Diagnostic setup error: '+str(error),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
