"""Resident generic ranks. CPU handles only job boundaries, never BFS states."""
import contextlib,json,os,subprocess,tempfile,time
from pathlib import Path
_active=None

class Job:
 def __init__(self,worker,sequence):self.worker=worker;self.sequence=sequence;self.pid=worker.process.pid;self.returncode=None;self.response=None;self.receipt_log=None;self.log_offset=0;self.logged=False
 def poll(self):
  path=self.worker.root/f'response-{self.sequence:08}.json'
  if self.returncode is None and path.exists():
   self.response=json.loads(path.read_text())
   if self.response.get('schema')!=1 or self.response.get('sequence')!=self.sequence:raise RuntimeError('GENERIC_SESSION_RESPONSE_IDENTITY')
   self.returncode=self.response.get('result',{}).get('exit_code',0) if self.response.get('ok') is True else 1
  if self.returncode is None and self.worker.process.poll() is not None:self.returncode=self.worker.process.returncode or 1
  if self.returncode is not None and self.receipt_log is not None and not self.logged:
   with (self.worker.root/'worker.log').open() as stream:
    stream.seek(self.log_offset);self.receipt_log.write(stream.read())
   if self.returncode:self.receipt_log.write('GENERIC_SESSION_ERROR '+repr(self.response)+'\n')
   self.receipt_log.flush();self.logged=True
  return self.returncode
 def wait(self,timeout=None):
  start=time.monotonic()
  while self.poll() is None:
   if timeout is not None and time.monotonic()-start>timeout:raise subprocess.TimeoutExpired('generic resident job',timeout)
   time.sleep(.001)
  return self.returncode
 def terminate(self):self.worker.process.terminate()
 def kill(self):self.worker.process.kill()

class Worker:
 def __init__(self,native,env,rank,device,root,kind):
  self.root=root;root.mkdir();self.sequence=0;self.job=None;self.log=(root/'worker.log').open('w')
  startup={k:v for k,v in env.items() if not k.startswith('MGBFS_') or k in ('MGBFS_CUDART_LIBRARY','MGBFS_CUDA_LIB_DIR','MGBFS_EXECUTABLE')}
  startup.update(RANK=str(rank),LOCAL_RANK=str(device),MGBFS_SESSION_KIND=kind)
  self.process=subprocess.Popen([str(native),'graph-session',str(root),str(rank),str(device)],env=startup,stdout=self.log,stderr=subprocess.STDOUT)
 def submit(self,command,args,env):
  if self.job is not None and self.job.poll() is None:raise RuntimeError('GENERIC_SESSION_CONCURRENT_JOB')
  if self.process.poll() is not None or self.job is not None and self.job.returncode and not (self.job.response or {}).get('result',{}).get('query_only'):raise RuntimeError('GENERIC_SESSION_FAILED_NO_RESTART')
  job=Job(self,self.sequence);self.job=job
  path=self.root/f'job-{self.sequence:08}.json';tmp=path.with_suffix('.tmp')
  tmp.write_text(json.dumps(dict(schema=1,sequence=self.sequence,command=command,args=args,env={k:v for k,v in env.items() if k.startswith(('MGBFS_','NCCL_'))})));os.replace(tmp,path);self.sequence+=1
  return job
 def close(self):
  (self.root/'shutdown').touch()
  try:self.process.wait(timeout=5)
  except subprocess.TimeoutExpired:
   self.process.terminate()
   try:self.process.wait(timeout=10)
   except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
  self.log.close()

class Pool:
 def __init__(self,kind='generic',owner=None):self.kind=kind;self.owner=owner;self.active_kind='generic';self.special=None;self.temporary=tempfile.TemporaryDirectory(prefix='mgbfs-generic-session-',dir=os.environ.get('MGBFS_GENERIC_SESSION_LOG_ROOT'));self.root=Path(self.temporary.name);self.visible_counts={};self.workers=[];self.identity=None;self.generation=0
 def release_memory(self):
  if not self.workers:return
  jobs=[w.submit('release-memory',[],{}) for w in self.workers]
  for job in jobs:
   if job.wait(30):raise RuntimeError('RESIDENT_MEMORY_HANDOFF_FAILED')
 def activate(self):
  root=self.owner or self
  if root.active_kind!=self.kind:
   previous=root if root.active_kind=='generic' else root.special
   if previous is not None:previous.release_memory()
   root.active_kind=self.kind
 def select(self,native,devices,env):
  self.activate()
  from .autotune import _dependency_identity,_stat_identity
  dependency=_dependency_identity(str(native),env)
  stamps=tuple((x[0],x[1],_stat_identity(x[1])) for x in (dependency or {}).get('resolved_dependencies',[]))
  key=(_stat_identity(native),stamps,str(native),tuple(devices),tuple((k,env.get(k)) for k in ('LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','CUDA_MODULE_LOADING','CUDA_DISABLE_PTX_JIT')),tuple(sorted((k,v) for k,v in env.items() if k.startswith('NCCL_'))))
  if key==self.identity and any(w.process.poll() is not None or w.job is not None and w.job.returncode not in (None,0) and not (w.job.response or {}).get('result',{}).get('query_only') for w in self.workers):raise RuntimeError('GENERIC_SESSION_FAILED_GROUP_NO_RESTART')
  if key!=self.identity:
   self.close_workers();root=self.root/str(self.generation);root.mkdir();self.generation+=1
   try:
    for rank,device in enumerate(devices):self.workers.append(Worker(native,env,rank,device,root/str(rank),self.kind))
    self.identity=key
   except BaseException:self.close_workers();raise
 def inventory(self,native,devices,env,seconds):
  self.select(native,devices,env);jobs=[w.submit('inventory',[],env) for w in self.workers];deadline=time.monotonic()+seconds
  for job in jobs:
   if job.wait(max(.001,deadline-time.monotonic())):raise RuntimeError('GENERIC_SESSION_INVENTORY_FAILED '+repr(job.response))
  return [job.response['result'] for job in jobs]
 def close_workers(self):
  for worker in self.workers:(worker.root/'shutdown').touch()
  for worker in self.workers:worker.close()
  self.workers=[];self.identity=None
 def close(self):
  if self.special is not None:self.special.close()
  self.close_workers();self.temporary.cleanup()

@contextlib.contextmanager
def persistent_graph_workers():
 global _active
 if _active is not None:raise RuntimeError('GENERIC_SESSION_NESTED')
 pool=Pool();_active=pool
 try:yield pool
 finally:_active=None;pool.close()

def query(command,env,seconds,kwargs):
 if _active is None or len(command)<5 or command[1] not in ('graph-info','graph-plan'):return None
 selected=command[3]
 if command[1]=='graph-plan':selected=','.join(str(x['device']) for x in json.loads(Path(selected).read_text()))
 if selected=='auto':
  key=(str(command[0]),env.get('CUDA_VISIBLE_DEVICES'),env.get('CUDA_DEVICE_ORDER'))
  if key not in _active.visible_counts:
   raw=subprocess.check_output([command[0],'graph-count'],env=env,text=True,timeout=seconds);_active.visible_counts[key]=int(raw.strip())
  devices=list(range(_active.visible_counts[key]))
 else:devices=[int(v) for v in selected.split(',')]
 started=time.monotonic();inventory=_active.inventory(command[0],devices,env,seconds)
 if env.get('MGBFS_GENERIC_HISTORY')=='sorted':
  definition=json.loads(Path(command[2]).read_text());generators=len(definition['action']['generators']);capacity=command[5] if len(command)>5 else 'auto';target=command[6] if len(command)>6 else str(max(1,(1<<20)//generators))
  if capacity!='auto':target=str(min(int(target),int(capacity)))
  for round in range(2):
   jobs=[w.submit('graph-local-plan',[command[2],str(device),str(len(devices)),command[4],capacity,target],env) for w,device in zip(_active.workers,devices)]
   for i,job in enumerate(jobs):
    if job.wait(max(.001,seconds-(time.monotonic()-started))):raise RuntimeError('GENERIC_SESSION_LOCAL_PLAN_FAILED '+repr(job.response))
    local=job.response['result'];inventory[i]['free_bytes']=local['free_bytes'];inventory[i]['local_sorted_plan']=local['plan']
   target=str(min(x['local_sorted_plan']['batch'] for x in inventory))
 with tempfile.TemporaryDirectory(prefix='mgbfs-live-inventory-') as d:
  path=Path(d)/'inventory.json';path.write_text(json.dumps(inventory))
  job=_active.workers[0].submit('graph-plan',[command[2],str(path),*command[4:]],env)
  if job.wait(max(.001,seconds-(time.monotonic()-started))):raise RuntimeError('GENERIC_SESSION_PLAN_FAILED '+repr(job.response))
  return subprocess.CompletedProcess(command,0,json.dumps(job.response['result']), '')

def launch(command,env,log,devices):
 if _active is None:return subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 _active.select(command[0],devices,env);rank=int(command[4])
 worker=_active.workers[rank];offset=(worker.root/'worker.log').stat().st_size
 job=worker.submit('graph-rank',command[2:],env);job.receipt_log=log;job.log_offset=offset
 log.write(json.dumps({'resident_pid':job.pid,'sequence':job.sequence,'worker_log':str(job.worker.root/'worker.log')})+'\n');log.flush()
 return job

def launch_specialized(command,env,log,devices):
 if _active is None:return subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 if env.get('MGBFS_TRANSPORT_BACKEND')=='NCCL_LSA':
  _active.release_memory()
  if _active.special is not None:_active.special.release_memory()
  return subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 if _active.special is None:_active.special=Pool(kind='bench',owner=_active)
 pool=_active.special;pool.select(command[0],devices,env);rank=int(env['RANK']);worker=pool.workers[rank];offset=(worker.root/'worker.log').stat().st_size
 job_env=dict(env)
 if command[-1]=='--search-only':job_env.update(MGBFS_SEARCH_ONLY='1',MGBFS_BENCH_SKIP_ARCHIVE='1',MGBFS_ARCHIVE_STREAM='0')
 else:job_env.pop('MGBFS_SEARCH_ONLY',None)
 job=worker.submit('bench',['resident-bench',*command[3:8]],job_env);job.receipt_log=log;job.log_offset=offset
 log.write(json.dumps({'resident_pid':job.pid,'sequence':job.sequence})+'\n');log.flush();return job
