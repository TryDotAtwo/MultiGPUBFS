"""Opt-in synthetic diagnostics. No production configuration changes or resets."""
import argparse,collections,ctypes as C,json,math,os,platform,shutil,subprocess,sys,tempfile,threading,time
from pathlib import Path

PROBE_BYTES=4096
def write(path,value):
 from .diagnostics import _write
 _write(path,value)

def process_snapshot(pid,destination):
 """Read only the process group created for this diagnostic probe."""
 value={'schema':1,'scope':'owned diagnostic process group only','processes':[],'unavailable':[]}
 if os.name!='posix' or not Path('/proc').is_dir():
  value['unavailable'].append('Linux /proc unavailable');write(destination,value);return value
 try:group=os.getpgid(pid)
 except ProcessLookupError:
  value['unavailable'].append('Probe process exited before snapshot');write(destination,value);return value
 candidates=[pid] if group==os.getpgrp() else [int(x.name) for x in Path('/proc').iterdir() if x.name.isdigit()]
 for number in candidates:
  try:
   if os.getpgid(number)!=group:continue
   root=Path('/proc')/str(number);item={'pid':number,'process_group':group,'threads':[]};value['processes'].append(item)
   item['executable']=str((root/'exe').resolve())
   for name in ('status','limits','wchan','syscall','stack'):
    try:item[name]=(root/name).read_text(errors='replace')[:12000]
    except OSError as e:item[name]={'unavailable':type(e).__name__,'errno':e.errno}
   try:item['loaded_libraries']=[x for x in (root/'maps').read_text(errors='replace').splitlines() if any(k in x for k in ('libcuda','libnccl','libmgbfs','libstdc++','libgcc','libpthread'))][:160]
   except OSError as e:item['loaded_libraries']={'unavailable':str(e)}
   for thread in sorted((root/'task').iterdir())[:64]:
    entry={'tid':int(thread.name)}
    for name in ('status','wchan','syscall','stack'):
     try:entry[name]=(thread/name).read_text(errors='replace')[:5000]
     except OSError as e:entry[name]={'unavailable':type(e).__name__,'errno':e.errno}
    item['threads'].append(entry)
  except (OSError,ValueError):continue
 write(destination,value);return value

def startup_cases(devices):
 cases=[]
 for device in devices:
  for module,data in [('inherited','inherited'),('EAGER','EAGER'),('LAZY','LAZY'),('EAGER','LAZY'),('LAZY','EAGER')]:
   for order in ('engine-first','context-first'):
    for thread in (False,True) if module=='EAGER' and data=='EAGER' else (False,):
     config={'kind':'api','device':device,'module':module,'data_loading':data,'order':order,'control_thread':thread}
     cases.append({'name':f'deep-gpu-{device}-{module}-{data}-{order}-thread{int(thread)}','config':config,'seconds':25})
 return cases

def fixture(name):
 from .graph_definition import GraphDefinition as G
 if name=='permutation':return G.permutation([[1,0,2],[0,2,1]],[0,1,2])
 if name=='repeated':return G.permutation([[1,0,2],[0,2,1]],[0,0,1])
 if name=='wide':return G.permutation([[1,0]+list(range(2,17))],list(range(17)))
 if name=='matrix':return G.matrix(2,1,[([0,1,1,0],3),([2,0,0,2],3)],[1,0])
 if name=='matrix-gemm':
  n=8;eye=[int(i==j) for i in range(n) for j in range(n)];perm=list(eye);perm[:8]=eye[8:16];perm[8:16]=eye[:8]
  return G.matrix(n,1,[(perm,2)],[1]+[0]*7)
 if name=='lrx':return G.permutation([[1,2,3,0],[3,0,1,2],[1,0,2,3]],[0,1,2,3])
 raise ValueError('Unknown fixture '+name)

def oracle(graph):
 start=tuple(graph.start);seen={start};frontier=[start];sizes=[]
 while frontier:
  sizes.append(len(frontier));last=frontier;next_layer=[]
  for state in frontier:
   for generator in range(graph.generator_count):
    child=tuple(graph.successor(list(state),generator))
    if child not in seen:seen.add(child);next_layer.append(child)
  if len(seen)>4096:raise ValueError('Diagnostic oracle exceeds bounded domain')
  frontier=next_layer
 return sizes,sorted(last)

def api_probe(config,root):
 from .diagnostics import _paths,_kernel
 state={'schema':1,'pid':os.getpid(),'config':config,'environment':{k:os.environ.get(k) for k in ('CUDA_MODULE_LOADING','CUDA_MODULE_DATA_LOADING','CUDA_VISIBLE_DEVICES','CUDA_DISABLE_PTX_JIT')},'operations':[]}
 def phase(name):
  state['last_operation']=name;state['operations'].append({'name':name,'status':'ENTERED','monotonic':time.monotonic()});write(root/'probe.json',state);print('PHASE '+name,flush=True)
 def done(code=0):
  state['operations'][-1].update(status='PASS' if code==0 else 'FAIL',code=code,finished_monotonic=time.monotonic());write(root/'probe.json',state)
  if code:raise RuntimeError('CUDA_API_STATUS_'+str(code)+' at '+state['last_operation'])
 phase('resolve private loader paths');native,env,paths=_paths();state['resolved_libraries']=paths;done()
 loaded={};stop=threading.Event();worker=None
 if config.get('control_thread'):
  phase('start concurrent file-polling control thread')
  def poll():
   while not stop.wait(.005):(root/'cancel').exists()
  worker=threading.Thread(target=poll);worker.start();done()
 try:
  phase('load CUDA runtime');cuda=C.CDLL(paths['libcudart.so.12'],mode=C.RTLD_GLOBAL);done()
  cuda.cudaSetDevice.argtypes=[C.c_int];cuda.cudaSetDevice.restype=C.c_int;cuda.cudaFree.argtypes=[C.c_void_p];cuda.cudaFree.restype=C.c_int
  def engine():
   phase('load NCCL');loaded['nccl']=C.CDLL(paths['libnccl.so.2'],mode=C.RTLD_GLOBAL);done()
   phase('load native CUDA library');loaded['engine']=C.CDLL(paths['libmgbfs_cuda.so'],mode=C.RTLD_GLOBAL);done()
  if config['order']=='engine-first':engine()
  for name,fn,args in [('cudaSetDevice',cuda.cudaSetDevice,[config['device']]),('cudaFree-null context',cuda.cudaFree,[None]),('repeat cudaSetDevice',cuda.cudaSetDevice,[config['device']])]:phase(name);done(fn(*args))
  if config['order']=='context-first':engine()
  phase('cuModuleGetLoadingMode');driver=C.CDLL('libcuda.so.1',mode=C.RTLD_GLOBAL);mode=C.c_int();fn=driver.cuModuleGetLoadingMode;fn.argtypes=[C.POINTER(C.c_int)];fn.restype=C.c_int;code=fn(C.byref(mode));state['actual_module_loading_mode']=mode.value;done(code)
  free=C.c_size_t();total=C.c_size_t();fn=cuda.cudaMemGetInfo;fn.argtypes=[C.POINTER(C.c_size_t)]*2;fn.restype=C.c_int
  phase('cudaMemGetInfo');code=fn(C.byref(free),C.byref(total));state['memory']={'free':free.value,'total':total.value};done(code)
  attributes={};fn=cuda.cudaDeviceGetAttribute;fn.argtypes=[C.POINTER(C.c_int),C.c_int,C.c_int];fn.restype=C.c_int
  for number,label in [(1,'max_threads_per_block'),(10,'warp_size'),(16,'multiprocessors'),(40,'async_engines'),(41,'unified_addressing'),(75,'compute_major'),(76,'compute_minor')]:
   phase('cudaDeviceGetAttribute '+label);v=C.c_int();code=fn(C.byref(v),number,config['device']);attributes[label]={'value':v.value,'code':code};state['attributes']=attributes;done(code)
  phase('full native hardware wrapper');values=(C.c_uint64*8)();fn=loaded['engine'].mgbfs_generic_hardware_info;fn.argtypes=[C.c_int,C.POINTER(C.c_uint64)];fn.restype=C.c_int;code=fn(config['device'],values);state['hardware']=list(values);done(code)
  phase('exact successor kernel');state['kernel']=_kernel(config['device']);done()
  phase('stream/event/graph and pinned-memory APIs');state['primitives']=primitives(cuda,phase,done);done()
  state['status']='PASS';write(root/'probe.json',state);return state
 finally:
  stop.set()
  if worker:worker.join(timeout=1)

def primitives(cuda,phase,done):
 result={};stream=C.c_void_p();event=C.c_void_p();memory=C.c_void_p();host=C.c_void_p();graph=C.c_void_p();executable=C.c_void_p()
 def invoke(name,types,args,required=True):
  phase(name);fn=getattr(cuda,name,None)
  if fn is None:
   result[name]={'status':'UNSUPPORTED','reason':'Symbol not exported'};done();return None
  fn.argtypes=types;fn.restype=C.c_int;code=fn(*args);result[name]={'code':code,'status':'PASS' if not code else 'FAIL'}
  if not required and code in (801,804):result[name]['status']='UNSUPPORTED';done();return code
  done(code);return code
 try:
  invoke('cudaStreamCreateWithFlags',[C.POINTER(C.c_void_p),C.c_uint],[C.byref(stream),1])
  invoke('cudaEventCreateWithFlags',[C.POINTER(C.c_void_p),C.c_uint],[C.byref(event),2])
  invoke('cudaMalloc',[C.POINTER(C.c_void_p),C.c_size_t],[C.byref(memory),PROBE_BYTES])
  invoke('cudaHostAlloc',[C.POINTER(C.c_void_p),C.c_size_t,C.c_uint],[C.byref(host),PROBE_BYTES,0])
  invoke('cudaMemsetAsync',[C.c_void_p,C.c_int,C.c_size_t,C.c_void_p],[memory,7,PROBE_BYTES,stream])
  invoke('cudaMemcpyAsync',[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int,C.c_void_p],[host,memory,PROBE_BYTES,2,stream])
  invoke('cudaEventRecord',[C.c_void_p,C.c_void_p],[event,stream]);invoke('cudaEventSynchronize',[C.c_void_p],[event])
  if bytes((C.c_ubyte*PROBE_BYTES).from_address(host.value))!=bytes([7])*PROBE_BYTES:raise RuntimeError('ASYNC_READBACK_MISMATCH')
  invoke('cudaGraphCreate',[C.POINTER(C.c_void_p),C.c_uint],[C.byref(graph),0])
  invoke('cudaGraphInstantiateWithFlags',[C.POINTER(C.c_void_p),C.c_void_p,C.c_ulonglong],[C.byref(executable),graph,0],False)
  if executable.value:invoke('cudaGraphLaunch',[C.c_void_p,C.c_void_p],[executable,stream])
  invoke('cudaStreamSynchronize',[C.c_void_p],[stream])
  result['readback_verified']=True;return result
 finally:
  for name,ptr in [('cudaGraphExecDestroy',executable),('cudaGraphDestroy',graph),('cudaFreeHost',host),('cudaFree',memory),('cudaEventDestroy',event),('cudaStreamDestroy',stream)]:
   if ptr.value:
    fn=getattr(cuda,name);fn.argtypes=[C.c_void_p];fn.restype=C.c_int;code=fn(ptr)
    if code:result[name]={'code':code,'status':'FAIL'}

def bfs_probe(config,root):
 from .native_distribution import native_runtime
 from .launch import run_graph
 from .generic_session import persistent_graph_workers
 graph=fixture(config['fixture']);layers,last=oracle(graph);_,env=native_runtime()
 env.update(MGBFS_GENERIC_HISTORY=config.get('history','hash'),MGBFS_GENERIC_GENERATOR=config.get('generator','cuda'),MGBFS_AUTOTUNE_CACHE_DIR=str(root/'tune-cache'))
 devices=config['devices'];kwargs=dict(devices=devices,capacity=256,shards=config.get('shards',1),autotune=config.get('autotune',False),max_seconds=30,backend=config.get('backend','generic'),transport=config.get('transport','full'),candidate_order=config.get('candidate_order','none'),_native_env=env)
 reports=[]
 def run(index):
  print('PHASE tiny BFS '+config['fixture']+' '+str(index),flush=True);output=root/('bfs-'+str(index));report=run_graph(graph,output,**kwargs)
  if report['status']!='COMPLETE' or report['layer_sizes']!=layers:raise RuntimeError('CPU_LAYER_ORACLE_MISMATCH')
  saved=json.loads((output/'states.json').read_text());payload=saved['current'] if isinstance(saved,dict) else saved
  if sorted(tuple(x) for x in payload)!=last:raise RuntimeError('CPU_TERMINAL_STATE_ORACLE_MISMATCH')
  reports.append({'status':report['status'],'layer_sizes':layers,'bfs_seconds':report['bfs_seconds'],'plan':report.get('plan'),'states_sha256':report['states_sha256']})
 if config.get('persistent'):
  with persistent_graph_workers():run(0);run(1)
 else:run(0)
 return {'cpu_layers_and_terminal_states_verified':True,'fixture':config['fixture'],'reports':reports}

def component(config,root):
 root=Path(root);os.environ['MGBFS_DIAGNOSTIC_TRACE_FILE']=str(root/'native-startup.log')
 for key in ('CUDA_MODULE_LOADING','CUDA_MODULE_DATA_LOADING'):
  setting=config.get('module' if key=='CUDA_MODULE_LOADING' else 'data_loading','inherited')
  if setting!='inherited':os.environ[key]=setting
 if config['kind']=='api':return api_probe(config,root)
 if config['kind']=='bfs':return bfs_probe(config,root)
 from .diagnostics import _component
 return _component(config['stage'],root,config.get('devices'),config.get('device'),config.get('seconds',30))

def run_case(runner,case,deadline):
 remaining=deadline-time.monotonic()-8
 if remaining<=.1:runner.skip(case['name'],'GLOBAL_DIAGNOSTIC_BUDGET_EXHAUSTED');return runner.report['steps'][-1]
 folder=runner.root/case['name'];folder.mkdir(exist_ok=True);write(folder/'case-input.json',case['config'])
 command=[sys.executable,'-m','multigpubfs.deep_diagnostics','--child',str(folder/'case-input.json'),'--output',str(folder)]
 if case.get('strace'):
  tool=shutil.which('strace')
  if not tool:runner.skip(case['name'],'strace is not installed; native checkpoints and /proc snapshots remain available');return runner.report['steps'][-1]
  command=[tool,'-f','-qq','-s','80','-e','trace=ioctl,futex,clone,clone3,mmap,openat,execve','-o',str(folder/'strace.log')]+command
 previous=runner.timeout;runner.timeout=min(case.get('seconds',30),remaining)
 try:entry=runner.run(case['name'],command,optional=case.get('optional',False))
 finally:runner.timeout=previous
 entry['diagnostic_case']=case['config']
 for name in ('probe.json','process-snapshot.json'):
  path=folder/name
  if path.is_file():
   try:entry.setdefault('partial_evidence',{})[name]=json.loads(path.read_text())
   except (OSError,ValueError):pass
 trace=folder/'native-startup.log'
 if trace.is_file():
  lines=trace.read_text(errors='replace').splitlines();entry['last_native_phase']=lines[-1] if lines else None
 runner.save();return entry

def summarize(steps):
 failures=[{'name':s['name'],'status':s['status'],'last_native_phase':s.get('last_native_phase'),'last_api':s.get('partial_evidence',{}).get('probe.json',{}).get('last_operation'),'error':s.get('error')} for s in steps if s['status'] in ('FAIL','TIMEOUT','WARNING')]
 comparisons=[]
 for s in steps:
  c=s.get('diagnostic_case',{})
  if c.get('kind')=='api':comparisons.append({'name':s['name'],'status':s['status'],'device':c['device'],'module':c['module'],'data':c['data_loading'],'order':c['order'],'thread':c['control_thread'],'last_api':s.get('partial_evidence',{}).get('probe.json',{}).get('last_operation')})
 return {'root_cause_confirmed':False,'scope':'Controlled observations; mode correlations do not prove a driver bug. Finite synthetic tests do not cover every possible graph or API parameter.','failures':failures,'startup_matrix':comparisons,'untested':[{'name':s['name'],'reason':s.get('reason')} for s in steps if s['status']=='SKIPPED']}

def run_deep(output=None,devices=None,timeout=60,total_timeout=1800):
 from .diagnostics import Runner
 import datetime
 if not math.isfinite(total_timeout) or total_timeout<1:raise ValueError('Total timeout must be positive and finite')
 if not math.isfinite(timeout) or timeout<=0:raise ValueError('Stage timeout must be positive and finite')
 if devices is not None and (not devices or len(set(devices))!=len(devices) or any(type(d) is not int or d<0 for d in devices)):raise ValueError('Devices must be distinct nonnegative IDs')
 root=Path(output or ('mgbfs-deep-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+str(os.getpid()))).absolute()
 runner=Runner(root,timeout);runner.deep_capture=True;runner.report.update(mode='deep',total_timeout_seconds=total_timeout,scope='isolated single-host synthetic test suite; no production BFS retry, reset, privilege changes or multi-node acceptance');deadline=time.monotonic()+total_timeout
 def run(name,stage,**kw):return run_case(runner,{'name':name,'config':dict(kind='component',stage=stage,**kw),'seconds':min(timeout,45)},deadline)
 try:
  run('system-info','system');run('filesystem','filesystem');run('environment-evidence','environment');run('nvml-evidence','nvml');run('torch-evidence','torch');run('driver-api','driver')
  runtime=run('native-runtime','runtime');count=run('cuda-discovery','count')
  if runtime['status']=='PASS' and count['status']=='PASS':
   selected=devices or list(range(count['result']['data']['visible_devices']))
   if any(d>=count['result']['data']['visible_devices'] for d in selected):runner.report['steps'].append({'name':'device-selection','status':'FAIL','error':'Selected GPU is not visible'})
   else:
    runner.report['devices']=selected
    for case in startup_cases(selected):run_case(runner,case,deadline)
    for device in selected:
     for stage in ('context','meminfo','hardware','memory','kernel','persistent'):
      run('gpu-'+str(device)+'-'+stage,stage,device=device,seconds=min(timeout,45))
     run_case(runner,{'name':f'deep-strace-gpu-{device}','strace':True,'config':{'kind':'component','stage':'persistent','device':device,'seconds':25},'seconds':30,'optional':True},deadline)
    if len(selected)>1:
     for transport,changes in [('inherited',{}),('socket',{'NCCL_P2P_DISABLE':'1','NCCL_SHM_DISABLE':'1','NCCL_IB_DISABLE':'1'}),('no-p2p',{'NCCL_P2P_DISABLE':'1'})]:
      case={'name':'deep-nccl-'+transport,'config':{'kind':'component','stage':'nccl_exchange','devices':selected,'environment_overrides':changes},'seconds':45}
      run_case(runner,case,deadline)
    # Change one execution dimension per bounded oracle case.
    base={'kind':'bfs','fixture':'permutation','devices':[selected[0]],'history':'hash','generator':'cuda','shards':1}
    variants=[('base',{}),('sorted',{'history':'sorted'}),('gemm',{'generator':'gemm'}),('parent',{'transport':'parent'}),('radix',{'candidate_order':'radix'}),('shards4',{'shards':4}),('persistent',{'persistent':True}),('tuner',{'autotune':True,'shards':None}),('repeated',{'fixture':'repeated'}),('matrix',{'fixture':'matrix'}),('matrix-gemm',{'fixture':'matrix-gemm','generator':'gemm'}),('wide',{'fixture':'wide'}),('shard-ab-hash',{'fixture':'lrx','backend':'shard_ab_hash'}),('shard-ab-sort',{'fixture':'lrx','backend':'shard_ab_sort_merge'})]
    if len(selected)>1:variants.extend([('multi',{'devices':selected}),('multi-sorted',{'devices':selected,'history':'sorted'}),('multi-parent',{'devices':selected,'transport':'parent'}),('multi-gemm',{'devices':selected,'generator':'gemm'}),('multi-persistent',{'devices':selected,'persistent':True}),('multi-shard-ab',{'devices':selected,'fixture':'lrx','backend':'shard_ab_hash'})])
    for label,changes in variants:run_case(runner,{'name':'deep-bfs-'+label,'config':dict(base,**changes),'seconds':min(timeout,45)},deadline)
  else:runner.skip('deep-gpu-suite','Native runtime or CUDA discovery failed')
 except KeyboardInterrupt:runner.report['steps'].append({'name':'interrupted','status':'FAIL','error':'Interrupted by caller'})
 except Exception as e:runner.report['steps'].append({'name':'deep-controller','status':'FAIL','error':str(e)})
 runner.report['deep_analysis']=summarize(runner.report['steps']);write(root/'deep-analysis.json',runner.report['deep_analysis'])
 rows=['name\tstatus\tlast native phase / API']
 for step in runner.report['steps']:rows.append(step['name']+'\t'+step['status']+'\t'+str(step.get('last_native_phase') or step.get('partial_evidence',{}).get('probe.json',{}).get('last_operation') or step.get('reason','')))
 (root/'deep-summary.txt').write_text('\n'.join(rows)+'\n',encoding='utf-8');return runner.finish()

def main(argv=None):
 parser=argparse.ArgumentParser();parser.add_argument('--child',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args(argv)
 root=args.output
 try:
  config=json.loads(Path(args.child).read_text())
  for k,v in config.get('environment_overrides',{}).items():
   if k not in ('NCCL_P2P_DISABLE','NCCL_SHM_DISABLE','NCCL_IB_DISABLE'):raise ValueError('Invalid diagnostic override')
   os.environ[k]=v
  result={'status':'PASS','data':component(config,root)};write(root/'result.json',result);return 0
 except Exception as e:
  import traceback
  result={'status':'FAIL','error':str(e),'exception_type':type(e).__name__,'traceback':traceback.format_exc()};print(result['traceback'],flush=True);write(root/'result.json',result);return 1

if __name__=='__main__':raise SystemExit(main())
