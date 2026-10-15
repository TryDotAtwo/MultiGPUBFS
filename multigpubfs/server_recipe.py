"""Find and explicitly launch a server-local, oracle-verified native BFS setup.

Recipes select only packaged builds and bounded allowlisted settings. They do
not repair a driver, change package defaults, or restart a production graph.
"""
import argparse,contextlib,hashlib,json,os,shlex,socket,sys,tempfile,time
from pathlib import Path
from .runtime_variants import PRODUCTION,SPECS,digest

FIELDS={'variant','devices','worker','module','data_loading','transport'}
FIXTURES=('permutation','repeated','matrix')

def overrides(settings):
 if set(settings)!=FIELDS:raise ValueError('SERVER_RECIPE_FIELDS')
 if settings['variant'] not in SPECS:raise ValueError('SERVER_RECIPE_BUILD')
 devices=settings['devices']
 if not isinstance(devices,list) or not devices or any(type(x) is not int or x<0 for x in devices) or len(set(devices))!=len(devices):raise ValueError('SERVER_RECIPE_DEVICES')
 if settings['worker'] not in ('persistent','fresh') or settings['transport'] not in ('inherited','socket'):raise ValueError('SERVER_RECIPE_MODE')
 env={'MGBFS_GENERIC_HISTORY':'hash','MGBFS_GENERIC_GENERATOR':'cuda'}
 for field,key in (('module','CUDA_MODULE_LOADING'),('data_loading','CUDA_MODULE_DATA_LOADING')):
  if settings[field] not in ('inherited','EAGER','LAZY'):raise ValueError('SERVER_RECIPE_LOADING')
  if settings[field]!='inherited':env[key]=settings[field]
 if settings['transport']=='socket':env.update(NCCL_P2P_DISABLE='1',NCCL_SHM_DISABLE='1',NCCL_IB_DISABLE='1')
 return env

def candidates(devices):
 # Exhaust common-device configurations before clearly labelled single-GPU fallbacks.
 groups=[list(devices)]+([[d] for d in devices] if len(devices)>1 else [])
 for group in groups:
  for build in (PRODUCTION,'static-sm120','shared-legacy','baseline-static-legacy'):
   for loading in ('inherited','LAZY','EAGER'):
    for transport in (('inherited','socket') if len(group)>1 else ('inherited',)):
     for worker in ('persistent','fresh'):
      yield dict(variant=build,devices=group,worker=worker,module=loading,data_loading=loading,transport=transport)

def accepted(step,settings):
 if step.get('status')!='PASS' or step.get('result',{}).get('status')!='PASS':return False
 if not step.get('partial_evidence',{}).get('cleanup.json',{}).get('complete'):return False
 data=step['result'].get('data',{})
 if data.get('settings')!=settings or data.get('cpu_oracles_verified') is not True:return False
 graphs=data.get('graphs',[])
 return set(x.get('fixture') for x in graphs)>=set(FIXTURES) and all(x.get('status')=='COMPLETE' and x.get('layers_verified') is True and x.get('terminal_states_verified') is True for x in graphs)

def server_identity():
 # Values of arbitrary environment variables are never copied into a recipe.
 env={k:v for k,v in os.environ.items() if k.startswith(('CUDA_','NCCL_','MGBFS_')) or k in ('LD_LIBRARY_PATH','LD_PRELOAD')}
 value={'host':socket.gethostname(),'python':str(Path(sys.executable).resolve()),'python_version':sys.version.split()[0],'environment_sha256':hashlib.sha256(json.dumps(env,sort_keys=True).encode()).hexdigest()}
 files=[Path('/proc/sys/kernel/random/boot_id'),Path('/proc/driver/nvidia/version')]
 directory=Path('/proc/driver/nvidia/gpus')
 try:files+=sorted(directory.glob('*/information'))[:128]
 except OSError:pass
 for path in files:
  try:value[str(path)]=hashlib.sha256(path.read_bytes()[:65536]).hexdigest()
  except OSError:value[str(path)]='unavailable'
 return value

def validate_identity(profile):
 if profile.get('server')!=server_identity():raise RuntimeError('SERVER_CHANGED: host, GPU/driver evidence, Python or CUDA/NCCL/loader environment changed; rerun doctor in this GPU job')

def code_digest():
 h=hashlib.sha256()
 for path in sorted(Path(__file__).parent.glob('*.py')):h.update(path.name.encode());h.update(path.read_bytes())
 return h.hexdigest()

def artifacts(verified,settings):
 selected=next(x for x in verified['variants'] if x['id']==settings['variant'])
 root=Path(verified['root'])
 return {'binary_sha256':verified['binary_sha256'],'cuda_library_sha256':selected['sha256'],'cudart_sha256':digest(root/'lib/libcudart.so.12'),'nccl_sha256':digest(root/'lib/libnccl.so.2'),'python_code_sha256':code_digest()}

def read_proof(profile,root):
 proof=profile.get('proof',{});path=(Path(root)/proof.get('path','')).resolve()
 if not path.is_relative_to(Path(root).resolve()) or not path.is_file() or path.is_symlink():raise RuntimeError('SERVER_RECIPE_PROOF_PATH')
 if digest(path)!=proof.get('sha256'):raise RuntimeError('SERVER_RECIPE_PROOF_CHECKSUM')
 result=json.loads(path.read_text(encoding='utf-8'))
 step={'status':'PASS','result':result,'partial_evidence':{'cleanup.json':{'complete':True}}}
 if not accepted(step,profile.get('settings')):raise RuntimeError('SERVER_RECIPE_PROOF_SETTINGS_OR_ORACLE')
 return result['data']

def read_profile(path,*,check_server=True):
 from .runtime_variants import inventory
 path=Path(path).absolute();value=json.loads(path.read_text(encoding='utf-8'))
 if value.get('schema')!=1 or value.get('status')!='VERIFIED':raise RuntimeError('SERVER_RECIPE_NOT_VERIFIED')
 overrides(value['settings']);read_proof(value,path.parent)
 cleanup=value.get('cleanup',{});proof=(path.parent/cleanup.get('path','')).resolve()
 if not proof.is_relative_to(path.parent.resolve()) or not proof.is_file() or digest(proof)!=cleanup.get('sha256') or json.loads(proof.read_text()).get('complete') is not True:raise RuntimeError('SERVER_RECIPE_CLEANUP_PROOF')
 if check_server:validate_identity(value)
 verified=inventory()
 if verified.get('status')!='AVAILABLE' or artifacts(verified,value['settings'])!=value.get('artifacts'):raise RuntimeError('SERVER_RECIPE_PACKAGE_CHANGED: rerun doctor after updating the installation')
 return value,verified

def execution_environment(settings,verified):
 from .runtime_variants import environment
 env=environment(settings['variant'],verified);env.update(overrides(settings))
 for name in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT','TORCHELASTIC_RUN_ID'):env.pop(name,None)
 return env

def command_argv(profile):
 return [sys.executable,'-m','multigpubfs.server_recipe','--profile',str(Path(profile).absolute()),'--check']

def command(profile):return shlex.join(command_argv(profile))

def probe(settings,root):
 """Run the actual selected launch path, including automatic admission/tuning."""
 from .deep_diagnostics import fixture,oracle,write
 from .launch import run_graph
 from .generic_session import persistent_graph_workers
 from .native_distribution import native_runtime
 from .runtime_variants import inventory,confirm_resolution
 from .diagnostics import _paths
 verified=inventory();_,_,libraries=_paths();resolution=confirm_resolution(settings['variant'],verified,libraries)
 overrides(settings);root=Path(root);root.mkdir(parents=True,exist_ok=True)
 _,env=native_runtime();env.update(overrides(settings));graphs=[]
 scope=persistent_graph_workers() if settings['worker']=='persistent' else contextlib.nullcontext()
 with scope:
  for index,name in enumerate(('permutation','permutation','repeated','matrix')):
   graph=fixture(name);layers,last=oracle(graph)
   print('PHASE server recipe exact '+name+' '+str(index)+' on GPUs '+str(settings['devices']),flush=True)
   output=root/('oracle-'+str(index))
   report=run_graph(graph,output,devices=settings['devices'],max_seconds=30,backend='generic',autotune=True,_native_env=env)
   if report.get('runtime_identity',{}).get('cuda_library_sha256')!=resolution['sha256']:raise RuntimeError('SERVER_RECIPE_LIBRARY_IDENTITY')
   if report['status']!='COMPLETE' or report['layer_sizes']!=layers:raise RuntimeError('SERVER_RECIPE_CPU_LAYERS_MISMATCH')
   saved=json.loads((output/'states.json').read_text());rows=saved['current'] if isinstance(saved,dict) else saved
   if sorted(tuple(x) for x in rows)!=last:raise RuntimeError('SERVER_RECIPE_CPU_TERMINAL_STATES_MISMATCH')
   graphs.append({'fixture':name,'status':report['status'],'layers_verified':True,'terminal_states_verified':True,'layer_sizes':layers,'states_sha256':report['states_sha256'],'bfs_seconds':report['bfs_seconds'],'runtime_identity':report.get('runtime_identity')})
 result={'cpu_oracles_verified':True,'settings':settings,'graphs':graphs,'scope':'Exact small permutation, repeated-symbol and matrix graphs; startup and functional acceptance, not large-graph capacity or throughput optimum'}
 write(root/'probe.json',result);return result

def find_working(runner,devices,deadline):
 from .runtime_variants import inventory
 from .deep_diagnostics import run_case,write
 def result(status,reason):return {'status':status,'reason':reason,'requested_devices':list(devices)}
 if getattr(runner,'deep_cleanup_blocked',False):return result('BLOCKED','Owned diagnostic processes remain; no further GPU work or ready command')
 if deadline-time.monotonic()<12:return result('INCOMPLETE','No time remains for exact BFS and verification of the printed launch command')
 verified=getattr(runner,'diagnostic_variants',None) or inventory()
 if verified.get('status')!='AVAILABLE':return result('UNAVAILABLE',verified.get('reason','No checksummed packaged build inventory'))
 runner.diagnostic_variants=verified
 identity=server_identity();remaining=deadline-time.monotonic();group_deadline=time.monotonic()+remaining*.7
 profile_path=runner.root/'working-config.json';attempts=[]
 for index,settings in enumerate(candidates(devices)):
  if getattr(runner,'deep_cleanup_blocked',False):return result('BLOCKED','Owned diagnostic process could not be reaped; no ready command')
  if deadline-time.monotonic()<12:break
  if len(devices)>1 and settings['devices']==list(devices) and time.monotonic()>group_deadline:continue
  name='server-recipe-'+str(index).zfill(3)
  config=dict(kind='server-recipe',settings=settings,variant=settings['variant'],module=settings['module'],data_loading=settings['data_loading'])
  if settings['transport']=='socket':config['environment_overrides']={k:v for k,v in overrides(settings).items() if k.startswith('NCCL_')}
  step=run_case(runner,{'name':name,'config':config,'seconds':min(60,runner.timeout)},deadline)
  attempts.append({'name':name,'settings':settings,'status':step['status']})
  if not accepted(step,settings):continue
  folder=runner.root/name;value={'schema':1,'status':'VERIFIED','settings':settings,'server':identity,'artifacts':artifacts(verified,settings),'proof':{'path':str((folder/'result.json').relative_to(runner.root)),'sha256':digest(folder/'result.json')},'cleanup':{'path':str((folder/'cleanup.json').relative_to(runner.root)),'sha256':digest(folder/'cleanup.json')},'scope':'Server-local single-host recipe; new job/environment/driver/package requires validation again; future graph capacity is not guaranteed'}
  write(profile_path,value)
  # Exercise the public entry point in a new process before recommending it.
  check_name=name+'-command';check_folder=runner.root/check_name
  previous=runner.timeout;runner.timeout=min(65,max(.1,deadline-time.monotonic()-8))
  try:check=runner.run(check_name,command_argv(profile_path)+['--output',str(check_folder)])
  finally:runner.timeout=previous
  cleanup=check_folder/'cleanup.json'
  if cleanup.is_file():check.setdefault('partial_evidence',{})['cleanup.json']=json.loads(cleanup.read_text())
  if accepted(check,settings):
   launch=command(profile_path)
   compute=shlex.join(command_argv(profile_path)[:-1]+['--catalog','PermutationGroups.coxeter','--catalog-kwargs','{"n":11}','--output','bfs-S11'])
   (runner.root/'server-command.txt').write_text('Verified small BFS check:\n'+launch+'\n\nExample real calculation (change graph parameters and output path):\n'+compute+'\n',encoding='utf-8')
   return {'status':'READY','selected':settings,'requested_devices':list(devices),'single_gpu_fallback':settings['devices']!=list(devices),'profile':str(profile_path),'command':launch,'calculation_example':compute,'proof_steps':[name,check_name],'attempts':attempts,'scope':'The printed launcher and exact small BFS passed on this server and selected GPU set; this does not prove fit or speed of every graph'}
  profile_path.unlink(missing_ok=True)
 runner.save()
 status='BLOCKED' if getattr(runner,'deep_cleanup_blocked',False) else 'INCOMPLETE' if deadline-time.monotonic()<12 else 'NOT_FOUND'
 return dict(result(status,'No candidate and its public launch command both passed exact BFS within the diagnostic budget'),attempts=attempts)

def main(argv=None):
 parser=argparse.ArgumentParser(description='Explicitly launch the server-local configuration verified by doctor; no driver changes or production retries')
 parser.add_argument('--profile',type=Path,required=True);parser.add_argument('--check',action='store_true')
 source=parser.add_mutually_exclusive_group();source.add_argument('--graph',type=Path);source.add_argument('--catalog')
 parser.add_argument('--catalog-args',default='[]');parser.add_argument('--catalog-kwargs',default='{}');parser.add_argument('--output',type=Path);parser.add_argument('--seconds',type=int,default=3600);parser.add_argument('--_child',action='store_true',help=argparse.SUPPRESS)
 args=parser.parse_args(argv)
 try:
  if args.seconds<1:raise ValueError('Seconds must be positive')
  if args.check and (args.graph is not None or args.catalog is not None):raise ValueError('Check mode does not accept a production graph')
  if not args.check and args.graph is None and args.catalog is None:raise ValueError('Choose --check, --graph or --catalog')
  profile,verified=read_profile(args.profile,check_server=not args._child)
  if not args._child:
   # exec preserves the diagnostic supervisor's owned process group. Every native
   # worker is still reaped by its one supervisor; no detached nested group.
   env=execution_environment(profile['settings'],verified)
   forwarded=list(sys.argv[1:] if argv is None else argv)
   os.execvpe(sys.executable,[sys.executable,'-m','multigpubfs.server_recipe',*forwarded,'--_child'],env)
   raise RuntimeError('SERVER_RECIPE_EXEC_DID_NOT_REPLACE_PROCESS')
  root=Path(args.output or tempfile.mkdtemp(prefix='mgbfs-verified-check-')).absolute()
  if args.check:
   root.mkdir(parents=True,exist_ok=True);data=probe(profile['settings'],root)
  else:
   if args.output is None:raise ValueError('A calculation requires a new --output directory')
   from .launch import run_graph
   from .generic_session import persistent_graph_workers
   from .graph_definition import GraphDefinition
   from .catalog import from_catalog
   graph=GraphDefinition.from_dict(json.loads(args.graph.read_text(encoding='utf-8'))) if args.graph else from_catalog(args.catalog,args=json.loads(args.catalog_args),kwargs=json.loads(args.catalog_kwargs))
   settings=profile['settings'];scope=persistent_graph_workers() if settings['worker']=='persistent' else contextlib.nullcontext()
   with scope:data=run_graph(graph,root,devices=settings['devices'],max_seconds=args.seconds,backend='generic',autotune=True)
  result={'status':'PASS','data':data}
  (root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
  return 0 if args.check or data['status']=='COMPLETE' else 2
 except (OSError,ValueError,RuntimeError,KeyError,TypeError) as error:
  print('SERVER_RECIPE_ERROR '+str(error),file=sys.stderr,flush=True);return 1

if __name__=='__main__':raise SystemExit(main())
