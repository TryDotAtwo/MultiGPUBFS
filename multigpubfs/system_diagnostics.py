"""Bounded user-space environment evidence. No allocation changes or privileged operations."""
import ctypes as C,errno,importlib.util,json,os,platform,resource,shutil,socket,stat,struct,subprocess,sys,tempfile,time
from pathlib import Path
from .driver_diagnostics import _read,_resolved,_phase

def limits():
 result={}
 for name in ('RLIMIT_AS','RLIMIT_NOFILE','RLIMIT_NPROC','RLIMIT_MEMLOCK','RLIMIT_FSIZE'):
  if hasattr(resource,name):result[name]=list(resource.getrlimit(getattr(resource,name)))
 return result

def device_access(path):
 item={'path':str(path)}
 try:
  s=path.stat();item.update(exists=True,character_device=stat.S_ISCHR(s.st_mode),major=os.major(s.st_rdev),minor=os.minor(s.st_rdev),mode=oct(stat.S_IMODE(s.st_mode)),owner_uid=s.st_uid,owner_gid=s.st_gid,read_access=os.access(path,os.R_OK),write_access=os.access(path,os.W_OK))
 except OSError as error:item.update(exists=False if error.errno==errno.ENOENT else None,error=str(error),errno=error.errno);return item
 if path.name in ('nvidiactl','nvidia-uvm') and item['character_device']:
  _phase('open and immediately close '+str(path)+'; no ioctl or GPU allocation')
  try:
   fd=os.open(path,os.O_RDWR|os.O_NONBLOCK|getattr(os,'O_CLOEXEC',0));os.close(fd);item['open']={'status':'PASS'}
  except OSError as error:item['open']={'status':'FAIL','errno':error.errno,'error':str(error)}
 return item

def elf(path):
 try:
  with path.open('rb') as f:header=f.read(20)
  if len(header)<20 or header[:4]!=b'\x7fELF':return {'valid':False,'reason':'not an ELF header'}
  endian='<' if header[5]==1 else '>';return {'valid':True,'bits':64 if header[4]==2 else 32,'machine':struct.unpack(endian+'H',header[18:20])[0]}
 except OSError as e:return {'valid':False,'error':str(e)}

def library_candidates():
 dirs=[Path(x) for x in os.environ.get('LD_LIBRARY_PATH','').split(':') if x]
 dirs += [Path(x) for x in ('/usr/lib64','/usr/lib','/usr/lib/x86_64-linux-gnu','/usr/lib64/nvidia','/usr/lib/x86_64-linux-gnu/nvidia/current','/usr/local/nvidia/lib64')]
 for root in Path('/usr/local').glob('cuda*'):dirs += [root/'compat',root/'lib64/stubs']
 cuda=os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
 if cuda:dirs += [Path(cuda)/'compat',Path(cuda)/'lib64/stubs']
 files=[];seen=set()
 for directory in dirs[:48]:
  try:matches=directory.glob('libcuda.so*')
  except OSError:continue
  try:
   for path in matches:
    if len(files)>=32:return files
    resolved=path.resolve()
    if resolved in seen or not resolved.is_file():continue
    seen.add(resolved);s=resolved.stat();files.append({'path':str(path),'resolved':str(resolved),'size':s.st_size,'inode':s.st_ino,'device':s.st_dev,'mtime_ns':s.st_mtime_ns,'elf':elf(resolved)})
  except OSError:continue
 return files

def cgroups():
 raw=_read('/proc/self/cgroup');out={'membership':raw,'device_policy_note':'cgroup v2 device BPF rules cannot be inferred from a missing devices.list; permission remains unknown without an actual operation.'}
 if raw['status']!='READ':return out
 records=[]
 for line in raw['text'].splitlines()[:16]:
  parts=line.split(':',2)
  if len(parts)!=3:continue
  ident,controllers,relative=parts;base=Path('/sys/fs/cgroup')
  if ident!='0':base/=controllers.split(',')[0]
  target=(base/relative.lstrip('/')).resolve()
  try:target.relative_to(base.resolve())
  except ValueError:continue
  item={'version':2 if ident=='0' else 1,'controllers':controllers,'path':str(target),'files':{}}
  for name in ('memory.max','memory.current','memory.events','memory.limit_in_bytes','pids.max','pids.current','devices.list'):
   value=_read(target/name,4096)
   if value['status']=='READ':item['files'][name]=value['text'].strip()
  records.append(item)
 out['records']=records;return out

def mount_for(path):
 data=_read('/proc/self/mountinfo',131072);best=None
 if data['status']!='READ':return data
 path=str(Path(path).resolve())
 for line in data['text'].splitlines():
  fields=line.split();
  if len(fields)<10 or '-' not in fields:continue
  point=fields[4].replace('\040',' ');sep=fields.index('-')
  if path==point or path.startswith(point.rstrip('/')+'/'):
   if best is None or len(point)>len(best['mountpoint']):best={'mountpoint':point,'options':fields[5].split(','),'filesystem':fields[sep+1]}
 return best or {'status':'UNKNOWN'}

def survey(root):
 data={'schema':1,'status':'RUNNING','python':sys.executable,'python_version':platform.python_version(),'machine':platform.machine(),'uid':os.geteuid(),'resource_limits':limits(),'environment':{k:os.environ[k][:4096] for k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','NVIDIA_VISIBLE_DEVICES','NVIDIA_DRIVER_CAPABILITIES','LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT','CUDA_HOME','CUDA_PATH','TMPDIR','NCCL_P2P_DISABLE','NCCL_SHM_DISABLE','NCCL_IB_DISABLE','NCCL_SOCKET_IFNAME','NCCL_NET','SLURM_GPUS_ON_NODE','SLURM_JOB_GPUS','SLURM_STEP_GPUS','SLURM_JOB_GRES') if k in os.environ}}
 def save():
  from .diagnostics import _write
  _write(root/'environment.json',data)
 save();_phase('read process security and cgroup metadata')
 status=_read('/proc/self/status');data['process_security']={}
 if status['status']=='READ':
  for line in status['text'].splitlines():
   key,_,value=line.partition(':')
   if key in ('NoNewPrivs','Seccomp','Seccomp_filters','CapEff','Threads','VmRSS','VmPeak'):data['process_security'][key]=value.strip()
 data['lsm_context']=_read('/proc/self/attr/current',1024);data['cgroups']=cgroups();data['container_markers']={p:Path(p).exists() for p in ('/.dockerenv','/run/.containerenv')};save()
 _phase('record NVIDIA device-node permission evidence');data['device_nodes']=[]
 nodes=[Path('/dev/nvidiactl'),Path('/dev/nvidia-uvm')]
 try:nodes += sorted(Path('/dev').glob('nvidia[0-9]*'))[:32]+sorted(Path('/dev/nvidia-caps').glob('nvidia-cap*'))[:32]
 except OSError as error:data['device_listing_error']=str(error)
 for path in nodes:data['device_nodes'].append(device_access(path));save()
 _phase('inventory bounded existing driver library candidates and ELF headers');data['driver_candidates']=library_candidates();save()
 manifest=Path(__file__).parent/'_native/manifest.json'
 try:
  declared=json.loads(manifest.read_text());data['native_build_declaration']={k:declared.get(k) for k in ('architectures','cuda_toolkit','source_commit','package_source_commit','elf_runtime_requirements')};data['native_executable_mount']=mount_for(manifest.parent/'bin/mgbfs')
 except (OSError,ValueError):data['native_build_declaration']={'status':'UNAVAILABLE'}
 data['filesystems']={}
 for path in (str(root),tempfile.gettempdir(),'/dev/shm'):
  item={'mount':mount_for(path)}
  try:
   disk=shutil.disk_usage(path);item['free_bytes']=disk.free
   with tempfile.TemporaryFile(dir=path) as f:f.write(b'mgbfs-doctor');f.flush();f.seek(0);item['write_read_verified']=f.read()==b'mgbfs-doctor'
  except OSError as error:item.update(error=str(error),errno=error.errno)
  data['filesystems'][path]=item;save()
 _phase('bounded local TCP bind/connect without external traffic')
 try:
  with socket.socket() as server,socket.socket() as client:
   server.settimeout(2);client.settimeout(2);server.bind(('127.0.0.1',0));server.listen(1);client.connect(server.getsockname());conn,_=server.accept()
   with conn:conn.settimeout(2);conn.sendall(b'OK');assert client.recv(2,socket.MSG_WAITALL)==b'OK'
  data['loopback']={'status':'PASS'}
 except (OSError,AssertionError) as error:data['loopback']={'status':'FAIL','error':str(error),'errno':getattr(error,'errno',None)}
 data['network_interfaces']=[]
 try:
  for path in Path('/sys/class/net').iterdir():data['network_interfaces'].append(path.name)
 except OSError as error:data['network_interfaces_error']=str(error)
 tools=[shutil.which('systemctl')];data['fabric_manager']={'status':'UNAVAILABLE','scope':'Daemon status does not prove Fabric Manager is required on this hardware.'}
 if tools[0]:
  try:
   state=subprocess.run([tools[0],'is-active','nvidia-fabricmanager'],capture_output=True,text=True,timeout=2);data['fabric_manager'].update(status='OBSERVED',exit_code=state.returncode,state=state.stdout.strip()[:256],error=state.stderr.strip()[:1024])
  except (OSError,subprocess.TimeoutExpired) as error:data['fabric_manager']['error']=str(error)
 data['status']='PASS';data['limitations']=['Security flags or cgroup membership alone do not prove a denial.','No device ioctl, GPU reset, scheduler modification, privileged BPF query or visibility-mask change is performed.'];save();return data

def nvml():
 _phase('NVML driver-version/management count; independent of CUDA and nvidia-smi')
 try:lib=C.CDLL('libnvidia-ml.so.1')
 except OSError as error:return {'status':'UNAVAILABLE','error':str(error)}
 data={'library':None}
 try:
  lib.nvmlInit_v2.restype=C.c_int;lib.nvmlErrorString.argtypes=[C.c_int];lib.nvmlErrorString.restype=C.c_char_p;data['library']=_resolved(lib.nvmlInit_v2);code=lib.nvmlInit_v2();data['initialization_code']=code
  if code:return dict(data,status='FAIL',error=lib.nvmlErrorString(code).decode(errors='replace'))
  try:
   buf=C.create_string_buffer(96);lib.nvmlSystemGetDriverVersion.argtypes=[C.c_void_p,C.c_uint];lib.nvmlSystemGetDriverVersion.restype=C.c_int;code=lib.nvmlSystemGetDriverVersion(buf,len(buf));data['driver_version_code']=code
   if code==0:data['driver_version']=buf.value.decode(errors='replace')
   count=C.c_uint();lib.nvmlDeviceGetCount_v2.argtypes=[C.POINTER(C.c_uint)];lib.nvmlDeviceGetCount_v2.restype=C.c_int;code=lib.nvmlDeviceGetCount_v2(C.byref(count));data['device_count_code']=code
   if code==0:data['management_device_count']=count.value
   data['status']='PASS';data['scope']='Management count is not CUDA visibility or proof of permission to use those GPUs.';return data
  finally:lib.nvmlShutdown()
 except (AttributeError,OSError) as error:return dict(data,status='FAIL',error=str(error))

def torch_probe():
 if importlib.util.find_spec('torch') is None:return {'status':'UNAVAILABLE','reason':'PyTorch is not installed; native BFS does not require it.'}
 for group in cgroups().get('records',[]):
  files=group.get('files',{});maximum=files.get('memory.max','max');current=files.get('memory.current','0')
  if maximum.isdigit() and current.isdigit() and int(maximum)-int(current)<768*(1<<20):return {'status':'UNAVAILABLE','reason':'Less than 768 MiB of observed cgroup memory headroom; optional Torch import skipped to avoid exhausting the job. This is not a measured Torch peak requirement.'}
 _phase('import installed PyTorch in the original client environment')
 import torch
 data={'torch_version':str(torch.__version__),'torch_cuda_build':torch.version.cuda}
 if torch.version.cuda is None:return dict(data,status='UNAVAILABLE',reason='CPU-only Torch build, not a native BFS failure.')
 try:
  _phase('Torch CUDA discovery and context initialization; no tensor allocations');data['available']=torch.cuda.is_available();data['visible_devices']=torch.cuda.device_count();torch.cuda.init();data['status']='PASS'
 except Exception as error:data.update(status='FAIL',error=str(error),exception_type=type(error).__name__)
 return data

def _result(root,name):
 try:return json.loads((root/name/'result.json').read_text()).get('data',{})
 except (OSError,ValueError):return {}

def _valid_contrast(env,baseline):
 return all(env.get(k)==baseline.get(k) for k in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER'))

def contrasts(root,seconds):
 parent=root.parent;base=_result(parent,'driver-api');envdata=_result(parent,'environment-evidence')
 try:discovery=json.loads((parent/'cuda-discovery'/'result.json').read_text()).get('status')
 except (OSError,ValueError):discovery=None
 if base.get('status')=='PASS' and discovery=='PASS':return {'status':'UNAVAILABLE','reason':'Baseline CUDA discovery passed; no loader changes are tested.'}
 from .native_distribution import native_runtime
 try:native,original=native_runtime()
 except Exception as error:return {'status':'UNAVAILABLE','reason':'Native environment unavailable: '+str(error)}
 variants=[]
 for key in ('LD_PRELOAD','LD_AUDIT'):
  if original.get(key):env=dict(original);env.pop(key);variants.append((f'without-{key}',env,[],[key]))
 selected=base.get('driver',{}).get('resolved_library');seen=set()
 for candidate in envdata.get('driver_candidates',[]):
  file=candidate['resolved'];header=candidate.get('elf',{})
  if file==selected or file in seen or header.get('bits')!=64 or header.get('machine')!=62 or '/stubs/' in file:continue
  seen.add(file);folder=root/('candidate-'+str(len(seen)));folder.mkdir();(folder/'libcuda.so.1').symlink_to(file)
  env=dict(original);env['LD_LIBRARY_PATH']=str(folder)+':'+str(Path(file).parent)+':'+env.get('LD_LIBRARY_PATH','');variants.append(('existing-driver-'+str(len(seen)),env,[],['LD_LIBRARY_PATH']))
  if len(seen)>=2:break
 deadline=time.monotonic()+min(30,max(1,seconds*.65));results=[]
 def run(name,env,extra,changed):
  assert _valid_contrast(env,original),'Diagnostic contrast must preserve GPU allocation/visibility'
  folder=root/name;folder.mkdir();out=folder/'driver.json';item={'name':name,'changed_environment_keys':changed,'status':'RUNNING','visibility_preserved':True}
  _phase('loader contrast '+name+'; visibility and GPU allocation unchanged')
  try:
   time_left=deadline-time.monotonic()
   if time_left<.3:return dict(item,status='SKIPPED',reason='Matrix budget exhausted')
   process=subprocess.run([sys.executable,'-m','multigpubfs.driver_diagnostics','--output',str(out)]+extra,env=env,capture_output=True,text=True,timeout=min(6,time_left));(folder/'probe.log').write_text((process.stdout+process.stderr)[-24000:]);item['exit_code']=process.returncode
   if out.is_file():item['driver_evidence']=json.loads(out.read_text())
   time_left=deadline-time.monotonic()
   if time_left<.3:return dict(item,status='INCOMPLETE',reason='No budget for native discovery confirmation')
   process=subprocess.run([native,'graph-count'],env=env,capture_output=True,text=True,timeout=min(6,time_left));item['native_discovery']={'exit_code':process.returncode,'stdout':process.stdout[-4000:],'stderr':process.stderr[-4000:]};item['status']='PASS' if process.returncode==0 and item.get('driver_evidence',{}).get('status')=='PASS' else 'FAIL'
  except subprocess.TimeoutExpired as error:item.update(status='TIMEOUT',error=str(error))
  except (OSError,ValueError) as error:item.update(status='FAIL',error=str(error))
  return item
 if not variants:return {'status':'UNAVAILABLE','reason':'No eligible distinct driver libraries or active loader overrides were found.'}
 for name,env,extra,changed in variants[:4]:
  results.append(run(name,env,extra,changed))
  if results[-1]['status']=='PASS':results.append(run('baseline-recheck',original,[],[]));break
 return {'status':'PASS','scope':'Bounded discovery-only contrast; no BFS, no permanent environment changes or privilege escalation. A successful query does not establish kernel/NCCL acceptance.','results':results,'reason':'No eligible distinct libraries or loader overrides found' if not variants else None}

def loader_trace(root,seconds):
 parent=root.parent
 try:result=json.loads((parent/'cuda-discovery'/'result.json').read_text())
 except (OSError,ValueError):result={}
 if result.get('status')=='PASS':return {'status':'UNAVAILABLE','reason':'Discovery passed; no loader trace needed.'}
 from .native_distribution import native_runtime
 try:native,env=native_runtime()
 except Exception as error:return {'status':'UNAVAILABLE','reason':str(error)}
 env=dict(env,LD_DEBUG='libs,versions');_phase('trace actual native graph-count loader; no BFS')
 try:
  process=subprocess.run([native,'graph-count'],env=env,text=True,capture_output=True,timeout=min(6,max(.1,seconds*.5)))
  lines=[line for line in process.stderr.splitlines() if any(key in line for key in ('libcuda','libnvidia','libcudart','libmgbfs','libnccl','not found','undefined symbol','error:'))]
  text='\n'.join(lines[:160]);(root/'loader.log').write_text(text);return {'status':'PASS','native_exit_code':process.returncode,'filtered_trace':text[-24000:],'stderr_tail':process.stderr[-6000:],'truncated':len(lines)>160,'scope':'Expected native failure is preserved; trace collection success is not CUDA acceptance.'}
 except subprocess.TimeoutExpired as error:return {'status':'FAIL','error':str(error)}
