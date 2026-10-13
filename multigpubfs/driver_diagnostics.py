"""Read-only CUDA driver evidence, independent of nvidia-smi or GPU kernels."""
import argparse,ctypes as C,ctypes.util,json,os,re,stat
from pathlib import Path

def _phase(text):print('PHASE '+text,flush=True)

def _read(path,limit=8192):
 try:
  with Path(path).open(errors='replace') as f:return {'status':'READ','text':f.read(limit)}
 except OSError as e:return {'status':'UNAVAILABLE','error':str(e)}

def _resolved(fn):
 class Info(C.Structure):_fields_=[('name',C.c_char_p),('base',C.c_void_p),('symbol',C.c_char_p),('address',C.c_void_p)]
 try:
  dl=C.CDLL(ctypes.util.find_library('dl') or None);dl.dladdr.argtypes=[C.c_void_p,C.POINTER(Info)];dl.dladdr.restype=C.c_int;info=Info()
  if dl.dladdr(C.cast(fn,C.c_void_p),C.byref(info)) and info.name:return str(Path(os.fsdecode(info.name)).resolve())
 except (AttributeError,OSError):pass
 return None

def _driver_status(lib,code):
 result={'code':code}
 for fn,key in (('cuGetErrorName','name'),('cuGetErrorString','description')):
  try:
   f=getattr(lib,fn);f.argtypes=[C.c_int,C.POINTER(C.c_char_p)];f.restype=C.c_int;value=C.c_char_p()
   if f(code,C.byref(value))==0 and value.value:result[key]=value.value.decode(errors='replace')
  except (AttributeError,OSError):pass
 return result

def _driver(checkpoint=lambda data:None):
 _phase('load libcuda.so.1 through the native process loader environment')
 try:lib=C.CDLL('libcuda.so.1')
 except OSError as e:return {'loaded':False,'error':str(e)}
 data={'loaded':True}
 try:
  lib.cuInit.argtypes=[C.c_uint];lib.cuInit.restype=C.c_int;data['resolved_library']=_resolved(lib.cuInit);checkpoint(data)
  lib.cuDriverGetVersion.argtypes=[C.POINTER(C.c_int)];lib.cuDriverGetVersion.restype=C.c_int;version=C.c_int()
  _phase('cuDriverGetVersion');data['version_query']=_driver_status(lib,lib.cuDriverGetVersion(C.byref(version)))
  if data['version_query']['code']==0:data['api_version']=version.value
  checkpoint(data)
  _phase('cuInit');data['initialization']=_driver_status(lib,lib.cuInit(0));checkpoint(data)
  if data['initialization']['code']==0:
   lib.cuDeviceGetCount.argtypes=[C.POINTER(C.c_int)];lib.cuDeviceGetCount.restype=C.c_int;count=C.c_int()
   _phase('cuDeviceGetCount');data['device_query']=_driver_status(lib,lib.cuDeviceGetCount(C.byref(count)))
   if data['device_query']['code']==0:
    data['visible_devices']=count.value;data['devices']=[]
    lib.cuDeviceGetName.argtypes=[C.c_void_p,C.c_int,C.c_int];lib.cuDeviceGetName.restype=C.c_int
    lib.cuDeviceGetAttribute.argtypes=[C.POINTER(C.c_int),C.c_int,C.c_int];lib.cuDeviceGetAttribute.restype=C.c_int
    for device in range(min(64,count.value)):
     name=C.create_string_buffer(256);item={'logical_id':device};code=lib.cuDeviceGetName(name,len(name),device)
     if code==0:item['name']=name.value.decode(errors='replace')
     caps=[]
     for attribute in (75,76):
      value=C.c_int();code=lib.cuDeviceGetAttribute(C.byref(value),attribute,device);caps.append(value.value if code==0 else None)
     if None not in caps:item['compute_capability']=caps
     data['devices'].append(item)
 except (AttributeError,OSError) as e:data['probe_error']=str(e)
 return data

def _runtime(checkpoint=lambda data:None):
 _phase('CUDA runtime version and independent cudaGetDeviceCount')
 try:lib=C.CDLL('libcudart.so.12')
 except OSError as e:return {'loaded':False,'error':str(e)}
 data={'loaded':True}
 try:
  lib.cudaGetErrorString.argtypes=[C.c_int];lib.cudaGetErrorString.restype=C.c_char_p
  for fn,key in (('cudaRuntimeGetVersion','runtime_version'),('cudaDriverGetVersion','driver_version'),('cudaGetDeviceCount','visible_devices')):
   f=getattr(lib,fn);f.argtypes=[C.POINTER(C.c_int)];f.restype=C.c_int;value=C.c_int();code=f(C.byref(value));description=lib.cudaGetErrorString(code)
   data[fn]={'code':code,'description':description.decode(errors='replace') if description else ''}
   if code==0:data[key]=value.value
   checkpoint(data)
  data['resolved_library']=_resolved(lib.cudaGetDeviceCount)
 except (AttributeError,OSError) as e:data['probe_error']=str(e)
 return data

def classify(data):
 d=data['driver'];r=data['runtime'];init=d.get('initialization',{});count=d.get('visible_devices',0);runtime_count=r.get('cudaGetDeviceCount',{});path=d.get('resolved_library','') or ''
 if not d.get('loaded'):return 'DRIVER_LIBRARY_UNAVAILABLE','libcuda.so.1 could not be loaded. GPU allocation, container driver exposure and loader paths remain to be checked.'
 if init.get('name')=='CUDA_ERROR_STUB_LIBRARY' or init.get('code')==34:return 'DRIVER_STUB_LOADED','A CUDA driver stub was loaded instead of a usable NVIDIA driver.'
 if init.get('name')=='CUDA_ERROR_SYSTEM_DRIVER_MISMATCH':return 'DRIVER_KERNEL_LIBRARY_MISMATCH','Driver API reports a mismatch between the loaded driver library and system driver.'
 if init.get('code') not in (None,0):return 'DRIVER_INITIALIZATION_FAILED','cuInit failed: '+init.get('name',str(init.get('code')))+'; inspect driver/device-node evidence. This does not establish an old driver by itself.'
 if 'probe_error' in d:return 'DRIVER_API_PROBE_FAILED',d['probe_error']
 if d.get('device_query',{}).get('code') not in (None,0):return 'DRIVER_DEVICE_QUERY_FAILED','Driver initialized but cuDeviceGetCount failed.'
 if count==0:return 'NO_VISIBLE_GPU','Driver API found no GPU visible to this process. Check the selected GPU allocation and CUDA_VISIBLE_DEVICES.'
 if not r.get('loaded'):return 'CUDA_RUNTIME_UNAVAILABLE','Driver API sees GPUs, but the native CUDA runtime cannot be loaded.'
 if 'probe_error' in r:return 'RUNTIME_API_PROBE_FAILED',r['probe_error']
 if runtime_count.get('code')==35:
  dv=d.get('api_version',0);rv=r.get('runtime_version',0)
  if dv and rv and dv//1000<rv//1000:return 'DRIVER_MAJOR_TOO_OLD','Driver API version '+str(dv)+' is from an older CUDA major family than runtime '+str(rv)+'; cudaGetDeviceCount returned insufficient-driver error 35.'
  return 'RUNTIME_DRIVER_COMPATIBILITY_FAILED','Driver API sees GPUs, but CUDA runtime reports insufficient-driver error 35. A lower minor API version alone does not prove incompatibility.'
 if runtime_count.get('code') not in (None,0):return 'RUNTIME_DEVICE_QUERY_FAILED','CUDA runtime device query failed: '+runtime_count.get('description',str(runtime_count.get('code')))
 if r.get('visible_devices')!=count:return 'DRIVER_RUNTIME_DEVICE_COUNT_MISMATCH','Driver and runtime report different visible GPU counts.'
 return 'DRIVER_RUNTIME_READY','Driver and runtime both initialize and agree on the visible GPU count. Native kernels/NCCL are checked separately.'

def probe(output=None):
 data={'schema':1,'scope':'read-only driver/runtime evidence, not a kernel or NCCL test','environment':{k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','LD_LIBRARY_PATH','LD_PRELOAD') if k in os.environ},'scheduler_variables_present':[k for k in ('SLURM_JOB_ID','SLURM_JOB_GPUS','PBS_JOBID','LSB_JOBID') if k in os.environ]}
 def save():
  if output is not None:
   temporary=Path(output).with_suffix('.tmp')
   with temporary.open('w',encoding='utf-8') as f:json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
   temporary.replace(output)
 def update(key,value):data[key]=value;save()
 data['status']='RUNNING';save()
 _phase('read NVIDIA kernel-driver metadata and device-node access')
 data['kernel_driver']=_read('/proc/driver/nvidia/version');data['device_nodes']=[]
 paths=set(Path('/dev').glob('nvidia[0-9]*'))|{Path('/dev/nvidiactl'),Path('/dev/nvidia-uvm')}
 for path in sorted(paths)[:66]:
  item={'path':str(path),'exists':path.exists()}
  try:
   info=path.stat();item.update(character_device=stat.S_ISCHR(info.st_mode),read_access=os.access(path,os.R_OK),write_access=os.access(path,os.W_OK))
  except OSError as e:item['error']=str(e)
  data['device_nodes'].append(item)
 save();data['driver']=_driver(lambda value:update('driver',value));save();data['runtime']=_runtime(lambda value:update('runtime',value));save()
 if '/stubs/' in (data['driver'].get('resolved_library') or ''):data['warnings']=['Driver library is in a stubs directory; directory name alone does not prove stub contents. Initialization/API results determine status.']
 data['classification'],data['summary']=classify(data);data['status']='PASS' if data['classification']=='DRIVER_RUNTIME_READY' else 'FAIL'
 data['access_note']='Device-node visibility and os.access do not prove cgroup/security-policy permission; the API result is the actual observation. No GPU allocation is inferred from scheduler-variable presence alone.'
 save();return data

def main(argv=None):
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args(argv)
 data=probe(args.output);temporary=args.output.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2),encoding='utf-8');temporary.replace(args.output)
 print(data['classification']+': '+data['summary'],flush=True)
 return 0

if __name__=='__main__':raise SystemExit(main())
