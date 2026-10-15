"""Small isolated CUDA Driver API query; no Torch, cudart, NCCL or allocation."""
import ctypes as C,json,sys
def discover():
 lib=C.CDLL('libcuda.so.1')
 def ck(code,operation):
  if code:raise RuntimeError(operation+' returned CUDA driver status '+str(code))
 lib.cuDriverGetVersion.argtypes=[C.POINTER(C.c_int)];lib.cuDriverGetVersion.restype=C.c_int
 version=C.c_int();ck(lib.cuDriverGetVersion(C.byref(version)),'cuDriverGetVersion')
 lib.cuInit.argtypes=[C.c_uint];lib.cuInit.restype=C.c_int;ck(lib.cuInit(0),'cuInit')
 lib.cuDeviceGetCount.argtypes=[C.POINTER(C.c_int)];lib.cuDeviceGetCount.restype=C.c_int
 count=C.c_int();ck(lib.cuDeviceGetCount(C.byref(count)),'cuDeviceGetCount')
 if not 0<count.value<=128:raise RuntimeError('No supported visible CUDA GPU set; run inside the allocated GPU job')
 lib.cuDeviceGet.argtypes=[C.POINTER(C.c_int),C.c_int];lib.cuDeviceGet.restype=C.c_int
 lib.cuDeviceGetAttribute.argtypes=[C.POINTER(C.c_int),C.c_int,C.c_int];lib.cuDeviceGetAttribute.restype=C.c_int
 lib.cuDeviceGetName.argtypes=[C.c_void_p,C.c_int,C.c_int];lib.cuDeviceGetName.restype=C.c_int
 devices=[]
 for logical in range(count.value):
  device=C.c_int();ck(lib.cuDeviceGet(C.byref(device),logical),'cuDeviceGet');cc=[]
  for attribute in (75,76):
   value=C.c_int();ck(lib.cuDeviceGetAttribute(C.byref(value),attribute,device.value),'cuDeviceGetAttribute');cc.append(value.value)
  name=C.create_string_buffer(256);ck(lib.cuDeviceGetName(name,len(name),device.value),'cuDeviceGetName')
  devices.append({'logical_id':logical,'name':name.value.decode(errors='replace'),'compute_capability':cc})
 return {'driver_api':version.value,'devices':devices}
if __name__=='__main__':
 try:print(json.dumps(discover()),flush=True)
 except Exception as error:print(str(error),file=sys.stderr,flush=True);raise SystemExit(1)
