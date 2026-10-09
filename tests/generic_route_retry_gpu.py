"""Retry votes must never hide owner or source metadata failures."""
import ctypes as C,json,sys
from pathlib import Path
p=C.c_void_p;u=C.c_uint32;cuda=C.CDLL('libcudart.so.12');lib=C.CDLL(sys.argv[1])
cuda.cudaMalloc.argtypes=[C.POINTER(p),C.c_size_t];cuda.cudaMemcpy.argtypes=[p,p,C.c_size_t,C.c_int];cuda.cudaFree.argtypes=[p]
lib.mgbfs_generic_route_retry_vote.argtypes=[p,p,p,p]
def ck(code):
 if code:raise RuntimeError('CUDA_STATUS_'+str(code))
checks=[]
for device in (0,1):
 ck(cuda.cudaSetDevice(device));buf=[]
 try:
  for _ in range(3):
   q=p();ck(cuda.cudaMalloc(C.byref(q),4));buf.append(q)
  for source in (0,1,2,4,5,6,7):
   for owner in (0,1,2,4,8):
    ck(cuda.cudaMemcpy(buf[0],C.byref(u(source)),4,1));ck(cuda.cudaMemcpy(buf[1],C.byref(u(owner)),4,1));ck(lib.mgbfs_generic_route_retry_vote(*buf,None));ck(cuda.cudaDeviceSynchronize());v=u();ck(cuda.cudaMemcpy(C.byref(v),buf[2],4,2))
    expected=(0x100|owner) if owner else (1 if source in (1,5) else ((0x10000|source) if source else 0));assert v.value==expected
    checks.append({'device':device,'source':source,'owner':owner,'vote':v.value})
 finally:
  for q in buf:ck(cuda.cudaFree(q))
receipt={'status':'VERIFIED_SOURCE_RETRY_ERROR_PRECEDENCE','checks':checks};Path('/root/universal/route-retry-vote-verification.json').write_text(json.dumps(receipt));print('VERIFIED',len(checks))
