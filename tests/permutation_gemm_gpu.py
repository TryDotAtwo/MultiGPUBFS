"""CUDA/GEMM parity and whole generation+hash+route timings on LRX n14 parents."""
import ctypes as C,os,json,hashlib,statistics
from pathlib import Path
import numpy as np
u=C.c_uint32;q=C.c_uint64;ptr=C.c_void_p
lib=C.CDLL(os.environ['MGBFS_CUDA_LIBRARY']);cuda=C.CDLL(os.environ['MGBFS_CUDART_LIBRARY'])
cuda.cudaMalloc.argtypes=[C.POINTER(ptr),C.c_size_t];cuda.cudaMemcpy.argtypes=[ptr,ptr,C.c_size_t,C.c_int];cuda.cudaFree.argtypes=[ptr];cuda.cudaMemset.argtypes=[ptr,C.c_int,C.c_size_t]
cuda.cudaEventCreate.argtypes=[C.POINTER(ptr)];cuda.cudaEventRecord.argtypes=[ptr,ptr];cuda.cudaEventSynchronize.argtypes=[ptr];cuda.cudaEventElapsedTime.argtypes=[C.POINTER(C.c_float),ptr,ptr]
create=lib.mgbfs_generic_gemm_create;create.argtypes=[u,u,u,u,ptr,C.POINTER(ptr)];destroy=lib.mgbfs_generic_gemm_destroy;destroy.argtypes=[ptr]
argtypes=[u]*5+[ptr,u,u]+[ptr]*3+[q]+[u]*5+[q]+[ptr]*6
plain=lib.mgbfs_generic_route_packed_u8;plain.argtypes=argtypes
gemm=lib.mgbfs_generic_route_gemm_u8;gemm.argtypes=[ptr]+argtypes

def ck(c):
 if c:raise RuntimeError('CUDA_STATUS_'+str(c))
def case(count,device,shards=4):
 ck(cuda.cudaSetDevice(device));n=14;g=3;world=2;queues=world*shards;stride=count+5;cap=count*g;rng=np.random.default_rng(count+device)
 perms=np.asarray([list(range(1,n))+[0],[n-1]+list(range(n-1)),[1,0]+list(range(2,n))],dtype=np.uint32)
 onehot=np.zeros((g,n,n),np.int64)
 for gg in range(g):onehot[gg,np.arange(n),perms[gg]]=1
 parents=np.zeros((n,stride),np.uint8);parents[:,:count]=np.argsort(rng.random((count,n)),axis=1).astype(np.uint8).T
 if count==33:parents[:,:count]=rng.integers(0,256,(n,count),dtype=np.uint8);parents[0,0]=255
 alloc=[];ctx=ptr();ck(create(n,1,g,count,onehot.ctypes.data,C.byref(ctx)))
 def mem(size,host=None):
  p=ptr();ck(cuda.cudaMalloc(C.byref(p),size));alloc.append(p)
  if host is not None:ck(cuda.cudaMemcpy(p,host,size,1))
  return p
 inputs=mem(parents.nbytes,parents.ctypes.data);ps=mem(perms.nbytes,perms.ctypes.data);records=mem(queues*cap*32);counts=mem(queues*4);error=mem(4)
 args=[0,n,n,1,g,inputs,count,stride,ps,None,None,12345,64,world,0,shards,cap,0,None,None,records,counts,error,None]
 start=ptr();end=ptr();ck(cuda.cudaEventCreate(C.byref(start)));ck(cuda.cudaEventCreate(C.byref(end)))
 def invoke(which):
  ck(cuda.cudaMemset(counts,0,queues*4));ck(cuda.cudaMemset(error,0,4));ck(cuda.cudaEventRecord(start,None));ck(plain(*args) if which=='cuda' else gemm(ctx,*args));ck(cuda.cudaEventRecord(end,None));ck(cuda.cudaEventSynchronize(end));ms=C.c_float();ck(cuda.cudaEventElapsedTime(C.byref(ms),start,end));return float(ms.value)
 def receipt():
  actual=np.empty(queues,np.uint32);ck(cuda.cudaMemcpy(actual.ctypes.data,counts,actual.nbytes,2));err=np.empty(1,np.uint32);ck(cuda.cudaMemcpy(err.ctypes.data,error,4,2));assert not err[0] and int(actual.sum())==count*g
  pieces=[]
  for i,nn in enumerate(actual):
   raw=np.empty(int(nn),dtype='V32');ck(cuda.cudaMemcpy(raw.ctypes.data,ptr(records.value+i*cap*32),raw.nbytes,2));pieces.append(raw)
  raw=np.concatenate(pieces);raw.sort();return hashlib.sha256(raw.tobytes()).hexdigest()
 try:
  invoke('cuda');a=receipt();invoke('gemm');b=receipt();assert a==b,(count,device,a,b)
  times={'cuda':[],'gemm':[]}
  for i in range(7):
   for which in (('cuda','gemm') if i%2==0 else ('gemm','cuda')):times[which].append(invoke(which))
  med={k:statistics.median(v) for k,v in times.items()}
  return {'requested_variant':os.environ.get('MGBFS_GEMM_VARIANT','auto'),'device':device,'parents':count,'children':count*g,'shards':shards,'median_ms':med,'samples_ms':times,'cuda_over_gemm':med['cuda']/med['gemm'],'exact_records_sha256':a}
 finally:
  ck(destroy(ctx));ck(cuda.cudaEventDestroy(start));ck(cuda.cudaEventDestroy(end))
  for p in alloc:ck(cuda.cudaFree(p))
if __name__=='__main__':
 rows=[]
 for device in (0,1):
  for count in (33,4096,32768,131072,349525,1048576):
   row=case(count,device);rows.append(row);print(json.dumps(row),flush=True)
 Path(os.environ.get('MGBFS_GEMM_REPORT','permutation-gemm-check.json')).write_text(json.dumps({'status':'VERIFIED_PERMUTATION_GEMM_ROUTE_PARITY','scope':'valid n14 permutations; exact records; event includes pack+GEMM+hash+route; excludes dedup/exchange','runs':rows},indent=2))
