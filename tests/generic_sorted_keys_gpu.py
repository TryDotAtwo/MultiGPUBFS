"""Exact origin regeneration under full hash collisions and malformed leases."""
import ctypes as C,json,os,sys
from pathlib import Path
from ctypes.util import find_library
u=C.c_uint32;h=C.c_uint64;p=C.c_void_p
class Route(C.Structure):_fields_=[('hash',h),('parent',h),('source',u),('generator',u),('shard',u),('reserved',u)]
lib=C.CDLL(sys.argv[1]);cuda=C.CDLL(find_library('cudart') or 'libcudart.so.13');cuda.cudaMalloc.argtypes=[C.POINTER(p),C.c_size_t];cuda.cudaMemcpy.argtypes=[p,p,C.c_size_t,C.c_int];cuda.cudaFree.argtypes=[p]
fn=lib.mgbfs_generic_accept_parent_origin;fn.argtypes=[u]*6+[p]*4+[u,u,h]+[p]*6+[u]*5+[p,u,p,u,u,u]+[p]*5+[u,p,p];fn.restype=C.c_int
sort=lib.mgbfs_generic_sort_origins;sort.argtypes=[p]*4+[u]*5+[p]*5+[h,p,p];sort.restype=C.c_int
checks=[]
def ck(x):
 if x:raise RuntimeError('CUDA_STATUS_'+str(x))

for device in (0,1):
 ck(cuda.cudaSetDevice(device));alloc=[]
 def upload(values,ctype):
  host=(ctype*len(values))(*values);v=p();ck(cuda.cudaMalloc(C.byref(v),max(1,C.sizeof(host))));alloc.append(v);ck(cuda.cudaMemcpy(v,host,C.sizeof(host),1));return v
 def read(v,n,ctype):
  host=(ctype*n)();ck(cuda.cudaMemcpy(host,v,C.sizeof(host),2));return list(host)
 try:
  world=2;shards=3;q=4;rank=device;shard=1;size=world*shards*q
  lm=[Route() for _ in range(size)];rm=[Route() for _ in range(size)];lc=[0]*6;rc=[0]*6
  hashes=[[0,(1<<64)-1,8],[7,6]]
  expected=[]
  for source,values in enumerate(hashes):
   queue=source*shards+shard;meta=lm if source==rank else rm;(lc if source==rank else rc)[queue]=len(values)
   for row,hashvalue in enumerate(values):meta[queue*q+row]=Route(hashvalue,0,source,0,shard,0);expected.append((hashvalue>>1,queue*q+row))
  meta=[upload(lm,Route),upload(rm,Route),upload(lc,u),upload(rc,u)];keys=upload([0]*8,h);out=upload([0]*8,h);orig=upload([0]*8,u);ordered=upload([0]*8,u);scratch=upload([0]*(8*64+65536),C.c_uint8);error=upload([0],u)
  args=[*meta,rank,world,shard,shards,q,keys,out,orig,ordered,scratch,8*64+65536,error,None];ck(sort(*args));ck(cuda.cudaDeviceSynchronize());assert read(error,1,u)==[0]
  got=list(zip(read(out,8,h),read(ordered,8,u)));assert sorted(got[:5])==sorted(expected);assert [x[0] for x in got]==sorted([x[0] for x in expected])+[(1<<64)-1]*3
  bad=[0]*6;bad[rank*shards+shard]=q+1;args[2]=upload(bad,u);ck(sort(*args));ck(cuda.cudaDeviceSynchronize());assert read(error,1,u)[0]&32
  checks.append({'device':device,'max_hash_not_padding':True,'index_permutation_exact':True,'invalid_count_rejected':True})
 finally:
  for v in reversed(alloc):ck(cuda.cudaFree(v))
v={'status':'VERIFIED_RADIX_ORIGIN_HASHMAX_PADDING_COUNTS','checks':checks};Path(sys.argv[2]).write_text(json.dumps(v));print(json.dumps(v))
