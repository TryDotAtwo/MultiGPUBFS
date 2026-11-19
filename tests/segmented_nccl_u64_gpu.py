"""Exercise the production NCCL u64 wrapper above the u32 boundary."""
import ctypes as C, multiprocessing as mp, os, traceback

LIB='/workspace/seg24-build/libmgbfs_cuda.so'
VALUES=((1<<32)+17,(1<<40)+42)

def prototypes(n,c):
 n.mgbfs_nccl_unique_id.argtypes=[C.c_void_p]
 n.mgbfs_nccl_create.argtypes=[C.c_uint,C.c_uint,C.c_uint,C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p,C.c_size_t]
 n.mgbfs_nccl_all_gather_u64.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p]
 n.mgbfs_nccl_destroy.argtypes=[C.c_void_p]
 c.cudaSetDevice.argtypes=[C.c_int];c.cudaMalloc.argtypes=[C.POINTER(C.c_void_p),C.c_size_t]
 c.cudaMemcpy.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int]
 c.cudaStreamCreate.argtypes=[C.POINTER(C.c_void_p)];c.cudaStreamSynchronize.argtypes=[C.c_void_p]
 c.cudaStreamDestroy.argtypes=[C.c_void_p];c.cudaFree.argtypes=[C.c_void_p]

def run(rank,raw,q):
 try:
  n=C.CDLL(LIB);c=C.CDLL('libcudart.so.12');prototypes(n,c)
  def ck(result,name):
   if result:raise RuntimeError((name,result))
  ck(c.cudaSetDevice(rank),'setDevice');stream=C.c_void_p();ck(c.cudaStreamCreate(C.byref(stream)),'stream')
  comm=C.c_void_p();err=C.create_string_buffer(512);uid=C.create_string_buffer(raw,128)
  z=n.mgbfs_nccl_create(rank,2,rank,uid,C.byref(comm),err,512)
  if z:raise RuntimeError(('nccl create',z,err.value))
  src=C.c_void_p();dst=C.c_void_p();ck(c.cudaMalloc(C.byref(src),8),'malloc src');ck(c.cudaMalloc(C.byref(dst),16),'malloc dst')
  value=C.c_uint64(VALUES[rank]);ck(c.cudaMemcpy(src,C.byref(value),8,1),'upload')
  ck(n.mgbfs_nccl_all_gather_u64(comm,src,dst,stream),'gather');ck(c.cudaStreamSynchronize(stream),'synchronize')
  got=(C.c_uint64*2)();ck(c.cudaMemcpy(got,dst,16,2),'download');assert tuple(got)==VALUES,(rank,tuple(got))
  ck(c.cudaFree(src),'free src');ck(c.cudaFree(dst),'free dst');n.mgbfs_nccl_destroy(comm);ck(c.cudaStreamDestroy(stream),'destroy stream')
  q.put((rank,'PASS',tuple(got)))
 except BaseException:q.put((rank,'FAIL',traceback.format_exc()))

if __name__=='__main__':
 n=C.CDLL(LIB);n.mgbfs_nccl_unique_id.argtypes=[C.c_void_p];uid=C.create_string_buffer(128);assert n.mgbfs_nccl_unique_id(uid)==0
 ctx=mp.get_context('spawn');q=ctx.Queue();jobs=[ctx.Process(target=run,args=(i,uid.raw,q)) for i in range(2)]
 for p in jobs:p.start()
 for p in jobs:p.join(60)
 for p in jobs:
  if p.is_alive():p.terminate();p.join();raise TimeoutError('NCCL gather')
 results=[q.get(timeout=5) for p in jobs];print(results,flush=True)
 assert all(p.exitcode==0 for p in jobs) and all(r[1]=='PASS' for r in results)
 print('PASS u64 gather above 2^32 on both ranks')
