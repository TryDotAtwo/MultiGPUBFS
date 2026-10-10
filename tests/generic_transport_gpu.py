import ctypes as C,json,sys,time,os
from pathlib import Path
rank=int(sys.argv[1]);root=Path(sys.argv[2]);p=C.c_void_p;u=C.c_uint32;h=C.c_uint64
lib=C.CDLL('/root/universal/native-generic/libmgbfs_cuda.so');cuda=C.CDLL(os.environ.get('MGBFS_CUDART_LIBRARY','libcudart.so.13'))
def ck(code):
 if code:raise RuntimeError('NATIVE_STATUS_'+str(code))
ck(cuda.cudaSetDevice(rank));cuda.cudaMalloc.argtypes=[C.POINTER(p),C.c_size_t];cuda.cudaMemcpy.argtypes=[p,p,C.c_size_t,C.c_int];cuda.cudaFree.argtypes=[p]
lib.mgbfs_nccl_unique_id.argtypes=[p];lib.mgbfs_nccl_create.argtypes=[u,u,u,p,C.POINTER(p),p,C.c_size_t];lib.mgbfs_nccl_destroy.argtypes=[p]
fn=lib.mgbfs_nccl_send_recv_triplet;fn.argtypes=[p,p,h,p,h,p,h,u,p,p,p,p];fn.restype=C.c_int
identity=root/'id';uid=(C.c_ubyte*128)()
if rank==0:
 ck(lib.mgbfs_nccl_unique_id(uid));temporary=root/'id.tmp';temporary.write_bytes(bytes(uid));temporary.rename(identity)
else:
 started=time.monotonic()
 while not identity.exists():
  if time.monotonic()-started>20:raise RuntimeError('BOOTSTRAP_TIMEOUT')
  time.sleep(.01)
 C.memmove(uid,identity.read_bytes(),128)
comm=p();error=C.create_string_buffer(512);ck(lib.mgbfs_nccl_create(rank,2,rank,uid,C.byref(comm),error,512));alloc=[]
def buffer(values):
 host=(u*len(values))(*values);q=p();ck(cuda.cudaMalloc(C.byref(q),C.sizeof(host)));alloc.append(q);ck(cuda.cudaMemcpy(q,host,C.sizeof(host),1));return q
try:
 sends=[buffer([rank+1]),buffer([100*rank+i for i in range(32)]),buffer([1000*rank+i for i in range(64)])];receives=[buffer([0]),buffer([0]*32),buffer([0]*64)]
 ck(fn(comm,sends[0],4,sends[1],128,sends[2],256,1-rank,*receives,None));ck(cuda.cudaDeviceSynchronize())
 for ptr,n,base in [(receives[0],1,1-rank+1),(receives[1],32,100*(1-rank)),(receives[2],64,1000*(1-rank))]:
  values=(u*n)();ck(cuda.cudaMemcpy(values,ptr,C.sizeof(values),2));assert list(values)==([base] if n==1 else list(range(base,base+n)))
 total_send=p();total_recv=p();ck(cuda.cudaMalloc(C.byref(total_send),8));alloc.append(total_send);ck(cuda.cudaMalloc(C.byref(total_recv),8));alloc.append(total_recv)
 large=(h*1)((1<<32)+rank);ck(cuda.cudaMemcpy(total_send,large,8,1))
 total=lib.mgbfs_nccl_all_reduce_sum_u64;total.argtypes=[p,p,p,p];ck(total(comm,total_send,total_recv,None));ck(cuda.cudaDeviceSynchronize());ck(cuda.cudaMemcpy(large,total_recv,8,2));assert large[0]==(1<<33)+1
 (root/f'rank-{rank}.json').write_text(json.dumps({'status':'VERIFIED_GENERIC_THREE_LANE_GPU_TRANSPORT','rank':rank,'scope':'NCCL two GPU count/metadata/payload primitive; not integrated BFS or throughput'}))
finally:
 lib.mgbfs_nccl_destroy(comm)
 for q in alloc:ck(cuda.cudaFree(q))
