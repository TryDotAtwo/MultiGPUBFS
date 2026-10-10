"""Actual native GEMM route and selected payload parity across exact-domain boundaries."""
import ctypes as C,random,os,json
u=C.c_uint32;q=C.c_uint64;ptr=C.c_void_p
lib=C.CDLL(os.environ['MGBFS_CUDA_LIBRARY']);cuda=C.CDLL(os.environ['MGBFS_CUDART_LIBRARY'])
cuda.cudaMalloc.argtypes=[C.POINTER(ptr),C.c_size_t];cuda.cudaMemcpy.argtypes=[ptr,ptr,C.c_size_t,C.c_int];cuda.cudaFree.argtypes=[ptr]
create=lib.mgbfs_generic_gemm_create;create.argtypes=[u,u,u,u,ptr,C.POINTER(ptr)]
destroy=lib.mgbfs_generic_gemm_destroy;destroy.argtypes=[ptr]
route_types=[u]*5+[ptr,u,u]+[ptr]*3+[q]+[u]*5+[q]+[ptr]*6
plain=lib.mgbfs_generic_route_i64;plain.argtypes=route_types
gemm=lib.mgbfs_generic_route_gemm_i64;gemm.argtypes=[ptr]+route_types
regen_types=[u]*5+[ptr,u,u]+[ptr]*3+[u,q,ptr,u,ptr,ptr,u,ptr,ptr]
regen=lib.mgbfs_generic_regenerate_gemm_routes_count_i64;regen.argtypes=[ptr]+regen_types
class Record(C.Structure):_fields_=[('hash',q),('parent',q),('source',u),('generator',u),('shard',u),('reserved',u)]
def ck(code):
 if code:raise RuntimeError('CUDA_STATUS_'+str(code))
def case(n,m,count,device):
 ck(cuda.cudaSetDevice(device));rng=random.Random(n*1000+m*100+count);g=3;stride=count+5;capacity=count*g;alloc=[];context=ptr()
 def upload(values,ty):
  host=(ty*len(values))(*values);p=ptr();ck(cuda.cudaMalloc(C.byref(p),max(1,C.sizeof(host))));alloc.append(p)
  if values:ck(cuda.cudaMemcpy(p,host,C.sizeof(host),1))
  return p
 def read(p,ty,length):
  host=(ty*length)();ck(cuda.cudaMemcpy(host,p,C.sizeof(host),2));return host
 parents=[rng.randrange(256) for _ in range(n*m*stride)];matrices=[rng.randrange(256) for _ in range(g*n*n)];mods=[2,251,256];mx=(C.c_int64*len(matrices))(*matrices)
 try:
  ck(create(n,m,g,count+2,mx,C.byref(context)));inputs=upload(parents,C.c_int64);matrix=upload(matrices,C.c_int64);mod=upload(mods,u);queues=upload([Record() for _ in range(capacity)],Record);counts=upload([0],u);error=upload([0],u);output=upload([0]*(capacity*n*m),C.c_int64)
  args=[1,n*m,n,m,g,inputs,count,stride,None,matrix,mod,12345,64,1,0,1,capacity,17,None,None,queues,counts,error,None]
  ck(plain(*args));ck(cuda.cudaDeviceSynchronize());assert read(error,u,1)[0]==0
  reference=sorted((v.hash,v.parent,v.generator) for v in read(queues,Record,capacity))
  ck(cuda.cudaMemset(counts,0,4));ck(gemm(context,*args));ck(cuda.cudaDeviceSynchronize());assert read(error,u,1)[0]==0
  records=read(queues,Record,capacity);assert sorted((v.hash,v.parent,v.generator) for v in records)==reference
  ck(regen(context,1,n*m,n,m,g,inputs,count,stride,None,matrix,mod,0,17,queues,capacity,counts,output,capacity,error,None));ck(cuda.cudaDeviceSynchronize());assert read(error,u,1)[0]==0
  actual=read(output,C.c_int64,n*m*capacity)
  for index,v in enumerate(records):
   parent=v.parent-17;generator=v.generator
   for element in range(n*m):
    row,col=divmod(element,m);expected=sum(matrices[generator*n*n+row*n+k]*parents[(k*m+col)*stride+parent] for k in range(n))%mods[generator]
    assert actual[element*capacity+index]==expected,(device,n,m,count,element,index)
 finally:
  destroy(context)
  for p in alloc:ck(cuda.cudaFree(p))
 return {'device':device,'n':n,'cols':m,'parents':count,'hashes_and_all_payload_elements_exact':True}
checks=[case(n,m,count,device) for device in (0,1) for n,m in [(8,1),(8,3),(16,8),(32,3),(64,1)] for count in (1,31,33)]
print(json.dumps({'status':'VERIFIED_NATIVE_GEMM_BOUNDARY_PARITY','checks':checks}))
