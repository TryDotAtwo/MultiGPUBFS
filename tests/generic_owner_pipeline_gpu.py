"""GPU route/regenerate/exact owner chain; CPU readback only at layer boundary."""
import ctypes as C,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from multigpubfs import GraphDefinition
p=C.c_void_p;u=C.c_uint32;h=C.c_uint64
lib=C.CDLL(sys.argv[1]);accept=lib.mgbfs_generic_accept_i64;accept.argtypes=[u,p,u,p,p,u,p,u,p,u,p,p,u,p,p,p];accept.restype=C.c_int
route=lib.mgbfs_generic_route_i64;route.argtypes=[u,u,u,u,u,p,u,u,p,p,p,h,u,u,u,u,u,h,p,p,p,p,p,p];route.restype=C.c_int
regen=lib.mgbfs_generic_regenerate_routes_count_i64;regen.argtypes=[u,u,u,u,u,p,u,u,p,p,p,u,h,p,u,p,p,u,p,p];regen.restype=C.c_int
seed=lib.mgbfs_generic_seed_i64;seed.argtypes=[u,p,u,u,p,u,h,u,p,p];seed.restype=C.c_int
gather=lib.mgbfs_generic_gather_i64;gather.argtypes=[u,p,u,p,u,p,u,p];gather.restype=C.c_int
cuda=C.CDLL('libcudart.so.12');cuda.cudaMalloc.argtypes=[C.POINTER(p),C.c_size_t];cuda.cudaMemcpy.argtypes=[p,p,C.c_size_t,C.c_int];cuda.cudaMemset.argtypes=[p,C.c_int,C.c_size_t];cuda.cudaFree.argtypes=[p]
def ck(code):
 if code:raise RuntimeError('CUDA_STATUS_'+str(code))
def run(g,bits,capacity=1024,resource=False):
 alloc=[];width=g.state_elements;a=g.action;kind=int(a['kind']=='matrix');ng=g.generator_count;batch=capacity*ng
 def upload(values,ty):
  host=(ty*len(values))(*values);q=p();ck(cuda.cudaMalloc(C.byref(q),max(1,C.sizeof(host))));alloc.append(q)
  if values:ck(cuda.cudaMemcpy(q,host,C.sizeof(host),1))
  return q
 def read(q,ty,n):
  host=(ty*n)();ck(cuda.cudaMemcpy(host,q,C.sizeof(host),2));return host
 try:
  state=[0]*(width*capacity)
  for e,v in enumerate(g.start):state[e*capacity]=v
  visited=upload(state,C.c_int64);parents=upload([0]*len(state),C.c_int64);front=upload([0]*capacity,u);future=upload([0]*capacity,u)
  slot_count=1
  while slot_count<capacity*2:slot_count*=2
  slots=upload([(1<<64)-1]*slot_count,h);vc=upload([1],u);fc=upload([0],u);error=upload([0],u)
  perm=upload([x for gen in a['generators'] for x in gen] if not kind else [],u);mx=upload([x for gen in a['generators'] for x in gen['matrix']] if kind else [],C.c_int64);mods=upload([gen['modulo'] for gen in a['generators']] if kind else [],u)
  banks=[(upload([0]*(batch*4),h),upload([0],u),upload([0]*(width*batch),C.c_int64)) for _ in range(2)]
  ck(seed(width,visited,capacity,1,slots,slot_count,123,bits,error,None));count=1;layers=[];fatal=0
  for depth in range(2048):
   ids=read(front,u,count);raw=read(visited,C.c_int64,width*capacity);layers.append(sorted([[raw[e*capacity+i] for e in range(width)] for i in ids]))
   records,received,incoming=banks[depth%2];ck(cuda.cudaMemset(received,0,4));ck(cuda.cudaMemset(fc,0,4))
   ck(gather(width,visited,capacity,front,count,parents,capacity,None))
   ck(route(kind,width,a.get('rows',width),a.get('cols',1),ng,parents,count,capacity,perm,mx,mods,123,bits,1,0,1,batch,0,None,None,records,received,error,None))
   ck(regen(kind,width,a.get('rows',width),a.get('cols',1),ng,parents,count,capacity,perm,mx,mods,0,0,records,batch,received,incoming,batch,error,None))
   ck(accept(width,incoming,batch,records,received,batch,slots,slot_count,visited,capacity,vc,future,capacity,fc,error,None))
   ck(cuda.cudaDeviceSynchronize());fatal=read(error,u,1)[0]
   if fatal:break
   count=read(fc,u,1)[0]
   if not count:break
   front,future=future,front
  else:raise AssertionError('depth bound')
  if resource:assert fatal&2
  else:assert fatal==0;assert layers==g.exact_layers(capacity)
  # Device-resident receive-count overflow cannot read beyond the inbox.
  ck(cuda.cudaMemset(error,0,4));bad=(u*1)(batch+1);ck(cuda.cudaMemcpy(received,bad,4,1));ck(accept(width,incoming,batch,records,received,batch,slots,slot_count,visited,capacity,vc,future,capacity,fc,error,None));ck(cuda.cudaDeviceSynchronize());assert read(error,u,1)[0]&32
  return {'states':sum(map(len,layers)),'layers':len(layers),'hash_bits':bits,'fatal':fatal}
 finally:
  for q in alloc:ck(cuda.cudaFree(q))
fixtures=[GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],[0,1,2,3]),GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],[0,0,1,2]),GraphDefinition.matrix(2,1,[([1,1,0,1],257)],[256,2])]
results=[]
for device in (0,1):
 ck(cuda.cudaSetDevice(device))
 for i,g in enumerate(fixtures):
  for bits in (64,0):results.append(dict(run(g,bits),device=device,fixture=i))
 results.append(dict(run(fixtures[0],0,3,True),device=device,fixture='resource'))
print(json.dumps({'status':'VERIFIED_GPU_SOURCE_ROUTE_EXACT_OWNER_CHAIN','runs':results,'device_counts_no_interstage_readback':True,'scope':'A/B source-route-owner chain on each of two GPUs; no inter-GPU transport or distributed SHARD_AB acceptance'}))
