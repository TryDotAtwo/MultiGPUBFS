"""GPU route/regenerate/exact owner chain; CPU readback only at layer boundary."""
import ctypes as C,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from multigpubfs import GraphDefinition
p=C.c_void_p;u=C.c_uint32;h=C.c_uint64
rank=int(sys.argv[1]);root=Path(sys.argv[2]);lib=C.CDLL('/root/universal/native-generic/libmgbfs_cuda.so');accept=lib.mgbfs_generic_accept_i64;accept.argtypes=[u,p,u,p,p,u,p,u,p,u,p,p,u,p,p,p];accept.restype=C.c_int
route=lib.mgbfs_generic_route_i64;route.argtypes=[u,u,u,u,u,p,u,u,p,p,p,h,u,u,u,u,u,h,p,p,p,p,p,p];route.restype=C.c_int
regen=lib.mgbfs_generic_regenerate_routes_count_i64;regen.argtypes=[u,u,u,u,u,p,u,u,p,p,p,u,h,p,u,p,p,u,p,p];regen.restype=C.c_int
seed=lib.mgbfs_generic_seed_i64;seed.argtypes=[u,p,u,u,p,u,h,u,p,p];seed.restype=C.c_int
gather=lib.mgbfs_generic_gather_i64;gather.argtypes=[u,p,u,p,u,p,u,p];gather.restype=C.c_int
cuda=C.CDLL('libcudart.so.12');cuda.cudaMalloc.argtypes=[C.POINTER(p),C.c_size_t];cuda.cudaMemcpy.argtypes=[p,p,C.c_size_t,C.c_int];cuda.cudaMemset.argtypes=[p,C.c_int,C.c_size_t];cuda.cudaFree.argtypes=[p]
def ck(code):
 if code:raise RuntimeError('CUDA_STATUS_'+str(code))
import time
ck(cuda.cudaSetDevice(rank))
lib.mgbfs_nccl_unique_id.argtypes=[p];lib.mgbfs_nccl_create.argtypes=[u,u,u,p,C.POINTER(p),p,C.c_size_t];lib.mgbfs_nccl_destroy.argtypes=[p]
triplet=lib.mgbfs_nccl_send_recv_triplet;triplet.argtypes=[p,p,h,p,h,p,h,u,p,p,p,p]
maximum=lib.mgbfs_nccl_all_reduce_max_u32;maximum.argtypes=[p,p,p,p]
summation=lib.mgbfs_nccl_all_reduce_sum_u32;summation.argtypes=[p,p,p,p]
uid=(C.c_ubyte*128)();identity=root/'id'
if rank==0:
 ck(lib.mgbfs_nccl_unique_id(uid));(root/'id.tmp').write_bytes(bytes(uid));(root/'id.tmp').rename(identity)
else:
 started=time.monotonic()
 while not identity.exists():
  if time.monotonic()-started>20:raise RuntimeError('bootstrap timeout')
  time.sleep(.01)
 C.memmove(uid,identity.read_bytes(),128)
comm=p();message=C.create_string_buffer(512);ck(lib.mgbfs_nccl_create(rank,2,rank,uid,C.byref(comm),message,512))
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
  slots=upload([(1<<64)-1]*slot_count,h);vc=upload([1 if rank==0 else 0],u);fc=upload([0],u);error=upload([0],u)
  perm=upload([x for gen in a['generators'] for x in gen] if not kind else [],u);mx=upload([x for gen in a['generators'] for x in gen['matrix']] if kind else [],C.c_int64);mods=upload([gen['modulo'] for gen in a['generators']] if kind else [],u)
  banks=[(upload([0]*(batch*8),h),upload([0,0],u),upload([0]*(width*batch*2),C.c_int64)) for _ in range(2)]
  inbox=(upload([0]*(batch*4),h),upload([0],u),upload([0]*(width*batch),C.c_int64))
  global_error=upload([0],u);global_count=upload([0],u)
  # Explicit owner map assigns the root to rank zero while routing both ranks.
  cuts=upload([0,1<<31,1<<32],h)
  def offset(q,n):return p(q.value+n)
  ck(seed(width,visited,capacity,1,slots,slot_count,123,bits,error,None))
  tokens=read(slots,h,slot_count);root_hash=next(x>>32 for x in tokens if x!=(1<<64)-1);root_owner=root_hash>>31
  owner_map=upload([root_owner,1-root_owner],u)
  ck(cuda.cudaMemset(slots,255,slot_count*8))
  if rank==0:ck(seed(width,visited,capacity,1,slots,slot_count,123,bits,error,None))
  count=1 if rank==0 else 0;layers=[];fatal=0;cursor=0
  for depth in range(2048):
   ids=read(front,u,count);raw=read(visited,C.c_int64,width*capacity);layers.append(sorted([[raw[e*capacity+i] for e in range(width)] for i in ids]))
   records,received,incoming=banks[depth%2];ck(cuda.cudaMemset(received,0,8));ck(cuda.cudaMemset(fc,0,4))
   ck(gather(width,visited,capacity,front,count,parents,capacity,None))
   ck(route(kind,width,a.get('rows',width),a.get('cols',1),ng,parents,count,capacity,perm,mx,mods,123,bits,2,rank,1,batch,cursor,owner_map,None,records,received,error,None))
   for peer in (0,1):
    ck(regen(kind,width,a.get('rows',width),a.get('cols',1),ng,parents,count,capacity,perm,mx,mods,rank,cursor,offset(records,peer*batch*32),batch,offset(received,peer*4),offset(incoming,peer*width*batch*8),batch,error,None))
   ck(maximum(comm,error,global_error,None));ck(cuda.cudaDeviceSynchronize());fatal=read(global_error,u,1)[0]
   if fatal:break
   peer=1-rank;remote_records,remote_count,remote_states=inbox
   ck(triplet(comm,offset(received,peer*4),4,offset(records,peer*batch*32),batch*32,offset(incoming,peer*width*batch*8),width*batch*8,peer,remote_count,remote_records,remote_states,None))
   for data,meta,rc in [(offset(incoming,rank*width*batch*8),offset(records,rank*batch*32),offset(received,rank*4)),(remote_states,remote_records,remote_count)]:
    ck(accept(width,data,batch,meta,rc,batch,slots,slot_count,visited,capacity,vc,future,capacity,fc,error,None))
   ck(maximum(comm,error,global_error,None));ck(summation(comm,fc,global_count,None));ck(cuda.cudaDeviceSynchronize());fatal=read(global_error,u,1)[0]
   if fatal:break
   cursor+=count;count=read(fc,u,1)[0]
   if not read(global_count,u,1)[0]:break
   front,future=future,front
  else:raise AssertionError('depth bound')
  return {'layers':layers,'hash_bits':bits,'fatal':fatal,'rank':rank,'source_parent_cursor':cursor}
 finally:
  for q in alloc:ck(cuda.cudaFree(q))
fixtures=[GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],[0,1,2,3]),GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],[0,0,1,2]),GraphDefinition.matrix(2,1,[([1,1,0,1],257)],[256,2])]
try:
 results=[]
 for i,g in enumerate(fixtures):
  for bits in (64,0):results.append(dict(run(g,bits),fixture=i))
 results.append(dict(run(fixtures[0],0,3,True),fixture='resource'))
 (root/f'rank-{rank}.json').write_text(json.dumps(results))
finally:lib.mgbfs_nccl_destroy(comm)
