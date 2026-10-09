"""Route metadata without materializing children; CPU oracle only in test."""
import ctypes as C,json,sys,bisect
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from multigpubfs.graph_definition import GraphDefinition
class Record(C.Structure):_fields_=[('hash',C.c_uint64),('parent',C.c_uint64),('source',C.c_uint32),('generator',C.c_uint32),('shard',C.c_uint32),('reserved',C.c_uint32)]
assert C.sizeof(Record)==32
lib=C.CDLL(sys.argv[1]);fn=lib.mgbfs_generic_route_i64
u=C.c_uint32;h=C.c_uint64;p=C.c_void_p
fn.argtypes=[u,u,u,u,u,p,u,u,p,p,p,h,u,u,u,u,u,h,p,p,p,p,p,p];fn.restype=C.c_int
regen=lib.mgbfs_generic_regenerate_routes_i64;regen.argtypes=[u,u,u,u,u,p,u,u,p,p,p,u,h,p,u,p,u,p,p];regen.restype=C.c_int
cuda=C.CDLL('libcudart.so.12');cuda.cudaMalloc.argtypes=[C.POINTER(p),C.c_size_t];cuda.cudaMemcpy.argtypes=[p,p,C.c_size_t,C.c_int];cuda.cudaFree.argtypes=[p]
def ck(n):
 if n:raise RuntimeError('CUDA_STATUS_'+str(n))
def mix(v):
 v&=(1<<64)-1;v^=v>>30;v=v*0xbf58476d1ce4e5b9&((1<<64)-1);v^=v>>27;v=v*0x94d049bb133111eb&((1<<64)-1);return v^(v>>31)
def hash_state(state,bits):
 value=123
 for e,x in enumerate(state):value=mix(value^(x&((1<<64)-1))^e)
 value=mix(value);return value if bits==64 else value&((1<<bits)-1)
def run(g,world,shards,bits,capacity=16,weighted=False,overflow=False):
 parents=[g.start,g.successor(g.start,0)];width=g.state_elements;action=g.action;kind=int(action['kind']=='matrix');count=len(parents);alloc=[];slots=world*shards
 def upload(values,ty):
  host=(ty*len(values))(*values);q=p();ck(cuda.cudaMalloc(C.byref(q),max(1,C.sizeof(host))));alloc.append(q)
  if values:ck(cuda.cudaMemcpy(q,host,C.sizeof(host),1))
  return q
 def read(q,ty,n):
  host=(ty*n)();ck(cuda.cudaMemcpy(host,q,C.sizeof(host),2));return host
 try:
  src=upload([row[e] for e in range(width) for row in parents],C.c_int64)
  perm=upload([x for gen in action['generators'] for x in gen] if not kind else [],u)
  mx=upload([x for gen in action['generators'] for x in gen['matrix']] if kind else [],C.c_int64)
  mod=upload([gen['modulo'] for gen in action['generators']] if kind else [],u)
  mapping=list(reversed(range(world)));rank_map=upload(mapping,u)
  cuts=None;cut_values=None
  if weighted:
   total=world*(world+1)//2;cut_values=[i*(i+1)//2*(1<<32)//total for i in range(world+1)];cuts=upload(cut_values,h)
  records=upload([Record() for _ in range(slots*capacity)],Record);counts=upload([0]*slots,u);error=upload([0],u)
  ck(fn(kind,width,action.get('rows',width),action.get('cols',1),g.generator_count,src,count,count,perm,mx,mod,123,bits,world,world-1,shards,capacity,1000000,rank_map,cuts,records,counts,error,None));ck(cuda.cudaDeviceSynchronize())
  fatal=read(error,u,1)[0]
  if overflow:assert fatal&1;return {'world':world,'shards':shards,'fatal':fatal}
  assert fatal==0,fatal
  sizes=read(counts,u,slots);data=read(records,Record,slots*capacity);actual=[]
  for queue,n in enumerate(sizes):
   assert n<=capacity
   for row in data[queue*capacity:queue*capacity+n]:actual.append((queue,row.hash,row.parent,row.source,row.generator,row.shard,row.reserved))
  expected=[]
  for i,parent in enumerate(parents):
   for gen in range(g.generator_count):
    value=hash_state(g.successor(parent,gen),bits);high=value>>32;owner=bisect.bisect_right(cut_values,high)-1 if weighted else high*world>>32;shard=(value&0xffffffff)*shards>>32;queue=mapping[owner]*shards+shard
    expected.append((queue,value,1000000+i,world-1,gen,shard,0))
  assert sorted(actual)==sorted(expected),(actual,expected)
  # Request only selected immutable origins, in reversed order. No dense
  # children buffer is used by either the routing or regeneration kernel.
  chosen=[actual[-1],actual[0]]
  requests=upload([Record(x[1],x[2],x[3],x[4],x[5],x[6]) for x in chosen],Record)
  regenerated=upload([0]*(width*2),C.c_int64)
  ck(regen(kind,width,action.get('rows',width),action.get('cols',1),g.generator_count,src,count,count,perm,mx,mod,world-1,1000000,requests,2,regenerated,2,error,None));ck(cuda.cudaDeviceSynchronize());assert read(error,u,1)[0]==0
  soa=read(regenerated,C.c_int64,width*2);states=[[soa[e*2+i] for e in range(width)] for i in range(2)]
  assert states==[g.successor(parents[x[2]-1000000],x[4]) for x in chosen]
  bad=chosen[0];bad_request=upload([Record(bad[1],bad[2],(bad[3]+1)%128,bad[4],bad[5],0)],Record);bad_error=upload([0],u)
  ck(regen(kind,width,action.get('rows',width),action.get('cols',1),g.generator_count,src,count,count,perm,mx,mod,world-1,1000000,bad_request,1,regenerated,2,bad_error,None));ck(cuda.cudaDeviceSynchronize());assert read(bad_error,u,1)[0]&2

  return {'world':world,'shards':shards,'hash_bits':bits,'weighted':weighted,'records':len(actual)}
 finally:
  for q in alloc:ck(cuda.cudaFree(q))
fixtures=[GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],[0,0,1,2]),GraphDefinition.matrix(2,1,[([1,1,0,1],257)],[256,2]),GraphDefinition.permutation([list(range(1,300))+[0]],list(range(300)))]
results=[]
for device in (0,1):
 ck(cuda.cudaSetDevice(device))
 for g in fixtures:
  for world in (1,2,8,128):
   for shards in (1,4):results.append(dict(run(g,world,shards,64,weighted=world>1),device=device))
  results.append(dict(run(g,128,4,0,capacity=1,overflow=True),device=device))
print(json.dumps({'status':'VERIFIED_GENERIC_HASH_ORIGIN_ROUTING','runs':results,'selected_regeneration_verified':True,'scope':'metadata route primitive on two GPUs; logical world128 is not 128-GPU or transport acceptance'}))
