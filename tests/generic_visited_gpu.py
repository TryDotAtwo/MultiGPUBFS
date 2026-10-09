"""Exact all-visited GPU BFS oracle; no CPU dedup or successors in GPU loop."""
import ctypes as C,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from multigpubfs.graph_definition import GraphDefinition
lib=C.CDLL(sys.argv[1]);expand=lib.mgbfs_generic_expand_i64;seed=lib.mgbfs_generic_seed_i64;gather=lib.mgbfs_generic_gather_i64
u=C.c_uint32;p=C.c_void_p;h=C.c_uint64
expand.argtypes=[u,u,u,u,u,p,u,u,p,p,p,p,u,p,u,p,p,u,p,h,u,p,p];expand.restype=C.c_int
seed.argtypes=[u,p,u,u,p,u,h,u,p,p];seed.restype=C.c_int
gather.argtypes=[u,p,u,p,u,p,u,p];gather.restype=C.c_int
cuda=C.CDLL('libcudart.so.12');cuda.cudaMalloc.argtypes=[C.POINTER(p),C.c_size_t];cuda.cudaMemcpy.argtypes=[p,p,C.c_size_t,C.c_int];cuda.cudaFree.argtypes=[p];cuda.cudaMemset.argtypes=[p,C.c_int,C.c_size_t]
def ck(status):
 if status:raise RuntimeError('CUDA_STATUS_'+str(status))
def run(graph,bits,capacity=1024,expect_resource=False):
 width=graph.state_elements;alloc=[];a=graph.action;kind=0 if a['kind']=='permutation' else 1;n=a['degree'] if not kind else a['rows'];m=1 if not kind else a['cols'];ng=graph.generator_count
 def upload(values,ty):
  v=(ty*len(values))(*values);q=p();ck(cuda.cudaMalloc(C.byref(q),max(1,C.sizeof(v))));alloc.append(q)
  if values:ck(cuda.cudaMemcpy(q,v,C.sizeof(v),1))
  return q
 def read(q,count,ty):
  v=(ty*count)();ck(cuda.cudaMemcpy(v,q,C.sizeof(v),2));return list(v)
 try:
  state=[0]*(capacity*width)
  for e,x in enumerate(graph.start):state[e*capacity]=x
  visited=upload(state,C.c_int64);parents=upload([0]*len(state),C.c_int64);front=upload([0]*capacity,u);future=upload([0]*capacity,u)
  size=1
  while size<capacity*4:size*=2
  slots=upload([(1<<64)-1]*size,h);vc=upload([1],u);fc=upload([0],u);error=upload([0],u)
  perms=upload([v for gen in a['generators'] for v in gen] if not kind else [],u)
  matrices=upload([v for gen in a['generators'] for v in gen['matrix']] if kind else [],C.c_int64)
  mods=upload([gen['modulo'] for gen in a['generators']] if kind else [],u)
  ck(seed(width,visited,capacity,1,slots,size,1234567,bits,error,None));count=1;layers=[];fatal=0
  for depth in range(2048):
   if not count:break
   ids=read(front,count,u);raw=read(visited,capacity*width,C.c_int64)
   layers.append(sorted([[raw[e*capacity+i] for e in range(width)] for i in ids]))
   ck(gather(width,visited,capacity,front,count,parents,capacity,None));ck(cuda.cudaMemset(fc,0,4))
   ck(expand(kind,width,n,m,ng,parents,count,capacity,perms,matrices,mods,slots,size,visited,capacity,vc,future,capacity,fc,1234567,bits,error,None));ck(cuda.cudaDeviceSynchronize())
   fatal=read(error,1,u)[0]
   if fatal:break
   count=read(fc,1,u)[0];front,future=future,front
  else:raise AssertionError('oracle depth bound')
  if expect_resource:assert fatal&2,('resource error not reported',fatal)
  else:
   assert not fatal,fatal;expected=graph.exact_layers(capacity);assert layers==expected,(layers,expected)
  return {'hash_bits':bits,'states':sum(map(len,layers)),'layers':len(layers),'fatal':fatal,'scope':'single-device exact GPU loop'}
 finally:
  for q in alloc:ck(cuda.cudaFree(q))
fixtures=[GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],[0,1,2,3]),
 GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],[0,0,1,2]),
 GraphDefinition.matrix(2,1,[([1,1,0,1],257)],[256,2]),
 GraphDefinition.matrix(1,1,[([2],0)],[1]),
 GraphDefinition.matrix(1,1,[([2],5),([3],7)],[4]),
 GraphDefinition.permutation([list(range(1,300))+[0]],list(range(300)))]
results=[]
for device in (0,1):
 ck(cuda.cudaSetDevice(device))
 for i,g in enumerate(fixtures):
  for bits in (64,0):results.append(dict(run(g,bits),device=device,fixture=i))
 results.append(dict(run(GraphDefinition.permutation([[1,2,3,0]],[0,1,2,3]),0,capacity=3,expect_resource=True),device=device,fixture='resource'))
print(json.dumps({'status':'VERIFIED_GENERIC_EXACT_ALL_VISITED_GPU','runs':results,'forced_collisions_verified':True,'scope':'single-device tests on both GPUs; not distributed SHARD_AB integration'}))
