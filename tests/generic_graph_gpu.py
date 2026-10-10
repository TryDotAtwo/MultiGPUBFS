"""Standalone CUDA successor gate; full exact comparison, not throughput acceptance."""
import ctypes as C,json,sys,os
from ctypes.util import find_library
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from multigpubfs.graph_definition import GraphDefinition
lib=C.CDLL(sys.argv[1]);fn=lib.mgbfs_generic_generate_i64
fn.argtypes=[C.c_uint32,C.c_uint32,C.c_uint32,C.c_uint32,C.c_uint32,C.c_void_p,C.c_uint32,C.c_uint32,C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_uint32,C.c_void_p,C.c_uint32,C.c_void_p,C.c_void_p];fn.restype=C.c_int
cuda=C.CDLL(os.environ.get('MGBFS_CUDART_LIBRARY') or find_library('cudart') or 'libcudart.so');cuda.cudaMalloc.argtypes=[C.POINTER(C.c_void_p),C.c_size_t];cuda.cudaMemcpy.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int];cuda.cudaFree.argtypes=[C.c_void_p]
def ck(v):
 if v:raise RuntimeError('CUDA_STATUS_'+str(v))
def array(values,ty):return (ty*len(values))(*values)
def case(g,parents,selection=None):
 width=g.state_elements;count=len(parents);children=count*g.generator_count;rows=children if selection is None else len(selection);alloc=[]
 def upload(vals,ty):
  host=array(vals,ty);ptr=C.c_void_p();ck(cuda.cudaMalloc(C.byref(ptr),max(1,C.sizeof(host))));alloc.append(ptr)
  if vals:ck(cuda.cudaMemcpy(ptr,host,C.sizeof(host),1))
  return ptr
 try:
  inputs=upload([parent[e] for e in range(width) for parent in parents],C.c_int64)
  a=g.action;kind=0 if a['kind']=='permutation' else 1;n=a['degree'] if kind==0 else a['rows'];m=1 if kind==0 else a['cols']
  perm=upload([x for gen in a['generators'] for x in gen] if kind==0 else [],C.c_uint32)
  matrix=upload([x for gen in a['generators'] for x in gen['matrix']] if kind else [],C.c_int64)
  mods=upload([gen['modulo'] for gen in a['generators']] if kind else [],C.c_uint32)
  indices=upload(selection or [],C.c_uint64) if selection is not None else None
  output=upload([0]*(width*rows),C.c_int64);error=upload([0],C.c_uint32)
  ck(fn(kind,width,n,m,g.generator_count,inputs,count,count,perm,matrix,mods,indices,rows,output,rows,error,None));ck(cuda.cudaDeviceSynchronize())
  host=array([0]*(width*rows),C.c_int64);ck(cuda.cudaMemcpy(host,output,C.sizeof(host),2));err=array([0],C.c_uint32);ck(cuda.cudaMemcpy(err,error,4,2));assert err[0]==0
  ids=list(range(children)) if selection is None else selection
  expected=[g.successor(parents[i//g.generator_count],i%g.generator_count) for i in ids]
  actual=[[host[e*rows+i] for e in range(width)] for i in range(rows)]
  assert actual==expected,(actual,expected)
 finally:
  for ptr in alloc:ck(cuda.cudaFree(ptr))
 return rows
cases=[(GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],[0,0,1,2]),[[0,0,1,2],[2,0,0,1]]),
 (GraphDefinition.matrix(2,1,[([1,1,0,1],257)],[256,2]),[[256,2],[1,2]]),
 (GraphDefinition.matrix(1,1,[([2],7),([2],0)],[2**63-1]),[[2**63-1],[-2]]),
 (GraphDefinition.matrix(1,1,[([2],5),([3],7)],[4]),[[4],[2]]),
 (GraphDefinition.permutation([list(range(1,300))+[0]],list(range(300))),[list(range(300))])]
rows=[]
for device in (0,1):
 ck(cuda.cudaSetDevice(device))
 for g,parents in cases:
  full=case(g,parents);selected=case(g,parents,[full-1,0]);rows.append({'device':device,'elements':g.state_elements,'full_children':full,'selected_children':selected})
print(json.dumps({'status':'VERIFIED_GENERIC_CUDA_SUCCESSORS','cases':rows,'scope':'primitive full-state and selected regeneration only; not integrated BFS or throughput'}))
