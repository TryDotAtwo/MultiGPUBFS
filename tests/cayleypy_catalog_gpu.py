"""Compare real pinned CayleyPy catalog actions with NumPy and CUDA."""
import sys,json,inspect,runpy
from pathlib import Path
import numpy as np
from cayleypy import PermutationGroups,MatrixGroups,Puzzles
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from multigpubfs.graph_definition import GraphDefinition,from_cayleypy
ns=runpy.run_path(str(Path(__file__).with_name('generic_graph_gpu.py')))
fixtures=[]
special={'conjugacy_classes':{'n':4,'classes':{(2,1,1):None}},'cubic_pancake':{'n':4,'subset':2},'koltsov3':{'n':6},'sheveleva2':{'n':4,'k':1},'top_spin':{'n':4,'k':3}}
for name in dir(PermutationGroups):
 if name.startswith('_') or not callable(getattr(PermutationGroups,name)):continue
 fn=getattr(PermutationGroups,name);kwargs=special.get(name,{})
 if not kwargs:
  for key,p in inspect.signature(fn).parameters.items():
   if p.default is inspect.Parameter.empty:kwargs[key]={'n':4,'k':2}[key]
 fixtures.append((name,fn(**kwargs)))
for name in dir(MatrixGroups):
 if name.startswith('_') or not callable(getattr(MatrixGroups,name)):continue
 fixtures.append((name,getattr(MatrixGroups,name)(n=3,modulo=3)))
for name in dir(Puzzles):
 if name.startswith('_') or not callable(getattr(Puzzles,name)):continue
 opts={'globe_puzzle':{'a':2,'b':3},'hungarian_rings':{'left_size':5,'left_index':2,'right_size':5,'right_index':2},'rubik_cube':{'cube_size':2,'metric':'QTM'}}.get(name,{})
 fixtures.append((name,getattr(Puzzles,name)(**opts)))
results=[]
for name,d in fixtures:
 g=from_cayleypy(d);parent=g.start
 for i in range(g.generator_count):
  if d.generators_type.name=='PERMUTATION':expected=np.asarray(parent,dtype=np.int64)[d.generators_permutations[i]].tolist()
  else:
   gen=d.generators_matrices[i];expected=gen.apply(np.asarray(parent,dtype=np.int64).reshape(gen.n,-1)).reshape(-1).tolist()
  assert g.successor(parent,i)==expected,name
 for device in (0,1):
  ns['ck'](ns['cuda'].cudaSetDevice(device));ns['case'](g,[parent])
 results.append({'name':name,'elements':g.state_elements,'generators':g.generator_count})
print(json.dumps({'status':'VERIFIED_CAYLEYPY_CATALOG_SUCCESSORS','catalog_cases':results,'scope':'catalog representative definitions; successor semantics on both GPUs, not full puzzle BFS'}))
