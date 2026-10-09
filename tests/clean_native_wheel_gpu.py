from pathlib import Path
import os,json,sys,hashlib,subprocess
import multigpubfs
from multigpubfs import GraphDefinition,from_cayleypy,run_graph
from multigpubfs.native_distribution import native_runtime
from cayleypy import PermutationGroups,CayleyGraphDef,MatrixGenerator
import numpy as np
r=Path('/root/universal/clean-installed-gate');r.mkdir(exist_ok=True)
assert 'site-packages' in str(Path(multigpubfs.__file__).resolve()),multigpubfs.__file__
assert not any(os.environ.get(k) for k in ('PYTHONPATH','MGBFS_EXECUTABLE','MGBFS_CUDA_LIB_DIR','MGBFS_CUDART_LIB_DIR','LD_LIBRARY_PATH'))
native,env=native_runtime();manifest=json.loads((Path(native).parents[1]/'manifest.json').read_text());assert manifest['source_commit']=='28d5aa3cab27bc7bdb7882742a33e7a0b68fc2b0'
fixtures=[PermutationGroups.lx(4),CayleyGraphDef.create([[1,2,3,0],[1,0,2,3]],central_state=[0,0,1,2]),CayleyGraphDef.for_matrix_group(generators=[MatrixGenerator.create([[1,1],[0,1]],257)],central_state=np.array([[256],[2]],dtype=np.int64))]
results=[]
for history in ('hash','sorted'):
 os.environ['MGBFS_GENERIC_HISTORY']=history
 for world in (1,2):
  for index,definition in enumerate(fixtures):
   g=from_cayleypy(definition);oracle=g.exact_layers(1024);out=r/f'{history}-{world}-{index}'
   x=run_graph(definition,out,devices=list(range(world)),backend='generic',max_seconds=30)
   assert x['status']=='COMPLETE' and x['layer_sizes']==list(map(len,oracle)),x
   state=json.loads((out/'states.json').read_text());assert sorted(state['current'])==oracle[-1]
   if len(oracle[-2])<1000:assert sorted(state['previous_small'])==oracle[-2]
   results.append({'history':history,'world':world,'fixture':index,'states':sum(x['layer_sizes']),'report':x})
 for world in (1,2):
  out=r/f'{history}-{world}-resource';g=from_cayleypy(fixtures[0]);oracle=g.exact_layers(1024);x=run_graph(g,out,devices=list(range(world)),capacity=3,shards=4,backend='generic',max_seconds=30,_batch=1)
  assert x['status']=='INCOMPLETE' and x['reason'].startswith('RESOURCE_'),x
  depth=len(x['layer_seconds']);assert x['layer_sizes']==list(map(len,oracle[:depth+1])),x
  state=json.loads((out/'states.json').read_text());assert sorted(state['current'])==oracle[depth]
  if depth and len(oracle[depth-1])<1000:assert sorted(state['previous_small'])==oracle[depth-1]
  results.append({'history':history,'world':world,'fixture':'resource','report':x})
os.environ.pop('MGBFS_GENERIC_HISTORY')
g=GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],[0,1,2,3]);definition=r/'cli-definition.json';definition.write_text(g.to_json());cli=Path(sys.executable).parent/'multigpubfs'
x=subprocess.run([str(cli),str(definition),str(r/'cli'), '--devices','0,1','--backend','generic','--seconds','30'],capture_output=True,text=True);assert x.returncode==0,x.stderr
report=json.loads(x.stdout);assert report['status']=='COMPLETE' and report['layer_sizes']==list(map(len,g.exact_layers(24)))
receipt={'status':'VERIFIED_CLEAN_NATIVE_WHEEL_CAYLEYPY_ONE_TWO_GPU','module':multigpubfs.__file__,'python':sys.executable,'manifest':manifest,'cases':results,'cli':report,'scope':'Fresh venv, bundled native CLI/CUDA, pip runtime dependencies, three CayleyPy definitions in hash/sorted on one/two RTX3060 and resource retention; no source overrides, no Blackwell or larger-rank claim'}
(r/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'status':receipt['status'],'cases':len(results),'module':multigpubfs.__file__}))
