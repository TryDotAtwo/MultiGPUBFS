"""End-to-end installed Python -> CLI -> Rust -> CUDA gate."""
import os,json,tempfile
from pathlib import Path
from cayleypy import PermutationGroups,CayleyGraphDef,MatrixGenerator
from multigpubfs import GraphDefinition,from_cayleypy,run_graph
import numpy as np
root=Path('/root/universal');os.environ['MGBFS_EXECUTABLE']=str(root/'src/target/release/mgbfs')
fixtures=[PermutationGroups.lx(4),CayleyGraphDef.create([[1,2,3,0],[1,0,2,3]],central_state=[0,0,1,2]),CayleyGraphDef.for_matrix_group(generators=[MatrixGenerator.create([[1,1],[0,1]],257)],central_state=np.array([[256],[2]],dtype=np.int64))]
results=[]
with tempfile.TemporaryDirectory(prefix='installed-gate-',dir=root) as temporary:
 for device in (0,1):
  for i,d in enumerate(fixtures):
   g=from_cayleypy(d);expected=g.exact_layers(1024);output=Path(temporary)/f'device-{device}-graph-{i}'
   report=run_graph(d,output,device=device,capacity=1024,max_seconds=10);assert report['status']=='COMPLETE',report;assert report['layer_sizes']==list(map(len,expected))
   states=json.loads((output/'states.json').read_text());assert sorted(states['current'])==expected[-1];assert sorted(states['previous_small'])==expected[-2]
   results.append({'device':device,'fixture':i,'status':report['status'],'states':sum(report['layer_sizes'])})
  output=Path(temporary)/f'device-{device}-resource';report=run_graph(fixtures[0],output,device=device,capacity=3,max_seconds=10);assert report['status']=='INCOMPLETE' and report['reason'].startswith('RESOURCE_');states=json.loads((output/'states.json').read_text());expected=from_cayleypy(fixtures[0]).exact_layers(100);assert sorted(states['current'])==expected[1];assert states['previous_small']==expected[0]
 g=GraphDefinition.permutation([[1,2,3,0]],[0,1,2,3],expected_max_unique_states=4);report=run_graph(g,Path(temporary)/'automatic-small',max_seconds=10);assert report['capacity']==4 and report['status']=='COMPLETE'
 os.environ['WORLD_SIZE']='2'
 try:run_graph(g,Path(temporary)/'invalid-topology')
 except RuntimeError as e:assert 'DISTRIBUTED_INTEGRATION_NOT_READY' in str(e)
 else:raise AssertionError('silently ignored requested GPUs')
print(json.dumps({'status':'VERIFIED_INSTALLED_PYTHON_CLI_GPU_GATE','cases':results,'resource_retention_verified':True,'automatic_small_bound_verified':True,'scope':'single-device general engine on both cards; not general distributed SHARD_AB'}))
