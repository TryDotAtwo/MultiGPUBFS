"""Public API and resident-worker correctness; bounded small GPU evidence only."""
import json,os,tempfile
from pathlib import Path
from cayleypy import CayleyGraph,PermutationGroups
import multigpubfs
from multigpubfs.cayleypy_backend import NativeOptions
from multigpubfs.graph_definition import from_cayleypy
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

root=Path(tempfile.mkdtemp(prefix='memory-presets-api-',dir='/workspace'))
native=os.environ['MGBFS_EXECUTABLE']
graph=CayleyGraph(PermutationGroups.coxeter(5),device='cpu')
definition=from_cayleypy(graph);expected=definition.exact_layers(1000)
expected_sizes=list(map(len,expected)); ids=None; records=[]
os.environ['MGBFS_SEGMENT_ROWS']='4'
os.environ['MGBFS_SEGMENT_TEST_BATCH']='1'
os.environ['MGBFS_SEGMENT_BATCH_TILE']='2'
multigpubfs.enable_native(recovery=False)
try:
 with persistent_graph_workers() as pool:
  for backend in ('segmented24','permutation_bitmap'):
   # Intentionally put balanced after max in the same worker pool.
   for preset in ('max_capacity','balanced'):
    options=NativeOptions(devices=(0,1),autotune=False,memory_backend=backend,memory_preset=preset,segment_rows=4,segment_strategy='batch')
    result=graph.bfs(native_options=options,output=root/(backend+'-'+preset+'-full'))
    assert result.bfs_completed and result.layer_sizes==expected_sizes,(backend,preset,result.report)
    geometry=result.report['segment_memory_by_rank'];assert len(geometry)==2
    if backend=='permutation_bitmap':assert all(x['banks']==(2 if preset=='max_capacity' else 3) for x in geometry),geometry
    else:
     expected_width=4
     widths=[w for x in geometry for bucket in x['position_item_bytes_by_shard'] for w in bucket]
     assert widths and all(w==expected_width for w in widths),(preset,widths)
    current=[w.process.pid for w in pool.workers]
    if ids is None:ids=current
    assert current==ids,(current,ids)
    out=root/(backend+'-'+preset+'-incomplete')
    report=run_graph(definition,out,devices=[0,1],shards=4,autotune=False,executable=native,
       memory_backend=backend,memory_preset=preset,segment_rows=4,segment_strategy='batch',capacity=3)
    assert report['status']=='INCOMPLETE' and report['reason']=='RESOURCE_10',report
    depth=len(report['layer_sizes'])-1;snapshot=json.loads((out/'states.json').read_text())
    assert report['layer_sizes']==expected_sizes[:depth+1]
    assert sorted(snapshot['current'])==expected[depth],(backend,preset,'current')
    assert sorted(snapshot['previous_small'])==(expected[depth-1] if depth else []),(backend,preset,'previous')
    assert [w.process.pid for w in pool.workers]==ids
    rec={'backend':backend,'preset':preset,'full_layers':len(expected_sizes),'incomplete_depth':depth,
         'banks':[x['banks'] for x in geometry],'peak_live_rows':[x['peak_live_rows'] for x in geometry],
         'workers':ids,'full_seconds':result.report['bfs_seconds']}
    records.append(rec);print(json.dumps(rec),flush=True)
  # Run another complete job after all four resource stops on the same workers.
  final=graph.bfs(native_options=NativeOptions(devices=(0,1),autotune=False,memory_backend='permutation_bitmap',memory_preset='balanced',segment_rows=4,segment_strategy='batch'),output=root/'after-stops')
  assert final.bfs_completed and final.layer_sizes==expected_sizes
  assert [w.process.pid for w in pool.workers]==ids
finally:
 multigpubfs.disable_native()
(root/'evidence.json').write_text(json.dumps(records,indent=2))
print('BATCH_PRESETS_TWO_GPU_PASS',root,flush=True)
