"""CayleyPy public selector and repeated persistent-worker reuse on two GPUs."""
import os,tempfile
from pathlib import Path
from cayleypy import CayleyGraph,PermutationGroups
import multigpubfs
from multigpubfs.cayleypy_backend import NativeOptions
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.graph_definition import from_cayleypy
root=Path(tempfile.mkdtemp(prefix='bitmap-api-',dir='/workspace'))
os.environ['MGBFS_EXPERIMENTAL_SEGMENTED24']='1'
os.environ['MGBFS_EXPERIMENTAL_BITMAP']='1'
multigpubfs.enable_native(NativeOptions(output_root=root,devices=(0,1),autotune=False,memory_backend='permutation_bitmap'),recovery=False)
try:
 with persistent_graph_workers() as pool:
  first=None
  for n,backend in [(4,'permutation_bitmap'),(4,'segmented24'),(5,'legacy'),(5,'permutation_bitmap'),(4,'legacy')]:
   graph=CayleyGraph(PermutationGroups.coxeter(n),device='cpu')
   options=NativeOptions(output_root=root,devices=(0,1),autotune=False,memory_backend=backend)
   result=graph.bfs(native_options=options);expected=from_cayleypy(graph).exact_layers(1000)
   assert result.bfs_completed and result.layer_sizes==list(map(len,expected)),result.report
   actual=result.report['backend']
   expected_name={'permutation_bitmap':'EXPERIMENTAL_PERMUTATION_BITMAP','segmented24':'EXPERIMENTAL_SEGMENTED24'}.get(backend)
   if expected_name:assert actual==expected_name,(backend,actual)
   else:assert actual not in ('EXPERIMENTAL_PERMUTATION_BITMAP','EXPERIMENTAL_SEGMENTED24'),actual
   ids=[w.process.pid for w in pool.workers]
   if first is None:first=ids
   assert ids==first,(ids,first)
   print('PASS',n,backend,'rank pids',ids,flush=True)
  graph=CayleyGraph(PermutationGroups.coxeter(4),device='cpu')
  options=NativeOptions(output_root=root,devices=(0,1),autotune=True,memory_backend='permutation_bitmap')
  result=graph.bfs(native_options=options)
  assert result.bfs_completed and result.report['backend']=='EXPERIMENTAL_PERMUTATION_BITMAP',result.report
  print('PASS bitmap autotune small',result.report.get('autotune',{}).get('status'),flush=True)
finally:multigpubfs.disable_native()
print('EVIDENCE',root)
