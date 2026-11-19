"""Public CayleyPy API, pool reuse, backend switching and real autotuner."""
import os,tempfile
from pathlib import Path
from cayleypy import CayleyGraph,PermutationGroups
import multigpubfs
from multigpubfs.cayleypy_backend import NativeOptions
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.graph_definition import from_cayleypy
os.environ['MGBFS_EXPERIMENTAL_SEGMENTED24']='1' # explicit legacy must override this
root=Path(tempfile.mkdtemp(prefix='seg24-cayleypy-',dir='/workspace'))
base=NativeOptions(output_root=root,devices=tuple(range(8)),autotune=False,memory_backend='segmented24')
multigpubfs.enable_native(base,recovery=False)
try:
 with persistent_graph_workers() as pool:
  first=None
  for name,n,backend in [('seg4',4,'segmented24'),('bitmap4',4,'permutation_bitmap'),('legacy4',4,'legacy'),('seg5',5,'segmented24'),('bitmap5',5,'permutation_bitmap'),('legacy5',5,'legacy')]:
   graph=CayleyGraph(PermutationGroups.coxeter(n),device='cpu')
   options=NativeOptions(output_root=root,devices=tuple(range(8)),autotune=False,memory_backend=backend)
   result=graph.bfs(native_options=options);expected=from_cayleypy(graph).exact_layers(1000)
   assert result.bfs_completed and result.layer_sizes==list(map(len,expected)),result.report
   actual=result.report['backend'];assert (actual=={'segmented24':'EXPERIMENTAL_SEGMENTED24','permutation_bitmap':'EXPERIMENTAL_PERMUTATION_BITMAP'}.get(backend,actual)),(backend,actual)
   if backend=='legacy':assert not actual.startswith('EXPERIMENTAL_'),actual
   ids=[w.process.pid for w in pool.workers]
   if first is None:first=ids
   assert ids==first,(ids,first)
   print('PASS CayleyPy',name,'backend',actual,'pids',ids,flush=True)
  graph=CayleyGraph(PermutationGroups.coxeter(4),device='cpu')
  opts=NativeOptions(output_root=root,devices=tuple(range(8)),autotune=True,memory_backend='segmented24')
  result=graph.bfs(native_options=opts);expected=from_cayleypy(graph).exact_layers(1000)
  assert result.bfs_completed and result.layer_sizes==list(map(len,expected)) and 'SEGMENTED24' in result.report['backend'],result.report
  print('PASS CayleyPy segmented autotune',result.report['status'],flush=True)
finally:multigpubfs.disable_native()
print('EVIDENCE',root)
