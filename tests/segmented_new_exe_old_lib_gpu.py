"""New executable against old ABI: supported defaults work, unsafe modes fail."""
import os,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
root=Path(tempfile.mkdtemp(prefix='segmented-old-lib-',dir='/workspace'))
native=os.environ['MGBFS_EXECUTABLE'];g=GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],[0,1,2,3]);want=[1,3,5,6,5,3,1]
with persistent_graph_workers():
 for backend in ('segmented24','permutation_bitmap'):
  r=run_graph(g,root/backend,devices=[0,1],shards=4,autotune=False,executable=native,memory_backend=backend,segment_strategy='atomic',segment_rows=64,_batch=8)
  assert r['status']=='COMPLETE' and r['layer_sizes']==want,(backend,r)
  print('PASS OLD LIB DEFAULT',backend,flush=True)
 for label,opts,expected in [
  ('warp',dict(memory_backend='segmented24',segment_strategy='warp'), 'SEGMENT_WARP_SAFE_NATIVE_ABI_MISSING'),
  ('page',dict(memory_backend='permutation_bitmap',segment_strategy='atomic',bitmap_owner_page=64),'BITMAP_OWNER_PAGE_NATIVE_ABI_MISSING'),
 ]:
  try:run_graph(g,root/label,devices=[0,1],shards=4,autotune=False,executable=native,segment_rows=64,_batch=8,**opts)
  except Exception as e:
   text=str(e)
   # The supervisor sometimes includes the native error in preserved rank logs.
   logs='\n'.join(p.read_text(errors='replace') for p in (root/label).rglob('rank-*.log')) if (root/label).exists() else ''
   assert expected in text or expected in logs,(label,text,logs)
   print('PASS OLD LIB EXPLICIT ERROR',label,expected,flush=True)
  else:raise AssertionError(f'old library silently accepted {label}')
print('NEW_EXECUTABLE_OLD_LIBRARY_PASS',root,flush=True)
