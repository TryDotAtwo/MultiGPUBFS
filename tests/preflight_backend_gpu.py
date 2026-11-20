"""Real two-GPU preflight must rehearse the requested memory backend."""
import json,os,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
root=Path(tempfile.mkdtemp(prefix='preflight-backend-',dir='/workspace'))
devices=list(range(int(os.environ.get('MGBFS_TEST_DEVICE_COUNT','2'))))
graph=GraphDefinition.permutation([[1,0,2],[0,2,1]],[0,1,2]); oracle=list(map(len,graph.exact_layers(100)))
with persistent_graph_workers():
 for idx,backend in enumerate(('legacy','segmented24','permutation_bitmap','permutation_bitmap')):
  r=run_graph(graph,root/str(idx),devices=devices,autotune=False,shards=1,
              memory_backend=backend,segment_rows=32)
  c=r['communication_preflight']
  assert r['status']=='COMPLETE' and r['layer_sizes']==oracle,(backend,r)
  assert c['rehearsal_backend']==backend,(backend,c)
  assert c['rehearsal_graph']==graph.digest(),c
  assert c['cache_hit']==(idx==3),c
  if idx==3:assert c['seconds']==0.0 and c['source_preflight_seconds']>0,c
  print(json.dumps(dict(backend=backend,status=r['status'],layers=r['layer_sizes'],preflight=c['bfs_prefix'],cache_hit=c['cache_hit'],preflight_seconds=c['seconds'])),flush=True)
 graph2=GraphDefinition.permutation([[1,0,2],[0,2,1]],[0,0,1])
 oracle2=list(map(len,graph2.exact_layers(100)))
 r=run_graph(graph2,root/'new-graph',devices=devices,autotune=False,shards=1,
             memory_backend='permutation_bitmap',segment_rows=32)
 c=r['communication_preflight']
 assert r['status']=='COMPLETE' and r['layer_sizes']==oracle2,r
 assert c['transport_cache_hit'] is True and c['rehearsal_graph']==graph2.digest(),c
 assert c['cache_source_graph']==graph.digest() and c['bfs_prefix'] is not None,c
 print(json.dumps(dict(backend='permutation_bitmap',case='new-graph',status=r['status'],layers=r['layer_sizes'],cache_hit=c['cache_hit'],transport_cache_hit=c['transport_cache_hit'],preflight_seconds=c['seconds'])),flush=True)
 # An execution-mode change must run a new prefix, while reusing the NCCL
 # transport probe already verified for these ranks and backend.
 os.environ['MGBFS_SEGMENT_EXACT_EXCHANGE']='0' if os.environ.get('MGBFS_SEGMENT_EXACT_EXCHANGE')=='1' else '1'
 r=run_graph(graph2,root/'new-mode',devices=devices,autotune=False,shards=1,
             memory_backend='permutation_bitmap',segment_rows=32)
 c=r['communication_preflight']
 assert r['status']=='COMPLETE' and r['layer_sizes']==oracle2,r
 assert c['transport_cache_hit'] is True and c['rehearsal_graph']==graph2.digest() and c['seconds']>0,c
 print(json.dumps(dict(backend='permutation_bitmap',case='new-mode',status=r['status'],transport_cache_hit=c['transport_cache_hit'],preflight_seconds=c['seconds'])),flush=True)
print('EVIDENCE',root,flush=True)
