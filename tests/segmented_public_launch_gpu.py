"""End-to-end Python launcher and resident rank reuse for private segmented mode."""
import json,os,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.launch import run_graph
from multigpubfs.generic_session import persistent_graph_workers
root=Path(tempfile.mkdtemp(prefix='seg24-launch-',dir='/workspace'))
env=dict(os.environ,MGBFS_SEGMENT_ROWS='3',MGBFS_GENERIC_GENERATOR='cuda',MGBFS_GENERIC_HISTORY='hash',MGBFS_GENERIC_TRANSPORT='full',MGBFS_GENERIC_SORT='none')
native=os.environ['MGBFS_EXECUTABLE']
cases=[('s4',GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],list(range(4)))),('directed',GraphDefinition.permutation([[1,2,3,0]],list(range(4)))),('repeated',GraphDefinition.permutation([[1,0,2,3,4],[0,2,1,3,4],[0,1,3,2,4],[0,1,2,4,3]],[0,0,1,2,3]))]
with persistent_graph_workers() as pool:
 ids=None
 for name,g in cases:
  report=run_graph(g,root/name,devices=[0,1],autotune=False,shards=4,executable=native,memory_backend='segmented24',_native_env=env)
  assert report['status']=='COMPLETE',report
  assert report['layer_sizes']==[len(x) for x in g.exact_layers(1000)],report
  assert 'SEGMENTED24' in report['backend'],report
  if name=='s4':assert max(json.loads((root/name/f'rank-{i}'/'report.json').read_text())['peak_owner_lanes'] for i in range(2))>1
  current=[w.process.pid for w in pool.workers]
  if ids is not None:assert ids==current,(ids,current)
  ids=current;print('PASS persistent',name,'pids',ids,flush=True)
# Explicit limits are per-rank, including values above 2^32; they do not
# narrow the live segmented address space or the configured limit.
g=cases[0][1];expected=g.exact_layers(1000)
with persistent_graph_workers():
 for cap in (3,(1<<32)+17):
  out=root/('cap-'+str(cap));report=run_graph(g,out,devices=[0,1],autotune=False,shards=4,capacity=cap,executable=native,memory_backend='segmented24',_native_env=env)
  assert report['segment_memory']['capacity_limit_per_rank']==cap,report
  if cap==3:
   assert report['status']=='INCOMPLETE' and report['reason']=='RESOURCE_10',report
   depth=len(report['layer_sizes'])-1;states=json.loads((out/'states.json').read_text())
   assert report['layer_sizes']==[len(v) for v in expected[:depth+1]]
   assert sorted(states['current'])==expected[depth]
   assert sorted(states['previous_small'])==(expected[depth-1] if depth else [])
  else:assert report['status']=='COMPLETE' and report['layer_sizes']==[len(v) for v in expected]
  print('PASS explicit capacity',cap,report['status'],flush=True)
print('EVIDENCE',root)
