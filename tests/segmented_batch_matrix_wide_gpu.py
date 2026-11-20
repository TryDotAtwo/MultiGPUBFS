"""Check segmented hash for 64-bit matrix actions and wide permutation states."""
import json,os,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
root=Path(tempfile.mkdtemp(prefix='segmented32-matrix-wide-',dir='/workspace'))
os.environ['MGBFS_SEGMENT_ROWS']='3'
os.environ['MGBFS_SEGMENT_TEST_BATCH']='1'
os.environ['MGBFS_SEGMENT_BATCH_TILE']='2'
tail=list(range(6,25))
fixtures=[
 ('matrix',GraphDefinition.matrix(2,1,[([1,1,0,1],3),([1,0,1,1],5)],[0,1])),
 ('signed',GraphDefinition.matrix(2,1,[([-1,0,0,1],0),([1,0,0,-1],0)],[-(1<<40),2])),
 ('directed',GraphDefinition.matrix(1,1,[([2],7)],[1])),
 ('wide-permutation',GraphDefinition.permutation([list(range(1,6))+[0]+tail,[1,0]+list(range(2,25))],list(range(25)))),
]
records=[]
with persistent_graph_workers():
 for name,graph in fixtures:
  expected=graph.exact_layers(1200)
  report=run_graph(graph,root/name,devices=[0,1],shards=4,autotune=False,
                   memory_backend='segmented',segment_rows=1<<32,segment_strategy='batch',_batch=2,max_seconds=60)
  assert report['status']=='COMPLETE',(name,report['reason'])
  assert report['layer_sizes']==list(map(len,expected)),name
  state=json.loads((root/name/'states.json').read_text())
  assert sorted(state['current'])==expected[-1],name
  assert sorted(state['previous_small'])==(expected[-2] if len(expected)>1 and len(expected[-2])<1000 else []),name
  item={'case':name,'states':sum(report['layer_sizes']),'layers':len(expected),
        'state_bytes':report['plan']['state_bytes'],'history_banks':report['plan']['history_layers'],
        'segments':[p['segment_count'] for p in report['segment_memory_by_rank']]}
  records.append(item);print('PASS',json.dumps(item),flush=True)
(root/'evidence.json').write_text(json.dumps(records,indent=2))
print('BATCH_HASH_MATRIX_WIDE_TWO_GPU_PASS',root,flush=True)
