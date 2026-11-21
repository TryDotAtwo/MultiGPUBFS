import os,json,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
root=Path(tempfile.mkdtemp(prefix='segmented-family-switch-'));native=os.environ['MGBFS_EXECUTABLE']
g=GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],[0,1,2,3]);layers=g.exact_layers(100)
with persistent_graph_workers() as pool:
 previous=None
 for i,mode in enumerate(('legacy','segmented24','permutation_bitmap','legacy')):
  args=dict(devices=[0,1],shards=2,autotune=False,_batch=4,memory_backend=mode,executable=native,_profile_layers=999)
  if mode!='legacy':args.update(segment_rows=3,segment_strategy='atomic')
  out=root/f'{i}-{mode}'
  result=run_graph(g,out,**args)
  assert result['status']=='COMPLETE' and result['layer_sizes']==list(map(len,layers)),result
  states=json.loads((out/'states.json').read_text());assert sorted(states['current'])==layers[-1] and sorted(states['previous_small'])==layers[-2]
  pids=[w.process.pid for w in pool.workers]
  if i in (1,3):assert pids!=previous,(mode,pids,previous)
  if i==2:assert pids==previous,(mode,pids,previous)
  print('PASS_FAMILY',mode,pids,'same_as_previous',pids==previous,flush=True);previous=pids
print('FAMILY_SWITCH_PASS',root,flush=True)
