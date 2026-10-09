from pathlib import Path
import json,os
from multigpubfs import GraphDefinition,run_graph
r=Path('/root/universal/sorted-public');r.mkdir(exist_ok=True);checks=[]
assert os.environ.get('MGBFS_GENERIC_SORT')=='radix'
fixtures=[]
for n in (25,31,300):
 fixtures.append((f'inverse{n}',GraphDefinition.permutation([list(range(1,n))+[0],[n-1]+list(range(n-1))],list(range(n))),4))
 fixtures.append((f'directed{n}',GraphDefinition.permutation([list(range(1,n))+[0]],list(range(n))),n))
n=25;tail=list(range(6,n));fixtures += [('wide-lrx',GraphDefinition.permutation([list(range(1,6))+[0]+tail,[5]+list(range(5))+tail,[1,0]+list(range(2,n))],list(range(n))),720),('wide-lx',GraphDefinition.permutation([list(range(1,6))+[0]+tail,[1,0]+list(range(2,n))],list(range(n))),720),('matrix',GraphDefinition.matrix(2,1,[([1,1,0,1],3),([1,0,1,1],5)],[0,1]),100),('signed',GraphDefinition.matrix(2,1,[([-1,0,0,1],0),([1,0,0,-1],0)],[-(1<<40),2]),4),('overflow',GraphDefinition.matrix(1,1,[([(1<<63)-1],3)],[2]),4)]
for devices in ([0],[1,0]):
 for name,g,capacity in fixtures:
  expected=g.exact_layers(1200);out=r/(name+'-'+str(len(devices)))
  v=run_graph(g,out,devices=devices,capacity=capacity,shards=3,_batch=2,autotune=False,max_seconds=30)
  assert v['plan']['sort_candidates'] and v['plan']['parent_transport'],(name,v['plan'])
  assert v['status']=='COMPLETE' and v['layer_sizes']==list(map(len,expected)),(name,v['status'],v['reason'],v['layer_sizes'])
  states=json.loads((out/'states.json').read_text());assert sorted(states['current'])==expected[-1];assert sorted(states['previous_small'])==(expected[-2] if len(expected)>1 and len(expected[-2])<1000 else [])
  checks.append({'case':name,'devices':devices,'states':sum(v['layer_sizes']),'history_layers':v['plan']['history_layers'],'state_bytes':v['plan']['state_bytes'],'planned_bytes':v['plan']['device_bytes'],'all_layers_terminal_exact':True})
v={'status':'VERIFIED_SORTED_PARENT_PUBLIC_ONE_TWO_GPU','checks':checks};(r/'verification.json').write_text(json.dumps(v));print(json.dumps(v))
