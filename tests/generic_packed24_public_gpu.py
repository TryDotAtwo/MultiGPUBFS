"""Automatic inverse-closed three-bank dispatch and exact terminal boundaries."""
import os,json,subprocess,time,signal,hashlib
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
r=Path(os.environ.get('MGBFS_ROLLING_PUBLIC_ROOT','/root/universal/packed24-public-gate'));r.mkdir(exist_ok=True);checks=[]
def prefix(g,layers):
 seen={tuple(g.start)};frontier=seen.copy();out=[]
 for _ in range(layers):
  out.append(sorted(map(list,frontier)));future={tuple(g.successor(list(x),i)) for x in frontier for i in range(len(g.generator_names))}-seen;seen|=future;frontier=future
 return out
for selection in ([0],[1,0]):
 for name,g,capacity in [('cycle',GraphDefinition.matrix(2,1,[([1,1,0,1],257),([1,-1,0,1],257)],[256,2]),3),*[(f'wide{n}',GraphDefinition.permutation([list(range(1,n))+[0],[n-1]+list(range(n-1))],list(range(n))),4) for n in (16,17,23,24,25)],('signed',GraphDefinition.matrix(2,1,[([-1,0,0,1],0),([1,0,0,-1],0)],[-(1<<40),2]),4)]:
  expected=g.exact_layers(300);folder=r/f'{name}-{len(selection)}';v=run_graph(g,folder,devices=selection,capacity=capacity,shards=1,max_seconds=30,autotune=False,_batch=2)
  assert v['status']=='COMPLETE' and v['plan']['history_layers']==3 and v['backend']=='GENERIC_DISTRIBUTED_EXACT_THREE_BANK_HISTORY';assert v['layer_sizes']==list(map(len,expected));data=json.loads((folder/'states.json').read_text());assert sorted(data['current'])==expected[-1] and sorted(data['previous_small'])==expected[-2]
  checks.append({'case':name,'devices':selection,'automatic_three_bank':True,'complete_exact':True})

receipt={'status':'VERIFIED_PACKED24_PUBLIC_BOUNDARIES','checks':checks};(r/'verification.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
