import json,os
from pathlib import Path
from unittest.mock import patch
from multigpubfs import GraphDefinition,run_graph
r=Path('/root/universal/history-selected-sorted-final-gate');r.mkdir(exist_ok=True)
g=GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],list(range(4)));oracle=g.exact_layers(24)
p={'backend':'generic','status':'DISPATCH_FIXTURE_ONLY_NOT_TIMING','shards':4,'batch_fraction':.25,'transport':'parent','candidate_order':'none','history_algorithm':'SORTED_RUNS','owner_lanes':2,'batch':2,'measured':True,'seconds':0}
with patch('multigpubfs.autotune.choose_profile',return_value=p):x=run_graph(g,r/'run',devices=[0,1],max_seconds=30,backend='generic')
assert x['status']=='COMPLETE' and x['layer_sizes']==list(map(len,oracle)),x
assert x['plan']['history_algorithm']=='SORTED_RUNS' and x['plan']['owner_lanes']==2 and x['plan']['batch']==2,x
states=json.loads((r/'run/states.json').read_text());assert sorted(states['current'])==oracle[-1] and sorted(states['previous_small'])==oracle[-2]
(r/'verification.json').write_text(json.dumps({'status':'VERIFIED_SELECTED_SORTED_PROFILE_DISPATCH','report':x,'scope':'Injected selection validates actual sorted worker dispatch and exact CPU oracle; no timing claim'},indent=2))
print('VERIFIED_SELECTED_SORTED_PROFILE_DISPATCH')

os.environ['MGBFS_GENERIC_HISTORY']='sorted';os.environ['MGBFS_GENERIC_OWNER_LANES']='8'
y=run_graph(g,r/'forced-small',devices=[0,1],max_seconds=30,backend='generic')
assert y['status']=='COMPLETE' and y['layer_sizes']==list(map(len,oracle)),y
assert y['plan']['history_algorithm']=='SORTED_RUNS' and y['plan']['owner_lanes']==8 and y['plan']['shards']>=8,y
(r/'forced-verification.json').write_text(json.dumps({'status':'VERIFIED_FORCED_SORTED_SMALL_PROFILE','report':y},indent=2))
print('VERIFIED_FORCED_SORTED_SMALL_PROFILE')
