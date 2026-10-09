import json
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
r=Path('/root/universal/unified-deadline-gate')
g=GraphDefinition.matrix(2,1,[([1,1,0,1],0)],[0,1])
report=run_graph(g,r,devices=[0,1],capacity=8192,max_seconds=1)
assert report['status']=='INCOMPLETE' and report['reason']=='DEADLINE';assert set(report['layer_sizes'])=={1}
states=json.loads((r/'states.json').read_text());depth=len(report['layer_sizes'])-1;assert states['current']==[[depth,1]];assert states['previous_small']==[[depth-1,1]]
(r/'verification.json').write_text(json.dumps({'status':'VERIFIED_UNIFIED_DISTRIBUTED_DEADLINE','completed_depth':depth,'last_two_layers_exact':True}));print('VERIFIED_DISTRIBUTED_DEADLINE',depth)
