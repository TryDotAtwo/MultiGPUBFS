"""Installed public interface, actual one/two GPU launch and final oracle checks."""
import json,os
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
r=Path('/root/universal/unified-launch-gate');r.mkdir(exist_ok=True)
fixtures=[GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],[0,1,2,3]),GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],[0,0,1,2]),GraphDefinition.matrix(2,1,[([1,1,0,1],257)],[256,2])]
checks=[]
for index,g in enumerate(fixtures):
 for selection in ('auto','one','swapped-four'):
  out=r/f'fixture-{index}-{selection}';options={} if selection=='auto' else {'device':1} if selection=='one' else {'devices':[1,0],'shards':4}
  report=run_graph(g,out,max_seconds=30,**options);oracle=g.exact_layers(1024);assert report['status']=='COMPLETE';assert report['layer_sizes']==list(map(len,oracle))
  states=json.loads((out/'states.json').read_text());assert sorted(states['current'])==oracle[-1];assert sorted(states['previous_small'])==oracle[-2]
  if selection!='one':assert report['devices']==([0,1] if selection=='auto' else [1,0]);assert report['world']==2
  else:assert report['device']==1 and report['world']==1
  if index==0:assert (report.get('plan') or report)['capacity']==24
  checks.append({'fixture':index,'selection':selection,'states':sum(report['layer_sizes']),'layers':len(oracle),'world':report['world']})
  (r/'status.json').write_text(json.dumps({'status':'RUNNING','checks':checks}))
resource=run_graph(fixtures[0],r/'resource',devices=[0,1],capacity=3,shards=4,max_seconds=30);assert resource['status']=='INCOMPLETE' and resource['reason'].startswith('RESOURCE_')
oracle=fixtures[0].exact_layers(1024);assert resource['layer_sizes']==list(map(len,oracle[:len(resource['layer_sizes'])]));states=json.loads((r/'resource/states.json').read_text());assert sorted(states['current'])==oracle[len(resource['layer_sizes'])-1];assert sorted(states['previous_small'])==oracle[len(resource['layer_sizes'])-2]
report={'status':'VERIFIED_INSTALLED_UNIFIED_SINGLE_HOST_ONE_TWO_GPU','checks':checks,'resource_current_previous_verified':True,'scope':'Automatic all-visible GPU selection, explicit one/swapped devices, 1/4 shards, complete finite graph counts and final states. Memory admission, not throughput autotuning; multihost not verified.'}
(r/'verification.json').write_text(json.dumps(report,indent=2));(r/'status.json').write_text(json.dumps(report));print(json.dumps(report))
