import json,os,time
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
from multigpubfs.autotune import choose_profile
from multigpubfs.native_distribution import native_runtime
r=Path('/root/universal/history-autotune-gate');r.mkdir(exist_ok=True)
os.environ['MGBFS_PROFILE_CACHE']=str(r/'cache')
g=GraphDefinition.permutation([[1,2,3,4,5,6,7,0]+list(range(8,25)),[1,0,2,3,4,5,6,7]+list(range(8,25))],list(range(25)))
oracle=g.exact_layers(40320)
report=run_graph(g,r/'automatic',max_seconds=90,backend='generic')
assert report['status']=='COMPLETE',report
assert report['plan']['state_bytes']==1 and report['plan']['capacity']<=40320,report
assert report['layer_sizes']==list(map(len,oracle)),report
p=report['autotune'];assert len(p['pilots'])==11;assert p['status']=='MEASURED_EQUAL_PREFIX_GPU_PROFILE',p
assert p['common_depth']>=4 and sum(p['common_layer_sizes'][2:])>=32768,p
assert {x.get('owner_lanes') for x in p['pilots'] if x.get('history_algorithm')=='SORTED_RUNS'}=={1,2,4,8},p
assert p['shards'] in (1,4,8,16) and p['batch_fraction'] in (1,.25)
states=json.loads((r/'automatic/states.json').read_text())
assert sorted(states['current'])==oracle[-1]
if len(oracle[-2])<1000:assert sorted(states['previous_small'])==oracle[-2]
else:assert not states['previous_small']
native,env=native_runtime();cached=choose_profile(g,None,None,90,native,env,allow_specialized=False)
assert cached['cache_hit'] and (cached['shards'],cached['batch_fraction'])==(p['shards'],p['batch_fraction'])
small=GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],[0,1,2,3])
manual=run_graph(small,r/'manual',devices=[1,0],shards=4,capacity=24,max_seconds=30,_batch=2,_profile_layers=2,autotune=False)
assert manual['reason']=='PROFILE_LAYER_LIMIT' and manual['layer_sizes']==[1,2,3],manual
small_report=run_graph(small,r/'small',max_seconds=30)
assert small_report['autotune']['status']=='SMALL_OR_SHORT_WORKLOAD_CONSERVATIVE_PROFILE'
receipt={'status':'VERIFIED_HISTORY_LANES_AUTOTUNE_TWO_GPU','automatic':report,'cache_hit_seconds':cached['seconds'],'profile_depth_limit':manual,'small':small_report,'scope':'actual two RTX3060 same-graph prefix choice, automatic production complete graph CPU oracle, cache and native batch/depth controls; global optimum and larger scales not established'}
(r/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
