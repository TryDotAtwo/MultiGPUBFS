import os,json
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
os.environ['CUDA_MODULE_LOADING']='EAGER';os.environ['MGBFS_GENERIC_HISTORY']='sorted'
r=Path(os.environ.get('MGBFS_SORTED_SINGLE_GATE_ROOT','/tmp/mgbfs-sorted-one-shard'));r.mkdir(parents=True,exist_ok=False)
g=GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],list(range(4)));expected=g.exact_layers(24)
report=run_graph(g,r/'result',devices=[0],shards=1,max_seconds=30,autotune=False)
assert report['status']=='COMPLETE' and report['layer_sizes']==list(map(len,expected))
assert report['plan']['history_algorithm']=='SORTED_RUNS' and report['plan']['table_slots']==0 and report['plan']['owner_lanes']==1
states=json.loads((r/'result/states.json').read_text());assert sorted(states['current'])==expected[-1] and sorted(states['previous_small'])==expected[-2]
proof={'status':'VERIFIED_ONE_SHARD_DIRECTED_PUBLIC_SORTED_DISPATCH','states':sum(map(len,expected)),'layers':len(expected),'single_native_hash_shortcut_bypassed':True};(r/'verification.json').write_text(json.dumps(proof));print(json.dumps(proof))
