import json,os,hashlib
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
r=Path(os.environ.get('MGBFS_SORTED_PUBLIC_GATE_ROOT','/tmp/mgbfs-sorted-public-admission'));r.mkdir(parents=True,exist_ok=False)
os.environ['CUDA_MODULE_LOADING']='EAGER';os.environ['MGBFS_GENERIC_HISTORY']='sorted';checks=[]
fixtures=[GraphDefinition.permutation([[1,2,3,4,0],[4,0,1,2,3],[1,0,2,3,4]],list(range(5))),GraphDefinition.matrix(2,1,[([1,1,0,1],257),([1,-1,0,1],257)],[256,2])]
for index,g in enumerate(fixtures):
 expected=g.exact_layers(1000)
 for devices in ([0],[0,1]):
  folder=r/f'graph{index}-world{len(devices)}';report=run_graph(g,folder,devices=devices,shards=8,max_seconds=60,autotune=False)
  assert report['status']=='COMPLETE' and report['layer_sizes']==list(map(len,expected))
  plans=report.get('rank_plans',[report['plan']]);assert len(plans)==len(devices)
  assert all(p['history_algorithm']=='SORTED_RUNS' and p['table_slots']==0 and p['owner_lanes']==4 and p['sorted_owner_bytes']>0 and p['driver_graph_reserve_bytes']>=128<<20 for p in plans)
  states=json.loads((folder/'states.json').read_text());assert sorted(states['current'])==expected[-1]
  assert sorted(states['previous_small'])==expected[-2]
  checks.append({'graph':index,'world':len(devices),'states':sum(map(len,expected)),'layers':len(expected),'plan_memory_includes_sorted_buffers':True,'hash_table_slots':0,'owner_lanes':4,'terminal_states_exact':True})
  print('PUBLIC_CASE_VERIFIED',index,len(devices),flush=True)
receipt={'status':'VERIFIED_PUBLIC_SORTED_AUTOMATIC_MEMORY_ONE_TWO_GPU','checks':checks,'scope':'Public run_graph, automatic capacity/matched batch, explicit sorted history; measured algorithm/profile choice and fresh installation pending'}
(r/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
