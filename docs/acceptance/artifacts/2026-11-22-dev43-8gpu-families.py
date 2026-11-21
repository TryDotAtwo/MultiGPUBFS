import json,os,subprocess,traceback
from collections import Counter
from pathlib import Path
from cayleypy import PermutationGroups
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
root=Path('/workspace/capacity-eight/results/families-exact-8gpu')
root.mkdir(parents=True,exist_ok=False)
os.environ.update(MGBFS_NATIVE_CACHE='/workspace/capacity-eight/native-cache',
 MGBFS_NATIVE_OFFLINE='1',MGBFS_SEGMENT_BATCH_RECOVERY='0')
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
assert not p.stdout.strip(),'FOREIGN_GPU_PROCESSES_AT_START'
n=5
x=list(range(n));x[0],x[1]=x[1],x[0]
even=list(range(n));odd=list(range(n))
for i in range(0,n-1,2):even[i],even[i+1]=even[i+1],even[i]
for i in range(1,n-1,2):odd[i],odd[i+1]=odd[i+1],odd[i]
cases=[('lrx-repeated',[list(range(1,n))+[0],[n-1]+list(range(n-1)),x],[0,0,1,2,3]),
       ('m2-repeated',[x,even,odd],[0,0,1,2,3])]
assert cases[0][1] == PermutationGroups.lrx(n).generators_permutations
assert cases[1][1] == PermutationGroups.rapaport_m2(n).generators_permutations
def oracle(gens,start):
 seen={tuple(start)};front={tuple(start)};layers=[]
 while front:
  layers.append(Counter(front));nxt={tuple(s[j] for j in g) for s in front for g in gens}-seen
  seen.update(nxt);front=nxt
 return layers
records=[];ids=None
try:
 with persistent_graph_workers() as pool:
  for backend in ('segmented24','permutation_bitmap'):
   for strategy in ('atomic','warp','batch'):
    for name,gens,start in cases:
     layers=oracle(gens,start);graph=GraphDefinition.permutation(gens,start)
     for depth in range(len(layers)):
      out=root/(backend+'-'+strategy+'-'+name+'-'+str(depth))
      r=run_graph(graph,out,devices=list(range(8)),autotune=False,shards=4,_batch=2,
       memory_backend=backend,memory_preset='max_capacity',segment_rows=3,
       segment_strategy=strategy,_profile_layers=depth)
      assert r['status']=='INCOMPLETE' and r['reason']=='PROFILE_LAYER_LIMIT',r
      assert r['layer_sizes']==[sum(x.values()) for x in layers[:depth+1]]
      snap=json.loads((out/'states.json').read_text())
      assert Counter(map(tuple,snap['current']))==layers[depth],(backend,strategy,name,depth)
      if depth:assert Counter(map(tuple,snap['previous_small']))==layers[depth-1]
      cur=[w.process.pid for w in pool.workers]
      if ids is None:ids=cur
      assert cur==ids and len(set(cur))==8
     out=root/(backend+'-'+strategy+'-'+name+'-complete')
     r=run_graph(graph,out,devices=list(range(8)),autotune=False,shards=4,_batch=2,
      memory_backend=backend,memory_preset='max_capacity',segment_rows=3,segment_strategy=strategy)
     assert r['status']=='COMPLETE' and r['layer_sizes']==[sum(x.values()) for x in layers]
     rec={'backend':backend,'strategy':strategy,'case':name,'layers':len(layers),
      'states':sum(sum(x.values()) for x in layers),'rank_pids':ids,
      'all_layer_exact_counters':True,'capacity_per_segment_bank':3,'generators':gens,'start':start}
     records.append(rec);(root/'records.json').write_text(json.dumps(records,indent=2))
     print('PASS',json.dumps(rec),flush=True)
except BaseException:
 (root/'failure.txt').write_text(traceback.format_exc());raise
(root/'complete').write_text('ok\n')