import json,os,subprocess,time,traceback
from collections import Counter
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

root=Path('/workspace/capacity-eight/results/exact-layers-8gpu')
root.mkdir(parents=True,exist_ok=False)
os.environ.update(MGBFS_NATIVE_CACHE='/workspace/capacity-eight/native-cache',
  MGBFS_NATIVE_OFFLINE='1',MGBFS_SEGMENT_BATCH_RECOVERY='0')
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
assert not p.stdout.strip(),'FOREIGN_GPU_PROCESSES_AT_START'
def rot(k):return [(i+k)%6 for i in range(6)]
cases=[('s4',[[1,0,2,3],[0,2,1,3],[0,1,3,2]],[0,1,2,3]),
       ('directed-c6',[rot(1),rot(4)],list(range(6))),
       ('repeated',[[1,0,2,3,4],[0,2,1,3,4],[0,1,3,2,4],[0,1,2,4,3]],[0,0,1,2,3])]
def oracle(gens,start):
 seen={tuple(start)};front={tuple(start)};layers=[]
 while front:
  layers.append(Counter(front))
  nxt={tuple(s[j] for j in g) for s in front for g in gens}-seen
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
      actual=Counter(map(tuple,snap['current']));assert actual==layers[depth],(backend,strategy,name,depth,actual,layers[depth])
      if depth:
       assert Counter(map(tuple,snap['previous_small']))==layers[depth-1]
      cur=[w.process.pid for w in pool.workers]
      if ids is None:ids=cur
      assert cur==ids and len(set(cur))==8
     out=root/(backend+'-'+strategy+'-'+name+'-complete')
     r=run_graph(graph,out,devices=list(range(8)),autotune=False,shards=4,_batch=2,
         memory_backend=backend,memory_preset='max_capacity',segment_rows=3,
         segment_strategy=strategy)
     assert r['status']=='COMPLETE' and r['layer_sizes']==[sum(x.values()) for x in layers]
     rec={'backend':backend,'strategy':strategy,'case':name,'layers':len(layers),
          'states':sum(sum(x.values()) for x in layers),'rank_pids':ids,
          'all_layer_exact_counters':True,'capacity_per_segment_bank':3}
     records.append(rec);(root/'records.json').write_text(json.dumps(records,indent=2))
     print('PASS',json.dumps(rec),flush=True)
except BaseException:
 (root/'failure.txt').write_text(traceback.format_exc());raise
(root/'complete').write_text('ok\n')
