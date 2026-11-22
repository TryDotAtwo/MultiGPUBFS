import gc,hashlib,json,math,os,subprocess,time,traceback
from collections import Counter
from pathlib import Path
from cayleypy import PermutationGroups
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
root=Path('/workspace/capacity-eight/results/repeated-large-8gpu')
root.mkdir(parents=True,exist_ok=False)
os.environ.update(MGBFS_NATIVE_CACHE='/workspace/capacity-eight/native-cache',
 MGBFS_NATIVE_OFFLINE='1',MGBFS_SEGMENT_ROWS='1048576',
 MGBFS_SEGMENT_OWNER_LANES='4',MGBFS_SEGMENT_EXACT_EXCHANGE='1',
 MGBFS_SEGMENT_BATCH_RECOVERY='0')
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
assert not p.stdout.strip(),'FOREIGN_GPU_PROCESSES_AT_START'
n=10;start=(0,0,1,2,3,4,5,6,7,8)
cases=[('lrx-repeated',PermutationGroups.lrx(n).generators_permutations),
       ('m2-repeated',PermutationGroups.rapaport_m2(n).generators_permutations)]
def oracle(gens):
 seen={start};front={start};counts=[];prior=set();t=time.monotonic()
 while front:
  counts.append(len(front))
  nxt={tuple(s[j] for j in g) for s in front for g in gens}-seen
  seen.update(nxt);prior,front=front,nxt
 print('ORACLE',len(seen),'states',len(counts),'layers',round(time.monotonic()-t,2),'s',flush=True)
 last=prior if len(prior)<=1000 else None
 del seen,front,prior,nxt;gc.collect()
 return counts,last
records=[];ids=None
try:
 with persistent_graph_workers() as pool:
  for name,gens in cases:
   print('START_ORACLE',name,flush=True);counts,last=oracle(gens)
   assert sum(counts)==math.factorial(n)//2,(name,sum(counts))
   graph=GraphDefinition.permutation(gens,list(start))
   (root/(name+'-oracle.json')).write_text(json.dumps({'layers':counts,'start':start,'generators':gens}))
   for backend in ('segmented24','permutation_bitmap'):
    for strategy in ('atomic','warp','batch'):
     label=name+'-'+backend+'-'+strategy;print('START',label,flush=True)
     t=time.monotonic()
     r=run_graph(graph,root/label,devices=list(range(8)),autotune=False,shards=8,_batch=8192,
      memory_backend=backend,memory_preset='balanced',segment_rows=2**32,segment_strategy=strategy)
     wall=time.monotonic()-t
     assert r['status']=='COMPLETE' and r['layer_sizes']==counts,(label,r['status'],r['reason'])
     actual=json.loads((root/label/'states.json').read_text())['current']
     full=last is not None
     if full:assert Counter(map(tuple,actual))==Counter(last)
     now=[w.process.pid for w in pool.workers]
     if ids is None:ids=now
     assert ids==now and len(set(now))==8
     rec={'name':label,'backend':backend,'strategy':strategy,'states':sum(counts),'layers':len(counts),
      'all_layer_sizes_match_independent_cpu_bfs':True,'last_layer_full_counter_checked':full,
      'bfs_seconds':r['bfs_seconds'],'wall_seconds':wall,'rank_pids':now,
      'rank_memory':r['segment_memory_by_rank'],'runtime_identity':r.get('runtime_identity')}
     records.append(rec);(root/'records.json').write_text(json.dumps(records,indent=2))
     print('PASS',label,'BFS',r['bfs_seconds'],'s',flush=True)
except BaseException:
 (root/'failure.txt').write_text(traceback.format_exc());raise
(root/'complete').write_text('ok\n')