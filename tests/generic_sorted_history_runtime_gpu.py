import json,os,subprocess,time,hashlib
from pathlib import Path
from multigpubfs import GraphDefinition
r=Path(os.environ.get('MGBFS_SORTED_HISTORY_GATE_ROOT','/tmp/mgbfs-sorted-history-gate'));r.mkdir(parents=True,exist_ok=False)
env=dict(os.environ);env['CUDA_MODULE_LOADING']='EAGER';env['MGBFS_GENERIC_HISTORY']='sorted';env['MGBFS_GENERIC_SORT']='none'
binary=Path(os.environ.get('MGBFS_SORTED_HISTORY_GATE_EXAMPLE','target/release/examples/generic_distributed_gate')).resolve()
checks=[]
def run(g,name,codec,bits,shards,cap,batch,history,transport='full',resource=False):
 out=r/name;out.mkdir();definition=out/'graph.json';definition.write_text(g.to_json());expected=g.exact_layers(6000)
 local=dict(env);local['MGBFS_GENERIC_TRANSPORT']=transport;jobs=[];logs=[]
 try:
  for rank in range(2):
   f=(out/f'rank-{rank}.log').open('w');logs.append(f)
   jobs.append(subprocess.Popen([str(binary),str(rank),str(out),str(definition),str(bits),str(shards),str(cap),str(codec),str(batch),json.dumps([0,1<<31,1<<32]),str(history)],env=local,stdout=f,stderr=subprocess.STDOUT))
  started=time.monotonic()
  while any(p.poll() is None for p in jobs):
   if any(p.poll() not in (None,0) for p in jobs):raise RuntimeError('actual rank failure '+name)
   if time.monotonic()-started>90:raise RuntimeError('test observation deadline; explicitly aborting fixture, no restart '+name)
   time.sleep(.1)
  assert all(p.returncode==0 for p in jobs)
 finally:
  for p in jobs:
   if p.poll() is None:p.terminate()
  for p in jobs:
   if p.poll() is None:
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:p.kill();p.wait()
  for f in logs:f.close()
 parts=[json.loads((out/f'rank-{i}.json').read_text()) for i in range(2)]
 assert all(p['history_algorithm']=='SORTED_RUNS' and p['memory_plan']['table_slots']==0 and p['memory_plan']['sorted_owner_bytes']>0 and p['owner_lanes']==min(shards,4) for p in parts)
 assert len(parts[0]['layers'])==len(parts[1]['layers'])
 actual=[sorted(parts[0]['layers'][i]+parts[1]['layers'][i]) for i in range(len(parts[0]['layers']))]
 assert actual==expected[:len(actual)],(name,'exact layer mismatch')
 if resource:
  assert all(p['fatal']&2 for p in parts),(name,'expected capacity resource',[p['fatal'] for p in parts])
  assert len(actual)>=2
  assert sorted(parts[0]['previous_small']+parts[1]['previous_small'])==expected[len(actual)-2]
 else:assert all(p['fatal']==0 for p in parts) and len(actual)==len(expected),(name,'not COMPLETE',[{'fatal':p['fatal'],'layers':len(p['layers'])} for p in parts])
 checks.append({'name':name,'states':sum(map(len,actual)),'layers':len(actual),'codec':codec,'bits':bits,'shards':shards,'history':history,'transport':transport,'resource':resource,'all_completed_layers_exact':True})
 print('CASE_VERIFIED',name,flush=True)
g=GraphDefinition.permutation([[1,2,3,0],[3,0,1,2],[1,0,2,3]],list(range(4)))
for bits,codec,shards in [(64,1,1),(0,1,8),(64,8,32)]:run(g,f'rolling-c{codec}-h{bits}-s{shards}',codec,bits,shards,24,2,3)
d=GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],[0,255,128,0]);run(d,'directed-packed',1,0,8,24,2,1)
w=25;g=GraphDefinition.permutation([list(range(1,w))+[0],[w-1]+list(range(w-1))],list(range(w)))
run(g,'wide-full',1,64,8,4,2,3,'full');run(g,'wide-parent',1,0,8,4,2,3,'parent');run(g,'wide-int64-parent',8,64,8,4,2,3,'parent')
g=GraphDefinition.matrix(2,1,[([1,1,0,1],257),([1,-1,0,1],257)],[256,2]);run(g,'matrix-parent',8,64,8,3,2,3,'parent')
g=GraphDefinition.matrix(2,1,[([-1,0,0,1],0),([1,0,0,-1],0)],[-(1<<40),2]);run(g,'signed-full',8,0,8,4,2,3)
n=6;g=GraphDefinition.permutation([list(range(1,n))+[0],[n-1]+list(range(n-1)),[1,0]+list(range(2,n))],list(range(n)));run(g,'incomplete-preserve',1,0,8,3,2,3,resource=True)
receipt={'status':'VERIFIED_SORTED_MAIN_BFS_TWO_GPU_ORACLES','checks':checks,'scope':'Explicit capacities, two RTX3060, sorted backend opt-in; unified AUTO admission and throughput pending','binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'cuda_sha256':hashlib.sha256((Path(os.environ['MGBFS_CUDA_LIB_DIR'])/'libmgbfs_cuda.so').read_bytes()).hexdigest()}
(r/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
