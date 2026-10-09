import json,os,subprocess,time,traceback
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
r=Path('/root/universal/compact-distributed-gate');r.mkdir(exist_ok=True)
env=dict(os.environ);checks=[]
g=GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],[0,255,128,0]);oracle=g.exact_layers(24)
def ranks(g,name,codec,bits,shards,capacity):
 out=r/name
 if out.exists():
  assert GraphDefinition.from_dict(json.loads((out/'graph.json').read_text())).digest()==g.digest()
  saved=[json.loads((out/f'rank-{rank}.json').read_text()) for rank in range(2)]
  assert all(v['rank']==rank and v['state_bytes']==codec and v['hash_bits']==bits and v['shards']==shards for rank,v in enumerate(saved))
  return saved
 out.mkdir();definition=out/'graph.json';definition.write_text(g.to_json());jobs=[];logs=[]
 try:
  for rank in range(2):
   log=(out/f'rank-{rank}.log').open('w');logs.append(log)
   jobs.append(subprocess.Popen(['/root/universal/src/target/release/examples/generic_distributed_gate',str(rank),str(out),str(definition),str(bits),str(shards),str(capacity),str(codec)],env=env,stdout=log,stderr=subprocess.STDOUT))
  started=time.monotonic()
  while any(p.poll() is None for p in jobs):
   if any(p.poll() not in (None,0) for p in jobs):raise RuntimeError('rank failure '+name)
   if time.monotonic()-started>60:raise RuntimeError('observation deadline; no restart '+name)
   time.sleep(.02)
  assert all(p.returncode==0 for p in jobs)
 finally:
  for p in jobs:
   if p.poll() is None:p.terminate()
  for p in jobs:
   if p.poll() is None:
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:p.kill();p.wait()
  for log in logs:log.close()
 return [json.loads((out/f'rank-{rank}.json').read_text()) for rank in range(2)]
for codec in (1,8):
 for bits in (0,64):
  for shards in (1,4):
   name=f'codec-{codec}-bits-{bits}-shards-{shards}';parts=ranks(g,name,codec,bits,shards,24)
   assert all(p['fatal']==0 and p['state_bytes']==codec for p in parts),parts
   actual=[sorted(parts[0]['layers'][i]+parts[1]['layers'][i]) for i in range(len(parts[0]['layers']))]
   assert actual==oracle,(name,actual,oracle)
   assert all(p['global_sizes']==list(map(len,oracle)) for p in parts)
   checks.append({'case':name,'all_layers_exact':True,'states':sum(map(len,oracle))})
parts=ranks(g,'resource',1,0,4,3);depth=len(parts[0]['layers'])
assert all(p['fatal']&2 for p in parts)
assert sorted(parts[0]['layers'][-1]+parts[1]['layers'][-1])==oracle[depth-1]
assert sorted(parts[0]['previous_small']+parts[1]['previous_small'])==oracle[depth-2]
fixtures=[g,GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],[-1,256,128,0]),GraphDefinition.matrix(2,1,[([1,1,0,1],257)],[256,2])]
for index,g in enumerate(fixtures):
 expected=g.exact_layers(1024)
 for selection in ('single','two-four'):
  out=r/f'public-{index}-{selection}';report=run_graph(g,out,device=0,max_seconds=30,autotune=False) if selection=='single' else run_graph(g,out,devices=[1,0],shards=4,max_seconds=30,autotune=False)
  assert report['status']=='COMPLETE' and report['layer_sizes']==list(map(len,expected)),report
  states=json.loads((out/'states.json').read_text());assert sorted(states['current'])==expected[-1];assert sorted(states['previous_small'])==expected[-2]
  codec=report.get('state_bytes',report.get('plan',{}).get('state_bytes'));assert codec==(1 if index==0 else 8),(codec,report)
  checks.append({'case':f'public-{index}-{selection}','state_bytes':codec,'all_layer_counts_and_terminal_states_exact':True})
receipt={'status':'VERIFIED_COMPACT_STORAGE_ONE_TWO_GPU','checks':checks,'resource_current_previous_exact':True,'scope':'storage codec and exact hash/equality/routing semantics only; throughput and specialized key-first dispatch pending'}
(r/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
