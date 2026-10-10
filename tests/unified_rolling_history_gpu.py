import json,os,subprocess,time,traceback,hashlib
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
r=Path(os.environ.get('MGBFS_GATE_ROOT','/root/universal/rolling-full-graph-gate'));r.mkdir(parents=True,exist_ok=True)
source=Path(os.environ.get('MGBFS_TEST_SOURCE','/root/universal/src'));native_library=Path(os.environ.get('MGBFS_TEST_CUDA_LIBRARY','/root/universal/native-generic/libmgbfs_cuda.so'))
identity={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in [('cli',str(source/'target/release/mgbfs')),('example',str(source/'target/release/examples/generic_distributed_gate')),('cuda',str(native_library))]}
identity_path=r/'native-identity.json'
if identity_path.exists():assert json.loads(identity_path.read_text())==identity,'VALIDATION_ARTIFACT_REUSE_MISMATCH'
else:identity_path.write_text(json.dumps(identity))
env=dict(os.environ);checks=[]
g=GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],[0,255,128,0]);oracle=g.exact_layers(24)
def ranks(g,name,codec,bits,shards,capacity,batch=2,history=3):
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
   jobs.append(subprocess.Popen([str(source/'target/release/examples/generic_distributed_gate'),str(rank),str(out),str(definition),str(bits),str(shards),str(capacity),str(codec),str(batch),json.dumps([0,1<<31,1<<32]),str(history)],env=env,stdout=log,stderr=subprocess.STDOUT))
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

fixtures=[]
for degree in (6,7):
 graph=GraphDefinition.permutation([list(range(1,degree))+[0],[degree-1]+list(range(degree-1)),[1,0]+list(range(2,degree))],list(range(degree)))
 expected=graph.exact_layers(6000);cap=max(map(len,expected));assert sum(map(len,expected))>cap
 fixtures.append((f'inverse-permutation-{degree}',graph,expected,cap))
wide=17;graph=GraphDefinition.permutation([list(range(1,wide))+[0],[wide-1]+list(range(wide-1))],list(range(wide)));fixtures.append(('wide-byte',graph,graph.exact_layers(32),4))
graph=GraphDefinition.matrix(2,1,[([1,1,0,1],257),([1,-1,0,1],257)],[256,2]);fixtures.append(('modular-matrix-cycle',graph,graph.exact_layers(300),3))
graph=GraphDefinition.matrix(2,1,[([-1,0,0,1],0),([1,0,0,-1],0)],[-(1<<40),2]);fixtures.append(('signed-wrap-involutions',graph,graph.exact_layers(8),4))
for name,graph,expected,capacity in fixtures:
 for bits in (0,64):
  for codec in ((1,8) if graph.action['kind']=='permutation' else (8,)):
   parts=ranks(graph,f'{name}-codec{codec}-bits{bits}',codec,bits,4,capacity,min(capacity,32))
   assert all(p['fatal']==0 for p in parts),parts
   actual=[sorted(parts[0]['layers'][i]+parts[1]['layers'][i]) for i in range(len(parts[0]['layers']))];assert actual==expected,name
   assert all(p['global_sizes']==list(map(len,expected)) for p in parts)
   checks.append({'case':name,'codec':codec,'bits':bits,'states':sum(map(len,expected)),'layers':len(expected),'capacity':capacity,'all_states_all_layers_exact':True})
# Failed future allocation leaves previous/current banks exact.
graph=fixtures[0][1];expected=fixtures[0][2];parts=ranks(graph,'rolling-resource',1,0,4,3,2);depth=len(parts[0]['layers']);assert all(p['fatal']&2 for p in parts)
assert sorted(parts[0]['layers'][-1]+parts[1]['layers'][-1])==expected[depth-1]
assert sorted(parts[0]['previous_small']+parts[1]['previous_small'])==expected[depth-2]
# A directed graph remains on the exact all-visited path.
parts=ranks(g,'directed-fallback',1,0,4,24,2,1);assert all(p['fatal']==0 for p in parts);assert [sorted(parts[0]['layers'][i]+parts[1]['layers'][i]) for i in range(len(parts[0]['layers']))]==oracle
receipt={'status':'VERIFIED_ROLLING_RUNTIME_COMPLETE_GRAPH_ORACLES_TWO_GPU','checks':checks,'resource_current_previous_exact':True,'directed_retained_history_exact':True,'scope':'two actual RTX3060; explicit runtime three-bank constructor, public automatic/network dispatch still pending'}
(r/'verification.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
